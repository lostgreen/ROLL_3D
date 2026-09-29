"""Real per-group backend smoke: follow returned IDs through original providers."""
import argparse
import json
from pathlib import Path
import time

from .env import ReconstructionEnv, SUCCESS
from .tool_guidance import native_example


CODE = ('import bpy\n'
        'bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.25, depth=0.8, location=(0,0,0.4))\n'
        'bpy.context.object.name="ProbeGeometry"\n'
        'mat=bpy.data.materials.new("ProbeMaterial"); mat.use_nodes=True\n'
        'mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(0.05,0.4,0.8,1)\n'
        'bpy.context.object.data.materials.append(mat)')


def generation_arguments(schema, reference_image):
    properties = schema['properties']
    texture = properties['texture'].get('default', True)
    allowed = properties['texture'].get('enum')
    if allowed and texture not in allowed:
        texture = allowed[0]
    presets = properties['quality_preset']['enum']
    preset = properties['quality_preset'].get('default', presets[0])
    if preset not in presets:
        preset = presets[0]
    return {'reference_image': reference_image, 'texture': texture,
            'quality_preset': preset, 'seed': 7}


def catalog_probe_query(task):
    """Derive a deterministic smoke query from a real importable catalog entry."""
    if task.get('probe', {}).get('asset_query'):
        return task['probe']['asset_query']
    from tools.retrieval import RetrievalProvider
    provider = RetrievalProvider(task['asset_index'], None)
    for item in provider.items:
        public = provider._public(item)
        if public['payload_available'] and public['format'] in ('glb', 'gltf'):
            return public['name'] or public['asset_id']
    raise ValueError('No catalog entry has an available GLB/GLTF payload')


def run_probe(task, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {'status': 'running', 'model_inference_tested': False, 'steps': [],
               'verified': [], 'probe_result': str(output / 'probe_result.json')}
    env = None
    try:
        env = ReconstructionEnv(task, output / 'episodes', max_steps=24)
        summary['group'] = env.task['group']
        env.reset(seed=7)
        summary['episode'] = str(env.episode)

        def call(name, arguments):
            before = time.monotonic()
            _, _, done, truncated, info = env.step(native_example(name, arguments))
            result = json.loads((env.episode / f'step_{env.step_count:03d}' / 'tool_result.json').read_text())
            summary['steps'].append({'step': env.step_count, 'tool': name,
                'status': result['status'], 'error_type': result.get('error_type'),
                'seconds': round(time.monotonic() - before, 3),
                'terminal_reason': info.get('termination_reason')})
            if result['status'] not in SUCCESS or truncated or (done and name != 'finish'):
                raise RuntimeError(f'Backend probe failed at {name}: {result.get("error_type") or info.get("termination_reason") or result["status"]}')
            data = result.get('structured')
            if not isinstance(data, dict) or not data:
                try:
                    data = json.loads(result.get('text', '').removeprefix('Code executed successfully: ').strip())
                except ValueError:
                    data = {}
            if result.get('artifact_ids'):
                data.setdefault('artifact_id', result['artifact_ids'][0])
            return data

        def import_and_place(artifact_id, location):
            imported = call('scene.import_artifact', {'artifact_id': artifact_id})
            root = imported['root_name']
            call('scene.set_transform', {'object_name': root, 'location': location,
                 'rotation': [0, 0, 0], 'scale': [0.5, 0.5, 0.5]})
            scene = call('scene.list_objects', {})
            if not any(obj.get('name') == root for obj in scene.get('objects', [])):
                raise RuntimeError('Imported root missing from scene.list_objects')
            call('render_scene_view', {'view': 'camera', 'camera': env.task['observation_camera'],
                 'lighting': 'scene', 'max_size': env.task.get('image_size', 640)})
            return root

        ref = env.task['references'][0].get('id', 'ref:0')
        crop = call('reference.crop', {'reference_image': ref,
                    'bbox': env.task.get('probe', {}).get('bbox', [0, 0, 1, 1])})
        summary['verified'].append('reference.crop')
        if env.task['group'] in ('blender', 'mixed'):
            call('execute_blender_code', {'code': CODE})
            scene = call('scene.list_objects', {})
            if not any(obj.get('name') == 'ProbeGeometry' for obj in scene.get('objects', [])):
                raise RuntimeError('Code probe did not create expected geometry')
            summary['verified'].append('blender_geometry_and_material')
        if env.task['group'] in ('hunyuan', 'mixed'):
            spec = env.session.registry.lookup('asset.generate_3d_from_image')[0]
            args = generation_arguments(spec.input_schema, crop['reference_image'])
            summary['generation_arguments'] = {k: v for k, v in args.items() if k != 'reference_image'}
            generated = call('asset.generate_3d_from_image', args)
            import_and_place(generated['artifact_id'], [0.5, 0, 0])
            summary['verified'].append('crop_generate_import_transform_render')
        if env.task['group'] in ('assets', 'mixed'):
            query = catalog_probe_query(env.task)
            candidates = call('asset.search', {'query': query, 'limit': 50})['assets']
            usable = [x for x in candidates if x.get('payload_available') and x.get('format') in ('glb', 'gltf')]
            if not usable:
                raise RuntimeError('Search returned no available importable GLB/GLTF candidate')
            asset_id = usable[0]['asset_id']
            call('asset.inspect', {'asset_id': asset_id})
            artifact = call('asset.import', {'asset_id': asset_id})
            import_and_place(artifact['artifact_id'], [-0.5, 0, 0])
            summary['verified'].append('search_inspect_stage_import_transform_render')
        call('finish', {'summary': 'Deterministic backend wiring probe; not a model reconstruction.'})
        summary['status'] = 'passed'
    except Exception as exc:
        summary['status'] = 'failed'
        summary['error'] = {'type': type(exc).__name__, 'message': str(exc)[:400]}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                summary['status'] = 'failed'
                summary['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)[:200]}
        summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'probe_result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_probe(args.task, args.output)
    compact = {key: result.get(key) for key in ('status', 'group', 'probe_result', 'elapsed_seconds', 'verified')}
    compact['error_type'] = result.get('error', {}).get('type')
    print(json.dumps(compact))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
