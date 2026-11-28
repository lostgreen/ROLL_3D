"""CPU-only fixed-camera final-scene previews; never save or fit model geometry."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

POLICY = 'eval30_fixed_camera_cutaway_v1'
VIEW_NAMES = ('public_0', 'public_1', 'hidden_0', 'hidden_1')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_views(root, row):
    """Only frozen GT camera metadata; public matrices come from the task."""
    if row['subset'] == 'test20':
        gt = read(root / 'ground_truth/scenes' / row['id'] / 'status.json')
        assert gt.get('status') == 'complete', 'GT view metadata is incomplete'
        views = gt.get('views4', gt.get('views', []))
    else:
        views = row['public_views'] + row['private_views']
    by_name = {v['name']: v for v in views}
    assert all(name in by_name for name in VIEW_NAMES), 'Four fixed views required'
    task = read(root / 'public' / row['id'] / 'task.json')
    protocol = task['public_camera_protocol']
    cameras = {v['name']: v for v in protocol['cameras']}
    result = []
    for i, name in enumerate(VIEW_NAMES):
        view = dict(by_name[name])
        if name.startswith('public_'):
            camera = cameras['PublicCamera' + name[-1]]
            view.update(matrix_world=camera['matrix_world'], lens_mm=camera['lens_mm'],
                        sensor_width_mm=camera['sensor_width_mm'], sensor_fit=camera['sensor_fit'])
        assert all(math.isfinite(float(x)) for key in ('camera_location', 'target') for x in view[key])
        assert math.isfinite(float(view['lens_mm'])) and 1 < float(view['lens_mm']) < 1000
        result.append(view)
    assert protocol['resolution'] == [640, 480], 'Unexpected frozen resolution'
    return result


def cutaway_functions(cfg):
    # Execute only these unchanged legacy functions, not its old runner imports.
    path = Path(cfg['repo']) / 'artifacts/indoor_qwen_200rollout_20261121/source/cutaway_preview.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {'bounds', 'candidates', 'camera_transparent_material', 'apply_cutaway'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names, 'Legacy cutaway functions missing'
    namespace = {'re': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['apply_cutaway'], path


def geometry_digest(scene):
    """Vertex/topology/transform fingerprint, excluding display material slots."""
    h = hashlib.sha256()
    for obj in sorted((o for o in scene.objects if o.type == 'MESH'), key=lambda o: o.name):
        h.update(obj.name.encode())
        h.update(repr(tuple(tuple(r) for r in obj.matrix_world)).encode())
        for vertex in obj.data.vertices:
            h.update(repr(tuple(vertex.co)).encode())
        for poly in obj.data.polygons:
            h.update(repr(tuple(poly.vertices)).encode())
    return h.hexdigest()


def set_camera(scene, view):
    import bpy
    from mathutils import Matrix, Vector
    data = bpy.data.cameras.new('FinalEvaluationCamera')
    camera = bpy.data.objects.new('FinalEvaluationCamera', data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    if 'matrix_world' in view:
        camera.matrix_world = Matrix(view['matrix_world'])
    else:
        camera.location = view['camera_location']
        camera.rotation_euler = (Vector(view['target']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    data.lens = float(view['lens_mm'])
    data.sensor_width = float(view.get('sensor_width_mm', 36.))
    data.sensor_fit = view.get('sensor_fit', 'AUTO')
    data.clip_start, data.clip_end = .05, 200.
    return camera


def render(root, variant, sid, threads):
    import bpy
    cfg = read(root / 'config.json')
    row = next(r for r in cfg['rows'] if r['id'] == sid)
    state = read(root / 'runs' / variant / sid / 'status.json')
    assert state['state'] == 'complete', 'Only complete scenes are rendered'
    source = Path(state['scene_blend'])
    assert source.is_file(), 'Final scene is missing'
    views = load_views(root, row)
    apply_cutaway, legacy = cutaway_functions(cfg)
    source_hash = sha(source)
    out = root / 'exports' / variant / sid
    out.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for view in views:
        # Restore the exact submitted scene for every view and before raw preview.
        bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False, load_ui=False)
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 12
        scene.cycles.use_denoising = True
        scene.cycles.seed = 20261123
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = threads
        scene.render.resolution_x, scene.render.resolution_y = 640, 480
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.
        scene.render.use_border = False
        scene.render.film_transparent = False
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.quality = 90
        camera = set_camera(scene, view)
        bpy.context.view_layer.update()
        before = geometry_digest(scene)
        if view['name'] == 'public_0':
            raw = out / 'raw_public_0.jpg'
            scene.render.filepath = str(raw)
            bpy.ops.render.render(write_still=True)
            records.append(dict(name='raw_public_0', file=raw.name, sha256=sha(raw), cutaway=False))
        audit = apply_cutaway(scene, view)
        bpy.context.view_layer.update()
        assert geometry_digest(scene) == before, 'Cutaway changed vertices, topology or transforms'
        path = out / (view['name'] + '.jpg')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        records.append(dict(name=view['name'], file=path.name, sha256=sha(path), cutaway=True,
                            camera_matrix_world=[list(r) for r in camera.matrix_world], lens_mm=camera.data.lens,
                            cutaway_objects=audit))
    assert sha(source) == source_hash, 'Submitted blend changed'
    dump(out / 'receipt.json', dict(state='rendered', policy=POLICY, variant=variant, id=sid, subset=row['subset'],
         source_path=str(source), source_sha256=source_hash, source_unchanged=True,
         geometry_unchanged=True, original_blend_saved=False, api_calls=0, device='CPU', threads=threads,
         views=records, seconds=time.monotonic() - started, fixed_views=views,
         cutaway_source=str(legacy), cutaway_source_sha256=sha(legacy), render_source_sha256=sha(__file__),
         display_policy='legacy near-side shell camera-only cutaway; raw_public_0 preserves submitted visibility',
         quality_evaluated=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=['base', 'short_cot_step303'], required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--threads', type=int, choices=[2, 3, 4], default=4)
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    try:
        render(args.root.resolve(), args.variant, args.id, args.threads)
        print('EVAL30_FIXED_RENDER_COMPLETE', flush=True)
    except BaseException as error:
        out = args.root / 'exports' / args.variant / args.id
        out.mkdir(parents=True, exist_ok=True)
        (out / 'render_traceback.log').write_text(traceback.format_exc())
        dump(out / 'render_error.json', dict(error_type=type(error).__name__, message=str(error)[:300]))
        raise


if __name__ == '__main__':
    main()
