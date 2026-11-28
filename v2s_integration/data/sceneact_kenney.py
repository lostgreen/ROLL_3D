"""Prepare a frozen Kenney pool and SceneActBench room tasks using CPU Blender.

Original data is read-only. Cameras/references are rendered from canonical
objects and metric layout GT, rather than assuming old render/meta filenames
and normalized camera frames match the current geometry. Run with --help.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_path(room, record):
    # The current release anonymizes filenames; src_glb is provenance only.
    anonymized = Path(room) / 'canonical' / (record['anon_id'] + '.glb')
    return anonymized if anonymized.is_file() else Path(room) / 'canonical' / record['src_glb']


def furniture_aliases(name):
    """Describe catalog semantics, independent of a particular target room."""
    lower = name.lower()
    aliases = set(re.findall('[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()))
    for token, values in {'bookcase': ['bookshelf', 'shelf', 'book shelf'],
                          'sofa': ['couch', 'loveseat', 'multi seat sofa'],
                          'lamp': ['lighting', 'light'],
                          'ceiling': ['pendant lamp', 'ceiling lamp'],
                          'sidetable': ['side table', 'end table', 'corner table'],
                          'cabinet': ['cupboard', 'storage cabinet'],
                          'lounge': ['living room'],
                          'coffeetable': ['coffee table'],
                          'dresser': ['drawer cabinet']}.items():
        if token in lower:
            aliases.update(values)
    return sorted(aliases)


def select_rooms(dataset, room_types):
    selected = []
    for room_type in room_types:
        candidates = []
        for folder in sorted((Path(dataset) / 'scenes').iterdir()):
            if not folder.is_dir() or '_' + room_type + '-' not in folder.name:
                continue
            gt = json.loads((folder / 'gt.json').read_text())
            objects = gt.get('objects', [])
            if objects and all(canonical_path(folder, x).is_file() for x in objects):
                candidates.append((len(objects), folder.name, folder))
        if not candidates:
            raise ValueError('No complete canonical/layout scene for ' + room_type)
        selected.append(str(min(candidates)[2]))
    return selected


def blender_worker(spec):
    import bpy
    from mathutils import Vector
    out = Path(spec['output'])
    started = time.monotonic()
    def bounds(objects):
        bpy.context.view_layer.update()
        points = [o.matrix_world @ Vector(c) for o in objects if o.type == 'MESH' for c in o.bound_box]
        if not points:
            raise ValueError('No mesh bounds')
        lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
        hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
        return lo, hi

    def clear():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        s = bpy.context.scene
        s.render.engine = 'CYCLES'
        s.cycles.device = 'CPU'
        s.cycles.samples = 8
        s.render.resolution_percentage = 100
        s.render.image_settings.file_format = 'PNG'
        s.render.image_settings.color_mode = 'RGB'
        s.world = bpy.data.worlds.new('NeutralWorld')
        s.world.use_nodes = True
        s.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.65, 0.65, 0.65, 1)
        s.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.8
        s.view_settings.view_transform = 'Standard'
        s.view_settings.look = 'None'
        return s

    def lighting(center, radius):
        for i, offset in enumerate(((1, -2, 3), (-2, -1, 2), (0, 2, 3))):
            data = bpy.data.lights.new('PublicLight' + str(i), 'AREA')
            data.energy, data.shape, data.size = 90 * radius**2, 'DISK', 2 * radius
            obj = bpy.data.objects.new(data.name, data)
            bpy.context.scene.collection.objects.link(obj)
            obj.location = center + Vector(offset) * radius
            obj.rotation_euler = (center - obj.location).to_track_quat('-Z', 'Y').to_euler()

    def camera(name, center, radius, offset, size):
        data = bpy.data.cameras.new(name)
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = center + Vector(offset).normalized() * radius * 3
        obj.rotation_euler = (center - obj.location).to_track_quat('-Z', 'Y').to_euler()
        data.type, data.ortho_scale = 'ORTHO', radius * 2.6
        data.clip_end = max(100, radius * 10)
        bpy.context.scene.render.resolution_x = size
        bpy.context.scene.render.resolution_y = size
        bpy.context.view_layer.update()
        return obj

    def render(cam, path):
        s = bpy.context.scene
        s.camera, s.render.filepath = cam, str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)

    raw = json.loads(Path(spec['kenney_index']).read_text())
    items = raw.get('assets', raw.get('items', []))
    prepared = []
    for i, item in enumerate(items):
        asset_id = str(item.get('asset_id', item.get('id')))
        source = Path(item.get('glb_path', item.get('path', '')))
        if not source.is_absolute():
            source = Path(spec['kenney_index']).parent / source
        digest = sha(source)
        if item.get('sha256') and item['sha256'] != digest:
            raise ValueError('Asset source changed: ' + asset_id)
        preview_dir = out / 'assets/previews' / asset_id
        metadata_file = preview_dir / 'metadata.json'
        if metadata_file.is_file():
            cached = json.loads(metadata_file.read_text())
            if cached.get('source_sha256') == digest and all((out / 'assets' / v['path']).is_file() and sha(out / 'assets' / v['path']) == v['sha256'] for v in cached['preview']['views']):
                entry = cached['item']
                entry['aliases'] = sorted(set(entry.get('aliases', [])) | set(furniture_aliases(entry['name'])))
                prepared.append(entry)
                continue
        clear()
        bpy.ops.import_scene.gltf(filepath=str(source))
        lo, hi = bounds(list(bpy.context.scene.objects))
        center, extent = (lo + hi) / 2, hi - lo
        radius = max(extent.length / 2, 0.05)
        lighting(center, radius)
        views = []
        for label, offset in [('front three-quarter', (1, -2, 1.2)), ('back three-quarter', (-1, 2, 1.2))]:
            name = 'front' if label.startswith('front') else 'back'
            path = preview_dir / (name + '.png')
            render(camera('PreviewCamera_' + name, center, radius, offset, 192), path)
            views.append({'path': str(path.relative_to(out / 'assets')), 'sha256': sha(path), 'label': label})
        entry = dict(item)
        entry.pop('metadata_path', None)
        entry['glb_path'], entry['sha256'] = str(source), digest
        entry['name'] = item.get('name', source.stem)
        entry['aliases'] = sorted(set(entry.get('aliases', [])) | set(furniture_aliases(entry['name'])))
        entry['dimensions'] = list(extent)
        entry['dimensions_info'] = {'axis_order': 'Blender X/Y/Z after glTF import', 'units': 'native GLB unit; physical size not independently calibrated'}
        entry['metadata'] = {**item.get('metadata', {}), 'dimensions': list(extent), 'dimensions_info': entry['dimensions_info']}
        entry['preview'] = {'source_sha256': digest, 'style': 'CPU Cycles, neutral lighting, individually fitted; no target scene', 'views': views}
        license_record = dict(entry.get('license', {}))
        license_record.setdefault('name', 'CC0-1.0')
        entry['license'] = license_record
        dump(metadata_file, {'source_sha256': digest, 'preview': entry['preview'], 'item': entry})
        prepared.append(entry)
        dump(out / 'prepare_status.json', {'stage': 'asset_previews', 'done': i + 1, 'total': len(items), 'seconds': time.monotonic() - started})
    index = out / 'assets/index.json'
    dump(index, {'pack_id': 'kenney_furniture_140_frozen', 'source_index': spec['kenney_index'], 'source_index_sha256': sha(spec['kenney_index']), 'assets': prepared})

    tasks = []
    for room_path in spec['rooms']:
        room = Path(room_path)
        gt = json.loads((room / 'gt.json').read_text())
        case = out / 'cases' / room.name
        private = case / 'private'
        public = case / 'public'
        clear()
        instance_records = []
        for record in gt['objects']:
            before = set(bpy.data.objects)
            path = canonical_path(room, record)
            bpy.ops.import_scene.gltf(filepath=str(path))
            imported = [o for o in bpy.data.objects if o not in before]
            lo, hi = bounds(imported)
            center, extent = (lo + hi) / 2, hi - lo
            root = bpy.data.objects.new('GT_' + record['anon_id'], None)
            bpy.context.scene.collection.objects.link(root)
            root.location = center
            bpy.context.view_layer.update()
            for obj in imported:
                if obj.parent not in imported:
                    world = obj.matrix_world.copy()
                    obj.parent, obj.matrix_world = root, world
            desired = record.get('canonical_extent', [v * 2 for v in record['size']])
            desired = Vector((desired[0], desired[2], desired[1]))
            root.scale = [desired[i] / extent[i] if extent[i] > 1e-8 else 1 for i in range(3)]
            loc = record['location']
            root.location = (loc[0], -loc[2], loc[1])
            root.rotation_euler = (0, 0, record['rotation'])
            root['gt_category'], root['gt_instance'] = record['category'], record['anon_id']
            lo, hi = bounds(imported)
            instance_records.append({'id': record['anon_id'], 'category': record['category'], 'center': list(root.location), 'rotation_z': record['rotation'], 'extent_before_rotation': list(desired), 'world_bounds': [list(lo), list(hi)], 'canonical_source': str(path)})
        lo, hi = bounds(list(bpy.context.scene.objects))
        center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.1)
        lighting(center, radius)
        protocol = {'world': 'right-handed Blender X/Y/Z, Z up, meters; dataset (x,y,z) maps to (x,-z,y)', 'projection': 'orthographic', 'resolution': [384, 384], 'cameras': []}
        refs = []
        for i, offset in enumerate(((1, -2, 1.4), (-2, -1, 1.4), (1, 2, 1.4), (-1, 2, 2))):
            cam = camera('PublicCamera' + str(i) if i < 3 else 'HiddenCamera', center, radius, offset, 384)
            path = (public if i < 3 else private) / f'view_{i:02d}.png'
            render(cam, path)
            if i < 3:
                protocol['cameras'].append({'reference_id': 'ref:' + str(i), 'name': cam.name, 'matrix_world': [list(r) for r in cam.matrix_world], 'ortho_scale': cam.data.ortho_scale})
                refs.append({'id': 'ref:' + str(i), 'image': str(path), 'timestamp': i, 'kind': 'static multiview; derived reference rendered from canonical/layout GT'})
        private.mkdir(parents=True, exist_ok=True)
        dump(private / 'instances.json', instance_records)
        dump(private / 'provenance.json', {'source_scene': str(room), 'gt_sha256': sha(room / 'gt.json'), 'source_gt': gt, 'adapter': 'canonical bbox-centered; canonical_extent scaled; metric Y-up to Blender Z-up; yaw around Z', 'input_protocol': 'new CPU renders from reconstructed source GT; not original render/*.png'})
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(private / 'gt.blend'))
        for obj in list(bpy.data.objects):
            if obj.type not in {'CAMERA', 'LIGHT'} or obj.name == 'HiddenCamera':
                bpy.data.objects.remove(obj, do_unlink=True)
        # Deleting objects alone leaves answer mesh datablocks in the file.
        # Public initial files should contain no recoverable target geometry.
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        if len(bpy.data.meshes) or len(bpy.data.curves):
            raise RuntimeError('Private geometry remained in public initial file')
        bpy.context.scene.camera = bpy.data.objects['PublicCamera0']
        bpy.ops.wm.save_as_mainfile(filepath=str(public / 'initial.blend'))
        dump(public / 'cameras.json', protocol)
        manifest = {'id': 'sceneact_kenney_' + room.name, 'group': 'assets', 'blender_bin': spec['blender'], 'initial_blend': str(public / 'initial.blend'), 'observation_camera': 'PublicCamera0', 'references': refs, 'public_camera_protocol': protocol, 'instruction': 'Reconstruct the visible furniture and arrangement using the local Kenney furniture candidates. The library has a stylized appearance and may only approximate the reference furniture. Inspect candidate previews, import suitable assets, and place/scale the returned roots. Assess what is visible in the images; do not assume an exact asset match. Preserve the supplied cameras/lights.', 'asset_index': str(index), 'image_size': 384, 'budget': {'max_wall_seconds': 3600}}
        dump(case / 'task.json', manifest)
        tasks.append({'task': str(case / 'task.json'), 'scene_id': room.name, 'objects': len(instance_records), 'categories': [x['category'] for x in instance_records], 'public_reference_count': len(refs), 'private_gt': str(private / 'gt.blend')})
        dump(out / 'prepare_status.json', {'stage': 'rooms', 'done': len(tasks), 'seconds': time.monotonic() - started})
    dump(out / 'prepared.json', {'assets': len(prepared), 'previews': sum(len(x['preview']['views']) for x in prepared), 'index': str(index), 'tasks': tasks, 'seconds': time.monotonic() - started, 'scope': '140 Kenney files; selected source scenes; raw source data unchanged'})
    dump(out / 'prepare_status.json', {'stage': 'complete', 'assets': len(prepared), 'rooms': len(tasks), 'seconds': time.monotonic() - started})
    print('PREPARED ' + json.dumps({'assets': len(prepared), 'rooms': len(tasks), 'seconds': time.monotonic() - started}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--kenney-index')
    parser.add_argument('--dataset')
    parser.add_argument('--blender')
    parser.add_argument('--output')
    parser.add_argument('--room-types', nargs='+', default=['Bedroom', 'DiningRoom', 'LivingRoom'])
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None
    args = parser.parse_args(argv)
    if args.worker:
        blender_worker(json.loads(Path(args.worker).read_text()))
        return
    if not all((args.kenney_index, args.dataset, args.blender, args.output)):
        parser.error('--kenney-index, --dataset, --blender, --output are required')
    out = Path(args.output).resolve()
    spec = {'kenney_index': str(Path(args.kenney_index).resolve()), 'rooms': select_rooms(args.dataset, args.room_types), 'output': str(out), 'blender': args.blender}
    dump(out / 'prepare_spec.json', spec)
    log = out / 'prepare_blender.log'
    with log.open('w') as output:
        done = subprocess.run([args.blender, '--background', '--threads', '8', '--factory-startup', '--python', str(Path(__file__).resolve()), '--', '--worker', str(out / 'prepare_spec.json')], stdout=output, stderr=subprocess.STDOUT)
    status_path = out / 'prepare_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if done.returncode or status.get('stage') != 'complete':
        dump(status_path, {'stage': 'failed', 'blender_exit': done.returncode, 'previous_status': status, 'log': str(log)})
        raise RuntimeError('Blender preparation failed; inspect filtered errors in ' + str(log))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
