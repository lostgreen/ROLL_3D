"""Prepare public, geometry-free starts inside Blender; never load source GT.

blender --background --factory-startup --python prepare_inputs.py -- --root ROOT
Only camera metadata and the two explicitly declared public JPEGs are consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector3(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + ': expected three finite numbers')
    values = [float(x) for x in value]
    if not all(math.isfinite(x) for x in values):
        raise ValueError(label + ': nonfinite value')
    return values


def validate_rows(root, config):
    rows = config['rows']
    ids = []
    for row in rows:
        ident = row['id']
        if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', ident):
            raise ValueError('Unsafe public scene ID')
        ids.append(ident)
        dest = root / 'public' / ident
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise FileExistsError('Refusing to overwrite nonempty output: ' + str(dest))
        refs, views = row['reference_files'], row['public_views']
        if len(refs) != 2 or len(views) != 2:
            raise ValueError('Exactly two public references and camera metadata records are required')
        for i, (ref, view) in enumerate(zip(refs, views)):
            source = Path(ref)
            if not source.is_absolute() or not source.is_file():
                raise FileNotFoundError('Declared public reference unavailable')
            if source.name != f'public_{i}.jpg' or view.get('name') != f'public_{i}':
                raise ValueError('Only the two declared public gallery views may be copied')
            if source.read_bytes()[:2] != b'\xff\xd8':
                raise ValueError('Declared public reference is not JPEG')
            location = vector3(view['camera_location'], 'camera_location')
            target = vector3(view['target'], 'target')
            if sum((a-b)**2 for a, b in zip(location, target)) <= 1e-12:
                raise ValueError('Camera coincides with target')
            lens = float(view['lens_mm'])
            if not math.isfinite(lens) or not 1 < lens < 1000:
                raise ValueError('Invalid public camera lens')
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Scene IDs must be nonempty and unique')
    return rows


def geometry_audit(bpy):
    objects = list(bpy.data.objects)
    cameras = [o for o in objects if o.type == 'CAMERA']
    lights = [o for o in objects if o.type == 'LIGHT']
    counts = dict(objects=len(objects), cameras=len(cameras), lights=len(lights),
                  meshes=len(bpy.data.meshes), materials=len(bpy.data.materials),
                  images=len(bpy.data.images), curves=len(bpy.data.curves),
                  other_objects=sum(o.type not in {'CAMERA', 'LIGHT'} for o in objects))
    expected = dict(objects=4, cameras=2, lights=2, meshes=0, materials=0,
                    images=0, curves=0, other_objects=0)
    if counts != expected:
        raise RuntimeError('Public start data audit failed: ' + json.dumps(counts))
    if sorted(o.name for o in cameras) != ['PublicCamera0', 'PublicCamera1']:
        raise RuntimeError('Unexpected or hidden camera')
    return counts


INSTRUCTION = '''Reconstruct the indoor scene from the two supplied public reference images.
Start from empty geometry. Use both views to infer visible objects and their spatial
arrangement, then build and inspect the result in Blender. Units are meters and Z is up.
You may edit the working cameras and lights; their consequences remain in your scene.
The gray background and cutaway presentation in the references are preview conventions;
you may use a cutaway presentation where it helps show the reconstructed interior.
The initial lighting is neutral engineering illumination, not a promise that it matches
the reference lighting. Use only the provided tools and public visual evidence.
Do not load external source scenes, original answer meshes, or hidden evaluation data.
When you decide to stop, use the finish tool and state any remaining limitations.'''


def prepare_one(bpy, Vector, root, config, row):
    ident = row['id']
    out = root / 'public' / ident
    out.mkdir(parents=True, exist_ok=True)
    # A new factory scene is essential: deleting source geometry would retain
    # answer meshes, materials, images, libraries and file paths as data blocks.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = 0
    scene.render.film_transparent = False
    world = bpy.data.worlds.new('NeutralPublicWorld')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (.78, .82, .9, 1)
    world.node_tree.nodes['Background'].inputs[1].default_value = .6
    scene.world = world
    for i, view in enumerate(row['public_views']):
        data = bpy.data.cameras.new(f'PublicCamera{i}')
        data.type, data.lens = 'PERSP', float(view['lens_mm'])
        data.clip_start, data.clip_end = .01, 1000
        camera = bpy.data.objects.new(f'PublicCamera{i}', data)
        scene.collection.objects.link(camera)
        camera.location = vector3(view['camera_location'], 'camera_location')
        target = Vector(vector3(view['target'], 'target'))
        camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
    # The light layout uses only public camera-target metadata and generic scale;
    # no source light, target object, bbox, geometry or furniture count is read.
    target = Vector(vector3(row['public_views'][0]['target'], 'target'))
    for name, offset, watts, size in [('NeutralKey', (0, -3, 6), 900, 5),
                                      ('NeutralFill', (3, 2, 4), 500, 4)]:
        light = bpy.data.lights.new(name, 'AREA')
        light.energy, light.shape, light.size = watts, 'DISK', size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = target + Vector(offset)
        obj.rotation_euler = (target-obj.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = bpy.data.objects['PublicCamera0']
    for source, i in zip(row['reference_files'], range(2)):
        shutil.copyfile(source, out / f'reference_{i}.jpg')
    # Render Result is transient, not a public image/texture data block.
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)
    before = geometry_audit(bpy)
    scene.render.filepath = str(out / 'initial.png')
    bpy.ops.render.render(write_still=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)
    initial = out / 'initial.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(initial), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(initial), load_ui=False, use_scripts=False)
    after = geometry_audit(bpy)
    camera_protocol = dict(resolution=[640, 480], projection='perspective', cameras=[])
    for name in ['PublicCamera0', 'PublicCamera1']:
        obj = bpy.data.objects[name]
        camera_protocol['cameras'].append(dict(name=name, location=list(obj.location),
            matrix_world=[list(r) for r in obj.matrix_world], lens_mm=float(obj.data.lens),
            sensor_width_mm=float(obj.data.sensor_width), sensor_fit=obj.data.sensor_fit))
    task = dict(id=ident, group='blender', blender_bin=config['blender'],
        initial_blend=str(initial), observation_camera='PublicCamera0', image_size=640,
        references=[dict(id=f'ref:{i}', image=str(out / f'reference_{i}.jpg')) for i in range(2)],
        instruction=INSTRUCTION, public_camera_protocol=camera_protocol,
        budget=dict(max_agent_steps=25, max_tool_calls=25, max_wall_seconds=10800))
    dump(out / 'task.json', task)
    receipt = dict(scene_id=ident, passed=True, geometry_source='new_factory_empty_scene',
        source_gt_opened=False, public_references=2, hidden_references=0,
        before_save=before, after_reload=after, render=dict(device='CPU', samples=12,
            threads=4, resolution=[640, 480], view_transform='AgX', denoise=True),
        initial_lighting='neutral; does not claim source-light matching',
        hashes={name: sha(out / name) for name in ['initial.blend', 'initial.png',
            'reference_0.jpg', 'reference_1.jpg', 'task.json']})
    dump(out / 'preparation_receipt.json', receipt)
    return dict(scene_id=ident, passed=True, initial_geometry_objects=0,
                cameras=2, lights=2, initial_blend_sha256=receipt['hashes']['initial.blend'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if not args.root.is_absolute():
        raise ValueError('--root must be absolute')
    root = args.root.resolve()
    config = json.loads((root / 'config.json').read_text())
    rows = validate_rows(root, config)  # Validate every row before creating output.
    import bpy
    from mathutils import Vector
    reports = [prepare_one(bpy, Vector, root, config, row) for row in rows]
    summary = dict(passed=True, scenes=len(reports), rows=reports,
                   API_calls=0, source_gt_opened=False, task_mode='from_empty')
    dump(root / 'preparation_receipt.json', summary)
    print(json.dumps(dict(passed=True, prepared=len(reports), API_calls=0,
                         source_gt_opened=False)))


if __name__ == '__main__':
    main()
