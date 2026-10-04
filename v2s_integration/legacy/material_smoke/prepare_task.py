"""Run with Blender to create a small, explicitly diagnostic material-edit task."""
import bpy
import json
from pathlib import Path
import sys
from mathutils import Vector

root=Path(sys.argv[sys.argv.index('--')+1]).resolve()
root.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=16
s.render.resolution_x=224;s.render.resolution_y=224;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB'
s.world=bpy.data.worlds.new('World');s.world.use_nodes=True
s.world.node_tree.nodes['Background'].inputs[0].default_value=(0.7,0.7,0.7,1)
s.world.node_tree.nodes['Background'].inputs[1].default_value=0.6
s.view_settings.view_transform='Standard'
def mat(name,color):
    m=bpy.data.materials.new(name);m.use_nodes=True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*color,1)
    m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.7
    return m
bpy.ops.mesh.primitive_cube_add(size=1.1,location=(-.8,0,.6));cube=bpy.context.object;cube.name='Cube'
cube.data.materials.append(mat('CubeMaterial',(.9,.12,.025)))
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=.65,location=(.8,0,.65));sphere=bpy.context.object;sphere.name='Sphere'
sphere.data.materials.append(mat('SphereMaterial',(.02,.18,.85)))
bpy.ops.mesh.primitive_plane_add(size=200);plane=bpy.context.object;plane.name='Floor';plane.data.materials.append(mat('FloorMaterial',(.55,.55,.55)))
bpy.ops.object.camera_add(location=(3,-6,4));cam=bpy.context.object;cam.name='EvaluationCamera'
cam.rotation_euler=(Vector((0,0,.5))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=4.0;s.camera=cam
bpy.ops.object.light_add(type='AREA',location=(-3,-4,7));bpy.context.object.data.energy=700;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=5
s.render.filepath=str(root/'reference.png');bpy.ops.render.render(write_still=True)
for o in (cube,sphere):o.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.35,.35,.35,1)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'initial.blend'))
s.render.filepath=str(root/'initial.png');bpy.ops.render.render(write_still=True)
(root/'task.json').write_text(json.dumps({'id':'material_edit_smoke','initial_blend':str(root/'initial.blend'),'reference_image':str(root/'reference.png'),'prompt':'Edit the existing Cube and Sphere materials to match the reference image. Keep geometry, positions, camera, lighting and floor unchanged. Use bpy material nodes and inspect the rendered feedback. Finish when satisfied.','reward_version':'diagnostic_observed_rgb_mse_v1','split':'train_smoke_only'},indent=2))
print('DONE prepare_task',flush=True)
