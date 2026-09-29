"""Model-facing usage help; execution schemas and provider handlers stay unchanged."""
import json


def native_example(name, arguments):
    """Qwen native function tags: raw string values, JSON for structured/scalar values."""
    values = []
    for key, value in arguments.items():
        payload = value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))
        values.append(f'<parameter={key}>{payload}</parameter>')
    return '<tool_call><function=' + name + '>' + ''.join(values) + '</function></tool_call>'


def guidance(name, schema, visible):
    """Return exact per-tool arguments and examples for the tools actually exposed."""
    help_text = {
        'execute_blender_code': (
            'Run multi-line Blender Python in the existing scene. Required code is a string; include import bpy. '
            'Create/edit meshes, curves, modifiers, materials and transforms freely; multiple objects per call are allowed. '
            'Preserve the environment cameras and lights. Calls are not transactional: edits before a Python exception '
            'can persist. Inspect the current scene before retrying; avoid deleting working geometry before replacement succeeds. '
            'For unknown shader socket names, inspect [socket.name for socket in node.inputs] before indexing. '
            'Wave Texture mode is node.wave_type (for example BANDS), not node.inputs["Wave Type"].',
            {'code': 'import bpy\nprint([(o.name, o.type) for o in bpy.context.scene.objects])'}),
        'reference.crop': (
            'Crop and observe a region of a registered public reference. Required reference_image is an ID from '
            'the observation; bbox is [left,top,right,bottom], normalized to image width/height in [0,1], origin top-left. '
            'Require left < right and top < bottom. Keep the complete visible object plus a small margin. '
            'Returns an image and a new reference_image ID. Cropping does not remove the background or recover occlusion.',
            {'reference_image': 'REFERENCE_ID_FROM_OBSERVATION', 'bbox': [0.1, 0.1, 0.9, 0.9]}),
        'reference.segment': (
            'Select one object using reference_image and bbox=[left,top,right,bottom] in normalized [0,1] '
            'image coordinates, origin top-left. Returns an RGBA object cutout with a new reference_image ID and '
            'an inspection overlay. Keep the whole visible object; missing parts are not recovered.',
            {'reference_image': 'REFERENCE_ID_FROM_OBSERVATION', 'bbox': [0.1, 0.1, 0.9, 0.9]}),
        'scene.list_objects': (
            'Inspect current geometry and imported Empty roots without changing the scene. No arguments. '
            'Returns names, parents, locations, dimensions and descendant world_bounds; use these for placement '
            'and to verify what survived a failed edit. Imported Empty root dimensions alone do not measure its mesh.', {}),
        'scene.set_transform': (
            'Set an existing object or imported root transform. Required object_name; also provide at least one of '
            'location, rotation or scale, each an array of exactly three finite numbers. Location uses meters; '
            'rotation is Euler XYZ in radians; scale is multiplicative, not target dimensions. Values replace the '
            'specified transform fields; omitted fields are unchanged. For an imported asset use returned root_name '
            'to move all parts. An unparented root uses world coordinates and has a bottom-center pivot. '
            'Use world_bounds to compute a scale factor from desired size / current size.',
            {'object_name': 'ROOT_NAME_FROM_IMPORT', 'location': [0.2, 0, 0.1], 'rotation': [0, 0, 0], 'scale': [0.5, 0.5, 0.5]}),
        'scene.import_artifact': (
            'Add a registered mesh artifact to Blender. Required artifact_id is the value returned by an asset tool, '
            'not its catalog asset_id, a reference_image ID or a filesystem path. Optional collection is a string. '
            'Only GLB/GLTF mesh artifacts can enter this scene. Returns root_name for the bottom-center Empty '
            'controlling all imported parts. Then inspect scene.list_objects, set scene.set_transform using root_name, '
            'and verify with render_scene_view. Each import creates a new instance.',
            {'artifact_id': 'ARTIFACT_ID_FROM_ASSET_TOOL'}),
        'asset.search': (
            'Search the local asset catalog. Required query is a short category/name string; optional limit is an '
            'integer 1..50. Returns catalog asset_id candidates and metadata; this does not change Blender. '
            'Inspect a candidate with asset.inspect, stage it with asset.import, then add its returned artifact_id '
            'using scene.import_artifact and place the returned root_name with scene.set_transform.',
            {'query': 'vase', 'limit': 5}),
        'asset.inspect': (
            'Inspect a catalog candidate before choosing it. Required asset_id is copied from asset.search. '
            'Returns metadata and available preview images; preview_status=unavailable means missing visual '
            'evidence, not a matching asset. This does not import anything. For a chosen candidate use asset.import.',
            {'asset_id': 'ASSET_ID_FROM_SEARCH'}),
        'asset.import': (
            'Stage a catalog asset in the run artifact store. Required asset_id comes from asset.search. '
            'Optional normalization is an object of metadata (coordinate_frame, units, scale, orientation); '
            'it does not physically resize or rotate the mesh. Omit it unless needed. Returns artifact_id; '
            'this tool alone does not add an object to Blender. Next call scene.import_artifact with that artifact_id, '
            'then place the returned root_name using scene.set_transform. Choose a GLB/GLTF candidate for scene import.',
            {'asset_id': 'ASSET_ID_FROM_SEARCH'}),
        'finish': (
            'End this trajectory. Optional summary is a short string describing completed work and remaining '
            'limitations. Call after checking the current scene; this performs no additional construction.',
            {'summary': 'Describe what is built and what remains inaccurate.'}),
    }
    if name == 'asset.generate_3d_from_image':
        properties = schema['properties']
        texture = properties['texture'].get('default', True)
        presets = properties['quality_preset']['enum']
        preset = properties['quality_preset'].get('default', presets[0])
        help_text[name] = (
            'Generate ONE 3D mesh asset using Hunyuan from a registered image. Required reference_image is an '
            'ID already returned in the observation or by reference.crop; it is not a path, URL, base64 image or '
            'text prompt. For an individual object, first crop that object with reference.crop, then pass the '
            'returned reference_image ID here. Optional texture is a JSON boolean, seed is an integer '
            '0..4294967295, quality_preset is one of ' + ', '.join(presets) + '. '
            + ('This backend supports shape only: texture must be false. ' if properties['texture'].get('enum') == [False]
               else 'texture=true requests textured output; texture=false requests geometry only. ')
            + 'No other arguments are accepted; do not pass prompt, negative_prompt, image_path, steps or a job ID. '
            'The environment waits for generation and returns success/failure; no polling tool is needed. '
            'Success returns artifact_id and mesh metadata, with provider-native orientation and unknown units. '
            'It does not add anything to Blender: next call scene.import_artifact with returned artifact_id, '
            'read root_name, inspect scene.list_objects bounds, set scene.set_transform, then render_scene_view. '
            'These are separate decisions; use actual returned IDs and assess the rendered result.',
            {'reference_image': 'REFERENCE_IMAGE_ID_FROM_CROP', 'texture': texture, 'seed': 7, 'quality_preset': preset})
    if name == 'render_scene_view':
        # Preserve the original MCP contract across versions rather than fabricate fields.
        return None
    if name not in help_text:
        return None
    text, example = help_text[name]
    if name == 'reference.crop' and 'asset.generate_3d_from_image' in visible:
        text += ' Its returned reference_image ID can be passed directly to asset.generate_3d_from_image.'
    return text + ' Example syntax (replace placeholder IDs and values using observed evidence): ' + native_example(name, example)
