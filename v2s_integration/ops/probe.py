"""Real backend contract probe; deterministic actions, no model or training."""
import argparse
import json

from v2s_integration.envs.reconstruction import ReconstructionEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    env = ReconstructionEnv(args.task, args.output, max_steps=5)
    try:
        env.reset(seed=0)
        ref = env.task['references'][0].get('id', 'ref:0')
        actions = [('reference.crop', {'reference_image': ref, 'bbox': [0, 0, 1, 1]})]
        if env.task['group'] in ('blender', 'mixed'):
            actions.append(('execute_blender_code', {'code':
                'import bpy\n'
                'bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.25, depth=0.8, location=(0,0,0.4))\n'
                'bpy.context.object.name="ProbeGeometry"\n'
                'mat=bpy.data.materials.new("ProbeMaterial"); mat.use_nodes=True\n'
                'mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(0.05,0.4,0.8,1)\n'
                'bpy.context.object.data.materials.append(mat)'}))
        actions.extend([('scene.list_objects', {}), ('finish', {'summary': 'Deterministic integration probe'})])
        for name, arguments in actions:
            _, _, done, truncated, info = env.step('<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>')
            if not info['metrics']['tool_success'] or truncated:
                raise RuntimeError(f'Probe failed at {name}; inspect episode artifacts')
            if done:
                break
        print(json.dumps({'status': 'passed', 'episode': str(env.episode), 'steps': env.step_count,
                          'model_inference_tested': False, 'generation_tested': False}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
