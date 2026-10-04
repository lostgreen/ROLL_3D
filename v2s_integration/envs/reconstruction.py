"""Multi-tool reconstruction environment, reusing the original provider runtime."""
import base64
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import uuid

from PIL import Image

from v2s_integration.runtime.session import (VENDOR, SCOPES, create_session, execute_validated, model_tools)
from v2s_integration.agent.actions import parse_action
from v2s_integration.evaluation.rewards import load_reward, evaluate_reward
from environment.budget import BudgetExceeded
from tools.base import ToolResult
from trajectory import TrajectoryRecorder

PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'
SUCCESS = {'succeeded', 'success', 'ok'}
MUTATING = {'execute_blender_code', 'scene.import_artifact', 'scene.set_transform'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def blender_json(text):
    # The original MCP wraps stdout from execute_blender_code in this prefix.
    return json.loads(text.removeprefix('Code executed successfully: ').strip())


class ReconstructionEnv:
    def __init__(self, task_manifest, output_root, max_steps=40, max_output_tokens=8192,
                 reward_function=None, training=False, training_target='reasoning_and_action', **kwargs):
        self.task = json.loads(Path(task_manifest).read_text())
        if self.task.get('group') not in SCOPES:
            raise ValueError('Manifest must select blender, hunyuan, assets, or mixed')
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 40:
            raise ValueError('max_steps must be in [1, 40]')
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError('max_output_tokens must be positive')
        self.max_steps, self.max_output_tokens = max_steps, max_output_tokens
        if training_target != 'reasoning_and_action':
            raise ValueError('Supported training_target: reasoning_and_action')
        self.training, self.training_target = training, training_target
        self.reward_function = load_reward(reward_function)
        if training and self.reward_function is None:
            raise ValueError('Training requires an explicit reward_function')
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.task['budget'] = {**self.task.get('budget', {}), 'max_agent_steps': max_steps,
                               'max_tool_calls': max_steps}
        self.task['budget'].setdefault('max_wall_seconds', 3600)
        self.mcp = self.blender = self.session = None
        self.closed = True

    def _start(self):
        from blender_process import HeadlessBlender
        from mcp_client import BlenderMCPClient
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.blender = HeadlessBlender(port=port, log_path=self.episode / 'blender.log',
                                       blender_bin=self.task['blender_bin']).start()
        self.mcp = BlenderMCPClient(command=sys.executable,
            args=['-c', 'from blender_mcp.server import main; main()'], cwd=str(VENDOR / 'blender-mcp'),
            env={'BLENDER_PORT': str(port), 'PYTHONPATH': str(VENDOR / 'blender-mcp/src'),
                 'PYTHONDONTWRITEBYTECODE': '1'})
        self.mcp.start()
        self._code('import bpy\nbpy.ops.wm.open_mainfile(filepath=' + repr(self.task['initial_blend']) + ')\n'
                   'for obj in list(bpy.data.objects):\n'
                   ' if obj.type not in {"CAMERA", "LIGHT"}: bpy.data.objects.remove(obj, do_unlink=True)\n')

    def _code(self, code):
        result = ToolResult.from_legacy(self.mcp.call_tool_rich('execute_blender_code', {'code': code}))
        if result.status not in SUCCESS:
            raise RuntimeError('Blender infrastructure command failed: ' + result.text[:240])
        return result

    def _save_image(self, item, label):
        raw = base64.b64decode(item['data'], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            picture = source.convert('RGB')
            path = self.episode / 'images' / f'{len(self.images):04d}.png'
            picture.save(path)
        ref = {'path': str(path), 'label': item.get('label') or label,
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.images.append(ref)
        return ref

    def _scene(self):
        result = self.session.scene.list_objects({})
        if result.status not in SUCCESS:
            raise RuntimeError('Scene observation failed: ' + result.text[:240])
        return blender_json(result.text)

    def _protected_state(self):
        result = self._code('import bpy, json\n'
            'items = [{"name": o.name, "type": o.type, "matrix": [list(r) for r in o.matrix_world], '
            '"data": {k: getattr(o.data, k) for k in '
            '("lens", "type", "ortho_scale", "clip_start", "clip_end") if hasattr(o.data, k)}, '
            '"energy": getattr(o.data, "energy", None), "color": list(o.data.color) if hasattr(o.data, "color") else None} '
            'for o in bpy.data.objects if o.type in {"CAMERA", "LIGHT"}]\n'
            'print(json.dumps(sorted(items, key=lambda x: x["name"])))')
        return blender_json(result.text)

    def _render(self):
        result = ToolResult.from_legacy(self.mcp.call_tool_rich('render_scene_view', {
            'view': 'camera', 'camera': self.task['observation_camera'],
            'lighting': 'scene', 'max_size': self.task.get('image_size', 640)}))
        if result.status not in SUCCESS and 'no finite visible geometry bounds' in result.text:
            # The original inspection tool fits geometry even in camera mode.
            # Empty reconstruction starts have no bounds: render the public camera
            # directly without adding a placeholder object to the agent's scene.
            path = self.episode / f'empty_observation_{self.step_count:03d}.png'
            size = int(self.task.get('image_size', 640))
            self._code('import bpy\ns=bpy.context.scene\n'
                's.camera=bpy.data.objects[' + repr(self.task['observation_camera']) + ']\n'
                's.render.engine="CYCLES"; s.cycles.device="CPU"; s.cycles.samples=16\n'
                f's.render.resolution_x={size}; s.render.resolution_y={size}; s.render.resolution_percentage=100\n'
                's.render.image_settings.file_format="PNG"; s.render.image_settings.color_mode="RGB"\n'
                's.render.filepath=' + repr(str(path)) + '\n'
                'bpy.ops.render.render(write_still=True)')
            return [self._save_image({'data': base64.b64encode(path.read_bytes()).decode()},
                                    f'Empty scene at turn {self.step_count}')]
        if result.status not in SUCCESS or not result.images:
            raise RuntimeError('Observation render failed: ' + result.text[:240])
        return [self._save_image(item, f'Current scene at turn {self.step_count}') for item in result.images]

    def reset(self, seed=None):
        self.close()
        self.episode = self.output_root / (self.task['group'] + '_' + uuid.uuid4().hex[:12])
        (self.episode / 'images').mkdir(parents=True)
        self.step_count, self.turns, self.images = 0, [], []
        self.closed = False
        try:
            references = {}
            self.reference_refs = []
            for i, ref in enumerate(self.task.get('references', [])):
                key = str(ref.get('id', f'ref:{i}'))
                if key in references:
                    raise ValueError('Duplicate reference image ID')
                path = Path(ref['image'])
                saved = self._save_image({'data': base64.b64encode(path.read_bytes()).decode()},
                                         f'Reference {key}; timestamp={ref.get("timestamp", "unspecified")}')
                references[key] = saved['path']
                self.reference_refs.append(saved)
            if not references:
                raise ValueError('references must contain actual public images')
            self._start()
            segmenter = None
            if self.task.get('segmentation'):
                from tools.segmentation import SAM2Segmenter
                cfg = self.task['segmentation']
                segmenter = SAM2Segmenter(cfg['checkpoint'], device=cfg.get('device', 'cuda:0'))
            self.session = create_session(self.task, self.mcp, self.episode / 'artifacts', references,
                                          segmenter=segmenter)
            self.recorder = TrajectoryRecorder(self.episode / 'trajectory', {
                'task_id': self.task['id'], 'group_id': self.task['group'], 'seed': seed,
                'tool_contracts': self.session.registry.contracts()})
            if self.session.generation:
                self.session.generation.recorder = self.recorder
            self.model_tools = model_tools(self.session)
            self.protected_state = self._protected_state()
            self.public_state = self._scene()
            if self.public_state.get('objects'):
                raise RuntimeError('Initial scene still contains geometry or object roots')
            self.initial_images = self._render()
            self.current_images = self.initial_images
            public = json.dumps(self.public_state, ensure_ascii=False)
            self.system_prompt = (PROMPTS / 'system.txt').read_text().format(
                observed_initial_scene='Empty geometry; preserve environment-owned objects ' +
                    ', '.join(o['name'] for o in self.protected_state) + '. ' + public,
                reference_image_ids=', '.join(references), capability_scope=SCOPES[self.task['group']])
            self.task_prompt = (PROMPTS / 'task.txt').read_text().format(
                reference_frame_labels_and_timestamps='; '.join(r['label'] for r in self.reference_refs),
                public_scene_state=public)
            dump(self.episode / 'environment.json', self.session.snapshot())
            dump(self.episode / 'tool_schemas.json', self.model_tools)
            dump(self.episode / 'manifest.json', {
                'task_id': self.task['id'], 'group': self.task['group'], 'seed': seed,
                'budget': self.task['budget'], 'max_output_tokens': self.max_output_tokens,
                'references': self.reference_refs, 'tool_contract_hash': self.session.registry.contract_hash(),
                'system_prompt': self.system_prompt, 'task_prompt': self.task_prompt,
                'prompt_version': 'native-tools-no-budget-v2',
                'prompt_files_sha256': {name: hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest()
                                        for name in ('system.txt', 'task.txt')},
                'evaluation_feedback': False, 'training': self.training,
                'training_target': self.training_target,
                'isolation': 'protocol allowlist; Blender Python is not a filesystem sandbox'})
            return self._observation('Initial scene', self.current_images), {'env_instruction': self.system_prompt}
        except BaseException:
            self.close()
            raise

    def _observation(self, text, image_refs):
        content, pictures = [{'type': 'text', 'text': text}], []
        for ref in image_refs:
            content.extend([{'type': 'text', 'text': ref['label']}, {'type': 'image'}])
            with Image.open(ref['path']) as source:
                pictures.append(source.convert('RGB'))
        return {'prompt': content, 'image': pictures}

    def step(self, action):
        if self.closed:
            raise RuntimeError('Episode is closed; call reset first')
        self.step_count += 1
        self.recorder.step = self.step_count
        folder = self.episode / f'step_{self.step_count:03d}'
        folder.mkdir()
        (folder / 'response.txt').write_text(str(action))
        self.recorder.event('model.response', raw=str(action))
        call, log, terminal_reason = None, None, None
        try:
            self.session.budget.begin_step()
            raw = str(action).strip().removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
            call = parse_action(raw, self.step_count, self.session.registry)
            result, log = execute_validated(self.session, call.name, call.arguments)
        except ValueError as exc:
            result = ToolResult(str(exc), status='rejected', error_type='invalid_action')
        except BudgetExceeded as exc:
            result = ToolResult(str(exc), status='rejected', error_type='budget_exceeded')
        tool_images = [self._save_image(item, 'Tool observation') for item in result.images]
        # Code can partially mutate the scene even when execution reports failure.
        try:
            self.public_state = self._scene()
            if self._protected_state() != self.protected_state:
                terminal_reason = 'environment_contract_violation'
            elif call and call.name in MUTATING:
                self.current_images = self._render()
            if terminal_reason:
                pass
            elif call and call.name == 'finish' and result.status in SUCCESS:
                terminal_reason = 'finish'
            elif result.error_type == 'budget_exceeded':
                terminal_reason = 'budget_exceeded'
            elif self.step_count >= self.max_steps:
                terminal_reason = 'max_steps'
        except Exception as exc:
            terminal_reason = 'environment_error'
            dump(folder / 'environment_error.json', {'type': type(exc).__name__, 'message': str(exc)[:500]})
        # Store full tool output outside conversational summaries, including structured data.
        dump(folder / 'tool_result.json', {**asdict(result), 'images': tool_images})
        self.recorder.event('tool.result', call=asdict(call) if call else None, result=asdict(result), execution=log)
        feedback = {'call_id': call.id if call else None, 'tool_name': call.name if call else None,
                    'status': result.status, 'error_type': result.error_type,
                    'text': result.for_model()['text'], 'artifact_ids': result.artifact_ids}
        turn = {'step': self.step_count, 'call': asdict(call) if call else None, 'feedback': feedback,
                'tool_images': tool_images, 'scene_images': self.current_images,
                'scene_state': self.public_state, 'terminal_reason': terminal_reason}
        if terminal_reason:
            terminal_reason = self.finish_episode(terminal_reason)
            turn['terminal_reason'] = terminal_reason
        self.turns.append(turn)
        reward = evaluate_reward(self.reward_function, env=self, turn=turn)
        turn['reward'] = reward
        dump(folder / 'reward.json', {'reward': reward, 'configured': self.reward_function is not None})
        dump(folder / 'event.json', {**turn, 'execution': log})
        dump(self.episode / 'events.json', [{'step': t['step'], 'tool': t['feedback']['tool_name'],
              'error': t['feedback']['error_type'], 'terminal_reason': t['terminal_reason']} for t in self.turns])
        info = {'metrics': {'invalid_action': float(result.error_type == 'invalid_action'),
                            'tool_success': float(result.status in SUCCESS)},
                'termination_reason': terminal_reason}
        return self._observation(json.dumps(feedback, ensure_ascii=False), tool_images + self.current_images), reward, terminal_reason == 'finish', bool(terminal_reason and terminal_reason != 'finish'), info

    def finish_episode(self, reason):
        try:
            self._code('import bpy\nbpy.ops.wm.save_as_mainfile(filepath=' + repr(str(self.episode / 'scene.blend')) + ')')
        except Exception as exc:
            dump(self.episode / 'save_error.json', {'type': type(exc).__name__, 'message': str(exc)[:500]})
            reason = 'environment_error'
        finally:
            dump(self.episode / 'result.json', {'task_id': self.task['id'], 'group': self.task['group'],
                 'steps': self.step_count, 'termination_reason': reason, 'quality_evaluated': False})
            self.recorder.event('episode.end', reason=reason, steps=self.step_count)
            self.close()
        return reason

    def close(self):
        try:
            if self.mcp:
                self.mcp.close()
        finally:
            self.mcp = None
            if self.blender:
                self.blender.stop()
            self.blender = None
            self.closed = True
