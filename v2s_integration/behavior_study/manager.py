"""ROLL multimodal decision adapter for collection, without trajectory repacking."""
import json
import threading
import time

from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager
from roll.distributed.scheduler.protocol import DataProto
from .inputs import record_request, render_request

_lock = threading.Lock()
_registered = False


def register():
    global _registered
    import gem
    with _lock:
        if not _registered:
            gem.register('video2scene_reconstruction', entry_point='behavior_study.env:ReconstructionEnv')
            _registered = True


class ContextBudgetExceeded(RuntimeError):
    pass


class ReconstructionManager(VLTrajEnvManager):
    def __init__(self, *args, **kwargs):
        register()
        super().__init__(*args, **kwargs)

    def format_messages(self, rollout_cache):
        reserve = self.env.max_output_tokens
        if self.env_config['max_tokens_per_step'] != reserve:
            raise ValueError('Environment and ROLL output budgets disagree')
        if self.worker_config.generating_args.max_new_tokens != reserve:
            raise ValueError('Inference worker and environment output budgets disagree')
        # Collate before counting: visual token expansion contributes to the budget.
        for keep in range(len(self.env.turns), -1, -1):
            prompt, messages, images, refs, state = render_request(self.tokenizer, self.env, keep)
            feature = {self.collator.prompt_key: prompt, self.collator.image_key: images}
            data = DataProto.from_single_dict(self.collator([feature]))
            if data.batch['input_ids'].shape[1] + reserve <= self.pipeline_config.sequence_length:
                break
        else:
            raise ContextBudgetExceeded('Reference images, state and output reserve do not fit')
        item = data.non_tensor_batch.get('multi_modal_data')
        infer_ids = item[0].get('prompt_token_ids') if item is not None else None
        self.request_dir = self.env.episode / 'requests' / f'{self.env.step_count + 1:03d}'
        record_request(self.request_dir, prompt, messages, self.env.model_tools, refs,
                       data.batch['input_ids'][0].tolist(), infer_ids, state)
        self.env.recorder.event('model.request', event_step=self.env.step_count + 1,
                                messages=messages, tools=self.env.model_tools,
                                input_manifest=str(self.request_dir / 'input_manifest.json'))
        return data, messages

    def make_decision(self, rollout_cache):
        started = time.monotonic()
        output = super().make_decision(rollout_cache)
        if output.batch is not None:
            (self.request_dir / 'generation.json').write_text(json.dumps({
                'response_ids': output.batch['responses'][0].tolist(),
                'decision_seconds': time.monotonic() - started,
                'stop_reason': str(output.meta_info.get('stop_reason'))}))
        return output

    def collect_episode(self):
        """Use after normal ROLL worker/proxy initialization; no optimizer or packing."""
        cache = self.reset()
        if cache is None:
            return None
        try:
            while not (cache.terminated or cache.truncated):
                try:
                    output = self.make_decision(cache)
                except ContextBudgetExceeded:
                    self.env.finish_episode('context_budget_exhausted')
                    break
                if output.batch is None:
                    self.env.finish_episode('generation_aborted')
                    break
                cache = self.step(output)
            return str(self.env.episode)
        except BaseException:
            if not self.env.closed:
                self.env.finish_episode('infrastructure_error')
            raise
        finally:
            self.env.close()
            self.trace_stack.close()

    def run(self):
        raise RuntimeError('Use collect_episode with the collection coordinator; RL training packing is not enabled')

    def formulate_rollouts(self, rollout_cache):
        raise RuntimeError('Collection stores per-turn requests; it must not repack them into a training trajectory')


def environment_config(task_manifest, output_root, max_steps=40, max_output_tokens=8192):
    """Explicit opt-in config for the upcoming collection coordinator."""
    return {'env_type': 'video2scene_reconstruction',
            'env_manager_cls': 'behavior_study.manager.ReconstructionManager',
            'max_steps': max_steps, 'max_tokens_per_step': max_output_tokens,
            'agent_system_template': '', 'pre_step_template': '', 'next_step_template': '',
            'env_config': {'task_manifest': str(task_manifest), 'output_root': str(output_root),
                           'max_steps': max_steps, 'max_output_tokens': max_output_tokens}}
