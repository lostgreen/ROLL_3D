"""One native ROLL visual manager for collection and decision-level training."""
import json
import threading
import time
import numpy as np
from tensordict import TensorDict

from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.utils.constants import GenerateStopReason
from v2s_integration.agent.messages import record_request, render_request
from v2s_integration.training.packing import pack_tensors

_lock = threading.Lock()
_registered = False


def register():
    global _registered
    import gem
    with _lock:
        if not _registered:
            gem.register('video2scene_reconstruction', entry_point='v2s_integration.envs.reconstruction:ReconstructionEnv')
            _registered = True


class ContextBudgetExceeded(RuntimeError):
    pass


class ReconstructionManager(VLTrajEnvManager):
    def __init__(self, *args, **kwargs):
        register()
        super().__init__(*args, **kwargs)
        self.decisions = []
        if self.mode == 'train' and not self.env.training:
            raise ValueError('Native training requires training=True and a reward function')
        self.env.training = self.mode == 'train'

    def reset(self):
        self.decisions = []
        if getattr(self, 'trace_stack', None):
            self.trace_stack.close()
        return super().reset()

    def begin_episode(self, seed, episode_id=0):
        self.decisions = []
        observation, info = self.env.reset(seed=seed)
        self.episode_id, self.current_step = episode_id, 0
        cache = RolloutCache(env_id=self.env_config['env_id'], group_id=self.env_config['group_id'],
                             tag=self.env_config['tag'])
        cache.history.append({'observation': observation, 'actions_left': self.env.max_steps,
                              'messages': None, **info})
        self.rollout_cache = cache
        return cache

    def run_rollout_loop(self, data):
        try:
            return super().run_rollout_loop(data)
        finally:
            self.env.close()
            if getattr(self, 'trace_stack', None):
                self.trace_stack.close()

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
        self.request_ids = data.batch['input_ids'][0].tolist()
        self.request_positions = data.batch['position_ids'].clone()
        self.request_messages = messages
        return data, messages

    def make_decision(self, rollout_cache):
        started = time.monotonic()
        try:
            output = super().make_decision(rollout_cache)
        except ContextBudgetExceeded:
            if self.mode != 'train':
                raise
            return DataProto(meta_info={'stop_reason': GenerateStopReason.MAX_LENGTH})
        if output.batch is None and self.mode == 'train':
            raise RuntimeError('Generation aborted; refusing an invalid training episode')
        if output.batch is not None:
            sampled = output.batch['responses'][0].tolist()
            if self.mode == 'train' or self.output_queue is not None:
                self.decisions.append({'output': output, 'prompt_ids': self.request_ids,
                                       'prompt_positions': self.request_positions,
                                       'sampled_ids': sampled, 'messages': self.request_messages,
                                       'directory': self.request_dir})
            (self.request_dir / 'generation.json').write_text(json.dumps({
                'response_ids': output.batch['responses'][0].tolist(),
                'decision_seconds': time.monotonic() - started,
                'stop_reason': str(output.meta_info.get('stop_reason'))}))
        return output

    def step(self, output):
        if output.batch is None:
            self.env.finish_episode('context_budget_exhausted')
            self.rollout_cache.terminated = self.rollout_cache.truncated = True
            return self.rollout_cache
        return super().step(output)

    def formulate_rollouts(self, rollout_cache):
        if not self.decisions:
            raise RuntimeError('Episode contains no sampled decisions')
        rewards = [item['reward'] for item in rollout_cache.history if 'reward' in item]
        if len(rewards) != len(self.decisions):
            raise ValueError('Sampled decisions and environment rewards do not align')
        episode_score = sum(rewards)
        forbidden = set()
        for token in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>', '<|im_start|>'):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(self.tokenizer, 'unk_token_id', None):
                forbidden.add(token_id)
        samples = []
        for step, (decision, reward) in enumerate(zip(self.decisions, rewards)):
            # GRPO learns an episode return; step methods consume step_scores.
            score = episode_score if self.pipeline_config.adv_estimator == 'grpo' else reward
            tensors = pack_tensors(decision['output'].batch, decision['prompt_ids'], decision['sampled_ids'],
                sequence_length=self.pipeline_config.sequence_length, pad_token_id=self.tokenizer.pad_token_id,
                reward=score, prompt_positions=decision['prompt_positions'], forbidden_token_ids=forbidden)
            extra = dict(decision['output'].non_tensor_batch)
            metadata = {'env_ids': rollout_cache.env_id, 'group_ids': rollout_cache.group_id,
                        'tags': rollout_cache.tag, 'step': step, 'step_scores': reward,
                        'episode_scores': episode_score, 'messages_list': decision['messages']}
            for key, value in metadata.items():
                array = np.empty(1, dtype=object)
                array[0] = value
                extra[key] = array
            samples.append(DataProto(batch=TensorDict(tensors, batch_size=[1]), non_tensor_batch=extra))
            (decision['directory'] / 'training_contract.json').write_text(json.dumps({
                'prompt_ids_match': True, 'response_ids_match': True, 'positions_match': True,
                'training_target': self.env.training_target,
                'loss_tokens': int(tensors['response_mask'].sum()),
                'prompt_tokens_in_loss': 0, 'padding_tokens_in_loss': 0,
                'engine_visual_expansion_verified': False,
                'packing': 'native_sampled_decision'}, indent=2))
        batch = DataProto.concat(samples)
        batch.meta_info = {'metrics': {'env/response_length': float(batch.batch['response_mask'].sum(-1).float().mean()),
                                      f'env/{rollout_cache.tag}/num_actions': len(samples)}}
        return batch
