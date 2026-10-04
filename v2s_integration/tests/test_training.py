"""Verify native sampled IDs, multimodal positions, masks, and shared config."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from tensordict import TensorDict
from v2s_integration.training.packing import pack_tensors
from v2s_integration.training.config import environment_config, load_training_config, MANAGER
from v2s_integration.evaluation.rewards import load_reward, evaluate_reward


def native_output(manager_module, native_postprocess, *, channels=3, prompt=(10, 11, 12, 13), response=(20, 21, 99)):
    ids = torch.tensor([prompt])
    positions = torch.arange(len(prompt))[None, :]
    if channels:
        positions = positions[:, None, :].repeat(1, channels, 1)
    data = manager_module.ContractDataProto(TensorDict({'input_ids': ids,
        'attention_mask': torch.ones_like(ids), 'position_ids': positions}, batch_size=[1]))
    output = native_postprocess(data, torch.tensor([(*prompt, *response)]), num_return_sequences=1,
        sequence_length=len(prompt) + len(response), eos_token_id=99, pad_token_id=0,
        pad_to_seq_len=False, output_logprobs=[[-0.1] * len(response)])
    return output, ids[0].tolist(), list(response), positions


@pytest.mark.parametrize('channels', [0, 3, 4])
def test_packing_preserves_native_ids_positions_and_logprobs(manager_module, native_postprocess, channels):
    output, prompt, response, positions = native_output(manager_module, native_postprocess, channels=channels)
    original = output.batch['input_ids'].clone()
    packed = pack_tensors(output.batch, prompt, response, sequence_length=12, pad_token_id=0,
                          reward=0.7, prompt_positions=positions)
    mask = packed['response_mask'][0].bool()
    assert packed['input_ids'][0][mask].tolist() == response
    assert packed['input_ids'][0][:len(prompt)].tolist() == prompt
    assert packed['response_mask'].sum() == 3
    assert not packed['response_mask'][..., :len(prompt)].any()
    assert not packed['response_mask'][..., 7:].any()
    assert torch.equal(packed['position_ids'][..., :7], output.batch['position_ids'])
    assert torch.equal(packed['infer_logprobs'][..., :6], output.batch['infer_logprobs'])
    assert packed['infer_logprobs'].shape[-1] == 11
    assert packed['scores'].sum().item() == pytest.approx(0.7)
    assert torch.equal(original, output.batch['input_ids'])


@pytest.mark.parametrize('fault', ['prompt', 'response', 'mask', 'positions', 'overflow'])
def test_contract_rejects_drift_and_training_truncation(manager_module, native_postprocess, fault):
    output, prompt, response, positions = native_output(manager_module, native_postprocess)
    length = 12
    if fault == 'prompt': prompt[0] += 1
    if fault == 'response': response[0] += 1
    if fault == 'mask': output.batch['response_mask'][0, 0] = 1
    if fault == 'positions': positions[..., 0] += 1
    if fault == 'overflow': length = 6
    with pytest.raises(ValueError):
        pack_tensors(output.batch, prompt, response, sequence_length=length, pad_token_id=0,
                     reward=1.0, prompt_positions=positions)


def test_each_decision_keeps_its_actual_context_and_visual_features(manager_module, native_postprocess, tmp_path):
    manager = manager_module.ReconstructionManager.__new__(manager_module.ReconstructionManager)
    manager.pipeline_config = SimpleNamespace(sequence_length=12, adv_estimator='step_reinforce')
    manager.tokenizer = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: None)
    manager.env = SimpleNamespace(training_target='reasoning_and_action')
    manager.decisions = []
    features = [object(), object()]
    for step, prompt in enumerate([(10, 11, 12, 13), (40, 41, 42)]):
        output, ids, response, positions = native_output(manager_module, native_postprocess, prompt=prompt)
        mm = np.empty(1, dtype=object)
        mm[0] = {'visual_feature': features[step]}
        output.non_tensor_batch = {'multi_modal_inputs': mm}
        folder = tmp_path / str(step)
        folder.mkdir()
        manager.decisions.append({'output': output, 'prompt_ids': ids, 'sampled_ids': response,
                                  'prompt_positions': positions, 'messages': [{'role': 'user'}] * (step + 1),
                                  'directory': folder})
    cache = SimpleNamespace(history=[{'reward': 0.}, {'reward': 1.}, {'observation': 'final'}],
                            env_id=2, group_id=1, tag='Reconstruction')
    batch = manager.formulate_rollouts(cache)
    assert batch.batch.batch_size == torch.Size([2])
    assert batch.batch['input_ids'][1, :3].tolist() == [40, 41, 42]
    assert batch.non_tensor_batch['multi_modal_inputs'][1]['visual_feature'] is features[1]
    assert batch.non_tensor_batch['step_scores'].tolist() == [0., 1.]
    assert batch.non_tensor_batch['episode_scores'].tolist() == [1., 1.]
    assert batch.non_tensor_batch['messages_list'].shape == (2,)
    assert batch.batch['scores'].sum(-1).tolist() == [0., 1.]
    manager.pipeline_config.adv_estimator = 'grpo'
    assert manager.formulate_rollouts(cache).batch['scores'].sum(-1).tolist() == [1., 1.]
    cache.history[0].pop('reward')
    with pytest.raises(ValueError, match='rewards'):
        manager.formulate_rollouts(cache)


def test_input_only_tokens_are_never_trained_as_responses(manager_module, native_postprocess):
    output, prompt, response, positions = native_output(manager_module, native_postprocess)
    with pytest.raises(ValueError, match='input-only'):
        pack_tensors(output.batch, prompt, response, sequence_length=12, pad_token_id=0,
                     reward=1.0, forbidden_token_ids=[response[0]])
    packed = pack_tensors(output.batch, prompt, response, sequence_length=12, pad_token_id=0,
                          reward=1.0, forbidden_token_ids=[prompt[0]])
    assert not packed['response_mask'][0, 0]


def test_train_and_collection_share_the_environment_contract():
    common = dict(task_manifest='/task.json', output_root='/episodes', max_steps=40, max_output_tokens=1024)
    collect = environment_config(**common)
    train = environment_config(**common, training=True, reward_function='test.rewards:quality')
    assert collect['env_type'] == train['env_type']
    assert collect['env_manager_cls'] == train['env_manager_cls'] == MANAGER
    for key in ('task_manifest', 'max_steps', 'max_output_tokens', 'training_target'):
        assert collect['env_config'][key] == train['env_config'][key]
    with pytest.raises(ValueError, match='reward_function'):
        environment_config(**common, training=True)
    with pytest.raises(ValueError, match='training_target'):
        environment_config(**common, training_target='action_only')


def test_example_builds_native_config_and_uses_explicit_target():
    root = Path(__file__).resolve().parents[2]
    cfg = load_training_config(root / 'v2s_integration/configs/train.example.yaml', root)
    assert cfg.custom_envs.Reconstruction.env_manager_cls == MANAGER
    assert cfg.custom_envs.Reconstruction.env_config.training_target == 'reasoning_and_action'
    assert cfg.actor_infer.generating_args.max_new_tokens == 1024
    assert cfg.actor_infer.strategy_args.strategy_config.limit_mm_per_prompt.image == 16
    assert cfg.train_env_manager.tags == ['Reconstruction']


def test_reward_is_explicit_and_finite():
    assert load_reward(None) is None
    assert evaluate_reward(None, env=None, turn=None) == 0.
    with pytest.raises(ValueError, match='module:function'):
        load_reward('not_a_plugin')
    with pytest.raises(ValueError, match='finite'):
        evaluate_reward(lambda **kw: float('nan'), env=None, turn={})
