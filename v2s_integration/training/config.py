"""One environment contract shared by native training and collection."""
from pathlib import Path

MANAGER = 'v2s_integration.rollout.manager.ReconstructionManager'


def environment_config(task_manifest, output_root, max_steps=40, max_output_tokens=8192,
                       *, training=False, reward_function=None, training_target='reasoning_and_action'):
    if training and not reward_function:
        raise ValueError('Training requires an explicit reward_function')
    if training_target != 'reasoning_and_action':
        raise ValueError('Supported training_target: reasoning_and_action')
    return {'env_type': 'video2scene_reconstruction', 'env_manager_cls': MANAGER,
            'max_steps': max_steps, 'max_tokens_per_step': max_output_tokens,
            'agent_system_template': '', 'pre_step_template': '', 'next_step_template': '',
            'env_config': {'task_manifest': str(task_manifest), 'output_root': str(output_root),
                           'max_steps': max_steps, 'max_output_tokens': max_output_tokens,
                           'training': training, 'reward_function': reward_function,
                           'training_target': training_target}}


def load_training_config(path, repo):
    from omegaconf import OmegaConf
    patch = OmegaConf.load(path)
    task = dict(patch.pop('task'))
    max_images = task.pop('max_images', 16)
    if 'custom_envs' in patch:
        raise ValueError('Configure the shared environment through task, not custom_envs')
    base = OmegaConf.load(Path(repo) / 'examples/qwen2.5-vl-3B-agentic/agentic_val_sokoban.yaml')
    for key in ('defaults', 'hydra', 'custom_envs', 'max_tokens_per_step'):
        if key in base:
            del base[key]
    config = OmegaConf.merge(base, patch)
    for worker in ('actor_train', 'actor_infer', 'reference'):
        if worker in patch and 'strategy_args' in patch[worker]:
            config[worker].strategy_args = patch[worker].strategy_args
    if config.adv_estimator not in ('grpo', 'step_reinforce'):
        raise ValueError('Decision packing currently supports grpo and step_reinforce')
    train = environment_config(**task, training=True)
    config.custom_envs = {'Reconstruction': train}
    output_tokens = task.get('max_output_tokens', 8192)
    config.actor_infer.generating_args.max_new_tokens = output_tokens
    # EnvManagerConfig owns its generation configuration separately from the worker.
    for name in ('train_env_manager', 'val_env_manager'):
        count = patch.get(name, {}).get('num_env_groups', 1)
        groups = {'tags': ['Reconstruction'], 'num_groups_partition': [count], 'num_env_groups': count}
        config[name] = OmegaConf.merge(config[name], groups)
    config.val_env_manager.group_size = 1
    config.actor_infer.strategy_args.strategy_config.max_model_len = config.sequence_length
    config.actor_infer.strategy_args.strategy_config.limit_mm_per_prompt = {'image': max_images}
    if config.sequence_length <= output_tokens:
        raise ValueError('sequence_length must leave room for the multimodal prompt')
    return config
