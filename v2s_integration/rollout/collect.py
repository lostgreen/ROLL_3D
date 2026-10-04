"""Native ROLL inference-only collection: real Cluster/Router/PolicyProxy/manager."""
import argparse
import json
import os
from pathlib import Path
import threading
import time


def write_json(path, value):
    """Readers never observe a half-written progress record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def validate_args(args):
    if not 0 <= args.gpu < 8:
        raise ValueError('--gpu must be a physical device index in [0, 7]')
    if not 1 <= args.steps <= 40:
        raise ValueError('--steps must be in [1, 40]')
    if args.context <= args.output_tokens or min(args.output_tokens, args.episodes, args.max_images) < 1:
        raise ValueError('Positive budgets/episodes/images and context > output tokens are required')
    if not 0 < args.memory_utilization < 1:
        raise ValueError('--memory-utilization must be between 0 and 1')
    if args.seed_start < 0:
        raise ValueError('--seed-start must be nonnegative')


def validate_gpu_visibility(environ):
    # ROLL Cluster sets worker CUDA_VISIBLE_DEVICES using physical device_mapping
    # ranks. A masked parent would otherwise make Ray and ROLL disagree on IDs.
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.replace(' ', '') != '0,1,2,3,4,5,6,7':
        raise ValueError('Launch with all 8 B300 GPUs visible; --gpu selects the physical inference device')


def episode_status(result):
    reason = result.get('termination_reason')
    if reason in {'finish', 'max_steps'}:
        return 'collected'
    if reason in {'context_budget_exhausted', 'budget_exceeded'}:
        return 'limited'
    return 'failed'


def run(args):
    validate_args(args)
    validate_gpu_visibility(os.environ)
    import ray
    from dacite import from_dict
    from omegaconf import OmegaConf
    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from roll.distributed.executor.cluster import Cluster
    from roll.distributed.scheduler.resource_manager import ResourceManager
    from roll.distributed.scheduler.router import RouterManager
    from roll.distributed.scheduler.transfer_backend import init_transfer_backend
    from roll.models.model_providers import default_tokenizer_provider, default_processor_provider, get_extra_data_provider
    from roll.utils.constants import RAY_NAMESPACE
    from .manager import ReconstructionManager, ContextBudgetExceeded
    from v2s_integration.training.config import environment_config
    from .seeded_proxy import SeededPolicyProxy, verify_seed_converter
    from roll.distributed.strategy.vllm_strategy import create_sampling_params_for_vllm
    verify_seed_converter(create_sampling_params_for_vllm)

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    def status(stage, **extra):
        write_json(out / 'status.json', {'stage': stage, 'updated_at_unix': time.time(), **extra})
    status('configuring')
    cfg = OmegaConf.load(Path(args.repo) / 'examples/qwen2.5-vl-3B-agentic/agentic_val_sokoban.yaml')
    for key in ('defaults', 'hydra', 'custom_envs', 'max_tokens_per_step'):
        if key in cfg:
            del cfg[key]
    cfg.seed = args.seed_start
    cfg.pretrain = args.model
    cfg.reward_pretrain = args.model
    cfg.exp_name = out.name
    cfg.output_dir = str(out)
    cfg.logging_dir = str(out / 'logs')
    cfg.render_save_dir = str(out / 'renders')
    cfg.track_with = 'tensorboard'
    cfg.tracker_kwargs = {'log_dir': str(out / 'tensorboard')}
    cfg.checkpoint_config = {'type': 'file_system', 'output_dir': str(out / 'unused_checkpoints')}
    cfg.num_gpus_per_node = 8
    cfg.max_steps = 1
    cfg.sequence_length = args.context
    cfg.rollout_batch_size = 1
    cfg.val_batch_size = 1
    cfg.system_envs = {key: os.environ[key] for key in (
        'PYTHONPATH', 'LD_LIBRARY_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
        'PYTHONDONTWRITEBYTECODE', 'ROLL_OTEL_ENABLED', 'V2S_VLLM_PORT_START') if key in os.environ}
    # Train/reference configs satisfy dataclass validation only; no such workers are created.
    cfg.actor_train.device_mapping = str([(args.gpu + 1) % 8])
    cfg.actor_train.model_args.attn_implementation = 'sdpa'
    cfg.actor_train.strategy_args = {'strategy_name': 'fsdp2_train', 'strategy_config': {'fsdp_size': 1}}
    cfg.reference.device_mapping = str([(args.gpu + 1) % 8])
    cfg.actor_infer.device_mapping = str([args.gpu])
    cfg.actor_infer.backend_timeout = 3
    cfg.actor_infer.generating_args.max_new_tokens = args.output_tokens
    cfg.actor_infer.generating_args.temperature = 0.7
    cfg.actor_infer.generating_args.top_p = 0.9
    cfg.actor_infer.generating_args.top_k = -1
    cfg.actor_infer.strategy_args = {'strategy_name': 'vllm', 'strategy_config': {
        'load_format': 'auto', 'gpu_memory_utilization': args.memory_utilization, 'enforce_eager': True,
        'max_model_len': args.context, 'enable_prefix_caching': False,
        'max_num_seqs': 1, 'limit_mm_per_prompt': {'image': args.max_images}}}
    template = environment_config(args.task, out / 'episodes', max_steps=args.steps,
                                  max_output_tokens=args.output_tokens)
    cfg.custom_envs = {'Reconstruction': template}
    env_manager_cfg = {'max_env_num_per_worker': 1, 'num_env_groups': 1, 'group_size': 1,
                       'tags': ['Reconstruction'], 'num_groups_partition': [1], 'format_penalty': 0.}
    cfg.train_env_manager = env_manager_cfg
    cfg.val_env_manager = env_manager_cfg
    OmegaConf.save(cfg, out / 'config.yaml')
    config = from_dict(AgenticConfig, OmegaConf.to_container(cfg, resolve=True))
    if config.num_gpus_per_node != 8:
        raise RuntimeError('This collector requires the validated 8-GPU B300 host')
    config.set_max_steps(max_steps=1)
    ray.init(address='local', num_cpus=16, num_gpus=8, namespace=RAY_NAMESPACE,
             include_dashboard=False, object_store_memory=2 * 1024**3,
             _temp_dir=f'/tmp/v2s_demo_ray_{os.getpid()}', runtime_env={'env_vars': config.system_envs})
    manager = None
    outcomes = []
    started = time.monotonic()
    try:
        for _ in range(30):
            if any(n['Resources'].get('GPU', 0) >= 8 for n in ray.nodes()):
                break
            time.sleep(1)
        resource = ResourceManager(num_nodes=1, num_gpus_per_node=8)
        init_transfer_backend(config.transfer_backend)
        status('loading_model')
        actor = Cluster(name=config.actor_infer.name, worker_cls=config.actor_infer.worker_cls,
                        resource_manager=resource, worker_config=config.actor_infer)
        actor.initialize(pipeline_config=config, blocking=True)
        router = ray.remote(RouterManager).options(max_concurrency=4).remote(
            actor_cluster=actor, router_args=config.router_args, num_gpus_per_node=8)
        ray.get(router.initialize.remote())
        tokenizer = default_tokenizer_provider(config.actor_infer.model_args, args.model)
        processor = default_processor_provider(config.actor_infer.model_args, args.model)
        env_config = OmegaConf.create({'env_id': 0, 'group_id': 0, 'tag': 'Reconstruction',
            'env_type': 'video2scene_reconstruction', 'config': template['env_config'],
            'max_steps': args.steps, 'max_tokens_per_step': args.output_tokens})
        manager = ReconstructionManager(worker_config=config.train_env_manager, pipeline_config=config,
            env_config=env_config, tokenizer=tokenizer, processor=processor, generate_scheduler=router,
            output_queue=None, thread_lock=threading.Lock(), mode='val',
            extra_data_provider=get_extra_data_provider(args.model, processor=processor))
        manager.llm_proxy = SeededPolicyProxy(manager.llm_proxy, manager)
        seed_policy = {'engine_initialization_seed': args.seed_start,
                       'per_request_sampling_seed': 'episode_seed * 1000 + zero_based_turn',
                       'sampling_seed_supported': True,
                       'note': 'Paired seeds do not guarantee identical outputs across different tool prompts'}
        write_json(out / 'collection_config.json', {
            'task': str(Path(args.task).resolve()), 'gpu': args.gpu,
            'context_tokens': args.context, 'output_tokens': args.output_tokens,
            'max_steps': args.steps, 'episodes': args.episodes,
            'environment_seeds': list(range(args.seed_start, args.seed_start + args.episodes)),
            'seed_policy': seed_policy, 'training': False})
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            episode_started = time.monotonic()
            phase = 'reset'
            previous_episode = getattr(manager.env, 'episode', None)
            manager.sampling_episode_seed = seed
            item = {'episode_index': episode, 'environment_seed': seed,
                    'seed_policy': seed_policy}
            try:
                status('resetting', episode_index=episode, environment_seed=seed, completed=len(outcomes))
                cache = manager.begin_episode(seed=seed, episode_id=episode)
                write_json(manager.env.episode / 'collection.json', item)
                while not (cache.terminated or cache.truncated):
                    status('rollout', episode_index=episode, environment_seed=seed,
                           episode=str(manager.env.episode), turn=cache.step + 1, completed=len(outcomes))
                    phase = 'inference'
                    try:
                        output = manager.make_decision(cache)
                    except ContextBudgetExceeded:
                        manager.env.finish_episode('context_budget_exhausted')
                        break
                    if output.batch is None:
                        manager.env.finish_episode('generation_aborted')
                        break
                    phase = 'environment_step'
                    cache = manager.step(output)
                item.update(json.loads((manager.env.episode / 'result.json').read_text()))
                item['status'] = episode_status(item)
                if item['status'] == 'failed':
                    item['recoverable'] = item.get('termination_reason') != 'generation_aborted'
                    if not item['recoverable']:
                        raise RuntimeError('Native generation aborted; refusing further episodes')
            except Exception as exc:
                item.update(status='failed', phase=phase, error_type=type(exc).__name__,
                            error=str(exc)[-1000:])
                # Tool API errors normally return structured feedback. An uncaught
                # environment error can be isolated by closing Blender and resetting
                # a fresh episode. Inference/Ray failures invalidate this worker.
                recoverable = (phase in ('reset', 'environment_step') and
                               not isinstance(exc, (ray.exceptions.RayError, MemoryError)))
                item['recoverable'] = recoverable
                if not recoverable:
                    raise
            finally:
                episode_path = getattr(manager.env, 'episode', None)
                item['episode'] = (str(episode_path) if episode_path != previous_episode else None)
                item['seconds'] = round(time.monotonic() - episode_started, 3)
                try:
                    manager.env.close()
                except Exception as exc:
                    item['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)[-500:]}
                outcomes.append(item)
                write_json(out / 'outcomes.json', outcomes)
                status('episode_complete', completed=len(outcomes), total=args.episodes, latest=item)
                if item.get('cleanup_error'):
                    raise RuntimeError('Environment cleanup failed; refusing another episode')
        failures = sum(item['status'] == 'failed' for item in outcomes)
        limited = sum(item['status'] == 'limited' for item in outcomes)
        status('complete_with_failures' if failures else 'complete', seconds=time.monotonic() - started,
               outcomes=outcomes, failed_episodes=failures, limited_episodes=limited)
        if failures:
            raise RuntimeError(f'{failures} episodes failed; see outcomes.json')
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc)[-1500:],
               completed=len(outcomes), failed_episodes=sum(x.get('status') == 'failed' for x in outcomes))
        raise
    finally:
        try:
            if manager:
                manager.env.close()
        finally:
            ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--memory-utilization', type=float, default=0.65)
    parser.add_argument('--max-images', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=8192)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=1)
    run(parser.parse_args())
