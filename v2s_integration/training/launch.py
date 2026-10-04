"""Launch ROLL's AgenticPipeline with the shared reconstruction task."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .config import load_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from omegaconf import OmegaConf
    cfg = load_training_config(args.config, args.repo)
    if args.output:
        cfg.output_dir = str(Path(args.output).resolve())
        cfg.logging_dir = str(Path(cfg.output_dir) / 'logs')
        cfg.checkpoint_config.output_dir = str(Path(cfg.output_dir) / 'checkpoints')
        cfg.tracker_kwargs.log_dir = str(Path(cfg.output_dir) / 'tensorboard')
        cfg.custom_envs.Reconstruction.env_config.output_root = str(Path(cfg.output_dir) / 'episodes')
    task = cfg.custom_envs.Reconstruction.env_config
    if args.config_only:
        print(json.dumps({'status': 'config_built', 'manager': cfg.custom_envs.Reconstruction.env_manager_cls,
                          'training_target': task.training_target, 'reward_function': task.reward_function,
                          'native_dataclass_validated': False, 'model_loaded': False}))
        return

    from v2s_integration.evaluation.rewards import load_reward
    load_reward(task.reward_function)
    if not Path(task.task_manifest).is_file():
        raise ValueError('task_manifest must exist')
    from dacite import from_dict
    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
    from roll.utils.constants import RAY_NAMESPACE
    import ray
    root = str(Path(__file__).resolve().parents[2])
    cfg.system_envs.PYTHONPATH = os.pathsep.join([str(Path(args.repo).resolve()), root,
                                               os.environ.get('PYTHONPATH', '')])
    config = from_dict(AgenticConfig, OmegaConf.to_container(cfg, resolve=True))
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / 'config.yaml')
    (output / 'launch.json').write_text(json.dumps({'pipeline': 'ROLL.AgenticPipeline',
        'manager': cfg.custom_envs.Reconstruction.env_manager_cls, 'training_target': task.training_target,
        'reward_function': task.reward_function, 'packing': 'native_sampled_decision'}, indent=2))
    try:
        ray.init(address='local', num_gpus=config.num_gpus_per_node, namespace=RAY_NAMESPACE,
                 include_dashboard=False, runtime_env={'env_vars': config.system_envs})
        for _ in range(30):
            if any(node['Resources'].get('GPU', 0) >= config.num_gpus_per_node for node in ray.nodes()):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Ray GPU resources did not become visible after startup')
        AgenticPipeline(pipeline_config=config).run()
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
