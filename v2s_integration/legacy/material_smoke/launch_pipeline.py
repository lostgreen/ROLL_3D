"""Bounded native ROLL AgenticPipeline experiment, isolated Ray runtime and files."""
import json
import os
from pathlib import Path
import sys
import time
import signal

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ROLL_OTEL_ENABLED='0')
os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+str(WORK)
sys.path[:0]=[str(ROOT),str(WORK)]
def stop_requested(signum,frame):
    raise KeyboardInterrupt('Bounded experiment termination requested')
signal.signal(signal.SIGTERM,stop_requested)

def config():
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'examples/qwen2.5-vl-3B-agentic/agentic_val_sokoban.yaml')
    for key in ('defaults','hydra','custom_envs','max_tokens_per_step'):
        if key in cfg:del cfg[key]
    output=Path(os.getenv('V2S_OUTPUT_DIR', str(WORK/'native_run')))
    cfg.exp_name=os.getenv('V2S_EXP_NAME','v2s_material_smoke')
    cfg.pretrain=os.getenv('V2S_PRETRAIN','/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct');cfg.reward_pretrain=cfg.pretrain
    cfg.output_dir=str(output);cfg.logging_dir=str(output/'logs');cfg.render_save_dir=str(output/'renders')
    cfg.system_envs={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','ROLL_OTEL_ENABLED':'0','PYTHONPATH':os.environ['PYTHONPATH']}
    cfg.track_with='tensorboard';cfg.tracker_kwargs={'log_dir':str(output/'tensorboard')}
    cfg.checkpoint_config={'type':'file_system','output_dir':str(output/'checkpoints')}
    cfg.num_gpus_per_node=int(os.getenv('V2S_NUM_GPUS','2'));cfg.max_steps=int(os.getenv('V2S_MAX_STEPS','2'));cfg.save_steps=1;cfg.eval_steps=0
    cfg.rollout_batch_size=int(os.getenv('V2S_ROLLOUT_BATCH','4'));cfg.val_batch_size=1;cfg.sequence_length=int(os.getenv('V2S_SEQUENCE_LENGTH','4096'))
    cfg.advantage_clip=10.;cfg.whiten_advantages=False;cfg.init_kl_coef=0.
    cfg.actor_train.model_args.attn_implementation='sdpa'
    cfg.actor_train.model_args.lora_target='q_proj,v_proj';cfg.actor_train.model_args.lora_rank=8;cfg.actor_train.model_args.lora_alpha=16
    cfg.actor_train.training_args.learning_rate=1e-5
    cfg.actor_train.training_args.per_device_train_batch_size=1;cfg.actor_train.training_args.gradient_accumulation_steps=4
    cfg.actor_train.training_args.warmup_steps=0;cfg.actor_train.training_args.lr_scheduler_type='constant'
    cfg.actor_train.strategy_args={'strategy_name':'fsdp2_train','strategy_config':{'fsdp_size':1,'param_dtype':'bf16','reduce_dtype':'fp32','offload_policy':False,'reshard_after_forward':True}}
    cfg.actor_train.device_mapping='[0]';cfg.actor_train.infer_batch_size=1
    cfg.actor_train.backend_timeout=3
    cfg.actor_infer.device_mapping='[1]'
    cfg.actor_infer.backend_timeout=3
    cfg.actor_infer.model_args.lora_target='q_proj,v_proj';cfg.actor_infer.model_args.lora_rank=8;cfg.actor_infer.model_args.lora_alpha=16
    cfg.actor_infer.generating_args.max_new_tokens=int(os.getenv('V2S_MAX_NEW_TOKENS','512'))
    cfg.actor_infer.generating_args.temperature=1.;cfg.actor_infer.generating_args.top_p=1.;cfg.actor_infer.generating_args.top_k=-1
    cfg.actor_infer.strategy_args={'strategy_name':'vllm','strategy_config':{'gpu_memory_utilization':.6,'enforce_eager':True,'max_model_len':4096,'enable_prefix_caching':False,'limit_mm_per_prompt':{'image':5}}}
    cfg.reference.device_mapping='[0]';cfg.reference.model_args.attn_implementation='sdpa'
    cfg.rollout_dump_dir=str(output/'rollouts')
    cfg.train_env_manager={'max_env_num_per_worker':2,'num_env_groups':1,'group_size':4,'tags':['Video2Scene'],'num_groups_partition':[1],'format_penalty':0.}
    cfg.val_env_manager={'max_env_num_per_worker':1,'num_env_groups':1,'group_size':1,'tags':['Video2Scene'],'num_groups_partition':[1]}
    task_manifest=os.getenv('V2S_TASK_MANIFEST',str(WORK/'task/task.json'))
    episode_root=os.getenv('V2S_EPISODE_ROOT',str(WORK/'native_episodes'))
    cfg.custom_envs={'Video2Scene':{'env_type':'video2scene','env_manager_cls':'v2s_manager.Video2SceneManager','max_steps':int(os.getenv('V2S_ENV_MAX_STEPS','3')),'max_tokens_per_step':int(os.getenv('V2S_MAX_TOKENS_PER_STEP','512')),'agent_system_template':'You are a visual Blender material editing agent. Output one plain JSON function call per turn.','pre_step_template':'\nTurn {turn_idx}:\n','next_step_template':'\nActions left: {actions_left}. Respond with exactly one plain JSON object; no XML tags or markdown.','env_config':{'task_manifest':task_manifest,'output_root':episode_root,'max_steps':int(os.getenv('V2S_ENV_MAX_STEPS','3'))}}}
    OmegaConf.save(cfg,WORK/'native_config.yaml')
    from dacite import from_dict
    from roll.pipeline.agentic.agentic_config import AgenticConfig
    return from_dict(AgenticConfig,OmegaConf.to_container(cfg,resolve=True))

if __name__=='__main__':
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
    import ray
    from roll.utils.constants import RAY_NAMESPACE
    started=time.time()
    state={'state':'starting','native_pipeline':True}
    try:
        cfg=config()
        if '--config-only' in sys.argv:
            print('DONE config validated',flush=True);sys.exit(0)
        ray.init(address='local',num_cpus=16,num_gpus=cfg.num_gpus_per_node,namespace=RAY_NAMESPACE,include_dashboard=False,object_store_memory=2*1024**3,_temp_dir=f'/tmp/v2s_roll_ray_{os.getpid()}',runtime_env={'env_vars':cfg.system_envs})
        # Ray may return from init before the local GPU node is visible to
        # ray.nodes(); wait for the resource view used by ROLL's scheduler.
        for _ in range(30):
            if any(float(n.get('Resources', {}).get('GPU', 0)) >= cfg.num_gpus_per_node for n in ray.nodes()):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Ray GPU resources did not become visible after startup')
        state['state']='initializing';(WORK/'native_status.json').write_text(json.dumps(state))
        pipeline=AgenticPipeline(pipeline_config=cfg)
        state['state']='training';(WORK/'native_status.json').write_text(json.dumps(state))
        pipeline.run()
        state.update(state='complete',seconds=time.time()-started)
        print('DONE native_pipeline',flush=True)
    except BaseException as exc:
        state.update(state='failed',error_type=type(exc).__name__,error=str(exc)[-1000:],seconds=time.time()-started)
        raise
    finally:
        (WORK/'native_status.json').write_text(json.dumps(state,indent=2))
        ray.shutdown()
