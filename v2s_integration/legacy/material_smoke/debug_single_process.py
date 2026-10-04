"""Diagnostic fallback: real ROLL MM/GRPO components, HF+LoRA, no Ray scheduler.

This is explicitly NOT the native distributed AgenticPipeline. It isolates model,
environment, token alignment, and an optimizer update from scheduler/backend issues.
"""
import json
import os
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK.parent),str(WORK)]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',ROLL_OTEL_ENABLED='0')
import numpy as np
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from peft import LoraConfig,get_peft_model
from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import RewardNormalizationConfig
from roll.pipeline.agentic.utils import agentic_reward_norm,agentic_compute_advantage
from roll.utils.functionals import agg_loss
from v2s_env import Video2SceneEnv

MODEL='/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct'
OUT=WORK/'component_run';OUT.mkdir(exist_ok=False)
state={'mode':'ROLL_components_HF_LoRA_not_native_pipeline','state':'loading'}
def status(**fields):
    state.update(fields);(OUT/'status.json').write_text(json.dumps(state,indent=2))
    print('STATUS',json.dumps(fields),flush=True)

def main():
    set_seed(20260922)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    tokenizer=processor.tokenizer
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:0')
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.eval()
    collator=DataCollatorWithPaddingForMM(tokenizer=tokenizer,processor=processor,answer_key=None,image_flag_key=None,video_flag_key=None,return_infer_inputs=False)
    def build(messages,images):
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=collator([{'prompt':text,'image':images}])
        inputs={k:batch[k] for k in ('input_ids','attention_mask')}
        inputs.update(dict(batch['multi_modal_inputs'][0]))
        return {k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in inputs.items()}
    def log_probs(sample):
        inputs={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in sample['inputs'].items()}
        ids=inputs['input_ids']
        # Explicit multimodal positions avoid cached rope deltas from previous generations.
        rope_model=base if hasattr(base,'get_rope_index') else base.model
        position_ids,_=rope_model.get_rope_index(ids,image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'])
        output=model(**inputs,position_ids=position_ids,use_cache=False)
        start=sample['prompt_len']-1
        logits=output.logits[:,start:-1,:].float()
        labels=ids[:,sample['prompt_len']:]
        selected=logits.gather(-1,labels.unsqueeze(-1)).squeeze(-1)-logits.logsumexp(-1)
        return selected
    episodes=[]
    for ep in range(4):
        env=Video2SceneEnv(output_root=WORK/'component_episodes',max_steps=3)
        samples=[];messages=[{'role':'system','content':'You are a visual Blender agent. Use the explicit tool_call protocol.'}];images=[]
        try:
            obs,_=env.reset(seed=42)
            for turn in range(3):
                messages.append({'role':'user','content':obs['prompt']});images.extend(obs['image'])
                inputs=build(messages,images);prompt_len=inputs['input_ids'].shape[1]
                image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
                assert (inputs['input_ids']==image_id).any() and 'pixel_values' in inputs
                with torch.no_grad():
                    ids=model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=1.,top_p=1.,top_k=0,use_cache=True,pad_token_id=tokenizer.pad_token_id)
                response=ids[0,prompt_len:]
                text=tokenizer.decode(response,skip_special_tokens=True)
                training_inputs=dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids))
                sample={'inputs':{k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in training_inputs.items()},'prompt_len':prompt_len,'response_tokens':len(response)}
                with torch.no_grad():sample['old_log_probs']=log_probs(sample).detach().cpu()
                assert sample['old_log_probs'].shape[1]==len(response)
                samples.append(sample)
                messages.append({'role':'assistant','content':text})
                obs,reward,done,truncated,info=env.step(text)
                status(state='rollout',episode=ep,turn=turn,response_tokens=len(response),done=done,reward=reward)
                if done:break
            torch.save(samples,env.episode/'training_samples.pt')
            episodes.append({'reward':reward,'samples':samples,'episode':str(env.episode)})
        finally:env.close()
    rewards=[e['reward'] for e in episodes]
    score_data=DataProto.from_dict({'scores':torch.tensor(rewards)},non_tensors={'traj_group_id':np.array(['same_task_seed42']*len(rewards),dtype=object)})
    advantages=agentic_reward_norm(score_data,RewardNormalizationConfig(grouping='traj_group_id',method='mean_std'))
    status(state='training',rewards=rewards,advantages=advantages.tolist())
    if not torch.isfinite(advantages).all() or float(advantages.abs().sum())==0:
        raise RuntimeError('No usable group reward variance; refusing to claim a policy update')
    trainable={name:param for name,param in model.named_parameters() if param.requires_grad}
    before={name:param.detach().cpu().clone() for name,param in trainable.items()}
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=1e-5,weight_decay=0.)
    model.gradient_checkpointing_enable();model.enable_input_require_grads()
    # Keep eval mode: LoRA dropout is zero and base-model dropout must match sampling.
    optimizer.zero_grad();losses=[];contracts=[]
    for index,episode in enumerate(episodes):
        for sample in episode['samples']:
            ids=sample['inputs']['input_ids']
            mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,sample['prompt_len']:]=True
            rewards_tensor=torch.zeros((1,ids.shape[1]-1));rewards_tensor[0,-1]=advantages[index]
            data=DataProto.from_dict({'input_ids':ids,'response_mask':mask,'token_level_rewards':rewards_tensor})
            data=agentic_compute_advantage(data,gamma=1.,lambd=1.,adv_estimator='grpo')
            adv=data.batch['advantages'][:,sample['prompt_len']-1:].to('cuda:0')
            new=log_probs(sample);old=sample['old_log_probs'].to('cuda:0')
            ratio=(new-old).exp()
            loss_matrix=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
            loss=agg_loss(loss_matrix,torch.ones_like(loss_matrix),'seq-mean-token-mean')/len(episodes)/len(episode['samples'])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite policy loss')
            loss.backward();losses.append(float(loss.detach()))
            image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
            contracts.append({'prompt_tokens':sample['prompt_len'],'response_tokens':sample['response_tokens'],'image_tokens':int((ids==image_id).sum()),'image_tokens_in_loss':int(((ids==image_id)&mask).sum()),'ratio_max_error':float((ratio.detach()-1).abs().max())})
    grad_norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
    if not torch.isfinite(grad_norm) or float(grad_norm)<=0:raise RuntimeError('Invalid or zero policy gradient')
    optimizer.step()
    delta=sum(float((param.detach().cpu()-before[name]).abs().sum()) for name,param in trainable.items())
    assert delta>0 and all(c['image_tokens_in_loss']==0 for c in contracts)
    checkpoint=OUT/'checkpoint';model.save_pretrained(checkpoint);processor.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt')
    # Read saved LoRA tensors and verify finite checkpoint contents (not a resume test).
    from safetensors.torch import load_file
    saved=load_file(str(checkpoint/'adapter_model.safetensors'))
    assert saved and all(torch.isfinite(t).all() for t in saved.values())
    summary={'mode':state['mode'],'rewards':rewards,'advantages':advantages.tolist(),'optimizer_steps':1,'loss':sum(losses),'grad_norm':float(grad_norm),'parameter_l1_delta':delta,'checkpoint_tensor_count':len(saved),'training_contracts':contracts,'episodes':[e['episode'] for e in episodes]}
    (OUT/'result.json').write_text(json.dumps(summary,indent=2))
    status(state='complete',optimizer_steps=1,grad_norm=float(grad_norm),parameter_l1_delta=delta)
    print('DONE component_train',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(state='failed',error_type=type(exc).__name__,error=str(exc)[:500]);raise
