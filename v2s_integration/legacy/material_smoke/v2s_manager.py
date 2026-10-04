from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager
from roll.utils.functionals import pad_to_length, aggregate_metrics
from token_history import preserve_responses
from v2s_env import register
import inspect
import json
import numpy as np
import torch

class Video2SceneManager(VLTrajEnvManager):
    def __init__(self,*args,**kwargs):
        register()
        super().__init__(*args,**kwargs)

    def reset(self):
        self._generated_ids=[]
        self._prompt_ids=[]
        return super().reset()

    def format_messages(self, rollout_cache):
        data, messages = super().format_messages(rollout_cache)
        header=self.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
        start_id=self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        newline=self.tokenizer.encode('\n',add_special_tokens=False)
        def restore(ids):
            return preserve_responses(ids,self._generated_ids,header,start_id,self.tokenizer.eos_token_id,newline)
        ids=restore(data.batch['input_ids'][0].tolist())
        data.batch['input_ids']=torch.tensor([ids],dtype=torch.long)
        data.batch['attention_mask']=torch.ones_like(data.batch['input_ids'])
        # Inference uses unexpanded image placeholders; training uses expanded IDs.
        if 'multi_modal_data' in data.non_tensor_batch:
            item=data.non_tensor_batch['multi_modal_data'][0]
            item['prompt_token_ids']=restore(item['prompt_token_ids'])
            item['_v2s_expected_prompt_ids']=ids
        kwargs={}
        mm=data.non_tensor_batch['multi_modal_inputs'][0]
        for name,param in inspect.signature(self.extra_data_provider).parameters.items():
            if name in data.batch:kwargs[name]=data.batch[name]
            elif name in mm:kwargs[name]=mm[name]
            elif param.default is not inspect.Parameter.empty:kwargs[name]=param.default
        data.batch.update(self.extra_data_provider(**kwargs))
        self._latest_prompt_ids=ids
        return data,messages

    def make_decision(self,rollout_cache):
        result=super().make_decision(rollout_cache)
        if result.batch is None:
            raise RuntimeError('No sampled response: context exhausted or generation aborted; refusing an invalid training trajectory')
        self._generated_ids.append(result.batch['responses'][0].tolist())
        self._prompt_ids.append(self._latest_prompt_ids)
        return result

    def formulate_rollouts(self,rollout_cache):
        if 'observation' in rollout_cache.history[-1]:rollout_cache.history.pop(-1)
        data,messages=self.format_messages(rollout_cache)
        ids=data.batch['input_ids'][0]
        mask=torch.zeros_like(ids,dtype=torch.bool)
        spans=[]
        ids_list=ids.tolist()
        cursor=0
        def locate(response, lower_bound):
            """Find the verbatim sampled response after template re-rendering.

            Qwen3.5's Transformers 5 chat template can tokenize the assistant
            boundary differently when `add_generation_prompt` changes between
            rollout and finalization.  The response tokens inserted by
            `preserve_responses` remain exact, so locate those subsequences
            instead of assuming the old prompt length is still an offset.
            """
            n=len(response)
            for start in range(max(0, lower_bound), len(ids_list)-n+1):
                if ids_list[start:start+n]==response:
                    return start
            return None
        for prompt,response in zip(self._prompt_ids,self._generated_ids):
            # Allow a small template-boundary shift, but never search before
            # the previous response. This keeps the loss on sampled tokens.
            start=locate(response,max(cursor,len(prompt)-16))
            end=None if start is None else start+len(response)
            prompt_match=start is not None and ids_list[:len(prompt)]==prompt
            response_match=start is not None and ids_list[start:end]==response
            spans.append({'start':start,'length':len(response),
                          'prompt_match':prompt_match,
                          'response_match':response_match})
            if start is not None:mask[start:end]=True;cursor=end
        expected=[t for turn in self._generated_ids for t in turn]
        image_id=self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        audit={'generated_tokens':len(expected),'loss_tokens':int(mask.sum()),'turns':spans,
               'exact_generated_token_match':ids[mask].tolist()==expected,
               'image_tokens':int((ids==image_id).sum()),
               'image_tokens_in_loss':int(((ids==image_id)&mask).sum()),
               'multimodal_features_present':'multi_modal_inputs' in data.non_tensor_batch,
               'position_ids_shape':list(data.batch['position_ids'].shape),
               'token_history_mode':'verbatim_sampled_responses'}
        (self.env.episode/'training_contract.json').write_text(json.dumps(audit,indent=2))
        assert mask.any() and audit['image_tokens']>0 and audit['image_tokens_in_loss']==0
        assert audit['multimodal_features_present']
        assert all(s['response_match'] for s in spans), 'Sampled response tokens changed during trajectory reconstruction'
        assert audit['exact_generated_token_match']
        first=int(mask.nonzero()[0]);last=int(mask.nonzero()[-1])
        assert last<self.pipeline_config.sequence_length
        scores=[h['reward'] for h in rollout_cache.history]
        score_tensor=torch.zeros_like(ids,dtype=torch.float);score_tensor[last]=sum(scores)
        prompt_mask=torch.arange(len(ids))<first
        data.batch['response_mask']=mask.unsqueeze(0)
        data.batch['prompt_mask']=prompt_mask.unsqueeze(0)
        data.batch['scores']=score_tensor.unsqueeze(0)
        for key in ('input_ids','attention_mask','position_ids','response_mask','prompt_mask','scores'):
            value=data.batch[key][...,:last+1]
            data.batch[key]=pad_to_length(value,length=self.pipeline_config.sequence_length,
                                         pad_value=self.tokenizer.pad_token_id if key=='input_ids' else 0)
        data.non_tensor_batch.update({
            'env_ids':np.array([rollout_cache.env_id],dtype=object),
            'group_ids':np.array([rollout_cache.group_id],dtype=object),
            'messages_list':np.array([messages],dtype=object),
            'tags':np.array([rollout_cache.tag],dtype=object),
            'step_scores':np.array([scores],dtype=object),
            'episode_scores':np.array([sum(scores)],dtype=object)})
        env_metric=aggregate_metrics(history_metrics=[h.get('metrics',{}) for h in rollout_cache.history],
                                    metrics_agg_mode=rollout_cache.history[-1].get('metrics_agg_mode',{}))
        env_metric['num_actions']=rollout_cache.step
        metrics={f'env/{rollout_cache.tag}/{k}':v for k,v in env_metric.items()}
        metrics['env/response_length']=float(mask.sum())
        data.meta_info={'metrics':metrics}
        return data
