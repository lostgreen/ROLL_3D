"""Regression checks for sampled-token loss spans; no model/GPU loading."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent.parent),str(Path(__file__).resolve().parent)]
from v2s_manager import Video2SceneManager
from token_history import preserve_responses

def case(prompt_ids, responses, full_ids, upstream_mask, should_fail=False):
    n=32
    ids=torch.tensor([full_ids+[0]*(n-len(full_ids))])
    mask=torch.zeros((1,n),dtype=torch.bool)
    for start,end in upstream_mask:mask[0,start:end]=True
    scores=torch.zeros((1,n));scores[0,int(mask[0].nonzero()[-1])]=.75
    data=SimpleNamespace(batch={'input_ids':ids,'response_mask':mask,
           'attention_mask':torch.ones((1,n),dtype=torch.long),'position_ids':torch.zeros((1,4,n)),
           'scores':scores},non_tensor_batch={'multi_modal_inputs':object()},meta_info={'metrics':{}})
    manager=object.__new__(Video2SceneManager)
    manager._prompt_ids=prompt_ids;manager._generated_ids=responses
    manager.tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:99,pad_token_id=0)
    manager.pipeline_config=SimpleNamespace(sequence_length=n)
    manager.format_messages=lambda _: (data,[])
    cache=SimpleNamespace(history=[{'reward':.75}],env_id=0,group_id=0,tag='test',step=len(responses))
    with TemporaryDirectory() as tmp:
        manager.env=SimpleNamespace(episode=Path(tmp))
        try:
            result=manager.formulate_rollouts(cache)
        except AssertionError:
            assert should_fail
            return
    assert not should_fail
    selected=result.batch['input_ids'][result.batch['response_mask']].tolist()
    assert selected==[t for turn in responses for t in turn]
    last=max(i for i,v in enumerate(result.batch['response_mask'][0]) if v)
    assert float(result.batch['scores'][0,last])==.75
    assert float(result.batch['scores'].sum())==.75
    assert not bool(result.batch['attention_mask'][0,last+1:].any())

# Length-truncated turn: EOS12 was inserted by the template, never sampled.
case([[1,99,2]],[[10,11]],[1,99,2,10,11,12],[(3,6)])
# Multi-turn history, template separator and image feedback stay outside loss.
case([[1,99,2],[1,99,2,10,11,12,3,99,2]],[[10,11],[15,16,12]],
     [1,99,2,10,11,12,3,99,2,15,16,12],[(3,6),(9,12)])
# Reject actual decode/re-encode drift or altered conditioning context.
case([[1,99,2]],[[10,11]],[1,99,2,10,17,12],[(3,6)],True)
case([[1,99,2]],[[10,11]],[4,99,2,10,11,12],[(3,6)],True)
# A decoded/re-encoded response can merge tokens; restore original IDs in
# both expanded training IDs and unexpanded inference prompt IDs.
assert preserve_responses([1,2,3,22,9,8,1,4,3], [[20,21]], [1,2,3],1,9,[8]) == [1,2,3,20,21,9,8,1,4,3]
assert preserve_responses([1,2,3,22,9,8], [[20,21,9]], [1,2,3],1,9,[8]) == [1,2,3,20,21,9,8]
try:
    preserve_responses([1,2,3,20,1,7,21,9,8], [[20,1,7,21,9]], [1,2,3],1,9,[8])
except AssertionError:
    pass
else:
    raise AssertionError('Generated ChatML boundaries must be rejected')
print('DONE token_contract 7 cases passed')
