"""Pack native PolicyProxy outputs without re-rendering conversation history."""
import torch


def pack_tensors(output, prompt_ids, sampled_ids, *, sequence_length, pad_token_id, reward,
                 prompt_positions=None, forbidden_token_ids=()):
    ids = output['input_ids'][0]
    attention = output['attention_mask'][0].bool()
    response = output['response_mask'][0].bool()
    if output['input_ids'].shape[0] != 1:
        raise ValueError('One generation per environment decision is required')
    length = int(attention.sum())
    if not attention[:length].all() or attention[length:].any():
        raise ValueError('Native output must use right padding')
    expected_mask = torch.zeros_like(response)
    expected_mask[len(prompt_ids):len(prompt_ids) + len(sampled_ids)] = True
    if length != len(prompt_ids) + len(sampled_ids) or not torch.equal(response, expected_mask):
        raise ValueError('Response positions differ from the actual sampled decision')
    if ids[:len(prompt_ids)].tolist() != prompt_ids or ids[response].tolist() != sampled_ids:
        raise ValueError('Native prompt or sampled response IDs changed')
    if not sampled_ids:
        raise ValueError('Cannot train an empty decision')
    if set(sampled_ids).intersection(forbidden_token_ids):
        raise ValueError('Sampled response contains input-only visual or message boundary tokens')
    if length > sequence_length:
        raise ValueError('Decision exceeds sequence_length; training must not truncate it')
    positions = output['position_ids']
    if prompt_positions is not None and not torch.equal(positions[..., :len(prompt_ids)], prompt_positions):
        raise ValueError('Multimodal prompt positions changed')

    tensors = {key: output[key][..., :length].clone() for key in
               ('input_ids', 'attention_mask', 'position_ids', 'response_mask', 'prompt_mask')}
    prompt_mask = attention & ~response
    if not torch.equal(output['prompt_mask'][0].bool(), prompt_mask):
        raise ValueError('Native prompt_mask is inconsistent')
    tensors['scores'] = torch.zeros_like(tensors['input_ids'], dtype=torch.float)
    tensors['scores'][0, length - 1] = reward
    for key, value in list(tensors.items()):
        padding = value.new_full((*value.shape[:-1], sequence_length - length),
                                 pad_token_id if key == 'input_ids' else 0)
        tensors[key] = torch.cat((value, padding), dim=-1)
    if 'infer_logprobs' in output:
        values = output['infer_logprobs'][..., :length - 1].clone()
        tensors['infer_logprobs'] = torch.cat((values, values.new_zeros(
            (*values.shape[:-1], sequence_length - length))), dim=-1)
    return tensors
