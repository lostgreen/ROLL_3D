"""Qwen ChatML history: retain sampled assistant tokens verbatim."""
def preserve_responses(ids, responses, header, start_id, eos_id, newline):
    assert all(start_id not in response for response in responses), 'Generated ChatML message boundaries are unsupported'
    starts=[i for i in range(len(ids)-len(header)+1) if ids[i:i+len(header)]==header]
    # An unfinished generation header may follow the completed assistant turns.
    assert len(starts) in (len(responses),len(responses)+1), 'Unexpected assistant boundary count'
    result=[];cursor=0
    for begin,response in zip(starts,responses):
        content=begin+len(header)
        end=next((i for i in range(content,len(ids)) if ids[i]==start_id),len(ids))
        result.extend(ids[cursor:content])
        result.extend(response)
        # End-of-turn tokens inserted by the template remain outside policy loss.
        if not response or response[-1]!=eos_id:result.append(eos_id)
        result.extend(newline)
        cursor=end
    result.extend(ids[cursor:])
    return result
