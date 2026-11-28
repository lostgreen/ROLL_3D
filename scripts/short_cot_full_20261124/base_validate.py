"""Strict local contract checks. Passing this is NOT semantic acceptance."""
import json
import re
import unicodedata
from common import VERSION, CONFIG


def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate_key:'+k)
            result[k]=v
        return result
    def invalid(value):
        raise ValueError('nonfinite_json:'+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def span_locations(span,text):
    def safe_boundary(location):
        a,b=location
        if a and text[a:a+1].isalnum() and (text[a-1].isalnum() or text[a-1]=='_'):return False
        if a and text[a:a+1].isdigit() and text[a-1] in '+-−.':return False
        if b<len(text) and text[b-1:b].isalnum() and (text[b].isalnum() or text[b]=='_'):return False
        if b<len(text) and text[b-1:b].isdigit() and text[b]=='.' and text[b+1:b+2].isdigit():return False
        return True
    exact=[m.span() for m in re.finditer(re.escape(span),text) if safe_boundary(m.span())]
    if len(exact)==1:return 'exact',exact[0]
    # Only whitespace and non-numeric sentence-edge punctuation may differ.
    value=unicodedata.normalize('NFC',span).strip()
    value=re.sub(r'^[,;:!?“”"\s]+|[,;:!?“”"\s]+$','',value)
    if value.endswith('.') and len(value)>1 and not value[-2].isdigit():value=value[:-1]
    pattern=r'\s+'.join(re.escape(t) for t in value.split())
    hits=[m.span() for m in re.finditer(pattern,unicodedata.normalize('NFC',text)) if safe_boundary(m.span())] if pattern else []
    return ('normalized',hits[0]) if len(hits)==1 else ('unbound',None)


def numerical_mentions(text,packet):
    """Skip actual reference IDs, retaining unknown IDs and physical numbers."""
    references=set()
    for image in packet.get('images',[]):
        references.update(re.findall(r'\bref:\d+\b',image.get('label','')))
    identifiers=[m.span() for m in re.finditer(
        r'(?<!\w)(?:ref:|reference[ \t]+)(\d+)(?!\w)',text,re.I)
                 if 'ref:'+m.group(1) in references]
    for match in re.finditer(r'(?<!\w)[+-]?\d+(?:\.\d+)?',text):
        if not any(a<=match.start()<match.end()<=b for a,b in identifiers):
            yield match


def validate(obj,packet,tokenizer):
    errors=[];warnings=[];bindings=[]
    def require(condition,name):
        if not condition:errors.append(name)
    keys={'schema_version','sample_id','status','route_type','short_cot','audit_sidecar'}
    require(isinstance(obj,dict) and set(obj)==keys,'root_schema')
    if errors:return dict(passed=False,errors=errors,warnings=[])
    require(obj['schema_version']==VERSION,'schema_version')
    require(obj['sample_id']==packet['sample_id'],'sample_id')
    require(obj['route_type']==packet['route_type'],'route_type')
    require(obj['status'] in ['ok','needs_human_review'],'status')
    side=obj['audit_sidecar']
    require(isinstance(side,dict) and set(side)=={'claims','dropped_ungrounded_raw_cot','flag_reason'},'sidecar_schema')
    if errors:return dict(passed=False,errors=errors,warnings=[])
    require(isinstance(side['claims'],list),'claims_list')
    require(isinstance(side['dropped_ungrounded_raw_cot'],list) and all(isinstance(x,str) for x in side['dropped_ungrounded_raw_cot']),'dropped_list')
    if obj['status']=='needs_human_review':
        require(obj['short_cot'] is None,'flagged_cot_not_null')
        require(isinstance(side['flag_reason'],str) and bool(side['flag_reason'].strip()),'flag_reason')
        return dict(passed=False,status='needs_human_review',errors=errors,warnings=[],reason=side['flag_reason'])
    text=obj['short_cot']
    require(isinstance(text,str) and bool(text.strip()),'empty_cot')
    require(side['flag_reason'] is None,'unexpected_flag_reason')
    if not isinstance(text,str):return dict(passed=False,errors=errors,warnings=[])
    for marker in ['<think>','</think>','<tool_call>','</tool_call>','<|im_start|>','<|im_end|>']:
        require(marker not in text,'protocol_marker:'+marker)
    ids={x['id'] for x in packet['evidence_catalog']}
    bound=[]
    if isinstance(side['claims'],list):
        for i,claim in enumerate(side['claims']):
            tag='claim_%d:'%i
            if not isinstance(claim,dict) or set(claim)!={'span','epistemic_type','evidence_ids','derivation'}:
                errors.append(tag+'schema');continue
            span=claim['span'];kind=claim['epistemic_type'];refs=claim['evidence_ids']
            require(kind in ['known','derived','estimated','hypothesis','planned'],tag+'epistemic_type')
            require(isinstance(refs,list) and bool(refs) and all(isinstance(x,str) and x in ids for x in refs),tag+'evidence_ids')
            if isinstance(span,str) and span:
                match,location=span_locations(span,text)
                require(location is not None,tag+'span_not_unique_or_missing')
                if match=='normalized':warnings.append(tag+'normalized_span')
                bindings.append(dict(index=i,span=span,span_match=match,location=location))
                if location:bound.append(location)
            else:errors.append(tag+'span_type')
            derivation=claim['derivation']
            if kind=='derived':
                require(isinstance(derivation,dict) and set(derivation)=={'expression','input_ids','units','assumptions'},tag+'derivation_schema')
                if isinstance(derivation,dict):
                    require(isinstance(derivation.get('expression'),str) and isinstance(derivation.get('units'),str),tag+'derivation_types')
                    inp=derivation.get('input_ids');require(isinstance(inp,list) and all(x in ids for x in inp),tag+'derivation_inputs')
                    require(isinstance(derivation.get('assumptions'),list) and all(isinstance(x,str) for x in derivation['assumptions']),tag+'derivation_assumptions')
                warnings.append(tag+'derivation_requires_semantic_check')
            else:require(derivation is None,tag+'unexpected_derivation')
            if isinstance(refs,list) and refs==['raw_cot:current'] and kind=='known':
                warnings.append(tag+'raw_cot_only_known_claim')
    tokens=len(tokenizer(text,add_special_tokens=False)['input_ids'])
    require(tokens<=CONFIG['budgets'][packet['route_type']],'soft_budget_exceeded')
    for m in numerical_mentions(text,packet):
        if not any(a<=m.start()<m.end()<=b for a,b in bound):warnings.append('number_outside_claim:'+m.group())
    if re.search(r'\b(fixed (?:next )?call|provided code|reviewer|annotation|teacher|as calculated earlier)\b',text,re.I):
        warnings.append('possible_annotation_context_leak')
    return dict(passed=not errors,status='contract_passed' if not errors else 'contract_failed',
                errors=errors,warnings=warnings,student_cot_tokens=tokens,
                budget=CONFIG['budgets'][packet['route_type']],span_bindings=bindings,
                semantic_accepted=False)
