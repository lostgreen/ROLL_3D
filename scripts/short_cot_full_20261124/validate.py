"""Closed schema plus catalog leakage gate and one narrow metadata cleanup."""
import copy
import re
import ast
import json
import unicodedata
from base_validate import strict_json,span_locations,numerical_mentions,validate as base_validate

def private_aliases(text,packet):
    if not isinstance(text,str):return []
    hits=set()
    for entry in packet.get('evidence_catalog',[]):
        key=entry['id']
        if re.search(r'(?<![\w:])'+re.escape(key)+r'(?![\w:])',text):hits.add(key)
    hits.update(m.group() for m in re.finditer(
        r'(?<![\w:])(?:image:\d+|msg:\d+|state:current|action:(?:current|L\d+)|tools:public|raw_cot:current)(?![\w:])',text))
    return sorted(hits)

def validate(obj,packet,tokenizer):
    result=base_validate(obj,packet,tokenizer)
    if isinstance(obj,dict):
        aliases=private_aliases(obj.get('short_cot'),packet)
        result['private_aliases']=aliases
        if aliases:
            result['errors'].append('private_catalog_keys_in_student_cot:'+','.join(aliases))
            result.update(passed=False,status='contract_failed')
        text=obj.get('short_cot')
        if isinstance(text,str):
            control=sorted(set('U+%04X'%ord(c) for c in text if unicodedata.category(c) in ['Cc','Cf'] and c not in '\n\t'))
            if control:result['errors'].append('student_control_characters:'+','.join(control))
            remainder=text
            for name in source_cjk_names(packet):
                for a,b in [('"','"'),("'","'"),('`','`'),('“','”'),('‘','’')]:remainder=remainder.replace(a+name+b,'')
            if re.search(r'[\u3400-\u9fff\uf900-\ufaff]',remainder):result['errors'].append('unquoted_or_non_source_CJK_in_student_cot')
            if control or re.search(r'[\u3400-\u9fff\uf900-\ufaff]',remainder):result.update(passed=False,status='contract_failed')
    result['schema_valid_review_flag']=bool(isinstance(obj,dict) and obj.get('status')=='needs_human_review'
        and result.get('status')=='needs_human_review' and not result.get('errors'))
    return result

def source_cjk_names(packet):
    names=set()
    try:
        state=json.loads(packet['student_messages'][-1]['content'].split(': ',1)[1])['public_scene_state']
        names.update(x['name'] for x in state.get('objects',[]) if isinstance(x.get('name'),str))
    except (KeyError,ValueError,IndexError,TypeError):pass
    for action in packet.get('current_action',[]):
        try:tree=ast.parse(action['function']['arguments'].get('code',''))
        except (SyntaxError,TypeError,KeyError):continue
        names.update(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
            and len(n.value)<=80 and re.fullmatch(r'[\w .:/-]+',n.value))
    return sorted((x for x in names if re.search(r'[\u3400-\u9fff\uf900-\ufaff]',x)),key=len,reverse=True)

def narrow_cleanup(obj):
    result=copy.deepcopy(obj);changes=[]
    if not isinstance(result,dict):return result,changes
    side=result.get('audit_sidecar')
    if not isinstance(side,dict) or not isinstance(side.get('claims'),list):return result,changes
    for i,c in enumerate(side['claims']):
        if isinstance(c,dict) and 'derivation_note' in c and c['derivation_note'] is None:
            del c['derivation_note'];changes.append('audit_sidecar.claims[%d].derivation_note:null'%i)
    return result,changes
