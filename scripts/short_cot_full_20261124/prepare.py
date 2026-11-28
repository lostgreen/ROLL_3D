"""Freeze membership, read-only sources, routing, and equivalent API packets."""
from collections import Counter
import copy
import functools
import gzip
import json
from pathlib import Path
import shutil
import time
from common import ROOT,DATA,OUT,INDEX,SOURCE_HASH,CONFIG,SYSTEM,EXCLUSIONS,dump,read,sha,digest
from routing import route_record
from presentation import sections

@functools.lru_cache(maxsize=8192)
def image_info(path,expected):
    from PIL import Image
    assert sha(path)==expected,('image_sha',path)
    with Image.open(path) as im:return list(im.size)

def load_packet(name):
    return json.load(gzip.open(OUT/'packages'/name/'step_context_pack.json.gz','rt'))

def pack(row,record):
    assert record['sample_id']==row['sample_id'] and record['step']==row['step']
    assert not record['content_escapes']
    assert all('reasoning_content' not in m and 'reasoning' not in m for m in record['messages'])
    route=route_record(record);action=record['target']['tool_calls'];code=action[0]['function']['arguments'].get('code','')
    messages=copy.deepcopy(record['messages']);images=[];iterator=iter(record['images'])
    catalog=[dict(id='tools:public',source='student_tools'),dict(id='state:current',source='student_messages',message_index=len(messages)-1),
        dict(id='action:current',source='current_action',action_sha256=digest(action)),
        dict(id='raw_cot:current',source='teacher_current_raw_cot',trust='fallible_candidate')]
    catalog.extend(dict(id='msg:%03d'%i,source='student_messages',message_index=i,role=m['role']) for i,m in enumerate(messages))
    catalog.extend(dict(id='action:L%04d'%i,source='current_action_code',line=i,line_sha256=digest(line)) for i,line in enumerate(code.splitlines(),1))
    for i,m in enumerate(messages):
        if not isinstance(m.get('content'),list):continue
        for part in m['content']:
            if part.get('type')!='image':continue
            ref=next(iterator);n=len(images);expected=row['image_hashes'][n]
            assert ref['sha256']==expected and (ref.get('generated_step') is None or ref['generated_step']<record['step'])
            image=dict(image_id='image:%d'%n,path=ref['path'],sha256=expected,size=image_info(ref['path'],expected),
                label=ref.get('label'),generated_step=ref.get('generated_step'),camera_id=ref.get('camera_id'),
                source_kind=ref.get('source_kind'),message_index=i)
            images.append(image);part['image_id']=image['image_id']
            catalog.append(dict(id=image['image_id'],source='student_visible_image',sha256=expected,message_index=i,
                generated_step=ref.get('generated_step'),label=ref.get('label')))
    assert next(iterator,None) is None and len(images)==len(row['image_hashes'])
    packet=dict(schema_version='step_context_pack_v1',sample_id=row['sample_id'],step=row['step'],
        route_type=route['type'],route_status=route['status'],soft_budget=CONFIG['budgets'].get(route['type']),
        student_messages=messages,student_tools=record['tools'],current_action=action,
        teacher_current_raw_cot=record['target']['reasoning_content'],images=images,evidence_catalog=catalog,
        camera_state={n:dict(state='unknown',basis='No static whole-history proof of current camera preservation; initial reference protocol remains valid') for n in ['PublicCamera0','PublicCamera1','render_selection']})
    _,canonical=sections(packet)
    folder=OUT/'packages'/row['sample_id'].replace(':','_');folder.mkdir(parents=True,exist_ok=True)
    with gzip.GzipFile(filename=str(folder/'step_context_pack.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(packet,ensure_ascii=False).encode())
    receipt=dict(sample_id=row['sample_id'],eligible_row=row,source_record_sha256=row['record_sha256'],
        source_assistant_sha256=record['source_assistant_sha256'],packet_sha256=digest(packet),
        canonical_api_packet_sha256=canonical,api_roundtrip_equal=True,route=route,
        current_and_future_results_included=False,previous_rewrites_injected=False,
        action_sha256=digest(action),normal_body_sha256=digest(record['target'].get('content','')),
        prefix_revalidated=False,prefix_check_required_before_request=True)
    dump(folder/'input_receipt.json',receipt)
    return dict(sample_id=row['sample_id'],scene_id=row['scene_id'],step=row['step'],route_type=route['type'],route_status=route['status'],
        packet_sha256=digest(packet),source_record_sha256=row['record_sha256'],raw_cot_tokens=row['cot_tokens'],
        student_prompt_tokens=row['prompt_tokens'],prior_error_count=len(route['prior_hard_errors']),
        action_tokens=row['action_tokens'],image_count=len(images),source_record_path=row['record_path'])

def refreeze_unlaunched():
    assert not any((OUT/'rewrites').glob('*')), 'Cannot change frozen inputs after API launch'
    old=read(OUT/'manifest.json');assert old['eligible_index_sha256']==sha(INDEX)
    previous_count=old['total'];previous_unresolved=old['unresolved_count']
    rows=[];unresolved=[]
    for item in old['rows']:
        if item['sample_id'] in EXCLUSIONS:continue
        if item['route_status']!='routed':
            receipt=read(OUT/'packages'/item['sample_id'].replace(':','_')/'input_receipt.json');source=receipt['eligible_row']
            assert sha(source['record_path'])==source['record_sha256']
            item=pack(source,json.load(gzip.open(source['record_path'],'rt')))
        rows.append(item)
        if item['route_status']!='routed':unresolved.append(item['sample_id'])
    assert len(rows)==3277-len(EXCLUSIONS)
    old.update(total=len(rows),preexcluded_count=len(EXCLUSIONS),rows=rows,unresolved_count=len(unresolved),
        route_counts=dict(Counter(x['route_type'] for x in rows)),config_sha256=digest(CONFIG),
        source_files={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()})
    dump(OUT/'manifest.json',old);dump(OUT/'rewrite_config.json',CONFIG);(OUT/'rewrite_system.txt').write_text(SYSTEM)
    dump(OUT/'pre_exclusions.json',dict(source_count=3277,items=EXCLUSIONS,target_only=True,history_removed=False))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,OUT/'source_snapshot'/p.name)
    dump(OUT/'routing_refreeze_receipt.json',dict(before_first_api=True,unresolved_before=previous_unresolved,unresolved_after=unresolved,
        original_candidates=previous_count,candidates=len(rows),source_exclusions=len(EXCLUSIONS),config_sha256=digest(CONFIG)))
    dump(OUT/'prepare_status.json',dict(stage='complete' if not unresolved else 'routing_needs_review',completed=len(rows),total=len(rows),unresolved=len(unresolved)))
    print(json.dumps(dict(candidates=len(rows),exclusions=len(EXCLUSIONS),routes=old['route_counts'],unresolved=unresolved)))
    assert not unresolved

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        manifest=read(OUT/'manifest.json');assert manifest['config_sha256']==digest(CONFIG) and manifest['eligible_index_sha256']==sha(INDEX)
        return
    allrows=[json.loads(x) for x in INDEX.open()];assert len(allrows)==3277 and len({x['scene_id'] for x in allrows})==199
    assert not any(x['scene_id']=='scene_171' for x in allrows)
    assert read(DATA/'formal_dry_run_body_clean_context_v2_fastpath_fix_001/manifest.json')['manifest_hash']==SOURCE_HASH
    policy=read(DATA/'all_body_clean_context_v2/context_policy_manifest.json')
    for name,h in policy['context_sources'].items():assert sha(DATA/'runtime_context_v2/v2s_integration/agent'/name)==h
    dump(OUT/'rewrite_config.json',CONFIG);(OUT/'rewrite_system.txt').write_text(SYSTEM)
    dump(OUT/'pre_exclusions.json',dict(source_count=3277,items=EXCLUSIONS,target_only=True,history_removed=False))
    source=OUT/'source_snapshot';source.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,source/p.name)
    rows=[];unresolved=[]
    for row in sorted(allrows,key=lambda x:x['sample_id']):
        if row['sample_id'] in EXCLUSIONS:continue
        assert sha(row['record_path'])==row['record_sha256']
        record=json.load(gzip.open(row['record_path'],'rt'))
        assert record['context_profile_sha256']==policy['context_profile_sha256']
        item=pack(row,record);rows.append(item)
        if item['route_status']!='routed':unresolved.append(dict(sample_id=row['sample_id'],action=record['target']['tool_calls'],route=read(OUT/'packages'/row['sample_id'].replace(':','_')/'input_receipt.json')['route']))
        if len(rows)%25==0:dump(OUT/'prepare_status.json',dict(stage='packing',completed=len(rows),total=3277-len(EXCLUSIONS),unresolved=len(unresolved),seconds=round(time.monotonic()-started,1)))
    dump(OUT/'routing_unresolved.json',dict(count=len(unresolved),items=unresolved))
    manifest=dict(version='short_cot_full_v32_cache_20261124',source_manifest_hash=SOURCE_HASH,source_count=3277,
        eligible_index_sha256=sha(INDEX),preexcluded_count=len(EXCLUSIONS),total=len(rows),scenes=199,
        config_sha256=digest(CONFIG),rows=rows,route_counts=dict(Counter(x['route_type'] for x in rows)),
        unresolved_count=len(unresolved),all_packets_api_equivalence_passed=True,
        source_files={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        source_policy_sha256=sha(DATA/'all_body_clean_context_v2/context_policy_manifest.json'),training_release=False)
    assert len(rows)==3277-len(EXCLUSIONS)
    dump(OUT/'manifest.json',manifest)
    dump(OUT/'prepare_status.json',dict(stage='complete' if not unresolved else 'routing_needs_review',completed=len(rows),total=len(rows),unresolved=len(unresolved),seconds=round(time.monotonic()-started,1)))
    print(json.dumps(dict(total=len(rows),routes=manifest['route_counts'],unresolved=len(unresolved))))
    if unresolved:raise RuntimeError('Unresolved routing must be decided before API launch; see routing_unresolved.json')

if __name__=='__main__':
    import sys
    refreeze_unlaunched() if '--refreeze-unlaunched' in sys.argv else main()
