"""CPU re-pack, target-only exclusions, and an independent post-run review sample."""
from collections import Counter
import copy
import gzip
import html
import json
import math
from pathlib import Path
import random
import re
import shutil
import time
from common import OUT,CONFIG,EXCLUSIONS,dump,read,sha,digest,cpu_processor
from prepare import load_packet
from validate import validate

def quantiles(values):
    if not values:return {}
    v=sorted(values)
    return dict(n=len(v),p50=v[max(0,math.ceil(len(v)*.50)-1)],p95=v[max(0,math.ceil(len(v)*.95)-1)],max=v[-1])

def sampling(rows,allrows):
    rng=random.Random(CONFIG['review_seed']);groups={}
    for row in rows:groups.setdefault(row['route_type'],[]).append(row)
    selected=[];scene_counts=Counter();remaining=55
    for route,group in sorted(groups.items(),key=lambda x:len(x[1])):
        quota=max(1,round(55*len(group)/len(rows)));quota=min(quota,len(group),remaining)
        ordered=sorted(group,key=lambda x:x['sample_id']);rng.shuffle(ordered)
        ordered.sort(key=lambda x:scene_counts[x['scene_id']])
        take=ordered[:quota];selected.extend(take);scene_counts.update(x['scene_id'] for x in take);remaining-=len(take)
    pool=[x for x in rows if x['sample_id'] not in {y['sample_id'] for y in selected}];rng.shuffle(pool)
    while len(selected)<min(55,len(rows)):
        pool.sort(key=lambda x:scene_counts[x['scene_id']]);x=pool.pop(0);selected.append(x);scene_counts[x['scene_id']]+=1
    if len(selected)>55:selected=selected[:55]
    chosen={x['sample_id'] for x in selected}
    def risk(x):return (int(x['state']!='ok'),(x.get('content_attempt') or 1)-1,int(x.get('prior_error_count',0)>0),len(x.get('warnings',[])),x.get('student_prompt_tokens',0))
    risks=sorted((x for x in allrows if x['sample_id'] not in chosen),key=risk,reverse=True)[:5]
    return dict(seed=CONFIG['review_seed'],random_ids=[x['sample_id'] for x in selected],risk_ids=[x['sample_id'] for x in risks],
        random_selection='Seeded route-stratified sample with room spread; selected without reading semantic outcomes.',
        risk_selection='Separate: model flags/failures, retries, historical errors, validator warnings, long context.',
        selected_after_annotation=True,reviewer='Agent; not API semantic checker',completed=False,training_release=False)

def main():
    import torch
    torch.set_num_threads(1)
    processor,tensor_features=cpu_processor();manifest=read(OUT/'manifest.json')
    decisions=read(OUT/'agent_review_decisions.json') if (OUT/'agent_review_decisions.json').exists() else {}
    rows=[];index=[];exclusions=[dict(sample_id=k,**v) for k,v in EXCLUSIONS.items()];started=time.monotonic()
    for item in manifest['rows']:
        sid=item['sample_id'];name=sid.replace(':','_');package=OUT/'packages'/name
        receipt=read(package/'input_receipt.json');old=receipt['eligible_row'];packet=load_packet(name)
        selected=read(OUT/'rewrites'/name/'selection.json')
        row=dict(item,state='ok' if selected.get('accepted_contract') else 'flagged' if selected.get('model_requested_review') else 'failed',
            content_attempt=selected.get('content_attempt'),local_cleanup=selected.get('local_cleanup',False),packing_passed=False)
        if selected.get('attempt'):
            attempt=Path(selected['attempt']);obj=read(attempt/'rewrite_output.json');v=validate(obj,packet,processor.tokenizer)
            row.update(warnings=v.get('warnings',[]),short_cot_tokens=v.get('student_cot_tokens'),flag_reason=obj['audit_sidecar'].get('flag_reason'),attempt=str(attempt))
            assert v['passed'] or v.get('schema_valid_review_flag'),sid
        else:obj=None
        decision=decisions.get(sid,{})
        if selected.get('accepted_contract') and decision.get('exclude'):
            row.update(state='excluded_by_agent',agent_evidence=decision)
            exclusions.append(dict(sample_id=sid,reason='explicit_fact_or_source_error',evidence=decision))
        elif selected.get('accepted_contract'):
            source=json.load(gzip.open(old['record_path'],'rt'));derived=copy.deepcopy(source)
            derived['target']['reasoning_content']=obj['short_cot']
            assert derived['messages']==source['messages'] and derived['images']==source['images']
            assert {k:v for k,v in derived['target'].items() if k!='reasoning_content'}=={k:v for k,v in source['target'].items() if k!='reasoning_content'}
            assert sha(old['record_path'])==old['record_sha256']
            candidate=OUT/'candidate_records'/name;candidate.mkdir(parents=True,exist_ok=True)
            path=candidate/'record.json.gz';validation_path=candidate/'packing_validation.json'
            previous=read(validation_path) if validation_path.exists() and path.exists() else None
            reuse=bool(previous and previous.get('passed') and previous.get('source_record_sha256')==old['record_sha256']
                and previous.get('short_cot_sha256')==digest(obj['short_cot'])
                and json.load(gzip.open(path,'rt'))==derived)
            if reuse:metrics=previous['metrics']
            else:feature,metrics,_=tensor_features(processor,derived)
            for k in ['prompt_tokens','prompt_sha256','prompt_ids_sha256','image_hashes','images','action_tokens','body_tokens']:assert metrics[k]==old[k],(sid,k)
            assert metrics['prefix_equal'] and metrics['parser_equal'] and metrics['total_tokens']<=98304
            if not reuse:
                labels=feature['batch']['labels'][0];n=metrics['prompt_tokens']
                assert labels[:n-1].eq(-100).all() and labels[-1].item()==-100
                assert int((labels!=-100).sum())==metrics['target_tokens'];del feature
                with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:f.write(json.dumps(derived,ensure_ascii=False).encode())
                dump(validation_path,dict(passed=True,metrics=metrics,source_record_sha256=old['record_sha256'],
                    short_cot_sha256=digest(obj['short_cot']),prefix_images_action_body_equal=True,loss_masks_recomputed=True,training_release=False))
            row['previous_verified_packing_reused']=reuse
            packed=dict(old);packed.update(metrics)
            packed.update(record_path=str(path),record_sha256=sha(path),
                source_record_path=old['record_path'],source_record_sha256=old['record_sha256'],
                annotation_version=CONFIG['prompt_version'],training_release=False)
            index.append(packed)
            row.update(packing_passed=True,short_cot_mask_tokens=metrics['cot_tokens'],short_cot=obj['short_cot'])
        else:
            exclusions.append(dict(sample_id=sid,reason='model_material_defect_flag' if selected.get('model_requested_review') else selected.get('reason'),
                evidence=row.get('flag_reason') or selected.get('last_errors')))
        rows.append(row)
        if len(rows)%25==0:dump(OUT/'export_status.json',dict(stage='cpu_repacking',finished=len(rows),total=len(manifest['rows']),retained=len(index),seconds=round(time.monotonic()-started,1)))
    index_path=OUT/'candidate_eligible_index.jsonl'
    with index_path.open('w') as f:
        for row in index:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    dump(OUT/'exclusion_manifest.json',dict(source_count=3277,retained=len(index),excluded=len(exclusions),items=exclusions,
        targets_only=True,discarded_steps_preserved_in_later_history=True,no_raw_fallback=True))
    review=sampling([r for r in rows if r['packing_passed']],rows);dump(OUT/'agent_review_sampling.json',review)
    reviewroot=OUT/'agent_review';(reviewroot/'images').mkdir(parents=True,exist_ok=True)
    chunks=['<!doctype html><meta charset="utf-8"><title>Full short-CoT review</title><style>body{max-width:1200px;margin:24px auto;font:15px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:45%;max-height:400px}section{border-top:2px solid #ccc;padding:20px 0}</style>',
        '<h1>55 stratified samples + 5 separate risk samples</h1><p>Program-validated candidates; Agent review pending. No API semantic checker or repair.</p>']
    byid={r['sample_id']:r for r in rows}
    for group in ['random_ids','risk_ids']:
        for sid in review[group]:
            row=byid[sid];packet=load_packet(sid.replace(':','_'))
            chunks.extend(['<section id="'+sid.replace(':','_')+'"><h2>'+sid+' / '+str(row['route_type'])+' / '+group+'</h2>',
                '<pre>'+html.escape(json.dumps({k:v for k,v in row.items() if k not in ['short_cot']},ensure_ascii=False,indent=2))+'</pre>',
                '<h3>Short CoT</h3><pre>'+html.escape(row.get('short_cot') or row.get('flag_reason') or row['state'])+'</pre>'])
            for image in packet['images']:
                dest=reviewroot/'images'/(image['sha256']+Path(image['path']).suffix)
                if not dest.exists():shutil.copy2(image['path'],dest)
                chunks.append('<img alt="'+html.escape(str(image.get('label')),quote=True)+'" src="images/'+dest.name+'">')
            for title,value in [('Immutable current tool action',packet['current_action']),('Original current Teacher CoT',packet['teacher_current_raw_cot'])]:
                chunks.append('<details><summary>'+title+'</summary><pre>'+html.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>')
            chunks.append('<p>Complete pre-decision packet: ../packages/'+sid.replace(':','_')+'/step_context_pack.json.gz</p></section>')
    (reviewroot/'index.html').write_text('\n'.join(chunks))
    lengths={route:quantiles([r['short_cot_tokens'] for r in rows if r.get('short_cot_tokens') is not None and r['route_type']==route]) for route in CONFIG['budgets']}
    headings=Counter(h for r in rows for h in re.findall(r'\[([A-Za-z /]+)\]',r.get('short_cot','')))
    ngrams=Counter()
    for r in rows:
        words=re.findall(r'[a-z]+',r.get('short_cot','').lower())
        ngrams.update(set(' '.join(words[i:i+8]) for i in range(max(0,len(words)-7))))
    stats=dict(source_count=3277,annotated=len(rows),preexcluded=len(EXCLUSIONS),retained=len(index),excluded=len(exclusions),states=dict(Counter(r['state'] for r in rows)),
        source_manifest_hash=manifest['source_manifest_hash'],config_sha256=digest(CONFIG),index_sha256=sha(index_path),
        route_lengths=lengths,headings=dict(headings),repeated_8grams=ngrams.most_common(20),
        all_retained_cpu_packing_passed=True,agent_review_pending=60,training_release=False,
        no_semantic_repair=True,no_api_checker=True,no_raw_fallback=True,completed_at=time.time())
    dump(OUT/'summary.json',stats);dump(OUT/'review_rows.json',rows)
    dump(OUT/'export_status.json',dict(stage='complete_pending_agent_review',finished=len(rows),retained=len(index),seconds=round(time.monotonic()-started,1)))
    progress=read(OUT/'progress.json');progress.update(stage='complete_pending_agent_review',retained=len(index),excluded=len(exclusions),updated_at=time.time());dump(OUT/'progress.json',progress)
    print(json.dumps({k:stats[k] for k in ['source_count','annotated','retained','excluded','agent_review_pending','training_release']}))

if __name__=='__main__':main()
