"""Deterministic request-local decision evidence. No model-based annotations."""
import argparse,json,re
from pathlib import Path
from trajectory_ir import read_events
from recording_checks import visible_result


def logical(name): return (name or '').replace('__','.')


def classify(calls):
    mapping={'asset.search':'search','asset.inspect':'inspect','asset.import':'import',
             'scene.import_artifact':'import','execute_blender_code':'native_code',
             'render_scene_view':'render_or_inspect_scene','render_scene_views':'render_or_inspect_scene',
             'get_scene_info':'render_or_inspect_scene','get_object_info':'render_or_inspect_scene','scene.list_objects':'render_or_inspect_scene'}
    return sorted({mapping.get(logical(c['name']),'other') for c in calls}) if calls else ['final_answer']


def project(root,rules=None):
    root=Path(root)
    if not (root/'events.jsonl').exists(): return {'status':'unsupported','reason':'summary_only'}
    rows=list(read_events(root));requests={p['request_id']:(e,p) for e,p in rows if e['event']=='model.request'}
    tools=[(e,p) for e,p in rows if e['event']=='tool.end'];ledger={};decisions=[]
    def entry(aid):
        return ledger.setdefault(aid,dict(asset_id=aid,first_returned_step=None,returned_by_queries=[],first_visible_step=None,
               inspected_steps=[],preview_first_visible_step=None,imported_steps=[],scene_import_steps=[],final_present=None,final_visible_px=None))
    for e,p in tools:
        name=logical(p.get('name'));args=p.get('arguments',{});aid=args.get('asset_id')
        if name=='asset.search':
            try: candidates=json.loads(p['result']['text']).get('assets',[])
            except (ValueError,KeyError): candidates=[]
            for a in candidates:
                row=entry(a['asset_id'])
                if row['first_returned_step'] is None:row['first_returned_step']=e.get('step')
                row['returned_by_queries'].append(args.get('query',''))
        elif aid and name=='asset.inspect':entry(aid)['inspected_steps'].append(e.get('step'))
        elif aid and name=='asset.import':entry(aid)['imported_steps'].append(e.get('step'))
        elif name=='scene.import_artifact':
            for te,tp in tools:
                if te['seq']>=e['seq'] or logical(tp.get('name'))!='asset.import':continue
                try:body=json.loads(tp['result']['text'])
                except (ValueError,KeyError):continue
                if body.get('artifact_id')==args.get('artifact_id') and body.get('asset_id'):
                    entry(body['asset_id'])['scene_import_steps'].append(e.get('step'))
    for e,p in rows:
        if e['event']!='model.response' or p.get('request_id') not in requests:continue
        qe,q=requests[p['request_id']]
        if q.get('purpose','main')!='main':continue
        visible=[];candidates=set();previews=set()
        for te,tp in tools:
            if te['seq']>=qe['seq']:continue
            match=visible_result(tp,q)
            if not match['text_visible']:continue
            visible.append(dict(call_id=tp.get('call_id'),name=tp.get('name'),**match))
            if logical(tp.get('name'))=='asset.search':
                try:candidates.update(a['asset_id'] for a in json.loads(tp['result']['text']).get('assets',[]))
                except (ValueError,KeyError):pass
            if logical(tp.get('name'))=='asset.inspect' and match['image_hashes'] and match['images_visible']:
                previews.add(tp['arguments']['asset_id'])
        calls=[dict(c,name=logical(c['name']),call_index=i,evidence='derived') for i,c in enumerate(p.get('tool_calls',[]))]
        inspected={c.get('arguments',{}).get('asset_id') for c in calls if c['name']=='asset.inspect'}
        queries=[c.get('arguments',{}).get('query','') for c in calls if c['name']=='asset.search']
        search_sets=[]
        for c in calls:
            aid=c.get('arguments',{}).get('asset_id')
            if c['name']=='asset.import':
                import_seq=next((te['seq'] for te,tp in tools if tp.get('call_id')==c.get('id')),e['seq'])
                c.update(import_without_seen_preview=aid not in previews,inspect_and_import_same_response=aid in inspected,
                         import_without_inspect_anywhere=not any(te['seq']<import_seq and logical(tp.get('name'))=='asset.inspect' and tp.get('arguments',{}).get('asset_id')==aid for te,tp in tools))
            if c['name']=='asset.search':
                query=c.get('arguments',{}).get('query','')
                c['intent']={'evidence':'heuristic','targets':[k for k,v in (rules or {}).items() if any(re.search(r'\b'+re.escape(word.lower())+r'\b',query.lower()) for word in v)] or ['unknown']}
                targets=set(c['intent']['targets'])-{'unknown'}
                visible_ids={v['call_id'] for v in visible}
                prior_queries=[tp.get('arguments',{}).get('query','') for te,tp in tools if tp.get('call_id') in visible_ids and logical(tp.get('name'))=='asset.search']
                c['query_reformulation']=bool(targets) and any(old!=query and any(re.search(r'\b'+re.escape(w.lower())+r'\b',old.lower()) for target in targets for w in (rules or {}).get(target,[])) for old in prior_queries)
                c['duplicate_query_same_response']=queries.count(query)>1
                result=next((tp for te,tp in tools if te['seq']>e['seq'] and tp.get('call_id')==c.get('id')),None)
                try:aset=tuple(sorted(a['asset_id'] for a in json.loads(result['result']['text']).get('assets',[]))) if result else None
                except (ValueError,KeyError):aset=None
                search_sets.append((c,aset))
        for c,aset in search_sets:c['duplicate_results_same_response']=aset is not None and sum(s==aset for _,s in search_sets)>1
        for aid in candidates:
            row=entry(aid)
            if row['first_visible_step'] is None:row['first_visible_step']=e.get('step')
        for aid in previews:
            row=entry(aid)
            if row['preview_first_visible_step'] is None:row['preview_first_visible_step']=e.get('step')
        decisions.append(dict(request_id=p['request_id'],step=e.get('step'),evidence='derived',calls=calls,batched=len(calls)>1,
            visible_tool_results=visible,visible_candidates=sorted(candidates),visible_previews=sorted(previews),
            text=p.get('text'),action_labels=classify(calls)))
    return dict(schema='behavior_ir.v1',status='supported',decisions=decisions,candidates=list(ledger.values()),
                limitations=['final visibility requires offline render audit; null is unknown','intent rules are heuristic','native_code labels tool choice, not the actual code operation'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('--out',required=True);p.add_argument('--rules');a=p.parse_args()
 result=project(a.episode,json.loads(Path(a.rules).read_text()) if a.rules else None);Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'status':result['status'],'decisions':len(result.get('decisions',[])),'candidates':len(result.get('candidates',[]))}))
