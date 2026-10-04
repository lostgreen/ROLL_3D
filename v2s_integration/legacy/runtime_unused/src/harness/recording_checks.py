"""Offline evidence checks shared by CLI and viewer; never infer missing history."""
import hashlib
import json
from pathlib import Path
from trajectory_ir import read_events, media_refs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refs(value):
    if isinstance(value, dict):
        if 'artifact_ref' in value:
            yield value
        for v in value.values():
            yield from refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from refs(v)


def episodes(path):
    path = Path(path)
    if (path / 'events.jsonl').is_file():
        return [path]
    return sorted(path.glob('trajectory/episode_*/events.jsonl')) and [p.parent for p in sorted(path.glob('trajectory/episode_*/events.jsonl'))]


def check_episode(root, *, events=None, require_scene=False, final_blend=None, verify_hashes=True):
    root = Path(root).resolve() if root else None
    if root is None or not (root/'events.jsonl').exists():
        return dict(kind='legacy_steps', level='summary_only', label='仅有步骤摘要',
                    missing=['events.jsonl'], reason='当前数据源没有完整 episode 事件；可能未记录，也可能未同步。图片计数不能替代图片文件。')
    missing, checked = [], set()
    rows = []
    try:
        rows = list(events if events is not None else read_events(root))
    except (OSError, ValueError, KeyError) as exc:
        missing.append('event_payload:' + type(exc).__name__)
    def verify(ref):
        key = (ref.get('artifact_ref'), ref.get('sha256'))
        if key in checked: return
        checked.add(key)
        p = (root / str(key[0])).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            missing.append('missing_blob:' + str(key[0]))
        elif verify_hashes and key[1] and sha(p) != key[1]:
            missing.append('hash_mismatch:' + str(key[0]))
    for event, payload in rows:
        verify(event)
        for ref in refs(payload): verify(ref)
    requests = [p for e,p in rows if e['event']=='model.request']
    responses = {p.get('request_id') for e,p in rows if e['event'] in ('model.response','model.error')}
    starts = {p.get('call_id') for e,p in rows if e['event']=='tool.start'}
    ends = {p.get('call_id') for e,p in rows if e['event']=='tool.end'}
    for q in requests:
        if q.get('request_id') not in responses: missing.append('response:'+str(q.get('request_id')))
        if 'messages' not in q or 'tools' not in q: missing.append('request_fields:'+str(q.get('request_id')))
    for c in starts - ends: missing.append('tool.end:'+str(c))
    if not requests: missing.append('model.request')
    if not any(e['event']=='episode.end' for e,p in rows): missing.append('episode.end')
    tool_images = sum(len(media_refs(p.get('result', {}))) for e,p in rows if e['event']=='tool.end')
    snapshot_count = sum(e['event']=='scene.snapshot' and p.get('status')=='ready' for e,p in rows)
    rendered = [(e,p) for e,p in rows if e['event']=='scene.render' and p.get('render_files')]
    for e,p in rendered:
        for file in p['render_files']:
            verify({'artifact_ref': file})
    mutations = [(e,p) for e,p in rows if e['event']=='tool.end' and p.get('name') in ('execute_blender_code','scene.import_artifact','scene.set_transform')]
    unavailable = []
    if not rendered: unavailable.append('scene_renders')
    if require_scene and mutations and not rendered: missing.append('scene_renders')
    if final_blend is not None and (not Path(final_blend).is_file() or Path(final_blend).stat().st_size == 0): missing.append('final.blend')
    return dict(kind='episode_events', label='episode 事件', level='partial' if missing else 'full',
                scope='interaction_and_requested_artifacts', missing=sorted(set(missing)), unavailable=unavailable,
                request_count=len(requests), tool_image_ref_count=tool_images, scene_snapshot_count=snapshot_count,
                scene_render_count=len(rendered), verified_hashes=verify_hashes,
                final_blend_checked=final_blend is not None)


def text_strings(v):
    if isinstance(v,str): yield v
    elif isinstance(v,dict):
        for k,x in v.items():
            if k not in ('artifact_ref','sha256'): yield from text_strings(x)
    elif isinstance(v,list):
        for x in v: yield from text_strings(x)


def visible_result(payload, request):
    """Call-id association plus exact text; images must be in this request."""
    messages = request.get('messages', [])
    call_id = payload.get('call_id')
    associated = []
    for msg in messages:
        if msg.get('tool_call_id') == call_id: associated.append(msg.get('content'))
        content = msg.get('content', [])
        if isinstance(content,list):
            associated.extend(b.get('content') for b in content if isinstance(b,dict) and b.get('type')=='tool_result' and b.get('tool_use_id')==call_id)
    # Non-ID adapters may only be matched by nonempty exact content, labelled accordingly.
    text = payload.get('result',{}).get('text','')
    haystack = list(text_strings(associated if associated else messages))
    match = any(text in s for s in haystack) if text else bool(associated)
    actual = {r.get('sha256') for r in media_refs(messages)}
    expected = {r.get('sha256') for r in media_refs(payload.get('result',{}))}
    return dict(text_visible=match, images_visible=expected <= actual,
                image_hashes=sorted(expected), missing_image_hashes=sorted(expected-actual),
                association='call_id' if associated else 'content_hash_fallback')


def feedback_check(root):
    rows=list(read_events(root)); checks=[]
    main=[(e,p) for e,p in rows if e['event']=='model.request' and p.get('purpose','main')=='main']
    for e,p in rows:
        if e['event']!='tool.end': continue
        later=next(((qe,q) for qe,q in main if qe['seq']>e['seq']),None)
        if later is None:
            checks.append(dict(call_id=p.get('call_id'),status='no_next_request')); continue
        qe,q=later; evidence=visible_result(p,q)
        compacted=any(x['event']=='model.request' and y.get('purpose')=='compaction' and e['seq']<x['seq']<qe['seq'] for x,y in rows)
        ok=evidence['text_visible'] and evidence['images_visible']
        checks.append(dict(call_id=p.get('call_id'), request_id=q.get('request_id'), step=e.get('step'),
                           name=p.get('name'), status='passed' if ok else 'not_delivered', compaction_between=compacted, **evidence))
    hashes={hashlib.sha256(json.dumps(q.get('tools',[]),sort_keys=True).encode()).hexdigest() for e,q in main}
    failures=sum(c['status']=='not_delivered' for c in checks)
    return dict(schema='feedback_delivery.v1', evidence='derived', status='passed' if not failures and len(hashes)<=1 else 'failed',
                failures=failures, toolset_constant=len(hashes)<=1,
                compactions=sum(e['event']=='model.request' and p.get('purpose')=='compaction' for e,p in rows),
                retries=sum(e['event']=='model.request' and p.get('attempt',0)>0 for e,p in rows), checks=checks)
