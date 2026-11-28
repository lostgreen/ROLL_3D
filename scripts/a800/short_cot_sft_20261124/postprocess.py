"""Final views and compact comparison; no per-step image montage."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import html
import shutil
import statistics
from chain_common import *

def main():
    cfg=read(EVAL/'config.json');rows=cfg['rows'];gallery=EVAL/'comparison';gallery.mkdir(exist_ok=True)
    old=Path(cfg['repo'])/'artifacts/raw_cot_eval30_20261123'
    def render(row):
        sid=row['id'];state=EVAL/'runs'/VARIANT/sid/'status.json'
        if not state.exists() or read(state).get('state')!='complete':return sid,'no_completed_scene'
        log=EVAL/'exports'/VARIANT/sid/'render.log'
        try:
            run([cfg['blender'],'--background','--factory-startup','--threads','4','--python',
                 CODE/'eval_source/render_final.py','--','--root',EVAL,'--variant',VARIANT,'--id',sid,'--threads','4'],
                 log,dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=1800)
            return sid,'rendered'
        except Exception as e:return sid,str(e)[:180]
    with ThreadPoolExecutor(max_workers=4) as ex:rendered=dict(ex.map(render,rows))
    cases=[];cards=[]
    for row in sorted(rows,key=lambda r:(r['subset'],r['id'])):
        sid=row['id'];p=EVAL/'runs'/VARIANT/sid
        state=read(p/'status.json') if (p/'status.json').exists() else {'state':'missing'}
        mm=read(p/'metrics.json') if (p/'metrics.json').exists() else []
        total_tokens=sum((m.get('generation') or {}).get('generated_tokens',0) or 0 for m in mm)
        total_seconds=sum((m.get('generation') or {}).get('generation_seconds',0) or 0 for m in mm)
        baseline=old/('test20_native0171graph' if row['subset']=='test20' else 'official_chat_eos_v2')
        bstate=baseline/'runs/sft_step309'/sid/'status.json';bm=baseline/'runs/sft_step309'/sid/'metrics.json'
        bmetrics=read(bm) if bm.exists() else []
        btokens=sum((m.get('generation') or {}).get('generated_tokens',0) or 0 for m in bmetrics)
        item=dict(id=sid,subset=row['subset'],state=state.get('state'),steps=state.get('steps',state.get('step',0)),
                  termination_reason=state.get('termination_reason'),tool_failures=state.get('tool_failures'),
                  invalid_actions=state.get('invalid_actions'),generated_tokens=total_tokens,generation_seconds=total_seconds,
                  render=rendered[sid],baseline_state=read(bstate) if bstate.exists() else None,
                  baseline_generated_tokens=btokens if bmetrics else None,
                  baseline_backend='vLLM0.17.1/H800' if row['subset']=='test20' else 'HF/H800 diagnostic only',
                  trajectory_directory=str(p),quality_evaluated=False)
        cases.append(item)
        dest=gallery/sid;dest.mkdir(exist_ok=True)
        viewrows=[]
        for i in (0,1):
            views=[]
            sources=[('Reference',EVAL/'public'/sid/('reference_'+str(i)+'.jpg')),
                     ('Raw CoT SFT',baseline/'exports/sft_step309'/sid/('public_'+str(i)+'.jpg')),
                     ('Short CoT SFT',EVAL/'exports'/VARIANT/sid/('public_'+str(i)+'.jpg'))]
            for j,(label,src) in enumerate(sources):
                if src.is_file():
                    file='view'+str(i)+'_'+str(j)+'.jpg';shutil.copy2(src,dest/file)
                    views.append('<figure><figcaption>'+html.escape(label)+'</figcaption><img loading="lazy" src="'+sid+'/'+file+'"></figure>')
                else:views.append('<figure><figcaption>'+html.escape(label)+'</figcaption><p>Unavailable</p></figure>')
            viewrows.append('<div class="views">'+''.join(views)+'</div>')
        text=f"{sid} · {row['subset']} · {item['termination_reason']} · steps {item['steps']} · output {total_tokens:,} tokens · tool errors {item['tool_failures']}"
        cards.append('<article><h2>'+html.escape(text)+'</h2>'+''.join(viewrows)+'</article>')
    groups={}
    for subset in ('test20','train10_diagnostic'):
        rr=[r for r in cases if r['subset']==subset]
        groups[subset]=dict(cases=len(rr),complete=sum(r['state']=='complete' for r in rr),
             termination_counts=dict(Counter(r['termination_reason'] for r in rr)),
             total_generated_tokens=sum(r['generated_tokens'] for r in rr),
             mean_steps=statistics.mean(r['steps'] for r in rr),
             renders=sum(r['render']=='rendered' for r in rr),
             quality_evaluated=False)
    summary=dict(groups=groups,cases=cases,selection_sha256=cfg['selection_sha256'],
         checkpoint=read(ROOT/'final_checkpoint.json')['path'],base_rerun=False,
         note='Execution completion is not a reconstruction quality score. Test20 uses the same vLLM protocol; hardware differs. Train10 HF baselines are diagnostic.',at=time.time())
    dump(EVAL/'summary.json',summary)
    page='<!doctype html><meta charset="utf-8"><title>Short CoT SFT train10 + test20</title><style>body{background:#151820;color:#eee;font:15px system-ui;margin:24px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%}h2{font-size:17px}article{margin:32px 0}figcaption{padding:8px}</style><h1>Short CoT SFT · train10 + test20</h1><p>Fixed reference cameras. Final previews only. Raw/Short test20 share vLLM 0.17.1 and decode protocol; A800 versus H800. Train10 HF baseline is diagnostic. Missing images remain marked.</p>'+''.join(cards)
    (gallery/'index.html').write_text(page)
    dump(EVAL/'postprocess_status.json',dict(stage='complete',summary=str(EVAL/'summary.json'),gallery=str(gallery/'index.html'),at=time.time()))

if __name__=='__main__':main()
