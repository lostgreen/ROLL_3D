"""Detached bounded-resume training, then CPU merge and thirty-scene inference."""
import fcntl
import math
import os
import sys
import traceback
from chain_common import *

STATUS=ROOT/'chain_status.json'
def stage(name,**kw):dump(STATUS,dict(stage=name,pid=os.getpid(),at=time.time(),**kw))
def newest(manifest):
    good=[]
    for p in (ROOT/'runs').glob('attempt_*/checkpoint_step_*/progress.json'):
        try:
            v=read(p)
            if v['manifest_hash']==manifest:good.append((v['global_step'],p.parent))
        except (OSError,ValueError,KeyError):continue
    for _,p in sorted(good,reverse=True):
        try:validate_checkpoint(p,manifest);return p
        except (OSError,ValueError,KeyError,AssertionError):continue
    return None

def main():
    lock=(ROOT/'chain.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        stage('waiting_for_data')
        receipt=wait_receipt(ROOT/'prepare_receipt.json')
        wait_receipt(ROOT/'fast_kernel_receipt.json')
        manifest=receipt['manifest_hash']
        stage('longest_pair_smoke')
        smoke=ROOT/'smoke_long_001'
        if not (ROOT/'smoke_receipt.json').exists():
            if not (smoke/'status.json').exists():
                assert not smoke.exists(),'SMOKE_OUTPUT_ALREADY_EXISTS_REVIEW_REQUIRED'
                run([TRAIN,CODE/'training_source/train_raw_cot.py','--config',ROOT/'smoke_config.json',
                     '--root',smoke,'--execute','--max-updates','1'],ROOT/'smoke_long_001.log',timeout=7200)
            from verify_smoke import main as verify_smoke
            verify_smoke()
        assert read(ROOT/'smoke_receipt.json')['passed']
        for attempt in range(1,5):
            out=ROOT/'runs'/('attempt_'+str(attempt).zfill(2))
            if out.exists():continue
            cp=newest(manifest)
            if cp and read(cp/'progress.json')['global_step']==303:break
            stage('training',attempt=attempt,run=str(out),resume=str(cp) if cp else None,total_updates=303)
            cmd=[TRAIN,CODE/'training_source/train_raw_cot.py','--config',ROOT/'train_config.json','--root',out,'--execute']
            if cp:cmd+=['--resume',cp]
            try:
                run(cmd,ROOT/('train_attempt_'+str(attempt).zfill(2)+'.log'),timeout=72*3600)
                cp=newest(manifest)
                assert cp and read(cp/'progress.json')['global_step']==303,'TRAIN_EXIT_WITHOUT_FINAL_CHECKPOINT'
                assert read(out/'status.json')['formal_training_complete']
                break
            except Exception as e:
                dump(ROOT/('attempt_'+str(attempt).zfill(2)+'_failure.json'),dict(type=type(e).__name__,message=str(e)[:300],at=time.time()))
                if attempt==4:raise
                stage('retry_pending',attempt=attempt,error=str(e)[:240]);time.sleep(30)
        cp=newest(manifest);assert cp
        progress=validate_checkpoint(cp,manifest,303)
        dump(ROOT/'final_checkpoint.json',dict(path=str(cp),**progress))
        stage('waiting_for_eval_environment',final_checkpoint=str(cp))
        wait_receipt(ROOT/'native_upstream_integrity.json',timeout=7200)
        prep=read(EVAL/'preparation_status.json');assert prep['state']=='complete' and prep['train_ready']==10 and prep['test_ready']==20
        stage('merging_final_adapter')
        if not (ROOT/'merge_receipt.json').exists():
            run([TRAIN,CODE/'merge_final.py'],ROOT/'merge_final.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=7200)
        stage('inference_train10_test20')
        run([TRAIN,CODE/'evaluate30.py'],ROOT/'evaluate30.log',timeout=49*3600)
        stage('rendering_and_comparison')
        run([TRAIN,CODE/'postprocess.py'],ROOT/'postprocess.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=12*3600)
        stage('complete',checkpoint=str(cp),evaluation=str(EVAL/'summary.json'),gallery=str(EVAL/'comparison/index.html'),occupancy_enabled=False)
    except BaseException as e:
        (ROOT/'chain_error_traceback.log').write_text(traceback.format_exc())
        stage('failed',error_type=type(e).__name__,message=str(e)[:300],occupancy_enabled=False)
        raise

if __name__=='__main__':main()
