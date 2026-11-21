"""Recover RGB-only SAM candidates on our reserved GPU, without touching labels.

Wait for an existing native-training claim to leave this GPU, then use the
already pinned local model. All outputs have a separate recovery directory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--hf-home',required=True);p.add_argument('--queue',type=Path,required=True)
    a=p.parse_args();out=a.work/'instance_cache_recovery_20261121';out.mkdir(exist_ok=True)
    state=out/'status.json';deadline=datetime.fromisoformat('2026-11-22T22:57:00+00:00').timestamp()
    def status(stage,**values):
        record=dict(stage=stage,host=socket.gethostname().split('.')[0],pid=os.getpid(),
                    updated_at=datetime.now(timezone.utc).isoformat(),**values)
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(state)
    recovery=a.queue/'jobs/rgb_sam_v2_s0_recovery_glaurung'
    try:
        while True:
            if time.time()>deadline:raise TimeoutError('Reservation/deadline limit reached')
            owner=recovery/'owner.json'
            if not owner.exists():status('waiting_for_training_claim');time.sleep(20);continue
            same_host=json.loads(owner.read_text())['host']==socket.gethostname().split('.')[0]
            if same_host and not (recovery/'done.json').exists():
                if (recovery/'failed.json').exists():raise RuntimeError('Training recovery failed; investigate before using GPU')
                status('waiting_for_existing_training');time.sleep(20);continue
            gpu=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
            memory,util=map(int,gpu.strip().split(','))
            if memory<500 and util<5:break
            status('waiting_for_idle_reserved_gpu',memory_mib=memory,utilization=util);time.sleep(20)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HF_HOME=a.hf_home,HF_HUB_OFFLINE='1',
                 TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        for dataset in ('isod','gte'):
            cache=out/f'{dataset}_cache'
            if (cache/'COMPLETE').exists():continue
            status('running_'+dataset+'_cache',frozen_semantic_labels_modified=False)
            cmd=[a.python,'-u',str(a.work/'code/cache_sam_prompt_study.py'),'--root',str(a.work),
                 '--config',str(out/f'{dataset}_config.json'),'--cache',str(cache),'--batch','2']
            with (out/f'{dataset}.log').open('a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',frozen_semantic_labels_modified=False,
               next_gate='Audit regenerated unions against accepted teacher masks; review instance proposals before training')
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
