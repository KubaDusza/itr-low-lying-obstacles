"""Server-owned labeling, validation gates, and distributed SAM-supervised training queue."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

HOST=socket.gethostname().split('.')[0]
SHARED=Path('/home/GTL/jdusza/itr_sam_v2_queue_20261121')
DEADLINE=datetime.fromisoformat('2026-11-22T22:57:00+00:00').timestamp()
COMMON=['--dataset','isod','--raw_depth','--encoder','resnet34','--encoder_block','NonBottleneck1D',
        '--nr_decoder_blocks','3','--context_module','ppm','--decoder_channels_mode','decreasing',
        '--encoder_decoder_fusion','add','--fuse_depth_in_rgb_encoder','SE-add',
        '--upsampling','learned-3x3-zeropad','--height','448','--width','640','--batch_size_valid','8','--workers','4']


def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)

def status(work,stage,**values):
    record=dict(host=HOST,pid=os.getpid(),stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**values)
    write(work/'status.json',record);write(SHARED/'workers'/f'{HOST}.json',record)

def wait_for(work,path):
    while not path.exists():
        if time.time()>DEADLINE:raise TimeoutError(path)
        status(work,'waiting',dependency=str(path));time.sleep(15)

def run(work,command,log,env=None,cwd=None,heartbeat=None):
    print('RUN',command,flush=True)
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as output:
        child=subprocess.Popen(list(map(str,command)),stdout=output,stderr=subprocess.STDOUT,
                               env=env,cwd=cwd or work,start_new_session=True)
        try:
            while child.poll() is None:
                status(work,'running',command=list(map(str,command)),child_pid=child.pid,log=str(log),heartbeat=heartbeat)
                if time.time()>DEADLINE:raise TimeoutError('Reservation deadline')
                time.sleep(10)
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def env_for(work,cfg,seed=0,kind='210'):
    env=os.environ.copy();env.update(ITR_DATA=str(work),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
      PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],
      ESANET_SEED=str(seed),PYTHONPATH=str(work/'code')+(':'+cfg['extra_pythonpath'] if cfg.get('extra_pythonpath') else ''),
      YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    return env

def link(source,dest):
    if dest.exists() or dest.is_symlink():return
    dest.symlink_to(Path(source).resolve())

def audit_masks(root,expected,split=None):
    import cv2
    import numpy as np
    manifest=read(root/'manifest.json');rows=manifest['frames']
    ids={(r['site'],r['frame']) for r in rows}
    assert ids==set(map(tuple,expected)) and len(rows)==len(ids)
    if split:assert manifest['split']==split
    assert manifest.get('complete') is True
    for r in rows:
        file=root/r['path'];image=cv2.imread(str(file),0)
        assert image is not None and set(np.unique(image)).issubset({0,255}),file
        assert hashlib.sha256(file.read_bytes()).hexdigest()==r['sha256'],file
    return manifest

def unlabeled_test(work,source,dest):
    import cv2
    import numpy as np
    if (dest/'VERIFIED').exists():return
    dest.mkdir(exist_ok=True)
    for item in ('train','train.txt','valid','valid.txt','depth_stats.json'):link(source/item,dest/item)
    for folder in ('rgb','depth','labels_2'):(dest/'test'/folder).mkdir(parents=True,exist_ok=True)
    names=[]
    for site in read(work/'splits.json')['test']:
        for frame in range(100):
            name=f'{site}_{frame}';names.append(name)
            for kind in ('rgb','depth'):link(work/'ISOD/data'/site/kind/f'{frame}.png',dest/'test'/kind/f'{name}.png')
            image=cv2.imread(str(work/'ISOD/data'/site/'depth'/f'{frame}.png'),-1)
            assert cv2.imwrite(str(dest/'test/labels_2'/f'{name}.png'),np.ones(image.shape,np.uint8))
    (dest/'test.txt').write_text('\n'.join(names)+'\n');(dest/'VERIFIED').touch()

def prepare(work,cfg,smoke=False):
    env=env_for(work,cfg);py=cfg['python210']
    if HOST!='morgoth':
        source=Path(cfg['source'])
        for item in ('ISOD','ISOD_reference_val_v1','ISOD_reference_test_v1','OWN_manual'):link(source/item,work/item)
        if not (work/'esanet').exists():shutil.copytree(source/'esanet',work/'esanet',symlinks=True)
        shutil.copy2(source/'esanet_data_full_reference/depth_stats.json',work/'depth_stats.json')
    target=work/'esanet/src/datasets/isod/pytorch_dataset.py'
    shutil.copy2(work/'code/esanet/isod_dataset.py',target)
    assert '# ITR selects by obstacle IoU' in (work/'esanet/train.py').read_text()
    link(SHARED/'ISOD_sam_train_v2',work/'ISOD_sam_train_v2')
    limits=['--train-limit','32','--val-limit','16'] if smoke else []
    suffix='smoke' if smoke else 'full'
    dest=work/f'esanet_data_{suffix}_sam_v2'
    if not (dest/'VERIFIED').exists():
        if dest.exists():shutil.rmtree(dest)
        command=[py,'code/esanet/export_isod.py','--out',dest,'--train-masks',work/'ISOD_sam_train_v2',
                 '--val-masks',work/'ISOD_reference_val_v1','--test-masks',work/'ISOD_reference_test_v1',
                 '--depth-stats',work/'depth_stats.json']+limits
        if HOST=='morgoth' and not smoke:
            gte=work/'esanet_gte'
            if gte.exists():shutil.rmtree(gte)
            command+=['--gte',work/'OWN','--gte-out',gte]
        run(work,command,work/f'prepare_esa_{suffix}.log',env)
        (dest/'VERIFIED').touch()
    if HOST!='morgoth':link(Path(cfg['source'])/'esanet_gte',work/'esanet_gte')
    if not smoke:unlabeled_test(work,dest,work/'esanet_test_all')
    if HOST=='morgoth':
        yolo=work/f'yolo_{suffix}_sam_v2'
        if not (yolo/'polygon_audit.json').exists():
            if yolo.exists():shutil.rmtree(yolo)
            run(work,[cfg['python214'],'code/to_yolo.py','--out',yolo,'--train-masks',work/'ISOD_sam_train_v2',
                      '--val-masks',work/'ISOD_reference_val_v1','--test-masks',work/'ISOD_reference_test_v1']+limits,
                      work/f'prepare_yolo_{suffix}.log',env)

def label(work,cfg):
    wait_for(work,work/'data.READY');wait_for(work,work/'runtime.READY');wait_for(work,work/'esa210_setup.exit')
    assert (work/'esa210_setup.exit').read_text().strip()=='0'
    env=env_for(work,cfg);env['HF_HOME']=cfg['hf_home'];py=cfg['python210']
    for name,config,output in [('isod','isod_inference.json','automatic_isod_all'),('gte','gte_inference.json','automatic_gte')]:
        if not (work/output/'VERIFIED').exists():
            for batch in (8,2):
                try:
                    run(work,[py,'code/cache_sam_prompt_study.py','--config',config,'--root',work,'--cache',work/f'cache_{name}','--batch',str(batch)],work/f'label_{name}.log',env)
                    break
                except subprocess.CalledProcessError:
                    tail=(work/f'label_{name}.log').read_text()[-12000:].lower()
                    if batch==2 or 'out of memory' not in tail:raise
            run(work,[py,'code/apply_sam_prompt_recipe.py','--cache',work/f'cache_{name}','--config',config,
                      '--recipe','recipe.json','--out',work/output],work/f'export_{name}.log',env)
            audit_masks(work/output,read(work/config)['frames']);(work/output/'VERIFIED').touch()
    splits=read(work/'splits.json');whole=read(work/'automatic_isod_all/manifest.json')
    target=SHARED/'ISOD_sam_train_v2'
    if not (target/'VERIFIED').exists():
        target.mkdir(parents=True,exist_ok=True);rows=[]
        for row in whole['frames']:
            if row['site'] not in splits['train']:continue
            relative=Path(row['path']);dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(work/'automatic_isod_all'/relative,dest)
            rows.append({**row,'review_status':'unattended_sam_three_prompt_v2'})
        manifest={**whole,'split':'train','version':2,'complete':True,'recipe':read(work/'recipe.json'),'frames':rows}
        write(target/'manifest.json',manifest)
        audit_masks(target,[(s,i) for s in splits['train'] for i in range(100)],'train')
        (target/'VERIFIED').touch()
    write(SHARED/'recipe.json',read(work/'recipe.json'))
    (SHARED/'LABELS_READY').touch()
    run(work,[py,'code/score_frozen.py','--isod-reference',work/'ISOD_reference_test_v1','--isod-predictions',work/'automatic_isod_all',
              '--gte-reference',work/'OWN_manual','--gte-predictions',work/'automatic_gte','--out',work/'teacher_scores.json'],work/'score_teacher.log',env)

def latest_checkpoint(directory):
    import torch
    files=list(directory.glob('isod/checkpoints_*/ckpt_latest.pth'))
    if not files:return None,None
    pairs=[(p,torch.load(p,map_location='cpu',weights_only=False)) for p in files]
    return max(pairs,key=lambda pair:pair[1]['epoch'])

def execute(work,cfg,job,smoke=False):
    kind=job['kind'];seed=job['seed'];epochs=job['epochs'];out=work/('smoke' if smoke else 'runs')/job['id'];out.mkdir(parents=True,exist_ok=True)
    env=env_for(work,cfg,seed);py=cfg['python214'] if kind in ('unet','yolo','rgbd') else cfg['python210']
    write(out/'provenance.json',dict(job=job,recipe_sha256=hashlib.sha256((SHARED/'recipe.json').read_bytes()).hexdigest(),
                                  labels_manifest_sha256=hashlib.sha256((SHARED/'ISOD_sam_train_v2/manifest.json').read_bytes()).hexdigest(),
                                  python=py,gpu_uuid=cfg['gpu_uuid'],host=HOST))
    suffix='smoke' if smoke else 'full';fit=out/'fit_run'
    if kind in ('rgb','depth','rgbd'):
        dataset=work/f'esanet_data_{suffix}_sam_v2';modality=['--modality',kind]
        # Inspect checkpoints with the training interpreter, not the queue interpreter.
        probe='import pathlib,torch,json,sys; ps=list(pathlib.Path(sys.argv[1]).glob("isod/checkpoints_*/ckpt_latest.pth")); pairs=[(str(p),torch.load(p,map_location="cpu",weights_only=False)) for p in ps]; p,s=max(pairs,key=lambda a:a[1]["epoch"]) if pairs else (None,{}); print(json.dumps({"path":p,"epoch":s.get("epoch",-1),"best":s.get("best_miou_epoch",-1)}))'
        state=json.loads(subprocess.check_output([py,'-c',probe,str(fit)],env=env,text=True))
        if state['epoch']<epochs-1:
            command=[py,'-u','train.py']+COMMON+modality+['--dataset_dir',dataset,'--pretrained_dir','./trained_models/imagenet',
                '--batch_size','8','--epochs',str(epochs),'--lr','0.01','--optimizer','SGD','--class_weighting','median_frequency','--results_dir',fit]
            if state['path']:command+=['--last_ckpt',state['path']]
            run(work,command,out/'training.log',env,work/'esanet',job['id'])
        state=json.loads(subprocess.check_output([py,'-c',probe,str(fit)],env=env,text=True));assert state['epoch']==epochs-1,state
        candidates=list(fit.glob(f'isod/checkpoints_*/ckpt_epoch_{state["best"]}.pth'));assert candidates
        checkpoint=candidates[-1]
        if smoke:
            run(work,[py,work/'code/esanet/predict_esanet.py']+COMMON+modality+['--dataset_dir',dataset,'--ckpt_path',checkpoint,'--pred_split','valid','--out',out/'pred'],out/'predict.log',env,work/'esanet')
            assert len(list((out/'pred').glob('*/*.png')))==16
        else:
            for destination,data in [('pred',work/'esanet_test_all'),('gte',work/'esanet_gte')]:
                run(work,[py,work/'code/esanet/predict_esanet.py']+COMMON+modality+['--dataset_dir',data,'--ckpt_path',checkpoint,'--pred_split','test','--out',out/destination],out/f'predict_{destination}.log',env,work/'esanet')
    elif kind=='unet':
        if fit.exists():shutil.move(fit,out/f'failed_fit_{int(time.time())}')
        command=[py,'-u','code/train_seg.py','--root',work,'--train-masks',work/'ISOD_sam_train_v2','--val-masks',work/'ISOD_reference_val_v1',
                 '--test-masks',work/'ISOD_reference_test_v1','--output',fit,'--seed',str(seed),'--workers','4','--epochs',str(epochs)]
        if smoke:command+=['--train-limit','32','--val-limit','16']
        else:command+=['--predict-all-test']
        run(work,command,out/'training.log',env,heartbeat=job['id'])
        shutil.copytree(fit/'pred',out/'pred',dirs_exist_ok=True)
        if not smoke:
            if (out/'gte').exists():shutil.rmtree(out/'gte')
            run(work,[py,'code/predict_unet_frozen.py','--gte',work/'OWN','--checkpoint',fit/'best.pt','--out',out/'gte'],out/'predict_gte.log',env)
        assert len((fit/'unet_mobilenet_v2_history.csv').read_text().splitlines())==epochs+1
    elif kind=='yolo':
        if fit.exists():shutil.move(fit,out/f'failed_fit_{int(time.time())}')
        command=[py,'-u','code/train_yolo.py','--root',work,'--model',cfg['yolo_weights'],'--data',work/f'yolo_{suffix}_sam_v2/data.yaml',
                 '--output',fit,'--seed',str(seed),'--workers','4','--epochs',str(epochs),'--val-masks',work/'ISOD_reference_val_v1','--test-masks',work/'ISOD_reference_test_v1']
        if smoke:command+=['--val-limit','16']
        else:command+=['--gte',work/'OWN','--predict-all-test']
        run(work,command,out/'training.log',env,heartbeat=job['id']);shutil.copytree(fit/'pred',out/'pred',dirs_exist_ok=True)
        if not smoke:shutil.copytree(fit/'gte',out/'gte',dirs_exist_ok=True)
    else:raise ValueError(kind)
    if not smoke:
        assert len(list((out/'pred').glob('*/*.png')))==300
        assert len(list((out/'gte').glob('*/*.png')))==320
        run(work,[py,'code/score_frozen.py','--isod-reference',work/'ISOD_reference_test_v1','--isod-predictions',out/'pred',
                  '--gte-reference',work/'OWN_manual','--gte-predictions',out/'gte','--out',out/'scores.json'],out/'score.log',env)
    (out/'VERIFIED').touch();return out

def memory_free(cfg):
    text=subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,util=map(int,text.strip().split(','));return memory,util

def worker(work,cfg):
    wait_for(work,work/'data.READY');wait_for(work,SHARED/'SMOKES_PASSED');prepare(work,cfg)
    while time.time()<DEADLINE:
        jobs=read(SHARED/'jobs.json');done=[(SHARED/'jobs'/j['id']/'done.json').exists() for j in jobs]
        if all(done):status(work,'complete',completed=len(jobs));return
        memory,util=memory_free(cfg)
        if cfg.get('require_idle') and (memory<22000 or util>10):
            status(work,'waiting_for_gpu',free_mib=memory,utilization=util);time.sleep(30);continue
        found=False
        for job in jobs:
            if job['kind'] not in cfg['allowed']:continue
            directory=SHARED/'jobs'/job['id']
            if (directory/'done.json').exists() or (directory/'failed.json').exists():continue
            if (directory/'owner.json').exists():
                owner=read(directory/'owner.json')
                if owner['host']==HOST:
                    try:os.kill(owner['pid'],0)
                    except ProcessLookupError:
                        directory.rename(directory.with_name(directory.name+f'.interrupted.{int(time.time())}'))
            minimum=12000 if job['kind']=='rgbd' else 7000 if job['kind'] in ('rgb','depth','yolo') else 4000
            if memory<minimum:continue
            try:directory.mkdir(parents=False)
            except FileExistsError:continue
            found=True
            write(directory/'owner.json',dict(host=HOST,pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
            error=None
            for attempt in (1,2):
                try:
                    out=execute(work,cfg,job);write(directory/'done.json',dict(host=HOST,out=str(out),attempt=attempt,scores=read(out/'scores.json')));error=None;break
                except Exception as exc:
                    error=repr(exc);write(directory/f'failure_attempt_{attempt}.json',dict(error=error,host=HOST,time=datetime.now(timezone.utc).isoformat()))
                    print('FAILED',job['id'],attempt,error,flush=True)
                    time.sleep(15)
            if error:write(directory/'failed.json',dict(host=HOST,error=error))
            break
        if not found:
            unfinished=[j for j in jobs if not (SHARED/'jobs'/j['id']/'done.json').exists()]
            if unfinished and all((SHARED/'jobs'/j['id']/'failed.json').exists() for j in unfinished):
                status(work,'completed_with_failures',failed=[j['id'] for j in unfinished]);return
            status(work,'waiting_for_jobs',completed=sum(done),total=len(jobs));time.sleep(20)
    status(work,'reservation_ended')

def master(work,cfg):
    label(work,cfg);prepare(work,cfg,smoke=True);prepare(work,cfg)
    # Verify every architecture/input path before releasing the overnight queue.
    for kind in ('rgb','depth','rgbd','unet','yolo'):
        job=dict(kind=kind,seed=0,epochs=1 if kind in ('unet','yolo') else 2,id='smoke_'+kind)
        if not (work/'smoke'/job['id']/'VERIFIED').exists():execute(work,cfg,job,smoke=True)
    jobs=[dict(id=f'{kind}_sam_v2_s{seed}',kind=kind,seed=seed,epochs=15 if kind=='unet' else 20 if kind=='yolo' else 500)
          for seed in (0,1,2) for kind in ('unet','yolo','rgb','depth','rgbd')]
    (SHARED/'jobs').mkdir(exist_ok=True);write(SHARED/'jobs.json',jobs);(SHARED/'SMOKES_PASSED').touch()
    worker(work,cfg)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['master','worker']);p.add_argument('--work',type=Path,required=True);p.add_argument('--host-config',type=Path,required=True);args=p.parse_args()
    work=args.work.resolve();cfg=read(args.host_config);SHARED.mkdir(exist_ok=True)
    os.chdir(work)
    if args.command=='master':master(work,cfg)
    else:worker(work,cfg)

if __name__=='__main__':main()
