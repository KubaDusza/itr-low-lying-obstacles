"""Prepare RGB-only SAM instance boxes on a GPU, without reference-mask inputs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    out=a.base/'automatic';out.mkdir(exist_ok=True)
    if (out/'VERIFIED').exists():return
    config=json.loads((a.base/'teacher_config.json').read_text());assert len(config['frames'])==1400
    assert config['inference_prompts']==['small object on the floor','cable on the floor','things on the floor']
    work=Path(os.environ['SLURM_TMPDIR'])/'teacher';work.mkdir();start=time.monotonic()
    # Only the 1,400 training RGB files are made available in the inference workspace.
    wanted={f'ISOD/data/{site}/rgb/{frame}.png' for site,frame in config['frames']}
    with tarfile.open('/scratch/work/duszaj1/itr_20261121/production/data.tar.gz') as tar:
        members=[m for m in tar.getmembers() if m.name in wanted];assert len(members)==1400
        tar.extractall(work,members=members,filter='data')
    os.environ['HF_HOME']=str(work/'hf');os.environ['HF_HUB_CACHE']=str(work/'hf/hub');os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi,snapshot_download
    write(out/'progress.json',dict(stage='downloading pinned SAM checkpoint to node-local temporary storage',reference_masks_read=False))
    files=HfApi().list_repo_files(config['model'],revision=config['revision'])
    weights=[f for f in files if f.endswith('.safetensors')]
    if not weights:weights=[f for f in files if f.startswith('pytorch_model') and f.endswith('.bin')]
    assert weights,'No checkpoint weight files found'
    allowed=weights+[f for f in files if f.endswith(('.json','.txt','.model','.tiktoken'))]
    snapshot_download(config['model'],revision=config['revision'],allow_patterns=allowed,cache_dir=os.environ['HF_HUB_CACHE'],max_workers=2)
    config_path=work/'config.json';config_path.write_text(json.dumps(config))
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    cache=out/'cache'
    for batch in (8,2):
        log=out/f'inference-batch{batch}.log'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,str(a.base/'cache_sam_prompt_study.py'),'--config',str(config_path),'--root',str(work),'--cache',str(cache),'--batch',str(batch)],stdout=stream,stderr=subprocess.STDOUT,env=env)
        if result.returncode==0:break
        if batch==2 or 'out of memory' not in log.read_text().lower():raise RuntimeError(log.read_text()[-5000:])
    assert (cache/'COMPLETE').exists()
    # Pure mask deduplication/parent resolution. No reference geometry is read.
    import numpy as np
    from export_instance_proposals import automatic_instances
    from refine_sam_instances import load_groups
    from prepare_instance_detection import write_labels
    records=[];units=Counter()
    for site,frame in config['frames']:
        path=cache/site/f'{frame}.npz';groups,_=load_groups(path);instances,events=automatic_instances(groups)
        with np.load(path) as data:shape=tuple(map(int,data['shape']))
        boxes=[]
        for ident,item in enumerate(instances,1):
            yy,xx=np.nonzero(item['mask']);assert len(xx)
            boxes.append(dict(id=ident,xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                unit=item['unit'],area_pixels=int(item['mask'].sum()),prompts=sorted({s['prompt'] for s in item['sources']})))
            units[item['unit']]+=1
        label=out/'automatic/labels/train'/f'{site}_{frame}.txt';write_labels(label,boxes,shape)
        records.append(dict(site=site,frame=frame,label=str(label.relative_to(out)),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(shape),boxes=boxes,parent_events=events))
    assert len(records)==1400
    result=dict(input='Full training RGB images and the frozen three prompts only',reference_masks_read=False,
        model=config['model'],revision=config['revision'],confidence=.6,mask_probability=.5,weights_trained=False,
        instance_policy='Mask-overlap duplicate resolution and development-fixed parent/child rules; ambiguous groups stay clusters',
        generation='Fresh pinned-checkpoint native instance inference; not assumed pixel-identical to previous semantic cache',
        config_sha256=hashlib.sha256((a.base/'teacher_config.json').read_bytes()).hexdigest(),unit_counts=dict(units),frames=records)
    write(out/'manifest.json',result)
    tmp=out/'labels.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz') as tar:
        for file in (out/'automatic/labels/train').glob('*.txt'):tar.add(file,arcname=str(file.relative_to(out)))
    tmp.replace(out/'labels.tar.gz')
    write(out/'progress.json',dict(stage='complete',frames=1400,units=dict(units),elapsed_seconds=time.monotonic()-start,reference_masks_read=False))
    (out/'VERIFIED').write_text('1400 RGB-only native SAM instance-box training labels; no reference masks used\n')


if __name__=='__main__':main()
