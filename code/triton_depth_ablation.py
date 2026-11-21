"""Inference-only depth interventions on a fixed ESANet RGB-D checkpoint.

Same RGB, weights, preprocessing and scoring domain. Zero means missing raw
depth; permutation preserves height/background/nominal-distance/frame-index.
These interventions measure model reliance, not the causal benefit of training
with depth. Full-frame prop ambiguities remain explicitly diagnostic.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np


def permutation(names):
    groups=defaultdict(list);mapping={}
    for name in names:
        run,frame=name.rsplit('_',1);parts=run.split('_')
        if parts[2]=='empty':mapping[name]=name
        else:groups[parts[0],parts[1],parts[3],frame].append(name)
    for values in groups.values():
        values=sorted(values);assert len(values)>1
        for i,name in enumerate(values):mapping[name]=values[(i+1)%len(values)]
    assert set(mapping)==set(names) and set(mapping.values())==set(names)
    assert all(mapping[n]!=n for n in names if '_empty_' not in n)
    return mapping


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import cv2
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gate',action='store_true');a=p.parse_args()
    source_run='real_data_gate' if a.gate else f'rgbd_reference210_s{a.seed}'
    source=a.base/'runs'/source_run;assert (source/'VERIFIED').exists()
    checkpoint=source/'selected_model.pth';assert checkpoint.exists()
    out=a.base/'depth_ablations'/('pipeline_gate' if a.gate else source_run);out.mkdir(parents=True,exist_ok=True)
    if (out/'VERIFIED').exists():return
    work=Path(os.environ['SLURM_TMPDIR'])/'itr';work.mkdir()
    for name in ('source.tar.gz','data.tar.gz'):
        with tarfile.open(a.base/name) as tar:tar.extractall(work,filter='data')
    shutil.copytree(a.base/'code',work/'code',dirs_exist_ok=True)
    spec=importlib.util.spec_from_file_location('controller',work/'code/overnight_sam_v2.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    cfg=dict(gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],extra_pythonpath=str(a.base/'deps'))
    env=controller.env_for(work,cfg,a.seed)
    dataset=work/'esanet_reference';gte=work/'esanet_gte'
    cmd=[sys.executable,'code/esanet/export_isod.py','--out',str(dataset),
         '--train-masks',str(work/'ISOD_reference_train_v1'),'--val-masks',str(work/'ISOD_reference_val_v1'),
         '--test-masks',str(work/'ISOD_reference_test_v1'),'--depth-stats',str(a.base/'depth_stats.json'),
         '--gte',str(work/'OWN'),'--gte-out',str(gte)]
    subprocess.run(cmd,cwd=work,env=env,check=True)
    names=(gte/'test.txt').read_text().splitlines();assert len(names)==320
    mapping=permutation(names);assert sum(k==v for k,v in mapping.items())==20
    manifest=dict(source_run=source_run,checkpoint_sha256=sha(checkpoint),
        pipeline_gate=a.gate,scientific_results_eligible=not a.gate,seed=a.seed,
        domain='full-frame diagnostic, recording props included',
        interpretation='model reliance under interventions; not an isolated causal effect of depth training',
        permutation='cyclic derangement across object sets within height/background/nominal-distance/frame-index; 20 empty frames unchanged',
        normal_reproduction={},variants={},rgb_identical_all_variants=True)
    for variant in ('normal','missing_raw_depth','permuted_depth'):
        target=gte if variant=='normal' else work/('gte_'+variant)
        if variant!='normal':
            shutil.copytree(gte,target,symlinks=True)
            for name in names:
                dest=target/'test/depth'/f'{name}.png';dest.unlink()
                if variant=='missing_raw_depth':
                    original=cv2.imread(str(gte/'test/depth'/f'{name}.png'),cv2.IMREAD_UNCHANGED)
                    assert original.dtype==np.uint16 and cv2.imwrite(str(dest),np.zeros_like(original))
                else:
                    original=gte/'test/depth'/f'{mapping[name]}.png'
                    dest.symlink_to(original.resolve())
                assert sha(target/'test/rgb'/f'{name}.png')==sha(gte/'test/rgb'/f'{name}.png')
        destination=out/variant;destination.mkdir(exist_ok=True)
        cmd=[sys.executable,str(work/'code/esanet/predict_esanet.py')]+controller.COMMON+[
            '--modality','rgbd','--dataset_dir',str(target),'--ckpt_path',str(checkpoint),
            '--pred_split','test','--out',str(destination/'gte')]
        with (out/f'{variant}.log').open('w') as log:
            subprocess.run(cmd,cwd=work/'esanet',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=list((destination/'gte').glob('*/*.png'));assert len(paths)==320
        subprocess.run([sys.executable,str(work/'code/score_frozen.py'),'--gte-reference',str(work/'OWN_manual'),
                        '--gte-predictions',str(destination/'gte'),'--out',str(destination/'scores.json')],env=env,check=True)
        manifest['variants'][variant]=json.loads((destination/'scores.json').read_text())['GTE-Floor']['pooled']
        if variant=='normal':
            changed=0;pixels=0
            for path in paths:
                baseline=source/'gte'/path.parent.name/path.name
                x=cv2.imread(str(path),0);y=cv2.imread(str(baseline),0)
                assert y is not None and x.shape==y.shape
                changed+=int((x!=y).sum());pixels+=x.size
            manifest['normal_reproduction']=dict(changed_pixels=changed,total_pixels=pixels,
                exact=changed==0,note='A new GPU allocation may change borderline floating-point decisions; all intervention effects use the fresh same-job normal baseline')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'permutation.json').write_text(json.dumps(mapping,indent=2)+'\n')
    import torch
    manifest['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                             slurm_job=os.environ['SLURM_JOB_ID'],runner_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'VERIFIED').write_text('Inference input/shape/count gates passed; scientific interpretation remains separate\n')
    print(json.dumps(dict(source_run=source_run,gate=a.gate,normal_reproduction=manifest['normal_reproduction'])))


if __name__=='__main__':main()
