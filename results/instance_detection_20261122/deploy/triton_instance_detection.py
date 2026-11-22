"""Disconnect-safe box-detector training from versioned per-instance labels."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=float)+'\n');temp.replace(path)


def copy_atomic(source,destination):
    if not source.exists():return
    destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.tmp');shutil.copy2(source,tmp);tmp.replace(destination)


def extract_images(archive,work):
    with tarfile.open(archive) as tar:
        members=[m for m in tar.getmembers() if m.isfile() and ('/rgb/' in m.name) and m.name.split('/')[0] in ('ISOD','OWN')]
        assert len(members)>=2320;tar.extractall(work,members=members,filter='data')


def make_dataset(base,work,source,gate=False):
    with tarfile.open(base/'reference_labels.tar.gz') as tar:tar.extractall(work/'labels_pack',filter='data')
    pack=work/'labels_pack';manifest=json.loads((pack/'reference_manifest.json').read_text())
    if source=='automatic':
        assert (base/'automatic/VERIFIED').exists()
        with tarfile.open(base/'automatic/labels.tar.gz') as tar:tar.extractall(pack,filter='data')
    dataset=work/'dataset';selected={}
    for split in ('train','val'):
        records=sorted([r for r in manifest['frames'] if r['split']==split],key=lambda r:(r['site'],r['frame']))
        if gate:records=records[:32 if split=='train' else 16]
        for row in records:
            stem=f'{row["site"]}_{row["frame"]}';image=work/row['image'];assert image.exists()
            for sub in ('images','labels'):(dataset/sub/split).mkdir(parents=True,exist_ok=True)
            (dataset/'images'/split/(stem+'.png')).symlink_to(image)
            label=pack/row['label'] if split=='val' or source=='reference' else pack/'automatic/labels/train'/f'{stem}.txt'
            assert label.exists();shutil.copy2(label,dataset/'labels'/split/(stem+'.txt'))
        selected[split]=len(records)
    if not gate:assert selected==dict(train=1400,val=300)
    yaml=dataset/'data.yaml';yaml.write_text(f'path: {dataset}\ntrain: images/train\nval: images/val\nnames:\n  0: floor_hazard\n')
    gte=work/'gte'
    for row in manifest['gte']:
        stem=f'{row["recording"]}_{row["frame"]}'
        for sub in ('images','labels'):(gte/sub/'test').mkdir(parents=True,exist_ok=True)
        (gte/'images/test'/f'{stem}.png').symlink_to(work/row['image']);shutil.copy2(pack/row['label'],gte/'labels/test'/f'{stem}.txt')
    gyaml=gte/'data.yaml';gyaml.write_text(f'path: {gte}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: floor_hazard\n')
    return yaml,gyaml,manifest,selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source',choices=['reference','automatic'],default='reference');p.add_argument('--seed',type=int,default=0);p.add_argument('--gate',action='store_true');a=p.parse_args()
    name='runtime_gate' if a.gate else f'yolov8n_detect_{a.source}_s{a.seed}';out=a.base/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'VERIFIED').exists():return
    work=Path(os.environ['SLURM_TMPDIR'])/'detect';work.mkdir()
    assert shutil.disk_usage(work).free>8*1024**3
    extract_images(Path('/scratch/work/duszaj1/itr_20261121/production/data.tar.gz'),work)
    yaml,gyaml,manifest,selected=make_dataset(a.base,work,a.source,a.gate)
    import torch,torchvision,ultralytics
    from ultralytics import YOLO
    assert torch.cuda.is_available()
    torchvision.ops.nms(torch.tensor([[0.,0.,2.,2.]],device='cuda'),torch.tensor([.9],device='cuda'),.5)
    weights=a.base/'assets/yolov8n.pt';weight_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    settings=dict(task='detect',model='yolov8n.pt',source=a.source,seed=a.seed,epochs=1 if a.gate else 20,
        pretrained_sha256=weight_hash,reference_manifest_sha256=hashlib.sha256((work/'labels_pack/reference_manifest.json').read_bytes()).hexdigest(),
        automatic_manifest_sha256=hashlib.sha256((a.base/'automatic/manifest.json').read_bytes()).hexdigest() if a.source=='automatic' else None,
        train_frames=selected['train'],validation_frames=selected['val'],gate=a.gate,
        torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(),slurm_job=os.environ['SLURM_JOB_ID'],
        validation_selection='shared ISOD reference-box validation mAP50:95',labels_quality='assisted hazard-unit boxes; some targets are groups',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gte_domain='full-frame diagnostic; props not exhaustively annotated')
    write(out/'settings.json',settings);begin=time.monotonic()
    model=YOLO(str(weights),task='detect')
    def checkpoint(trainer):
        for field in ('last','best'):
            value=getattr(trainer,field,None)
            if value:copy_atomic(Path(value),out/'weights'/f'{field}.pt')
        copy_atomic(Path(trainer.save_dir)/'results.csv',out/'results.csv')
        write(out/'progress.json',dict(stage='training',completed_epochs=trainer.epoch+1,total_epochs=settings['epochs'],
            elapsed_seconds=time.monotonic()-begin,updated_at=datetime.now(timezone.utc).isoformat(),metrics=trainer.metrics))
        if trainer.epoch>=0:(out/'FIRST_EPOCH_OK').write_text('Training, validation and checkpoint export passed\n')
    model.add_callback('on_fit_epoch_end',checkpoint)
    model.train(data=str(yaml),epochs=settings['epochs'],imgsz=640,batch=16,device=0,workers=4,
        optimizer='SGD',lr0=.01,momentum=.937,weight_decay=.0005,seed=a.seed,deterministic=True,
        patience=0,save=True,save_period=-1,project=str(work/'fit'),name='model',exist_ok=False,
        cache=False,plots=False,verbose=False,amp=True,max_det=300,close_mosaic=5)
    fit=work/'fit/model';copy_atomic(fit/'results.csv',out/'results.csv');copy_atomic(fit/'args.yaml',out/'args.yaml')
    for field in ('best','last'):copy_atomic(fit/'weights'/f'{field}.pt',out/'weights'/f'{field}.pt')
    best=YOLO(str(out/'weights/best.pt'))
    if a.gate:
        prediction=best.predict(str(work/manifest['gte'][0]['image']),device=0,conf=.25,verbose=False)[0]
        write(out/'native_box_probe.json',dict(boxes=prediction.boxes.xyxy.cpu().tolist(),scores=prediction.boxes.conf.cpu().tolist()))
    else:
        results=best.val(data=str(gyaml),split='test',imgsz=640,batch=16,device=0,workers=4,conf=.001,iou=.7,max_det=300,plots=False,verbose=False)
        write(out/'gte_metrics.json',dict(metrics=results.results_dict,domain=settings['gte_domain'],reference_units_include_groups=True))
        prediction_records=[]
        images=sorted((work/'gte/images/test').glob('*.png'))
        for result in best.predict([str(p) for p in images],stream=True,device=0,imgsz=640,conf=.001,iou=.7,max_det=300,verbose=False):
            prediction_records.append(dict(image=Path(result.path).stem,shape=result.orig_shape,boxes_xyxy=result.boxes.xyxy.cpu().tolist(),
                scores=result.boxes.conf.cpu().tolist(),classes=result.boxes.cls.cpu().tolist()))
        assert len(prediction_records)==320
        write(out/'gte_boxes.json',dict(confidence_export_threshold=.001,visualization_threshold=.25,frames=prediction_records))
    write(out/'completion.json',dict(settings=settings,elapsed_seconds=time.monotonic()-begin,selected_checkpoint='best.pt',complete=True))
    (out/'VERIFIED').write_text('Runtime gate passed\n' if a.gate else '20 epochs, native detector boxes and external diagnostic scoring completed\n')


if __name__=='__main__':main()
