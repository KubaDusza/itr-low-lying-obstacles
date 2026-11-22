"""YOLOv8n-seg on automatic SAM 3 labels, fixed 60-epoch schedule; checkpoint by reference-validation obstacle IoU."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def copy(p,q):
 if p.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--gate',action='store_true');a=ap.parse_args();start=time.monotonic()
 out=a.base/'runs'/('yoloseg_e60_sam_gate' if a.gate else f'yoloseg_e60_sam_s{a.seed}');out.mkdir(parents=True,exist_ok=True)
 if (out/'VERIFIED').exists():return
 work=Path(os.environ['SLURM_TMPDIR'])/'seg';work.mkdir()
 with tarfile.open('/scratch/work/duszaj1/itr_20261121/production/data.tar.gz') as tar:
  members=[m for m in tar.getmembers() if m.isfile() and (m.name.startswith(('ISOD_sam_train_v2/','ISOD_reference_train_v1/','ISOD_reference_val_v1/','ISOD_reference_test_v1/','OWN_manual/')) or m.name.endswith('splits.json') or '/rgb/' in m.name and m.name.startswith(('ISOD/','OWN/')))];tar.extractall(work,members=members,filter='data')
 import cv2,numpy as np
 cv2.setNumThreads(1);dataset=work/'dataset';manifests={}
 for split,count in [('train',1400),('val',300),('test',30)]:
  root=work/('ISOD_sam_train_v2' if split=='train' else f'ISOD_reference_{split}_v1');manifest=json.loads((root/'manifest.json').read_text());assert len(manifest['frames'])==count;manifests[split]=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
  records=manifest['frames'][:32 if split=='train' else 16] if a.gate and split!='test' else manifest['frames']
  for r in records:
   p=root/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];raw=cv2.imread(str(p),0);mask=(raw>0).astype(np.uint8);h,w=mask.shape;stem=f'{r["site"]}_{r["frame"]}'
   for sub in ['images','labels']:(dataset/sub/split).mkdir(parents=True,exist_ok=True)
   (dataset/'images'/split/f'{stem}.png').symlink_to(work/'ISOD/data'/r['site']/'rgb'/f'{r["frame"]}.png')
   contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);lines=[]
   for c in contours:
    if cv2.contourArea(c)<20 or len(c)<3:continue
    c=cv2.approxPolyDP(c,1.,True).reshape(-1,2).astype(float)
    if len(c)<3:continue
    c/=np.array([w,h]);c=c.clip(0,1);lines.append('0 '+' '.join(f'{v:.5f}' for v in c.ravel()))
   (dataset/'labels'/split/f'{stem}.txt').write_text('\n'.join(lines)+('\n' if lines else ''))
 yaml=dataset/'data.yaml';yaml.write_text(f'path: {dataset}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: obstacle\n')
 asset=a.base/'segassets/yolov8n-seg.pt';asset.parent.mkdir(exist_ok=True)
 from ultralytics import YOLO
 if not asset.exists():
  YOLO(str(asset))
 assert asset.exists()
 fit=work/'fit';cmd=[sys.executable,'-u',str(a.base/'helpers/train_yolo_long.py'),'--root',str(work),'--model',str(asset),'--data',str(yaml),'--epochs','1' if a.gate else '60','--patience','0','--optimizer','AdamW','--lr0','0.002','--output',str(fit),'--val-masks',str(work/'ISOD_reference_val_v1'),'--test-masks',str(work/'ISOD_reference_test_v1'),'--seed',str(a.seed),'--workers','4']
 if a.gate:cmd+=['--val-limit','16']
 else:cmd+=['--gte',str(work/'OWN'),'--predict-all-test']
 write(out/'provenance.json',dict(command=cmd,seed=a.seed,gate=a.gate,max_epochs=1 if a.gate else 60,patience=0,stopping='none: fixed 60-epoch schedule',selection='Highest pooled validation obstacle IoU among five-epoch and final checkpoints',optimizer='AdamW',lr0=.002,manifests=manifests,pretrained_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),slurm_job=os.environ['SLURM_JOB_ID'],controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 with (out/'training.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT)
  while child.poll() is None:
   copy(fit/'fit/results.csv',out/'results.csv');write(out/'progress.json',dict(stage='training or validation checkpoint selection',seconds=time.monotonic()-start));time.sleep(10)
  if child.returncode:raise RuntimeError((out/'training.log').read_text()[-5000:])
 copy(fit/'fit/results.csv',out/'results.csv');copy(fit/'selection.json',out/'selection.json');copy(fit/'settings.json',out/'settings.json');copy(fit/'fit/args.yaml',out/'args.yaml')
 selection=json.loads((fit/'selection.json').read_text());copy(fit/'fit/weights'/selection['best'],out/'weights/selected.pt');copy(fit/'fit/weights/last.pt',out/'weights/last.pt')
 if not a.gate:
  shutil.copytree(fit/'pred',out/'pred',dirs_exist_ok=True);shutil.copytree(fit/'gte',out/'gte',dirs_exist_ok=True)
  assert len(list((out/'gte').glob('*/*.png')))==320 and len(list((out/'pred').glob('*/*.png')))==300
 h=list(csv.DictReader((out/'results.csv').open()));assert a.gate or len(h)==60,len(h);write(out/'completion.json',dict(complete=True,epochs=len(h),selected_checkpoint=selection['best'],criterion='validation obstacle IoU',elapsed_seconds=time.monotonic()-start));(out/'VERIFIED').write_text('Training, checkpoint selection and mask export completed\n')
if __name__=='__main__':main()
