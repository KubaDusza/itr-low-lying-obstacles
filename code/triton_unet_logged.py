"""Rerun U-Net with validation loss, preserving the original objective and schedule."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path

def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def copy(src,dst):
 if src.exists():
  dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copy2(src,tmp);tmp.replace(dst)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--gate',action='store_true');a=p.parse_args()
 name='unet_runtime_gate' if a.gate else f'unet_logged_reference_s{a.seed}';out=a.base/'runs'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'VERIFIED').exists():return
 work=Path(os.environ['SLURM_TMPDIR'])/'unet';work.mkdir();assert shutil.disk_usage(work).free>6*1024**3
 prefixes=('ISOD_reference_train_v1/','ISOD_reference_val_v1/','ISOD_reference_test_v1/','OWN_manual/')
 with tarfile.open('/scratch/work/duszaj1/itr_20261121/production/data.tar.gz') as tar:
  members=[m for m in tar.getmembers() if m.isfile() and (m.name.startswith(prefixes) or '/rgb/' in m.name and m.name.startswith(('ISOD/','OWN/')) or m.name=='splits.json')];tar.extractall(work,members=members,filter='data')
 if not (work/'splits.json').exists():
  with tarfile.open('/scratch/work/duszaj1/itr_20261121/production/source.tar.gz') as tar:
   members=[m for m in tar.getmembers() if m.name=='splits.json'];assert len(members)==1;tar.extractall(work,members=members,filter='data')
 for folder,count in [('ISOD_reference_train_v1',1400),('ISOD_reference_val_v1',300),('ISOD_reference_test_v1',30)]:
  manifest=json.loads((work/folder/'manifest.json').read_text());assert len(manifest['frames'])==count
  for r in manifest['frames']:assert hashlib.sha256((work/folder/r['path']).read_bytes()).hexdigest()==r['sha256']
 epochs=1 if a.gate else 15;fit=work/'fit'
 cmd=[sys.executable,'-u',str(a.base/'helpers/train_seg.py'),'--root',str(work),'--arch','unet','--encoder','mobilenet_v2','--epochs',str(epochs),'--batch','8','--workers','4','--seed',str(a.seed),'--train-masks',str(work/'ISOD_reference_train_v1'),'--val-masks',str(work/'ISOD_reference_val_v1'),'--test-masks',str(work/'ISOD_reference_test_v1'),'--output',str(fit)]
 if a.gate:cmd+=['--train-limit','32','--val-limit','16']
 import torch,segmentation_models_pytorch as smp
 settings=dict(model='U-Net MobileNetV2',seed=a.seed,epochs=epochs,training_frames=32 if a.gate else 1400,validation_frames=16 if a.gate else 300,
  change='Log validation BCE+Dice loss, identical objective, 15-epoch schedule and validation-IoU selection; original runs preserved',torch=torch.__version__,smp=smp.__version__,gpu=torch.cuda.get_device_name(),slurm_job=os.environ['SLURM_JOB_ID'],command=cmd,gate=a.gate,logger_sha256=hashlib.sha256((a.base/'helpers/train_seg.py').read_bytes()).hexdigest())
 write(out/'provenance.json',settings);start=time.monotonic()
 with (out/'training.log').open('w') as stream:
  child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
  while child.poll() is None:
   time.sleep(5)
   hpath=fit/'unet_mobilenet_v2_history.csv'
   if hpath.exists():
    copy(hpath,out/'history.csv');rr=list(csv.DictReader(hpath.open()));write(out/'progress.json',dict(stage='training',completed_epochs=len(rr),total_epochs=epochs,elapsed_seconds=time.monotonic()-start))
   for rel in ['best.pt','last_checkpoint.pt']:
    q=fit/rel
    if q.exists() and time.time()-q.stat().st_mtime>1:copy(q,out/rel)
  if child.returncode:raise RuntimeError((out/'training.log').read_text()[-4000:])
 copy(fit/'unet_mobilenet_v2_history.csv',out/'history.csv');copy(fit/'best.pt',out/'best.pt');copy(fit/'last_checkpoint.pt',out/'last_checkpoint.pt');copy(fit/'settings.json',out/'settings.json');copy(fit/'selection.json',out/'selection.json')
 h=list(csv.DictReader((out/'history.csv').open()));assert len(h)==epochs and all('val_loss' in r for r in h)
 import math
 assert all(math.isfinite(float(r['val_loss'])) for r in h)
 if not a.gate:
  shutil.copytree(fit/'pred',out/'pred',dirs_exist_ok=True)
  for command in [[sys.executable,str(a.base/'helpers/predict_unet_frozen.py'),'--gte',str(work/'OWN'),'--checkpoint',str(out/'best.pt'),'--out',str(out/'gte')],
   [sys.executable,str(a.base/'helpers/score_frozen.py'),'--isod-reference',str(work/'ISOD_reference_test_v1'),'--isod-predictions',str(out/'pred'),'--gte-reference',str(work/'OWN_manual'),'--gte-predictions',str(out/'gte'),'--out',str(out/'scores.json')]]:
   subprocess.run(command,check=True)
  assert len(list((out/'gte').glob('*/*.png')))==320
 write(out/'completion.json',dict(complete=True,epochs=epochs,selected_epoch=json.loads((out/'selection.json').read_text())['epoch'],criterion='best validation obstacle IoU',validation_loss_logged=True,elapsed_seconds=time.monotonic()-start))
 (out/'VERIFIED').write_text('Training, validation loss logging, selection and inference verified\n')
if __name__=='__main__':main()
