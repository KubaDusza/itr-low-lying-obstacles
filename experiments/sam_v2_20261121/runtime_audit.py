from pathlib import Path
import hashlib,json,os,subprocess
probe='import json,torch,numpy,sys; t=torch.ones(1,device="cuda"); print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),cuda_check=float(t.sum().item()))))'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='GPU-8babf310-56d2-66c4-5cf8-cec9ed2cab1f';env['OMP_NUM_THREADS']='1'
records={}
for name,python in [('210','/opt/venv/jdusza_itr_prompt_v2/bin/python'),('214','/opt/venv/jdusza_itr_train214/bin/python')]:
    records[name]=json.loads(subprocess.check_output([python,'-c',probe],env=env,text=True))
assets=[Path('esanet/trained_models/imagenet/r34_NBt1D.pth'),Path('/opt/venv/jdusza_itr_train214/itr_assets/yolov8n-seg.pt')]
records['initialization_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
Path('runtimes.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
