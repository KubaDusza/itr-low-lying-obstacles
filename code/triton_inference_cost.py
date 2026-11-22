"""Batch-one SAM teacher versus YOLO inference cost on GPU or server CPU."""
import argparse,gc,json,os,platform,resource,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
 work=Path(os.environ['SLURM_TMPDIR'])/'cost';work.mkdir();out=a.base/'cost';out.mkdir(exist_ok=True)
 import cv2,torch
 torch.set_num_threads(8);cv2.setNumThreads(1)
 hardware=torch.cuda.get_device_name() if a.device=='cuda' else next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),platform.processor())
 names=['OWN/h52_poster_rags_075/rgb/2.png','OWN/h25_floor_cables_250/rgb/2.png','OWN/h25_floor_random_150/rgb/2.png']
 with tarfile.open('/scratch/work/duszaj1/itr_20261121/production/data.tar.gz') as tar:
  members=[m for m in tar.getmembers() if m.name in names];assert len(members)==3;tar.extractall(work,members=members,filter='data')
 images=[cv2.imread(str(work/n)) for n in names];assert all(im is not None for im in images)
 rows=[];warmup=3 if a.device=='cuda' else 1;repeats=5 if a.device=='cuda' else 1
 def measure(name,run,parameters,precision):
  for _ in range(warmup):run(images[0])
  if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  times=[]
  for im in images:
   for _ in range(repeats):
    if a.device=='cuda':torch.cuda.synchronize()
    begin=time.perf_counter();run(im)
    if a.device=='cuda':torch.cuda.synchronize()
    times.append((time.perf_counter()-begin)*1000)
  peak=torch.cuda.max_memory_allocated()/1024**3 if a.device=='cuda' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
  record=dict(model=name,device=a.device,hardware=hardware,precision=precision,parameters=parameters,warmup=warmup,measured_frames=len(times),images=names,
   median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),times_ms=times,peak_memory_GiB=float(peak),memory_measure='peak allocated CUDA memory' if a.device=='cuda' else 'process peak RSS')
  rows.append(record);(out/f'{a.device}-progress.json').write_text(json.dumps(rows,indent=2));print(name,record['median_ms'],record['peak_memory_GiB'],flush=True)
 from ultralytics import YOLO
 weights=Path('/scratch/work/duszaj1/itr_instances_20261122/runs/yolov8n_detect_reference_s0/weights/best.pt');yolo=YOLO(str(weights))
 parameters=sum(p.numel() for p in yolo.model.parameters())
 def run_yolo(im):
  result=yolo.predict(im,imgsz=640,device=0 if a.device=='cuda' else 'cpu',half=a.device=='cuda',conf=.25,iou=.7,max_det=300,verbose=False)[0]
  result.boxes.xyxy.cpu().numpy()
 measure('YOLOv8n box detector',run_yolo,parameters,'float16' if a.device=='cuda' else 'float32')
 del yolo;gc.collect()
 if a.device=='cuda':torch.cuda.empty_cache()
 os.environ['HF_HOME']=str(work/'hf');os.environ['HF_HUB_CACHE']=str(work/'hf/hub');os.environ['HF_HUB_DISABLE_XET']='1'
 from huggingface_hub import HfApi,snapshot_download
 model_id='bodhicitta/sam3';revision='cba430d22f6fdc3f06ad3841274ec7bb55885f2f';files=HfApi().list_repo_files(model_id,revision=revision)
 allowed=[f for f in files if f.endswith(('.safetensors','.json','.txt','.model','.tiktoken'))]
 snapshot_download(model_id,revision=revision,allow_patterns=allowed,cache_dir=os.environ['HF_HUB_CACHE'],max_workers=2)
 from transformers import Sam3Model,Sam3Processor
 sam=Sam3Model.from_pretrained(model_id,revision=revision,local_files_only=True,dtype=torch.bfloat16 if a.device=='cuda' else torch.float32).to(a.device).eval()
 processor=Sam3Processor.from_pretrained(model_id,revision=revision,local_files_only=True)
 prompts=['small object on the floor','cable on the floor','things on the floor'];parameters=sum(p.numel() for p in sam.parameters())
 @torch.inference_mode()
 def run_sam(im):
  rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);inputs=processor(images=[rgb],text=[prompts[0]],return_tensors='pt').to(a.device)
  vision=sam.get_vision_features(inputs['pixel_values']);sizes=inputs['original_sizes'].tolist()
  for prompt in prompts:
   token=processor(text=[prompt],return_tensors='pt').to(a.device)
   result=sam(vision_embeds=vision,input_ids=token['input_ids'],attention_mask=token['attention_mask'])
   processed=processor.post_process_instance_segmentation(result,threshold=.6,mask_threshold=.5,target_sizes=sizes)
   processed[0]['masks'].cpu().numpy()
 measure('SAM 3, three prompts with shared image features',run_sam,parameters,'bfloat16' if a.device=='cuda' else 'float32')
 result=dict(complete=True,protocol='Batch 1, images already loaded, preprocessing plus model plus postprocessing plus output-to-CPU. Model download/load excluded. Same fixed weights and three images on each hardware; precision follows optimized GPU and float32 CPU modes.',torch=torch.__version__,cpu_threads=8,slurm_job=os.environ['SLURM_JOB_ID'],rows=rows,sam_revision=revision)
 (out/f'{a.device}.json').write_text(json.dumps(result,indent=2)+'\n');(out/f'{a.device}.VERIFIED').write_text('Both models measured\n')
if __name__=='__main__':main()
