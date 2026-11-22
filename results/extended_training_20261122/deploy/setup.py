from pathlib import Path
import subprocess,sys,importlib.util,json,hashlib
base=Path('/scratch/work/duszaj1/itr_extended_20261122');old=Path('/scratch/work/duszaj1/itr_instances_20261122')
for rel in ['reference_labels.tar.gz','automatic','assets']:
 p=base/rel
 if not p.exists():p.symlink_to(old/rel,target_is_directory=(old/rel).is_dir())
(base/'runs').mkdir(exist_ok=True)
needed=[]
for module,package in [('segmentation_models_pytorch','segmentation-models-pytorch==0.5.0'),('timm','timm==1.0.22'),('huggingface_hub','huggingface-hub==0.36.0'),('safetensors','safetensors==0.6.2')]:
 if importlib.util.find_spec(module) is None:needed.append(package)
if needed:subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(base/'deps'),*needed],check=True)
import importlib
sys.path.insert(0,str(base/'deps'));importlib.invalidate_caches()
print('Installed helpers:',needed)
for module in ['segmentation_models_pytorch','timm','huggingface_hub','safetensors','cv2']:
 __import__(module)
print('U-Net dependencies import successfully')
(base/'submission_policy.json').write_text(json.dumps(dict(boxes=dict(max_epochs=200,early_stopping_patience=20,seeds=[0,1,2],sources=['reference','automatic'],initialization='same COCO checkpoint as 20-epoch study'),unet=dict(epochs=15,seeds=[0,1,2],source='same frozen reference labels',change='log same BCE+Dice on validation every epoch'),previous_runs_preserved=True),indent=2)+'\n')
