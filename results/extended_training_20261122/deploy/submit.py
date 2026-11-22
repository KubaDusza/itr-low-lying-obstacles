from pathlib import Path
import subprocess,json
b=Path('/scratch/work/duszaj1/itr_extended_20261122');out=b/'submitted.json'
assert not out.exists(),'Jobs already submitted; inspect receipt instead of resubmitting'
def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],cwd=b,text=True).strip().split(';')[0]
result={}
result['unet_gate']=submit('train.slurm','unet-gate')
result['reference_boxes']=submit('--array=0-2%3','train.slurm','reference')
result['automatic_boxes']=submit('--array=0-2%3','train.slurm','automatic')
result['unet_logged']=submit('--array=0-2%3','--dependency=afterok:'+result['unet_gate'],'train.slurm','unet')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
