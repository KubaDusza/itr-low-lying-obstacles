#!/bin/bash -l
set -euo pipefail
BASE=/scratch/work/duszaj1/itr_20261121/production
cd "$BASE"
module load scicomp-pytorch-env/2026.1
python - <<'CHECK'
from pathlib import Path
import hashlib,json,shutil
p=Path('.')
for name,expected in json.loads((p/'checksums.json').read_text()).items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==expected,name
assert shutil.disk_usage(p).free>2*1024**3,'Insufficient free space after staging'
print('Archive hashes and storage gate passed')
CHECK
python - <<'DEPS'
import importlib.util,subprocess,sys
from pathlib import Path
packages={'cv2':'opencv-python-headless','mock':'mock','cityscapesscripts':'cityscapesScripts','h5py':'h5py','scipy':'scipy','pandas':'pandas','matplotlib':'matplotlib'}
missing=[p for m,p in packages.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target','deps']+missing,check=True)
DEPS
if [[ -f submitted.json ]]; then cat submitted.json; exit 0; fi
gate=$(sbatch --parsable --partition=gpu-debug --gpus=a100:1 --time=00:20:00 --job-name=itr-rgbd-gate production.slurm gate)
array=$(sbatch --parsable --array=0-5%2 --dependency="afterok:$gate" production.slurm)
printf '{"gate_job":"%s","array_job":"%s","runs":6,"max_parallel_gpus":2}
' "$gate" "$array" > submitted.json
cat submitted.json
