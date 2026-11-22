"""Refresh the verified 36-run evidence and assets, preserving per-step logs."""
import hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/results_report_20261122/final_refresh';OUT.mkdir(exist_ok=True)
steps=['prepare_report_revision','score_isod_test300','make_isod_scale_analysis','bootstrap_gte_contrasts','collect_report_numbers','make_final_report_assets','make_qualitative_cases']
for name in [f'unet_logged_sam_s{s}' for s in range(3)]+[f'yoloseg_e60_sam_s{s}' for s in range(3)]:
 p=ROOT/'results/extended_training_20261122/runs'/name;assert (p/'VERIFIED').exists(),name
 assert len(list((p/'gte').glob('*/*.png')))==320,name
 assert len(list((p/('pred300' if name.startswith('unet') else 'pred')).glob('*/*.png')))==300,name
state=dict(started_at=datetime.now(timezone.utc).isoformat(),status='running',steps=[])
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/itr-mpl')
for name in steps:
 script=ROOT/'code'/f'{name}.py';start=time.monotonic();row=dict(script=str(script.relative_to(ROOT)),sha256=hashlib.sha256(script.read_bytes()).hexdigest(),status='running');state['steps'].append(row);(OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 with (OUT/f'{name}.log').open('w') as log:result=subprocess.run([sys.executable,str(script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 row.update(seconds=time.monotonic()-start,exit_code=result.returncode,status='complete' if result.returncode==0 else 'failed');(OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 if result.returncode:
  state['status']='failed';(OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n');print((OUT/f'{name}.log').read_text()[-3500:]);raise SystemExit(result.returncode)
 print(f'{name}: complete ({row["seconds"]:.1f}s)',flush=True)
shutil.copy2(ROOT/'results/results_report_20261122/qualitative_cases.pdf',ROOT/'results_report/figures/qualitative_cases.pdf')
state.update(status='analysis and assets complete; hand-written report prose/build still required',completed_at=datetime.now(timezone.utc).isoformat());(OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
