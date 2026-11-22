"""Recalculate the teammate's attached/isolated extra-pixel and empty-scene analysis."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/results_report_20261122/revision'
sources=json.loads((R/'summary.json').read_text())['source_runs'];results=[]
for mode in ['rgb','rgbd']:
 for source in ['reference','automatic']:
  for seed in range(3):
   name=f'rgbd_{"reference" if source=="reference" else "sam"}210_s{seed}' if mode=='rgbd' else f'esanet_full_rgb_reference_s{seed}' if source=='reference' else 'rgb_sam_v2_s0_recovery_glaurung' if seed==0 else f'rgb_sam_v2_s{seed}'
   folder=ROOT/sources[name]['predictions'];total=0;isolated=0;empty=[]
   for p in sorted((ROOT/'data/OWN_manual').glob('*.png')):
    if p.stem.endswith('_id'):continue
    rec,frame=p.stem.rsplit('_',1);gt=np.array(Image.open(p))>0;pred=np.array(Image.open(folder/rec/f'{frame}.png'))>0
    if pred.ndim==3:pred=pred[...,0]
    if gt.ndim==3:gt=gt[...,0]
    labels,n=ndimage.label(pred,structure=np.ones((3,3),np.uint8));sizes=np.bincount(labels.ravel(),minlength=n+1);hits=np.bincount(labels[gt].ravel(),minlength=n+1);extra=sizes-hits
    extra[0]=0;total+=int(extra.sum());isolated+=int(extra[hits==0].sum())
    if '_empty' in rec:empty.append(int(pred.sum()))
   expected=sources[name]['legacy_full_frame_pixel'];assert total==expected['pred']-expected['inter'] and len(empty)==20
   results.append(dict(model=name,modality=mode,source=source,seed=seed,extra_pixels=total,isolated_extra_pixels=isolated,isolated_fraction=isolated/total,
       empty_frames=20,empty_frames_with_foreground=sum(v>0 for v in empty),empty_foreground_pixels_per_frame=float(np.mean(empty))))
   print(name,round(isolated/total,3),flush=True)
(R/'error_types.json').write_text(json.dumps(dict(definition='An extra predicted pixel is isolated if its 8-connected predicted component touches no reference foreground; otherwise it is attached. Attached does not necessarily mean a narrow boundary error.',scope='All 320 full GTE frames; recording props can contribute extra pixels.',runs=results),indent=2)+'\n')
lines=[r'\begin{table}[H]',r'\caption{Attached versus isolated excess foreground and empty-control behavior, recomputed from current predictions. Values are three-seed means. Isolated means the predicted component touches no reference foreground; attached components can include more than a narrow boundary. All counts are relative to the current full-frame reference, so props can contribute.}\label{tab:error-types}',r'\centering\small',r'\begin{tabular}{llrrr}',r'\toprule Input & Training labels & Isolated extra pixels (\%) & Empty frames with foreground (/20) & Empty predicted pixels/frame\\\midrule']
for mode in ['rgb','rgbd']:
 for source in ['reference','automatic']:
  rr=[r for r in results if r['modality']==mode and r['source']==source]
  lines.append(('RGB-D' if mode=='rgbd' else 'RGB')+' & '+source+f' & {np.mean([r["isolated_fraction"] for r in rr])*100:.1f} & {np.mean([r["empty_frames_with_foreground"] for r in rr]):.1f} & {np.mean([r["empty_foreground_pixels_per_frame"] for r in rr]):.0f}'+r'\\')
lines +=[r'\bottomrule\end{tabular}',r'\end{table}'];(ROOT/'results_report/error_types_table.tex').write_text('\n'.join(lines)+'\n')
