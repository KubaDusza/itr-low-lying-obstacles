import sys,json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,'code')
from make_detection_report_figures import COLORS, ROOT, BASE, EVIDENCE
import matplotlib.pyplot as plt
models={}
for mode in ['rgb','depth','rgbd']:
 for seed in range(3):models[f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}']=(mode,seed)
units=defaultdict(list)
with (BASE/'gte_full_diagnostic/objects.csv').open() as f:
 for r in csv.DictReader(f):
  if r['model'] in models:units[r['model'],r['recording'],r['id']].append((float(r['coverage'])>=.5,r['localized50']=='True'))
records=defaultdict(list)
for (model,rec,ident),v in units.items():
 assert len(v)==5
 records[model,rec].append(np.mean(v,axis=0))
cats=defaultdict(list)
for (model,rec),v in records.items():cats[model,rec.split('_')[2]].append(np.mean(v,axis=0))
rows=[]
for (model,cat),v in cats.items():
 mode,seed=models[model];means=np.mean(v,axis=0)
 rows.append(dict(model=model,modality=mode,seed=seed,object_set=cat,recordings=len(v),coverage=means[0],localization=means[1]))
empty=defaultdict(list)
with (BASE/'gte_full_diagnostic/frames.csv').open() as f:
 for r in csv.DictReader(f):
  if r['model'] in models and r['empty_reference']=='True':empty[r['model']].append(int(r['fp50']))
fig,axes=plt.subplots(1,3,figsize=(10,3.2));plt.rcParams.update({'pdf.fonttype':42})
keys=['cables','chess','rags','hexkeys','random','screws','tapes']
for mode,label in [('rgb','RGB'),('depth','D'),('rgbd','RGB-D')]:
 for i,key in enumerate(['coverage','localization']):
  a=np.array([[r[key] for r in rows if r['modality']==mode and r['object_set']==cat] for cat in keys])
  axes[i].errorbar(range(7),a.mean(1)*100,yerr=a.std(1,ddof=1)*100,color=COLORS[mode],label=label,marker='o',lw=1,capsize=2,ms=3)
 vals=[np.mean(v) for m,v in empty.items() if models[m][0]==mode]
 x=['rgb','depth','rgbd'].index(mode);axes[2].bar(x,np.mean(vals),color=COLORS[mode]);axes[2].errorbar(x,np.mean(vals),yerr=np.std(vals,ddof=1),capsize=3,color=COLORS[mode])
for i,title in enumerate(['Coverage by object set','Localization by object set']):
 axes[i].set_title(title);axes[i].set_ylim(0,100);axes[i].set_xticks(range(7),['Cables','Chess','Cloths','Hex keys','Mixed','Screws','Tape'],rotation=45,ha='right');axes[i].set_ylabel('Recall (%)')
axes[0].legend(frameon=False,fontsize=8);axes[2].set_title('Empty-scene diagnostic');axes[2].set_xticks(range(3),['RGB','D','RGB-D']);axes[2].set_ylabel('False components / frame')
for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Reference-supervised ESANet; three-seed means and SD; same 320 GTE frames',fontsize=10)
fig.tight_layout();fig.savefig(EVIDENCE/'category_and_empty_analysis.pdf');plt.close(fig)
with (EVIDENCE/'category_and_empty_analysis.json').open('w') as f:json.dump(dict(category_records=rows,empty_scenes={k:dict(frames=len(v),false_components_per_frame=float(np.mean(v)),frames_with_false_component=sum(x>0 for x in v)) for k,v in empty.items()},scope='Full-frame diagnostic: recording props can be flagged. No prop-aware masking or inference crops.'),f,indent=2)
print('Created category and empty-scene supporting analysis')
