"""Plot independently trained box detectors from verified, matched runs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from analyse_instance_detection import match

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/instance_detection_20261122'
OUT=ROOT/'results_report/figures'
COLORS={'reference':'#222222','automatic':'#0072b2'}


def main():
    summary=json.loads((BASE/'analysis/summary.json').read_text());assert summary['complete']
    refs=json.loads((BASE/'input/reference_manifest.json').read_text())
    settings=[];audit=[]
    for source in COLORS:
        for seed in range(3):
            run=BASE/'runs'/f'yolov8n_detect_{source}_s{seed}'
            s=json.loads((run/'settings.json').read_text());settings.append(s)
            history=list(csv.DictReader((run/'results.csv').open()))
            assert [int(r['epoch']) for r in history]==list(range(1,21))
            assert s['train_frames']==1400 and s['validation_frames']==300
            assert s['source']==source and s['seed']==seed and s['task']=='detect'
            audit.append(dict(run=run.name,epochs=len(history),best_validation_AP50_95=max(float(r['metrics/mAP50-95(B)']) for r in history),
                selected_epoch=max(range(20),key=lambda i:float(history[i]['metrics/mAP50-95(B)']))+1))
    for key in ['model','pretrained_sha256','reference_manifest_sha256','torch','ultralytics','validation_selection']:
        assert len({s[key] for s in settings})==1,key
    assert len({s['automatic_manifest_sha256'] for s in settings if s['source']=='automatic'})==1
    means={}
    for source in COLORS:
        rr=[r for r in summary['runs'].values() if r['source']==source]
        means[source]={key:dict(mean=float(np.mean([r[key] for r in rr])),sd=float(np.std([r[key] for r in rr],ddof=1))) for key in
            ['box_AP50','box_AP50_95','precision','recall','false_boxes_per_frame','empty_false_boxes_per_frame']}
    plt.rcParams.update({'font.size':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(6.8,2.5),gridspec_kw={'height_ratios':[.75,1]})
    for ax,metric,title in zip(axes[0],['box_AP50','box_AP50_95','recall'],['Box AP50','Box AP50:95','Recall at conf. 0.25 / IoU 0.5']):
        for x,source in enumerate(COLORS):
            vals=[r[metric]*100 for r in summary['runs'].values() if r['source']==source]
            ax.bar(x,np.mean(vals),color=COLORS[source],width=.52,alpha=.85)
            ax.errorbar(x,np.mean(vals),yerr=np.std(vals,ddof=1),color=COLORS[source],capsize=3,lw=.8)
            ax.scatter(x+np.linspace(-.09,.09,3),vals,s=8,c='white',edgecolors=COLORS[source],linewidths=.6,zorder=4)
        ax.set_xticks([0,1],['Reference labels','SAM labels']);ax.set_ylim(0,100);ax.set_yticks([0,50,100]);ax.set_ylabel('%');ax.set_title(title,fontsize=7.5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    # Prespecified matched cloth scene: shows native detections alongside all three segmentation modalities elsewhere.
    recording='h52_poster_rags_075';frame=2
    rgb=np.array(Image.open(ROOT/'data/OWN'/recording/'rgb/2.png').convert('RGB'))
    ref=next(r for r in refs['gte'] if r['recording']==recording and r['frame']==frame)
    gt=[b['xyxy'] for b in ref['boxes']]
    axes[1,0].imshow(rgb)
    for ident,box in enumerate(gt,1):
        x0,y0,x1,y1=box;axes[1,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#00bfc4',lw=.8))
        axes[1,0].text(x0,y0-3,str(ident),fontsize=6,color='#00ffff')
    axes[1,0].set_title('Reference boxes: same poster scene',fontsize=7.5)
    examples=[]
    for ax,source in zip(axes[1,1:],COLORS):
        pred=next(r for r in json.loads((BASE/f'runs/yolov8n_detect_{source}_s0/gte_boxes.json').read_text())['frames'] if r['image']==f'{recording}_{frame}')
        result=match(gt,pred['boxes_xyxy'],pred['scores']);matched={r['prediction'] for r in result['matched']}
        ax.imshow(rgb)
        for i,(box,conf) in enumerate(zip(pred['boxes_xyxy'],pred['scores'])):
            if conf<.25:continue
            x0,y0,x1,y1=box;color='#00dd60' if i in matched else '#ff604f'
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=color,lw=.8))
            ax.text(x0,y0-2,f'{conf:.2f}',fontsize=5,color=color)
        ax.set_title(('Reference' if source=='reference' else 'SAM')+f'-trained: {result["tp"]}/4 matched',fontsize=7.5)
        examples.append(dict(source=source,recording=recording,frame=frame,seed=0,**result))
    for ax in axes[1]:ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=.07,right=.99,top=.92,bottom=.02,hspace=.62,wspace=.29)
    fig.savefig(OUT/'native_box_detection.pdf');fig.savefig(OUT/'native_box_detection.png',dpi=240);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,4.8))
    fields=['train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','metrics/mAP50-95(B)']
    for ax,field in zip(axes.flat,fields):
        for s in settings:
            h=list(csv.DictReader((BASE/f'runs/yolov8n_detect_{s["source"]}_s{s["seed"]}/results.csv').open()))
            ax.plot([int(r['epoch']) for r in h],[float(r[field]) for r in h],color=COLORS[s['source']],alpha=.65,lw=.9,label=s['source'] if s['seed']==0 else None)
        ax.set_title(field);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(BASE/'analysis/training_curves.pdf');plt.close(fig)
    result=dict(complete=True,matched_configuration_verified=True,epochs=20,training_audit=audit,cohorts=means,qualitative_examples=examples,
        warning='Box AP and fixed-confidence box recall use a separate task and matching rule from segmentation coverage/localization; do not rank these directly.')
    (BASE/'analysis/presentation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
