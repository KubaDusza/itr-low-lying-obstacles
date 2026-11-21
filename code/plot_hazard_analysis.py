"""Reproducible diagnostic plots and examples from verified cached outputs."""
import csv
from collections import defaultdict
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evaluation_v2_20261121/figures'


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def modality_plots(frames):
    fig,axes=plt.subplots(1,3,figsize=(11,3.5));rows=[]
    for mode,label,color,marker in [('rgb','RGB','#222222','o'),('depth','Depth','#777777','s'),('rgbd','RGB-D','#0072B2','^')]:
        values=defaultdict(list)
        for seed in range(3):
            selected=[r for r in frames if r['model']==f'esanet_full_{mode}_reference_s{seed}' and r['empty_reference']=='False']
            for d in [.75,1.5,2.5]:
                group=[r for r in selected if float(r['nominal_distance_m'])==d]
                assert len(group)==100
                by_recording=defaultdict(list)
                for r in group:by_recording[r['recording']].append(float(r['localization_recall50']))
                measures=[sum(int(r['pixel_inter']) for r in group)/sum(int(r['pixel_union']) for r in group),
                          float(np.mean([np.mean(v) for v in by_recording.values()])),
                          float(np.mean([int(r['fp50']) for r in group]))]
                for i,v in enumerate(measures):values[i,d].append(v)
                rows.append(dict(modality=mode,seed=seed,nominal_distance_m=d,recordings=len(by_recording),frames=len(group),
                                 pooled_pixel_iou=measures[0],macro_recording_localization_recall=measures[1],false_components_per_frame=measures[2]))
        for i,ax in enumerate(axes):
            x=[.75,1.5,2.5];y=[np.mean(values[i,d]) for d in x];err=[np.std(values[i,d],ddof=1) for d in x]
            ax.errorbar(x,y,yerr=err,color=color,marker=marker,label=label,capsize=3,lw=1.4)
    for ax,title in zip(axes,['Pixel IoU','Localization recall (recording mean)','False components / frame']):
        ax.set_title(title,fontsize=10);ax.set_xticks([.75,1.5,2.5]);ax.set_xlabel('Nominal floor distance (m)')
        ax.grid(axis='y',color='.9');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1);axes[2].set_ylim(bottom=0);axes[0].legend(frameon=False)
    fig.suptitle('ESANet reference-trained models: all three modalities, three seeds',fontsize=12)
    fig.text(.5,-.02,'Full-frame diagnostic; recording props included. Error bars: seed s.d. (n=3), not scene confidence intervals.\nID-aware localization at 0.50; fastener separation provisional; no predicted-component size filter.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'modalities_by_distance')
    with (OUT/'modality_distance_values.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def size_plots(objects):
    fig,axes=plt.subplots(1,2,figsize=(8,3.5));counts={}
    for stem,label,color in [('esanet_full_rgb_reference','ESANet RGB','#222222'),('esanet_full_depth_reference','ESANet D','#777777'),('esanet_full_rgbd_reference','ESANet RGB-D','#0072B2')]:
        series=[]
        for seed in range(3):
            rows=[r for r in objects if r['model']==f'{stem}_s{seed}']
            bins=[[],[],[]]
            for r in rows:
                area=int(r['area_pixels']);bins[0 if area<100 else 1 if area<1000 else 2].append(r)
            vals=[]
            for i,b in enumerate(bins):
                # Average the five static repeats before pooling observed units.
                by_unit=defaultdict(list)
                for r in b:by_unit[r['recording'],r['id']].append(r['localized50']=='True')
                vals.append(np.mean([np.mean(v) for v in by_unit.values()]))
                counts[i]=len(by_unit)
            series.append(vals)
        arr=np.array(series);axes[0].errorbar(range(3),arr.mean(0),yerr=arr.std(0,ddof=1),marker='o',color=color,capsize=3,label=label)
    axes[0].set_xticks(range(3),[f'<100\n(n={counts[0]})',f'100–999\n(n={counts[1]})',f'≥1000\n(n={counts[2]})'])
    axes[0].set_xlabel('Reference unit area (pixels)');axes[0].set_ylabel('Localization recall');axes[0].set_ylim(0,1);axes[0].legend(frameon=False,fontsize=8)
    teacher=[r for r in objects if r['model']=='sam_three_prompt_v2']
    bins=[[],[],[]]
    for r in teacher:
        area=int(r['area_pixels']);bins[0 if area<100 else 1 if area<1000 else 2].append(r)
    for key,label,style in [('coverage','Coverage ≥0.50','--'),('adjusted_iou','Localization ≥0.50','-')]:
        vals=[]
        for b in bins:
            by_unit=defaultdict(list)
            for r in b:by_unit[r['recording'],r['id']].append(float(r[key])>=.5)
            vals.append(np.mean([np.mean(v) for v in by_unit.values()]))
        axes[1].plot(range(3),vals,style,marker='o',color='#222222',label=label)
    axes[1].set_xticks(range(3),['<100','100–999','≥1000']);axes[1].set_ylim(0,1);axes[1].set_xlabel('Reference unit area (pixels)')
    axes[1].set_title('Direct three-prompt SAM teacher',fontsize=10);axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',color='.9');ax.spines[['top','right']].set_visible(False)
    fig.text(.5,-.035,'Full-frame diagnostic. Size, distance and object type are confounded; these bins are descriptive.\nCounts are recording–reference-unit pairs, not independent physical objects.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'size_and_teacher')


def color_instances(rgb,ids,objects):
    out=np.array(rgb).astype(float);legend={o['id']:o['unit'] for o in objects}
    for ident in np.unique(ids):
        if ident==0:continue
        unit=legend[int(ident)]
        color=(255,0,0) if unit=='unseeded_reference_region' else (255,0,200) if unit=='unresolved_cluster' else tuple(255*v for v in colorsys.hsv_to_rgb((ident*.618)%1,.8,1))
        mask=ids==ident;out[mask]=.35*out[mask]+.65*np.array(color)
    return Image.fromarray(out.astype(np.uint8))


def instance_qa():
    base=ROOT/'results/instance_proposals_20261121';m=json.loads((base/'manifest.json').read_text())
    choices=[('BT_LM',34),('BT_BM',12),('FL_LM',64),('HDCL_RM_A',4),('HB_MM',60),('HL_BM',87)]
    canvas=Image.new('RGB',(1280,6*245+35),'white');draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(['RGB','Accepted binary reference','Reference instance proposals','RGB-only automatic proposals']):draw.text((col*320+5,5),title,fill='black')
    protocol=json.loads((ROOT/'results/sam_prompt_v2/expanded_protocol.json').read_text())
    definitions={(r['site'],r['frame']):r for split in ('development','validation') for r in protocol[split]}
    for row,key in enumerate(choices):
        r=next(r for r in m['frames'] if (r['site'],r['frame'])==key)
        rgb=Image.open(ROOT/'data/ISOD/data'/key[0]/'rgb'/f'{key[1]}.png').convert('RGB')
        ref=np.array(Image.open(base/r['reference']['path']));gt=ref>0
        binary=np.array(rgb).astype(float);binary[gt]=binary[gt]*.35+np.array([0,220,220])*.65
        with np.load(base/r['automatic']['path']) as data:
            shape=tuple(data['shape']);masks=np.unpackbits(data['masks'],axis=1)[:,:int(np.prod(shape))].reshape((-1,*shape))
        auto=np.zeros(shape,np.uint16)
        for i,mask in enumerate(masks,1):auto[mask>0]=i
        panels=[rgb,Image.fromarray(binary.astype(np.uint8)),color_instances(rgb,ref,r['reference']['objects']),color_instances(rgb,auto,r['automatic']['objects'])]
        for col,im in enumerate(panels):
            im.thumbnail((316,210));canvas.paste(im,(col*320,row*245+55))
        draw.text((4,row*245+33),f'{key[0]} / {key[1]} — {r["dataset_split"]}; reference={len(r["reference"]["objects"])} / auto={len(r["automatic"]["objects"])}',fill='black')
    canvas.save(OUT/'instance_proposal_qa.jpg')
    (OUT/'instance_qa_selection.json').write_text(json.dumps(dict(selection='Two human-identified pile cases; largest development residuals; an unflagged development case; largest validation residual (GT-assistance diagnosis only).',
        frames=choices,red='reference foreground without a reliable SAM seed',magenta='explicit unresolved cluster',
        automatic_display_overlap='Painter display only; underlying NPZ preserves overlapping masks',
        quality_approved=False),indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'results/evaluation_v2_20261121/gte_full_diagnostic'
    frames=list(csv.DictReader((base/'frames.csv').open()));objects=list(csv.DictReader((base/'objects.csv').open()))
    modality_plots(frames);size_plots(objects);instance_qa()
    print('Wrote modality/range, size/teacher and instance QA figures with source tables')
