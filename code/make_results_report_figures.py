"""Publication-size figures, with fixed and outcome-selected examples separated."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from floor_plane import floor_gap

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'results/results_report_20261122'
ANALYSIS=ROOT/'results/triton_20261121/production/analysis_final'
OUT=ROOT/'results_report/figures'


def gray(path):
    a=np.array(Image.open(path));return a[...,0] if a.ndim==3 else a


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42})
    distance=list(csv.DictReader((ANALYSIS/'figures/modality_distance_values.csv').open()))
    objects=list(csv.DictReader((ANALYSIS/'gte_full_diagnostic/objects.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.1));binrows=[]
    for mode,label,color,marker in [('rgb','RGB','#222222','o'),('depth','D','#888888','s'),('rgbd','RGB-D','#0072b2','^')]:
        values=[]
        for d in [.75,1.5,2.5]:
            values.append([float(r['macro_recording_localization_recall']) for r in distance if r['modality']==mode and float(r['nominal_distance_m'])==d])
        values=np.array(values);assert values.shape==(3,3)
        axes[0].errorbar([.75,1.5,2.5],values.mean(1),yerr=values.std(1,ddof=1),color=color,marker=marker,label=label,capsize=2,lw=1)
        bins=[]
        for seed in range(3):
            name=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}'
            units=defaultdict(list)
            for r in objects:
                if r['model']==name:units[r['recording'],r['id']].append(r)
            b=[[],[],[]]
            for unit,rr in units.items():
                assert len(rr)==5
                area=np.mean([int(r['area_pixels']) for r in rr]);i=0 if area<100 else 1 if area<1000 else 2
                b[i].append(np.mean([r['localized50']=='True' for r in rr]))
            counts=[len(v) for v in b];bins.append([np.mean(v) for v in b])
            for i,v in enumerate(b):binrows.append(dict(modality=mode,seed=seed,area_bin=i,unit_recording_pairs=len(v),localization_recall=float(np.mean(v))))
        arr=np.array(bins);axes[1].errorbar(range(3),arr.mean(0),yerr=arr.std(0,ddof=1),color=color,marker=marker,label=label,capsize=2,lw=1)
    axes[0].set_xlabel('Nominal floor distance (m)');axes[0].set_xticks([.75,1.5,2.5]);axes[0].set_title('(a) Same collection conditions by range')
    axes[1].set_xlabel('Reference unit area (pixels)');axes[1].set_xticks(range(3),[f'<100\n(n={counts[0]})',f'100–999\n(n={counts[1]})',f'≥1000\n(n={counts[2]})']);axes[1].set_title('(b) Apparent target size')
    for ax in axes:ax.set_ylim(0,1);ax.set_ylabel('Localization recall');ax.grid(axis='y',color='.9');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,loc='upper right');fig.tight_layout(pad=.4,w_pad=1)
    fig.savefig(OUT/'range_and_size.pdf');fig.savefig(OUT/'range_and_size.png',dpi=220);plt.close(fig)
    with (EVIDENCE/'report_size_bins.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(binrows[0]));w.writeheader();w.writerows(binrows)
    sources=json.loads((ANALYSIS/'gte_full_diagnostic/summary.json').read_text())['source_runs']
    settings=json.loads((ROOT/'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m']
    gapmap=plt.get_cmap('coolwarm').copy();gapmap.set_bad('black')
    fig,axes=plt.subplots(3,5,figsize=(6.8,2.85));records=[]
    for row,(background,dist) in enumerate([('floor','075'),('floor','250'),('poster','075')]):
        recording=f'h52_{background}_rags_{dist}';rgb=np.array(Image.open(ROOT/'data/OWN'/recording/'rgb/2.png').convert('RGB'))
        gt=gray(ROOT/'data/OWN_manual'/f'{recording}_2.png')>0
        raw=gray(ROOT/'data/OWN'/recording/'depth/2.png');gap,valid=floor_gap(raw,settings,seed_from=0)
        axes[row,0].imshow(rgb);axes[row,0].contour(gt,levels=[.5],colors=['cyan'],linewidths=.35)
        depthimage=axes[row,1].imshow(np.ma.masked_where(~valid,gap*1000),cmap=gapmap,vmin=-50,vmax=50)
        axes[row,0].set_ylabel(f'{background}\n{int(dist)/100:g} m',fontsize=8)
        for col,mode in enumerate(['rgb','depth','rgbd'],2):
            name='rgbd_reference210_s0' if mode=='rgbd' else f'esanet_full_{mode}_reference_s0'
            pred=gray(ROOT/sources[name]['predictions']/recording/'2.png')>0
            painted=rgb.astype(float)*.55
            painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
            axes[row,col].imshow(painted.astype(np.uint8))
        records.append(dict(recording=recording,frame=2,seed=0,display_crop=None))
    for ax,title in zip(axes[0],['RGB + reference','Depth residual','RGB prediction','D prediction','RGB-D prediction']):ax.set_title(title,fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.04,right=.995,bottom=.12,top=.92,wspace=.025,hspace=.04)
    pos=axes[2,1].get_position();cax=fig.add_axes([pos.x0,.065,pos.width,.015]);cb=fig.colorbar(depthimage,cax=cax,orientation='horizontal',ticks=[-50,0,50]);cb.ax.tick_params(labelsize=6,pad=1)
    fig.savefig(OUT/'cloth_depth_examples.pdf');fig.savefig(OUT/'cloth_depth_examples.png',dpi=240);plt.close(fig)
    (OUT/'selection.json').write_text(json.dumps(dict(
        range_and_size='All three matched-runtime reference seeds; average five static repeats before unit/recording summaries. Seed SD, not scene CI.',
        cloth_examples=dict(selection='Fixed h52/cloth set, frame 2 and seed 0: near/far bare floor plus near poster to combine scale, shape and background diagnostics. Not selected for fusion advantage. Full three-mark floor sequence retained in server analysis figures.',
            records=records,residual_scale_mm=[-50,50],same_full_frame_scale=True,interpretation='Camera-axis residual, not physical object height.'),
        supporting_teacher_examples='results/results_report_20261122/automatic_teacher_examples.pdf'),indent=2)+'\n')
    print('Created two publication-size figures and their selection manifest')


if __name__=='__main__':main()
