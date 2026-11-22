"""Object-focused report figures from existing verified predictions; CPU only."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from floor_plane import floor_gap

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/triton_20261121/production/analysis_final'
EVIDENCE=ROOT/'results/results_report_20261122'
OUT=ROOT/'results_report/figures'
COLORS={'rgb':'#222222','depth':'#888888','rgbd':'#0072b2'}


def rows(p):return list(csv.DictReader(p.open()))


def gray(p):
    a=np.array(Image.open(p));return a[...,0] if a.ndim==3 else a


def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.025);fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.025);plt.close(fig)


def pipeline():
    fig,ax=plt.subplots(figsize=(3.3,1.65));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.96,'Same 1,400 ISOD training images',ha='center',va='center',fontsize=7.5)
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',ec='#444444',fc='white',lw=.65))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)
    box(.015,.62,.46,.22,'Reference labels\n(SAM-initialized, corrected)')
    box(.525,.62,.46,.22,'SAM 3 labels\nRGB + 3 text prompts')
    box(.015,.34,.46,.17,'Train model')
    box(.525,.34,.46,.17,'Train same model')
    box(.10,.015,.80,.18,'Test on ISOD (300 frames)\nand GTE-Floor (320 frames)')
    for x in [.245,.755]:
        ax.annotate('',xy=(x,.51),xytext=(x,.62),arrowprops=dict(arrowstyle='->',lw=.6))
        ax.annotate('',xy=(.5,.195),xytext=(x,.34),arrowprops=dict(arrowstyle='->',lw=.6))
    fig.tight_layout(pad=.1);save(fig,'supervision_pipeline')


def comparison(evidence):
    groups=[('ESANet RGB','rgb_reference','rgb_automatic','rgb'),
            ('ESANet D','depth_reference','depth_automatic','depth'),
            ('ESANet RGB-D','rgbd_reference','rgbd_automatic','rgbd'),
            ('U-Net RGB','unet_reference',None,'rgb'),('YOLOv8n-seg','yolo_reference',None,'rgb'),
            ('Direct SAM','sam_three_prompt_v2',None,'rgb'),('Floor fit','floor_plane',None,'depth')]
    metrics=[('coverage_recall50','Objects covered','Recall (%)'),
             ('micro_localization_recall','Objects localized','Recall (%)'),
             ('pixel_iou','Mask quality','IoU (%)'),
             ('false_components_per_frame','Excess detections','Components/frame (log)')]
    fig,axes=plt.subplots(1,4,figsize=(6.8,2.2),sharey=True)
    for y,(label,ref,auto,color) in enumerate(groups):
        for kind,key,offset in [('reference',ref,-.12 if auto else 0),('automatic',auto,.12)]:
            if key is None:continue
            fixed=not evidence[key]['seeds'];marker='D' if fixed else 'o' if kind=='reference' else 's'
            for ax,(metric,title,xlabel) in zip(axes,metrics):
                m=evidence[key]['metrics'][metric];scale=1 if metric=='false_components_per_frame' else 100
                ax.errorbar(m['mean']*scale,y+offset,xerr=None if m['sd'] is None else m['sd']*scale,
                    marker=marker,markersize=3.5,color=COLORS[color],mfc=COLORS[color] if fixed or kind=='reference' else 'white',
                    capsize=2,lw=.9,elinewidth=.8)
    for ax,(metric,title,xlabel) in zip(axes,metrics):
        ax.set_title(title,fontsize=8);ax.set_xlabel(xlabel,fontsize=7.5)
        ax.set_yticks(range(len(groups)),[g[0] for g in groups]);ax.tick_params(axis='both',labelsize=7)
        ax.grid(axis='x',color='.9');ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        if metric=='false_components_per_frame':ax.set_xscale('log');ax.set_xlim(.5,600);ax.set_xticks([1,10,100],['1','10','100'])
        else:ax.set_xlim(0,100);ax.set_xticks([0,50,100])
    axes[0].invert_yaxis()
    legend=[Line2D([],[],color='#333333',marker='o',ls='none',markersize=4,label='Reference-label training'),
            Line2D([],[],color='#333333',marker='s',mfc='white',ls='none',markersize=4,label='SAM-label training'),
            Line2D([],[],color='#333333',marker='D',ls='none',markersize=4,label='Direct method')]
    fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(.56,0))
    fig.subplots_adjust(left=.15,right=.995,top=.86,bottom=.30,wspace=.24)
    save(fig,'detection_comparison')


def conditions(objects):
    fig,axes=plt.subplots(1,3,figsize=(6.8,1.65));out=[];counts=None
    for mode,label,marker in [('rgb','RGB','o'),('depth','D','s'),('rgbd','RGB-D','^')]:
        curves=defaultdict(list);areas=[]
        for seed in range(3):
            model=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}'
            units=defaultdict(list)
            for r in objects:
                if r['model']==model:units[r['recording'],r['id']].append(r)
            byrecord=defaultdict(list);bins=[[],[],[]]
            for (recording,ident),rr in units.items():
                assert len(rr)==5
                cov=float(np.mean([float(r['coverage'])>=.5 for r in rr]));loc=float(np.mean([r['localized50']=='True' for r in rr]));area=np.mean([int(r['area_pixels']) for r in rr])
                byrecord[recording].append((cov,loc));bins[0 if area<100 else 1 if area<1000 else 2].append(cov)
            for d in [.75,1.5,2.5]:
                selected=[np.mean(v,axis=0) for k,v in byrecord.items() if int(k.split('_')[-1])/100==d]
                assert len(selected)==20
                value=np.mean(selected,axis=0)
                for i in range(2):curves[i,d].append(value[i])
                out.append(dict(modality=mode,seed=seed,nominal_floor_distance_m=d,recordings=20,coverage_recall=value[0],localization_recall=value[1]))
            counts=[len(v) for v in bins];areas.append([np.mean(v) for v in bins])
        for i in range(2):
            x=[.75,1.5,2.5];a=np.array([curves[i,d] for d in x])
            axes[i].errorbar(x,a.mean(1)*100,yerr=a.std(1,ddof=1)*100,marker=marker,color=COLORS[mode],capsize=2,lw=1,label=label,markersize=3.5)
        a=np.array(areas);axes[2].errorbar(range(3),a.mean(0)*100,yerr=a.std(0,ddof=1)*100,marker=marker,color=COLORS[mode],capsize=2,lw=1,markersize=3.5)
    for ax,title in zip(axes,['(a) Covered by distance','(b) Localized by distance','(c) Covered by apparent size']):
        ax.set_title(title,fontsize=8);ax.set_ylim(0,100);ax.set_yticks([0,50,100]);ax.tick_params(labelsize=7)
        ax.grid(axis='y',color='.9');ax.spines[['top','right']].set_visible(False);ax.set_ylabel('Recall (%)',fontsize=7.5)
    for ax in axes[:2]:ax.set_xticks([.75,1.5,2.5]);ax.set_xlabel('Nominal floor distance (m)',fontsize=7.5)
    axes[2].set_xticks(range(3),[f'<100\n(n={counts[0]})',f'100–999\n(n={counts[1]})',f'≥1000\n(n={counts[2]})'])
    axes[2].set_xlabel('Annotated foreground area (px)',fontsize=7.5);axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    fig.tight_layout(pad=.35,w_pad=.65);save(fig,'detection_by_distance_size')
    with (EVIDENCE/'visual_range_size.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)


def crop_for(mask):
    yy,xx=np.where(mask);h,w=mask.shape
    pad=max(8,.18*max(np.ptp(xx)+1,np.ptp(yy)+1))
    cw=min(w,int(np.ceil(max(np.ptp(xx)+1+2*pad,(np.ptp(yy)+1+2*pad)*w/h))))
    ch=min(h,int(np.ceil(cw*h/w)))
    x=int(np.clip((xx.min()+xx.max()+1-cw)/2,0,w-cw));y=int(np.clip((yy.min()+yy.max()+1-ch)/2,0,h-ch))
    box=(x,y,x+cw,y+ch)
    assert mask[y:y+ch,x:x+cw].sum()==mask.sum()
    return box


def examples(objects,evidence):
    units=rows(EVIDENCE/'matched_units.csv')
    recovered=max([r for r in units if float(r['rgbd_localization_rate'])>=.8 and float(r['rgb_localization_rate'])<=.2],key=lambda r:float(r['rgbd_coverage_rate'])-float(r['rgb_coverage_rate']))
    loss=min(units,key=lambda r:float(r['rgbd_coverage_rate'])-float(r['rgb_coverage_rate']))
    choices=[('RGB-D recovery',recovered['recording'],[int(recovered['id'])]),
             ('Fusion miss',loss['recording'],[int(loss['id'])]),
             ('Cloths on bare floor','h52_floor_rags_075',[1,2,3,4]),
             ('Same cloths on poster','h52_poster_rags_075',[1,2,3,4])]
    definitions=json.loads((ROOT/'results/gte_instance_proposals_20261121/manifest.json').read_text())['frames'];lookup={(r['recording'],r['frame']):r for r in definitions}
    sources=json.loads((BASE/'gte_full_diagnostic/summary.json').read_text())['source_runs'];obs={(r['model'],r['recording'],r['frame'],r['id']):r for r in objects}
    settings=json.loads((ROOT/'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m']
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('black')
    choices=choices[2:]
    fig,axes=plt.subplots(2,5,figsize=(6.8,2.45));manifest=[]
    for row,(kind,recording,idents) in enumerate(choices):
        rgb=np.array(Image.open(ROOT/'data/OWN'/recording/'rgb/2.png').convert('RGB'));gt=gray(ROOT/'data/OWN_manual'/f'{recording}_2.png')>0
        source=lookup[recording,2];ids=gray(ROOT/'results/gte_instance_proposals_20261121'/source['path']);target=np.isin(ids,idents)
        x0,y0,x1,y1=crop_for(target);clip=np.s_[y0:y1,x0:x1]
        axes[row,0].imshow(rgb);axes[row,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#df36bc',lw=.8))
        axes[row,0].contour(target,levels=[.5],colors='cyan',linewidths=.45)
        raw=gray(ROOT/'data/OWN'/recording/'depth/2.png');gap,valid=floor_gap(raw,settings,seed_from=0)
        depth_artist=axes[row,1].imshow(np.ma.masked_where(~valid[clip],gap[clip]*1000),cmap=cmap,vmin=-50,vmax=50)
        axes[row,1].contour(target[clip],levels=[.5],colors='yellow',linewidths=.45)
        shown=[]
        for col,name in enumerate(['esanet_full_rgb_reference_s0','esanet_full_depth_reference_s0','rgbd_reference210_s0'],2):
            pred=gray(ROOT/sources[name]['predictions']/recording/'2.png')>0
            painted=rgb.astype(float)*.55;painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
            axes[row,col].imshow(painted.astype(np.uint8)[clip]);axes[row,col].contour(target[clip],levels=[.5],colors='yellow',linewidths=.4)
            metrics=[]
            for ident in idents:
                metric=obs[name,recording,'2',str(ident)];coverage=float(metric['coverage']);localized=metric['localized50']=='True'
                metrics.append(dict(id=ident,coverage=coverage,adjusted_iou=float(metric['adjusted_iou']),localized=localized))
            label=(f'{metrics[0]["coverage"]:.0%} covered' if len(idents)==1 else
                   'Covered: '+''.join(str(v['id'])+' ' for v in metrics if v['coverage']>=.5).strip())
            localized=[str(v['id']) for v in metrics if v['localized']]
            label+='\nLocalized: '+(','.join(localized) if localized else 'none')
            axes[row,col].text(.5,-.07,label,transform=axes[row,col].transAxes,ha='center',va='top',fontsize=5.9,linespacing=1.0)
            shown.append(dict(model=name,objects=metrics))
            if len(idents)==4:
                for ident in idents:
                    yy,xx=np.where(ids==ident)
                    axes[row,col].text(float(np.mean(xx))-x0,float(np.min(yy))-y0-3,str(ident),color='yellow',fontsize=6,ha='center',weight='bold')
        distance=int(recording.split('_')[-1])/100
        axes[row,0].text(0,1.10,f'({chr(97+row)}) {kind}, {distance:g} m',transform=axes[row,0].transAxes,ha='left',fontsize=7,clip_on=False)
        manifest.append(dict(selection=kind,recording=recording,reference_ids=idents,frame=2,seed=0,crop=[x0,y0,x1,y1],
            display_only=True,full_frame_scoring=True,shown=shown))
    for ax,title in zip(axes[0],['Full scene + reference','Depth residual (zoom)','ESANet RGB','ESANet D','ESANet RGB-D']):ax.set_title(title,fontsize=7,pad=17)
    for ax in axes.flat:
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.01,right=.995,bottom=.105,top=.89,wspace=.025,hspace=.63)
    fig.canvas.draw();pos=axes[1,1].get_position();cax=fig.add_axes([pos.x0,.026,pos.width,.01]);cb=fig.colorbar(depth_artist,cax=cax,orientation='horizontal',ticks=[-50,0,50]);cb.ax.tick_params(labelsize=5.5,pad=1)
    save(fig,'detection_examples')
    return manifest

def main():
    OUT.mkdir(exist_ok=True,parents=True);plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    evidence=json.loads((EVIDENCE/'evidence.json').read_text());objects=rows(BASE/'gte_full_diagnostic/objects.csv')
    pipeline();comparison(evidence);conditions(objects);selected=examples(objects,evidence)
    (OUT/'detection_visuals_manifest.json').write_text(json.dumps(dict(
        figures=['detection_comparison','detection_by_distance_size','detection_examples'],
        graph_errors='Training-seed SD, not partner-report bootstrap intervals; RGB automatic n=2, remaining learned cohorts n=3.',
        metrics='Coverage >=0.5 reference pixels; localization adjusted IoU >=0.5; false prediction components; not native instance AP.',
        image_area='Reference foreground pixels on original 640x362 GTE evaluation grid, not square centimetres.',
        physical_size='Supplementary only when measured or calibrated/validated; inverse-depth floor residual is not object height.',
        inspiration='Partner report: emphasize object recall, show performance graphically and enlarge matched failure/success crops. All numbers/predictions here use the current frozen experiments.',
        examples=selected),indent=2)+'\n')
    print('Created detection comparison, detection by distance/size, and enlarged found/missed examples')


if __name__=='__main__':main()
