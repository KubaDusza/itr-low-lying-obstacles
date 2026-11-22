"""Teammate-style figures, current tables and appendix evidence from saved runs."""
import csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from floor_plane import floor_gap
from analyse_instance_detection import match

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'results/results_report_20261122';R=E/'revision';P=ROOT/'results_report';F=P/'figures'
COLORS={'rgb':'#2878d4','depth':'#777777','rgbd':'#f46c24','reference':'#2878d4','automatic':'#f46c24'}
MODES=[('rgb','RGB'),('depth','D'),('rgbd','RGB-D')]
EXTENDED=ROOT/'results/extended_training_20261122'
NATIVE=EXTENDED if all((EXTENDED/'runs'/f'yolov8n_detect_{src}_s{s}'/'VERIFIED').exists() for src in ['reference','automatic'] for s in range(3)) else ROOT/'results/instance_detection_20261122'
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':8,'pdf.fonttype':42,'axes.linewidth':.7})

def rows(p):return list(csv.DictReader(p.open()))
def save(fig,name):
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight',pad_inches=.03);fig.savefig(F/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.03);plt.close(fig)
def mean(v):return float(np.mean(v))
def sd(v):return float(np.std(v,ddof=1)) if len(v)>1 else 0
def stat(v,scale=1,precision=3):return f'{mean(v)*scale:.{precision}f}'
def gray(p):
    a=np.array(Image.open(p));return a[...,0] if a.ndim==3 else a

def tables():
    domain=json.loads((R/'domain_comparison.json').read_text());cat=rows(R/'category_scores.csv');comp=json.loads((R/'component_comparison.json').read_text())['runs']
    lines=[r'\begin{table*}[!b]',r'\caption{Reference-trained ESANet on the reviewed ISOD test subset (30 frames) and external GTE-Floor (320 frames, including 20 empty). Values are means of three seeds. Object recall is the fraction of annotated units at least half covered; these units include groups. Pixel FP/FN are normalized per frame. Differences in scene, sensor and annotations accompany the domain change. Seed variation is given in Appendix~\ref{app:fullscores}.}',r'\label{tab:domain-results}',r'\centering\footnotesize',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'& \multicolumn{3}{c}{ISOD (30 frames)} & \multicolumn{3}{c}{GTE-Floor (320 frames)}\\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Metric & RGB & D & RGB-D & RGB & D & RGB-D\\',r'\midrule']
    for key,title in [('iou','Pixel IoU'),('precision','Pixel precision'),('recall','Pixel recall'),('f1','Pixel F1'),('object_recall','Object recall'),('fp_pixels_per_frame','FP pixels/frame'),('fn_pixels_per_frame','FN pixels/frame'),('empty_false_pixels_per_frame','Empty scenes: predicted pixels/frame')]:
        vals=[]
        for dataset in ['ISOD','GTE']:
            for mode,label in MODES:
                d=domain[mode][dataset].get(key);vals.append('--' if d is None else f'{d["mean"]:.0f}' if 'pixels' in key else f'{d["mean"]:.3f}')
        lines.append(title+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}'];(P/'domain_table.tex').write_text('\n'.join(lines)+'\n')
    labels={'rags':'Cloths/tissues','tapes':'Tape rolls','random':'Mixed items','chess':'Chess pieces','screws':'Screws','cables':'Cables','hexkeys':'Hex keys'}
    lines=[r'\begin{table}[!t]',r'\caption{GTE-Floor results by object set, reference-trained ESANet. $n$ counts recordings, each with five static repeats. Entries are three-seed means. Higher recall means more annotated units covered, not necessarily more precise masks.}',r'\label{tab:object-sets}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lr*{6}{r}}',r'\toprule',r' & & \multicolumn{3}{c}{Pixel IoU} & \multicolumn{3}{c}{Object recall}\\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',r'Object set & $n$ & RGB & D & RGB-D & RGB & D & RGB-D\\',r'\midrule']
    for key,label in labels.items():
        selected=[r for r in cat if r['object_set']==key];n=selected[0]['recordings'];vals=[]
        for metric in ['pixel_iou','object_recall']:
            vv=[mean([float(r[metric]) for r in selected if r['modality']==mode]) for mode,_ in MODES]
            for value in vv:vals.append((r'\textbf{'+f'{value:.3f}'+'}') if value==max(vv) else f'{value:.3f}')
        lines.append(label+' & '+n+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'object_set_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[!t]',r'\caption{Common obstacle-region comparison on GTE-Floor. Recall credits combined coverage; component F1 also penalizes extra regions and poor spatial extent using the published split/merge-tolerant metric~\cite{chan2021segment}. Native boxes are filled and united first. Learned entries average three seeds; direct methods are untrained.}',r'\label{tab:shared-detection}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Object recall} & \multicolumn{2}{c}{Component F1}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r'Method & Ref. & SAM & Ref. & SAM\\',r'\midrule']
    fams=[('ESANet RGB','ESANet RGB'),('ESANet D','ESANet D'),('ESANet RGB-D','ESANet RGB-D'),('U-Net RGB','U-Net RGB'),('YOLOv8n-seg RGB','YOLOv8n-seg'),('YOLOv8n RGB','YOLOv8n boxes')]
    for family,label in fams:
        vals=[]
        for metric in ['object_recall','component_mean_f1']:
            for source in ['reference','automatic']:
                rr=[r[metric] for r in comp.values() if r['family']==family and r['source']==source];vals.append(stat(rr) if rr else '--')
        lines.append(label+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\midrule']
    for family,label in [('SAM 3 direct','Direct SAM 3'),('Floor plane D','Floor-plane fit')]:
        d=next(r for r in comp.values() if r['family']==family);lines.append(label+r' & \multicolumn{2}{c}{'+f'{d["object_recall"]:.3f}'+r'} & \multicolumn{2}{c}{'+f'{d["component_mean_f1"]:.3f}'+r'}\\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'shared_detection_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{longtable}{llrrr}',r'\caption{Reference-trained ESANet domain scores with training-seed standard deviation. Each dataset entry is pooled within each seed, then averaged. ISOD has 30 reviewed test frames; GTE has 320 full frames.}\label{tab:domain-spread}\\',r'\toprule Dataset & Metric & RGB & D & RGB-D\\\midrule\endfirsthead',r'\toprule Dataset & Metric & RGB & D & RGB-D\\\midrule\endhead']
    for dataset in ['ISOD','GTE']:
        for metric,title in [('iou','Pixel IoU'),('precision','Pixel precision'),('recall','Pixel recall'),('f1','Pixel F1'),('object_recall','Object recall')]:
            if metric not in domain['rgb'][dataset]:continue
            vals=[]
            for mode,label in MODES:
                d=domain[mode][dataset][metric];vals.append(f'{d["mean"]:.3f}'+r'$\pm$'+f'{d["sd"]:.3f}')
            lines.append(dataset+' & '+title+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\bottomrule\end{longtable}'];(P/'domain_spread_table.tex').write_text('\n'.join(lines)+'\n')
    # Full values with seed SD and explicit target representation belong in the appendix.
    lines=[r'\begin{longtable}{llrrrr}',r'\caption{Complete obstacle-region scores on GTE-Floor: means $\pm$ seed standard deviation. The F1 metric follows SegmentMeIfYouCan with the dataset adaptations in Appendix~\ref{app:evaluation}. False components use adjusted component precision threshold 0.5 and include unannotated props.}\label{tab:full-component}\\',r'\toprule Method & Training labels & Object recall & Component recall & False regions/frame & Mean component F1\\\midrule\endfirsthead',r'\toprule Method & Training labels & Object recall & Component recall & False regions/frame & Mean component F1\\\midrule\endhead']
    for family in [x[0] for x in fams]+['SAM 3 direct','Floor plane D']:
        for source in ['reference','automatic','direct']:
            rr=[r for r in comp.values() if r['family']==family and r['source']==source]
            if not rr:continue
            vals=[]
            for metric in ['object_recall','component_recall','false_components_per_frame','component_mean_f1']:
                vv=[r[metric] for r in rr];vals.append(f'{mean(vv):.3f}'+(r'$\pm$'+f'{sd(vv):.3f}' if len(vv)>1 else ''))
            lines.append(family+' & '+source+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\bottomrule\end{longtable}'];(P/'full_component_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{longtable}{lrrrr}',r'\caption{Per-item reference-supervised recovery on GTE-Floor. $n$ is the number of annotated unit/recording pairs; repeats are averaged within each seed. Item names and existing groups follow the collection inventory, rather than automatically inferred semantic classes.}\label{tab:per-item}\\',r'\toprule Annotated item & $n$ & RGB & D & RGB-D\\\midrule\endfirsthead',r'\toprule Annotated item & $n$ & RGB & D & RGB-D\\\midrule\endhead']
    obs=rows(R/'objects.csv');inventory={r['item_id']:r['label'] for r in rows(ROOT/'data_collection/items.csv')}
    models={f'rgbd_reference210_s{s}' if m=='rgbd' else f'esanet_full_{m}_reference_s{s}':(m,s) for m,_ in MODES for s in range(3)}
    items=defaultdict(list)
    for r in obs:
        if r['model'] in models:items[r['parent_item_id']].append(r)
    for item,rr in sorted(items.items()):
        n=len({(r['recording'],r['id']) for r in rr});vals=[]
        for mode,_ in MODES:
            perseed=[mean([float(r['coverage'])>=.5 for r in rr if models[r['model']]==(mode,seed)]) for seed in range(3)];vals.append(f'{mean(perseed):.3f}')
        label=inventory.get(item,item).replace('&',r'\&').replace('_',r'\_');lines.append(label+' & '+str(n)+' & '+' & '.join(vals)+r'\\')
    lines +=[r'\bottomrule\end{longtable}'];(P/'per_item_table.tex').write_text('\n'.join(lines)+'\n')


def range_plot():
    objects=rows(R/'objects.csv');frames=rows(R/'frames.csv');fig,axes=plt.subplots(1,3,figsize=(6.8,1.8));counts=None
    for offset,(mode,label) in zip([-.06,0,.06],MODES):
        vals=defaultdict(list);area=[]
        for seed in range(3):
            model=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}';oo=[r for r in objects if r['model']==model];ff=[r for r in frames if r['model']==model]
            for distance in [.75,1.5,2.5]:
                selected=[r for r in ff if r['nominal_distance_m'] and float(r['nominal_distance_m'])==distance]
                vals[0,distance].append(sum(int(r['pixel_inter']) for r in selected)/sum(int(r['pixel_union']) for r in selected))
                byrec=defaultdict(list)
                for r in oo:
                    if int(r['recording'].split('_')[-1])/100==distance:byrec[r['recording']].append(float(r['coverage'])>=.5)
                assert len(byrec)==20;vals[1,distance].append(mean([mean(v) for v in byrec.values()]))
            units=defaultdict(list)
            for r in oo:units[r['recording'],r['id']].append(r)
            bins=[[],[],[]]
            for rr in units.values():
                a=mean([int(r['area_pixels']) for r in rr]);b=0 if a<100 else 1 if a<1000 else 2;bins[b].append(mean([float(r['coverage'])>=.5 for r in rr]))
            counts=[len(v) for v in bins];area.append([mean(v) for v in bins])
        for panel in [0,1]:
            for x,d in enumerate([.75,1.5,2.5]):
                v=vals[panel,d];axes[panel].errorbar(x+offset,mean(v),yerr=sd(v),fmt='o',ms=3.5,color=COLORS[mode],capsize=0,elinewidth=1.1,label=label if x==0 else None)
        for x in range(3):
            v=[a[x] for a in area];axes[2].errorbar(x+offset,mean(v),yerr=sd(v),fmt='o',ms=3.5,color=COLORS[mode],capsize=0,elinewidth=1.1)
    for ax,title in zip(axes,['(a) Pooled pixel IoU vs. distance','(b) Object recall vs. distance','(c) Object recall vs. image size']):
        ax.set_title(title,fontsize=8);ax.set_ylim(0,1);ax.set_yticks(np.linspace(0,1,6));ax.set_xlim(-.4,2.4);ax.tick_params(labelsize=7)
        ax.grid(axis='y',color='.85',lw=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for ax in axes[:2]:ax.set_xticks(range(3),['0.75 m\n(n=20)','1.5 m\n(n=20)','2.5 m\n(n=20)'])
    axes[2].set_xticks(range(3),[f'<100 px\n(n={counts[0]})',f'100–999 px\n(n={counts[1]})',f'≥1000 px\n(n={counts[2]})']);axes[0].set_ylabel('Score',fontsize=7.5);axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    fig.tight_layout(pad=.35,w_pad=.7);save(fig,'detection_by_distance_size')


def cloth_figures():
    sources=json.loads((R/'summary.json').read_text())['source_runs'];objects=rows(R/'objects.csv');obs={(r['model'],r['recording'],r['frame'],r['id']):r for r in objects}
    definitions=json.loads((ROOT/'results/gte_instance_proposals_20261121/manifest.json').read_text())['frames'];lookup={(r['recording'],r['frame']):r for r in definitions}
    settings=json.loads((ROOT/'results/frozen_v1/floor_plane_v1/settings.json').read_text())['depth_range_m'];cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('black')
    for seed in range(3):
        fig,axes=plt.subplots(2,6,figsize=(6.8,2.3));case=[]
        for row,recording in enumerate(['h52_floor_rags_075','h52_poster_rags_075']):
            rgb=np.array(Image.open(ROOT/'data/OWN'/recording/'rgb/2.png').convert('RGB'));gt=gray(ROOT/'data/OWN_manual'/f'{recording}_2.png')>0
            ids=gray(ROOT/'results/gte_instance_proposals_20261121'/lookup[recording,2]['path']);raw=gray(ROOT/'data/OWN'/recording/'depth/2.png');gap,valid=floor_gap(raw,settings,seed_from=0)
            axes[row,0].imshow(rgb);axes[row,0].contour(gt,levels=[.5],colors='cyan',linewidths=.4)
            depth_artist=axes[row,1].imshow(np.ma.masked_where(~valid,gap*1000),cmap=cmap,vmin=-50,vmax=50)
            axes[row,1].contour(gt,levels=[.5],colors='yellow',linewidths=.35)
            for col,mode in enumerate(['rgb','depth','rgbd'],2):
                model=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}';pred=gray(ROOT/sources[model]['predictions']/recording/'2.png')>0
                painted=rgb.astype(float)*.55;painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
                axes[row,col].imshow(painted.astype(np.uint8));axes[row,col].contour(gt,levels=[.5],colors='yellow',linewidths=.35)
                covered=[]
                for ident in [1,2,3,4]:
                    yy,xx=np.where(ids==ident);axes[row,col].text(float(np.mean(xx)),float(np.min(yy))-5,str(ident),fontsize=5.7,color='yellow',ha='center')
                    metric=obs[model,recording,'2',str(ident)]
                    if float(metric['coverage'])>=.5:covered.append(str(ident))
                axes[row,col].text(.5,-.07,'Covered: '+(','.join(covered) if covered else 'none'),transform=axes[row,col].transAxes,ha='center',va='top',fontsize=5.8)
                case.append(dict(seed=seed,recording=recording,model=model,covered=covered))
            rr=next(r for r in json.loads((NATIVE/f'runs/yolov8n_detect_reference_s{seed}/gte_boxes.json').read_text())['frames'] if r['image']==recording+'_2')
            refs=next(r for r in json.loads((ROOT/'results/instance_detection_20261122/input/reference_manifest.json').read_text())['gte'] if r['recording']==recording and r['frame']==2)
            result=match([b['xyxy'] for b in refs['boxes']],rr['boxes_xyxy'],rr['scores']);matched={m['prediction'] for m in result['matched']};axes[row,5].imshow(rgb)
            for i,(box,conf) in enumerate(zip(rr['boxes_xyxy'],rr['scores'])):
                if conf<.25:continue
                x0,y0,x1,y1=box;axes[row,5].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=.7,ec='#f46c24'))
            box_union=np.zeros(gt.shape,bool)
            for box,conf in zip(rr['boxes_xyxy'],rr['scores']):
                if conf>=.25:
                    x0,y0,x1,y1=box;box_union[max(0,int(np.floor(y0))):min(gt.shape[0],int(np.ceil(y1))),max(0,int(np.floor(x0))):min(gt.shape[1],int(np.ceil(x1)))]=True
            covered=[str(i) for i in [1,2,3,4] if np.mean(box_union[ids==i])>=.5]
            axes[row,5].text(.5,-.07,'Covered: '+(','.join(covered) if covered else 'none'),transform=axes[row,5].transAxes,ha='center',va='top',fontsize=5.8)
            axes[row,0].text(0,1.13,('Bare floor' if row==0 else 'White poster')+f', seed {seed}',transform=axes[row,0].transAxes,ha='left',fontsize=6.8,clip_on=False)
        for ax,title in zip(axes[0],['RGB + reference','Depth residual','ESANet RGB','ESANet D','ESANet RGB-D','YOLOv8n boxes']):ax.set_title(title,fontsize=6.7,pad=19)
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
        fig.subplots_adjust(left=.01,right=.995,bottom=.16,top=.83,wspace=.02,hspace=.6);fig.canvas.draw();pos=axes[1,1].get_position();cax=fig.add_axes([pos.x0,.045,pos.width,.012]);cb=fig.colorbar(depth_artist,cax=cax,orientation='horizontal',ticks=[-50,0,50]);cb.ax.tick_params(labelsize=5.5,pad=1)
        save(fig,'detection_examples' if seed==0 else f'cloth_seed{seed}')
    grouped_cloth_pdf()
    (F/'cloth_display_notes.json').write_text(json.dumps(dict(frame=2,main_seed=0,all_seeds=[0,1,2],no_empty_labels=True,primary_labels='Coverage >=50% of each reference mask; no localization-status labels',native_boxes='Confidence .25; orange boxes. Coverage and component scoring use the filled union, not box counts.'),indent=2)+'\n')


def grouped_cloth_pdf():
    import fitz
    paths=[F/'detection_examples.pdf',F/'cloth_seed1.pdf',F/'cloth_seed2.pdf']
    docs=[fitz.open(p) for p in paths];width=docs[0][0].rect.width;clips=[]
    for doc in docs:
        page=doc[0];blocks=page.get_text('blocks')
        floor=next(b[1] for b in blocks if b[4].startswith('Bare floor, seed'))
        poster=next(b[1] for b in blocks if b[4].startswith('White poster, seed'))
        lower=max(b[3] for b in blocks if 'Covered:' in b[4] and b[1]>poster)
        clips.append([fitz.Rect(0,floor-1.5,width,poster-2),fitz.Rect(0,poster-1.5,width,lower+2)])
    header=32.;gap=7.;height=2*header+sum(clips[s][scene].height for scene in range(2) for s in range(3))+7*gap
    result=fitz.open();page=result.new_page(width=width,height=height);y=0
    columns=['RGB + reference','Depth residual','ESANet RGB','ESANet D','ESANet RGB-D','YOLOv8n boxes']
    for scene,title in enumerate(['Bare floor: seeds 0, 1, 2','White poster: seeds 0, 1, 2']):
        page.insert_text((4,y+10),title,fontsize=9,fontname='tiro')
        for c,label in enumerate(columns):page.insert_textbox(fitz.Rect(c*width/6,y+17,(c+1)*width/6,y+30),label,fontsize=7,fontname='tiro',align=1)
        y+=header
        for seed,doc in enumerate(docs):
            clip=clips[seed][scene];dest=fitz.Rect(0,y,width,y+clip.height);page.show_pdf_page(dest,doc,0,clip=clip);y+=clip.height+gap
    result.save(F/'cloth_scenes_all_seeds.pdf');result.close()
    for doc in docs:doc.close()


def curves():
    audit=[];fig,axes=plt.subplots(3,3,figsize=(8.2,6.5))
    for row,(mode,label) in enumerate(MODES):
        for source in ['reference','automatic']:
            arrays=[]
            for seed in range(3):
                if mode=='rgbd':run=ROOT/'results/triton_20261121/production/runs'/f'rgbd_{"reference" if source=="reference" else "sam"}210_s{seed}';hpath=next(run.glob('history/**/logs.csv'))
                elif source=='reference':hpath=next(ROOT.glob(f'results/frozen_v1/*/runs/esanet_full_{mode}_reference_s{seed}/**/logs.csv'))
                else:
                    name='rgb_sam_v2_s0_recovery_glaurung' if mode=='rgb' and seed==0 else f'{mode}_sam_v2_s{seed}';hpath=next(ROOT.glob(f'results/sam_v2_20261121/*/runs/{name}/**/logs.csv'))
                h=rows(hpath);assert len(h)==500 and [int(r['epoch']) for r in h]==list(range(500))
                v=np.array([[float(r[k]) for k in ['loss_train_full_size','loss_valid','IoU_valid_class_1']] for r in h]);assert np.isfinite(v).all();arrays.append(v)
                audit.append(dict(modality=mode,source=source,seed=seed,history=str(hpath.relative_to(ROOT)),epochs=len(h),selected_epoch_zero_based=max(range(500),key=lambda i:v[i,2])))
            a=np.array(arrays)
            for col in range(3):
                ax=axes[row,col]
                for x in a:ax.plot(np.arange(1,501),x[:,col],color=COLORS[source],alpha=.20,lw=.5)
                ax.plot(np.arange(1,501),a[:,:,col].mean(0),color=COLORS[source],lw=1,label='Reference' if source=='reference' else 'SAM')
        for col,metric in enumerate(['Training full-res. loss','Validation loss','Validation obstacle IoU']):
            ax=axes[row,col];ax.set_title(f'{label}: {metric}',fontsize=9);ax.set_xlabel('Epoch');ax.grid(axis='y',color='.88',lw=.5);ax.spines[['top','right']].set_visible(False)
            if col==2:ax.set_ylim(0,1)
    loss_max=max(axes[r,c].dataLim.ymax for r in range(3) for c in [0,1])*1.04
    for r in range(3):
        for c in [0,1]:axes[r,c].set_ylim(0,loss_max)
    axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout(pad=.7);save(fig,'appendix_esanet_curves')
    fig,axes=plt.subplots(2,3,figsize=(8.2,4.5))
    for seed in range(3):
        path=ROOT/f'results/extended_training_20261122/runs/unet_logged_reference_s{seed}/history.csv'
        h=rows(path);assert len(h)==15 and all('val_loss' in r for r in h)
        for ax,key in zip(axes[0],['train_loss','val_loss','val_IoU']):ax.plot([int(r['epoch']) for r in h],[float(r[key]) for r in h],label=f'Seed {seed}',lw=.9)
        h=rows(next(ROOT.glob(f'results/frozen_v1/*/runs/yolo_full_reference_s{seed}/**/results.csv')));assert len(h)==20
        for ax,key in zip(axes[1],['train/seg_loss','val/seg_loss','metrics/mAP50-95(M)']):ax.plot([int(r['epoch']) for r in h],[float(r[key]) for r in h],lw=.9)
    for ax,title in zip(axes.flat,['U-Net: training loss','U-Net: validation loss','U-Net: validation IoU','YOLOv8n-seg: training mask loss','YOLOv8n-seg: validation mask loss','YOLOv8n-seg: validation mask AP50:95']):
        ax.set_title(title,fontsize=8.5);ax.set_xlabel('Epoch');ax.grid(axis='y',color='.88',lw=.5);ax.spines[['top','right']].set_visible(False)
    for row in [0,1]:
        limit=max(axes[row,0].dataLim.ymax,axes[row,1].dataLim.ymax)*1.04
        for col in [0,1]:axes[row,col].set_ylim(0,limit)
        axes[row,2].set_ylim(0,1)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.tight_layout(pad=.7);save(fig,'appendix_rgb_baseline_curves');(R/'training_audit.json').write_text(json.dumps(dict(ESANet=audit,UNet_epochs=15,UNet_validation_loss_logged=True,YOLOseg_epochs=20),indent=2)+'\n')
    # Native loss pairs share y limits; AP retains its separate 0--1 scale.
    fig,axes=plt.subplots(2,4,figsize=(8.2,4.5))
    fields=['train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss','metrics/mAP50-95(B)']
    positions=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(0,3)]
    for source in ['reference','automatic']:
        vv=[]
        for seed in range(3):
            h=rows(NATIVE/f'runs/yolov8n_detect_{source}_s{seed}/results.csv');assert 1<=len(h)<=200
            vv.append((np.array([int(r['epoch']) for r in h]),np.array([[float(r[k]) for k in fields] for r in h])))
        common=min(len(y) for x,y in vv)
        a=np.array([y[:common] for x,y in vv])
        for col,pos in enumerate(positions):
            ax=axes[pos]
            for x,y in vv:ax.plot(x,y[:,col],color=COLORS[source],lw=.6,alpha=.25)
            ax.plot(np.arange(1,common+1),a[:,:,col].mean(0),color=COLORS[source],lw=1,label='Reference' if source=='reference' else 'SAM')
    for pos,title in zip(positions,['Train: box loss','Train: classification loss','Train: distribution loss','Validation: box loss','Validation: classification loss','Validation: distribution loss','Validation box AP50:95']):
        ax=axes[pos];ax.set_title(title,fontsize=8.3);ax.set_xlabel('Epoch');ax.grid(axis='y',color='.88',lw=.5);ax.spines[['top','right']].set_visible(False)
    for c in range(3):
        limit=max(axes[0,c].dataLim.ymax,axes[1,c].dataLim.ymax)*1.04
        for r in range(2):axes[r,c].set_ylim(0,limit)
    axes[0,3].set_ylim(0,1);axes[1,3].axis('off');axes[1,3].text(.5,.5,'Matching train/validation\ny-axis limits within\neach loss component.',ha='center',va='center',fontsize=8)
    axes[0,0].legend(frameon=False,fontsize=7.5);fig.tight_layout(pad=.7);save(fig,'appendix_native_curves')


def shorten_table_titles():
    titles={'domain_table.tex':'ESANet on ISOD and GTE-Floor.', 'object_set_table.tex':'GTE-Floor results by object set.',
        'shared_detection_table.tex':'Obstacle recovery with reference and SAM labels.', 'full_component_table.tex':'Full obstacle-region results on GTE-Floor.',
        'domain_spread_table.tex':'ESANet variability across training seeds.', 'per_item_table.tex':'Recovery by annotated GTE-Floor item.'}
    for name,title in titles.items():
        p=P/name;t=p.read_text();start=t.index('\\caption{')+len('\\caption{');level=1;end=start
        while level:
            if t[end]=='{':level+=1
            elif t[end]=='}':level-=1
            end+=1
        t=t[:start]+title+t[end-1:]
        notes={'domain_table.tex':'Three-seed means. Object recall uses annotated units, including groups. Pixel FP/FN are normalized per frame.',
            'object_set_table.tex':'$n$: recordings; three-seed means. Bold marks the largest mean in each metric block.',
            'shared_detection_table.tex':'Ref.: assisted reference labels; SAM: RGB-only automatic labels. Means of three seeds; direct methods are fixed. Boxes are filled and united for common scoring.'}
        if name in notes:t=t.replace(r'\end{tabular}',r'\end{tabular}'+r'\par\smallskip{\footnotesize '+notes[name]+'}')
        p.write_text(t)


def main():
    assert (R/'COMPLETE').exists();tables();shorten_table_titles();range_plot();cloth_figures();curves();print('Generated current tables, teammate-style range plot, all-seed cloth views and complete training curves')
if __name__=='__main__':main()
