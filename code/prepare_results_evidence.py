"""Summarize completed, versioned experiments for the results report.

CPU-only: reads existing predictions/scores; never runs a model or changes labels.
Historical SAM runs are not substituted for the accepted three-prompt teacher.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'pdf.fonttype': 42})

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/'results/triton_20261121/production/analysis_final'
OUT = ROOT/'results/results_report_20261122'


def rows(path):
    return list(csv.DictReader(path.open()))


def write_csv(path, data):
    assert data
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def stats(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                n=len(values), values=[float(v) for v in values])


def teacher_audit():
    """Exact accepted binary teacher, compared with separately stored references."""
    manifest=json.loads((ROOT/'data/ISOD_sam_all_v2/manifest.json').read_text())
    assert manifest['complete'] and manifest['reference_masks_read'] is False
    protocol=json.loads((ROOT/'results/sam_prompt_v2/expanded_protocol.json').read_text())
    development={(r['site'],r['frame']) for r in protocol['development']}
    measured=[]
    for phase,base in [('development','ISOD_reference_train_v1'),('validation','ISOD_reference_val_v1'),('test_subset','ISOD_reference_test_v1')]:
        counts=defaultdict(int);ious=[]
        for p in sorted((ROOT/'data'/base).glob('*/*.png')):
            if phase=='development' and (p.parent.name,int(p.stem)) not in development:continue
            gt=np.array(Image.open(p))>0
            teacher=ROOT/'data/ISOD_sam_all_v2'/p.parent.name/p.name
            pred=np.array(Image.open(teacher))>0
            assert gt.shape==pred.shape
            inter=int((gt&pred).sum());union=int((gt|pred).sum())
            counts['intersection']+=inter;counts['union']+=union;counts['frames']+=1
            if union:ious.append(inter/union)
        assert counts['frames']=={'development':30,'validation':300,'test_subset':30}[phase]
        measured.append(dict(phase=phase,**counts,pooled_iou=counts['intersection']/counts['union'],
                             mean_image_iou=float(np.mean(ious)),empty_union_frames=counts['frames']-len(ious)))
    write_csv(OUT/'automatic_teacher_quality.csv',measured)
    choices=[('development','BT_BM',12),('development','BT_LM',34),
             ('validation','HL_BM',87),('validation','MEB_HW_A',44),
             ('test_subset','HDCL_HW_A',14),('test_subset','MEB_CM',44)]
    with PdfPages(OUT/'automatic_teacher_examples.pdf') as pdf:
        for start in range(0,len(choices),2):
            fig,axes=plt.subplots(2,3,figsize=(10,5.3))
            for row,(phase,site,frame) in enumerate(choices[start:start+2]):
                base={'development':'ISOD_reference_train_v1','validation':'ISOD_reference_val_v1','test_subset':'ISOD_reference_test_v1'}[phase]
                rgb=np.array(Image.open(ROOT/'data/ISOD/data'/site/'rgb'/f'{frame}.png').convert('RGB'))
                gt=np.array(Image.open(ROOT/'data'/base/site/f'{frame}.png'))>0
                pred=np.array(Image.open(ROOT/'data/ISOD_sam_all_v2'/site/f'{frame}.png'))>0
                reference=rgb.astype(float)*.55;reference[gt]=[20,205,205]
                painted=rgb.astype(float)*.55;painted[pred&gt]=[40,185,95];painted[pred&~gt]=[235,70,60];painted[~pred&gt]=[45,130,230]
                for col,image in enumerate([rgb,reference.astype(np.uint8),painted.astype(np.uint8)]):axes[row,col].imshow(image)
                axes[row,0].set_ylabel(f'{phase}\n{site} / {frame}',fontsize=9)
            for ax,title in zip(axes[0],['RGB','Reviewed reference','Exact RGB-only teacher']):ax.set_title(title,fontsize=10)
            for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle('Three-prompt automatic label examples',fontsize=13)
            fig.text(.5,.025,'Fixed examples, not an estimate of prevalence. Green: correct foreground; red: extra; blue: missed.\nPrompts: small object on the floor / cable on the floor / things on the floor. Confidence 0.6; mask threshold 0.5.\nDevelopment and validation informed prompts; the 30 reviewed test frames are a limited held-out subset.',ha='center',fontsize=8)
            fig.subplots_adjust(left=.085,right=.99,top=.89,bottom=.16,wspace=.04,hspace=.10)
            pdf.savefig(fig)
            fig.savefig(OUT/f'automatic_teacher_examples_{start//2+1}.png',dpi=150)
            plt.close(fig)
    (OUT/'automatic_teacher_example_selection.json').write_text(json.dumps(dict(
        selection='Two previously discussed training pile cases, two fixed validation frames, two fixed reviewed test frames. Selected before examining these panels.',
        frames=choices,teacher_manifest_sha256=hashlib.sha256((ROOT/'data/ISOD_sam_all_v2/manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (ANALYSIS/'COMPLETE').exists(), 'Final server analysis must finish first'
    completion = json.loads((ANALYSIS/'completion.json').read_text())
    assert completion['full_RGBD_training_and_depth_interventions_required']
    source_dir=OUT/'revision' if (OUT/'revision/COMPLETE').exists() else ANALYSIS/'gte_full_diagnostic'
    source = json.loads((source_dir/'summary.json').read_text())['source_runs']
    recordings = rows(ANALYSIS/'recording_summary.csv')
    if source_dir==OUT/'revision':
        grouped=defaultdict(list)
        for row in rows(source_dir/'frames.csv'):grouped[row['model'],row['recording']].append(row)
        recordings=[]
        for (model,recording),rr in grouped.items():
            tp=sum(int(r['tp50']) for r in rr);fn=sum(int(r['fn50']) for r in rr)
            recordings.append(dict(model=model,recording=recording,nominal_distance_m=rr[0]['nominal_distance_m'],localization_recall=tp/(tp+fn) if tp+fn else None))
    objects = rows(source_dir/'objects.csv')
    object_rows=defaultdict(list)
    for r in objects:object_rows[r['model']].append(r)
    def metrics(name):
        s = source[name]; op = s['pooled']['operating_points']['50']
        rr = [r for r in recordings if r['model']==name and r['nominal_distance_m']]
        assert len(rr)==60 and s['frames']==320
        return dict(pixel_iou=s['legacy_full_frame_pixel']['iou'],
                    coverage_recall50=float(np.mean([float(r['coverage'])>=.5 for r in object_rows[name]])),
                    micro_localization_recall=op['recall'],
                    macro_recording_localization_recall=float(np.mean([float(r['localization_recall']) for r in rr])),
                    false_components_per_frame=op['fp']/320)
    names = {}
    for mode in ('rgb','depth','rgbd'):
        names[mode,'reference'] = {s:(f'rgbd_reference210_s{s}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{s}') for s in range(3)}
        names[mode,'automatic'] = {}
        for s in range(3):
            stem = f'rgbd_sam210_s{s}' if mode=='rgbd' else f'{mode}_sam_v2_s{s}'
            name = next((n for n in [stem+'_recovery_glaurung',stem] if n in source),None)
            if name: names[mode,'automatic'][s] = name
    table=[]; summary={}; paired=[]
    for (mode,labels),cohort in names.items():
        present={s:n for s,n in cohort.items() if n in source}
        assert present
        measures=[metrics(n) for n in present.values()]
        summary[mode+'_'+labels] = dict(seeds=list(present), models=list(present.values()),
            metrics={k:stats([m[k] for m in measures]) for k in measures[0]})
        for s,name in present.items(): table.append(dict(modality=mode,labels=labels,seed=s,model=name,**metrics(name)))
    for family in ('unet','yolo'):
        cohort=[f'unet_logged_reference_s{s}' for s in range(3)] if family=='unet' and 'unet_logged_reference_s0' in source else [f'{family}_full_reference_s{s}' for s in range(3)]
        assert all(n in source for n in cohort)
        measures=[metrics(n) for n in cohort]
        summary[family+'_reference']=dict(seeds=[0,1,2],models=cohort,
            metrics={k:stats([m[k] for m in measures]) for k in measures[0]})
        for s,name in enumerate(cohort):table.append(dict(modality=family,labels='reference',seed=s,model=name,**metrics(name)))
    for name in ('sam_three_prompt_v2','floor_plane'):
        summary[name]=dict(seeds=[],models=[name],metrics={k:stats([v]) for k,v in metrics(name).items()})
        table.append(dict(modality=name,labels='direct',seed='',model=name,**metrics(name)))
    for mode in ('rgb','depth','rgbd'):
        for s in sorted(names[mode,'automatic']):
            ref,auto=names[mode,'reference'][s],names[mode,'automatic'][s]
            if ref not in source: continue
            before,after=metrics(ref),metrics(auto)
            for k in before: paired.append(dict(modality=mode,seed=s,metric=k,
                reference=before[k],automatic=after[k],change=after[k]-before[k]))
    write_csv(OUT/'cohort_scores.csv',table);write_csv(OUT/'paired_supervision.csv',paired)
    # Keep the paired modality difference for each seed, not just best-run scores.
    modality=[]
    for s in range(3):
        rgb=metrics(names['rgb','reference'][s]);rgbd=metrics(names['rgbd','reference'][s])
        for k in rgb: modality.append(dict(seed=s,metric=k,rgb=rgb[k],rgbd=rgbd[k],change=rgbd[k]-rgb[k]))
    write_csv(OUT/'paired_modalities.csv',modality)
    conditions=[]
    for s in range(3):
        p=ROOT/'results/triton_20261121/production/depth_ablations'/f'rgbd_reference210_s{s}'/'manifest.json'
        d=json.loads(p.read_text());assert d['scientific_results_eligible'] and d['rgb_identical_all_variants']
        for variant,m in d['variants'].items(): conditions.append(dict(seed=s,variant=variant,**m))
    write_csv(OUT/'depth_interventions.csv',conditions)
    # All six RGB-D histories must have reached all 500 epochs.
    histories=[]; fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for labels,color in [('reference','#222222'),('sam','#0072b2')]:
        values=[]
        for seed in range(3):
            run=ROOT/'results/triton_20261121/production/runs'/f'rgbd_{labels}210_s{seed}'
            assert (run/'VERIFIED').exists()
            h=rows(next(run.glob('history/**/logs.csv')))
            assert len(h)==500 and [int(r['epoch']) for r in h]==list(range(500))
            best=max(h,key=lambda r:float(r['IoU_valid_class_1']))
            histories.append(dict(model=run.name,epochs=len(h),selected_epoch_zero_based=int(best['epoch']),
                                  best_validation_obstacle_iou=float(best['IoU_valid_class_1'])))
            v=np.array([[float(r[k]) for k in ('loss_train_total','loss_valid','IoU_valid_class_1')] for r in h])
            values.append(v)
        arr=np.array(values)
        for i,ax in enumerate(axes):
            for v in arr: ax.plot(np.arange(1,501),v[:,i],color=color,alpha=.18,lw=.5)
            ax.plot(np.arange(1,501),arr[:,:,i].mean(0),color=color,label=labels,lw=1.2)
    for ax,title in zip(axes,['Training objective','Validation loss','Validation obstacle IoU']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(axis='y',color='.9');ax.spines[['top','right']].set_visible(False)
    axes[2].set_ylim(0,1);axes[0].legend(frameon=False,fontsize=8)
    fig.text(.5,-.03,'Six RGB-D runs, 500 epochs each. Bold: mean; faint: seeds. Training and validation objectives have different loss components.',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'training_curves.pdf',bbox_inches='tight');plt.close(fig)
    write_csv(OUT/'training_completion.csv',histories)
    # Match each reference unit across all three seeds and five static repeats.
    byunit=defaultdict(dict)
    selected={names[m,'reference'][s]:(m,s) for m in ('rgb','depth','rgbd') for s in range(3)}
    for r in objects:
        if r['model'] in selected:
            mode,seed=selected[r['model']]
            key=(r['recording'],int(r['id']))
            group=byunit[key].setdefault(mode,[])
            group.append(dict(seed=seed,frame=int(r['frame']),localized=r['localized50']=='True',coverage=float(r['coverage']),
                              adjusted_iou=float(r['adjusted_iou']),area=int(r['area_pixels']),parent=r['parent_item_id'],unit=r['unit']))
    units=[]
    for (recording,ident),modes in sorted(byunit.items()):
        assert set(modes)=={'rgb','depth','rgbd'} and all(len(v)==15 for v in modes.values())
        example=modes['rgb'][0]
        r=dict(recording=recording,id=ident,parent_item_id=example['parent'],unit=example['unit'],
               mean_area_pixels=float(np.mean([v['area'] for v in modes['rgb']])))
        for mode,v in modes.items():
            r[mode+'_localization_rate']=float(np.mean([x['localized'] for x in v]))
            r[mode+'_coverage_rate']=float(np.mean([x['coverage']>=.5 for x in v]))
            r[mode+'_mean_adjusted_iou']=float(np.mean([x['adjusted_iou'] for x in v]))
        r['rgbd_minus_rgb']=r['rgbd_localization_rate']-r['rgb_localization_rate'];units.append(r)
    write_csv(OUT/'matched_units.csv',units)
    sensitivity=[]
    for name in sorted({r['model'] for r in table}):
        m=metrics(name)
        sensitivity.append(dict(model=name,reference_observations=len(object_rows[name]),coverage_recall50=m['coverage_recall50'],
            **{f'localization_recall{t}':source[name]['pooled']['operating_points'][str(t)]['recall'] for t in (25,50,75)}))
    write_csv(OUT/'coverage_and_thresholds.csv',sensitivity)
    # Existing object IDs include explicit groups; do not relabel them physical individuals.
    summary['limits'] = dict(gte='Full-frame diagnostic: recording props included; 64 recordings in one room, five repeat frames each.',
        isod='Only 30 reviewed test frames, 10/site, of the 300-frame test split; provisional evidence.',
        instances='GTE IDs include individuals and groups; not native instance AP.',
        automatic='RGB-only generation, but prompt development/validation used reviewed references.',
        new_corrections='Semantic runs use frozen 70 reviewed + 1330 assisted masks; the new box detector uses the updated 85 reviewed + 1315 assisted instance-derived reference source.')
    summary['source_analysis_completion']=completion
    summary['source_summary_sha256']=hashlib.sha256((source_dir/'summary.json').read_bytes()).hexdigest()
    summary['matched_reference_unit_recording_pairs']=len(units)
    (OUT/'evidence.json').write_text(json.dumps(summary,indent=2)+'\n')
    teacher_audit()
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_analysis_completion',)},indent=2))


if __name__=='__main__': main()
