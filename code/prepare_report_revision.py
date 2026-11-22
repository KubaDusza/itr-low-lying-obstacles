"""Current report tables and common box comparison from saved predictions only."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from hazard_metrics import evaluate,aggregate
from prepare_instance_detection import boxes_from_ids

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/triton_20261121/production/analysis_final/gte_full_diagnostic'
OUT=ROOT/'results/results_report_20261122/revision'
NAME='rgb_sam_v2_s0_recovery_glaurung'
RUN=ROOT/'results/sam_v2_20261121/glaurung/runs'/NAME


def readcsv(p):return list(csv.DictReader(p.open()))
def csvwrite(p,rr):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def raster(p):
    a=np.array(Image.open(p));return a[...,0] if a.ndim==3 else a

def stats(v):return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else None,values=[float(x) for x in v])
def px(rr):
    c={k:sum(int(r[k]) for r in rr) for k in ['inter','union','pred','gt']}
    p=c['inter']/c['pred'] if c['pred'] else 0;r=c['inter']/c['gt'] if c['gt'] else 0
    return dict(iou=c['inter']/c['union'],precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0,
        fp_pixels_per_frame=(c['pred']-c['inter'])/len(rr),fn_pixels_per_frame=(c['gt']-c['inter'])/len(rr))

def fixed_match(gt,pred):
    """Maximum-cardinality one-to-one box matching at IoU .5; no score ordering."""
    from analyse_instance_detection import iou_matrix
    matrix=iou_matrix(pred,gt)
    if not matrix.size:return dict(tp=0,fp=len(pred),fn=len(gt))
    eligible=matrix>=.5;gain=eligible*(min(matrix.shape)+1+matrix)
    ii,jj=linear_sum_assignment(-gain);tp=int(eligible[ii,jj].sum())
    return dict(tp=tp,fp=len(pred)-tp,fn=len(gt)-tp)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert (RUN/'VERIFIED').exists()
    history=readcsv(next(RUN.glob('**/logs.csv')));assert len(history)==500 and [int(r['epoch']) for r in history]==list(range(500))
    provenance=json.loads((RUN/'provenance.json').read_text());other=json.loads((ROOT/'results/sam_v2_20261121/rivendell/runs/rgb_sam_v2_s1/provenance.json').read_text())
    for key in ['recipe_sha256','labels_manifest_sha256']:assert provenance[key]==other[key]
    summary=json.loads((OLD/'summary.json').read_text());sources=summary['source_runs']
    manifest=json.loads((ROOT/'results/gte_instance_proposals_20261121/manifest.json').read_text())
    frames=readcsv(OLD/'frames.csv');objects=readcsv(OLD/'objects.csv')
    extra_runs=[(NAME,RUN)]
    unet_names=[f'unet_logged_reference_s{s}' for s in range(3)]
    extended=ROOT/'results/extended_training_20261122'
    updated_unet=all((extended/'runs'/n/'VERIFIED').exists() for n in unet_names)
    if updated_unet:extra_runs += [(n,extended/'runs'/n) for n in unet_names]
    seg_names=[f'yoloseg_e60_reference_s{s}' for s in range(3)]
    updated_seg=all((extended/'runs'/n/'VERIFIED').exists() for n in seg_names)
    if updated_seg:extra_runs += [(n,extended/'runs'/n) for n in seg_names]
    for new_name,new_dir in extra_runs:
        newframes=[];newobjects=[];evaluations=[];full=defaultdict(int)
        for entry in manifest['frames']:
            rec=entry['recording'];frame=entry['frame'];gt=raster(ROOT/'data/OWN_manual'/f'{rec}_{frame}.png')>0
            ids=raster(ROOT/'results/gte_instance_proposals_20261121'/entry['path']).copy();ignore=np.zeros_like(gt)
            attrs={r['id']:r for r in entry['objects']}
            for attr in attrs.values():
                if attr['unit']=='uncertain_reference_fragment':ignore|=ids==attr['id']
            ids[ignore]=0;pred=raster(new_dir/'gte'/rec/f'{frame}.png')>0
            assert pred.shape==ids.shape
            counts=dict(inter=int((gt&pred).sum()),union=int((gt|pred).sum()),pred=int(pred.sum()),gt=int(gt.sum()))
            for k,v in counts.items():full[k]+=v
            result=evaluate(ids,pred,ignore=ignore);op=result['operating_points']['50'];parts=rec.split('_')
            newframes.append(dict(model=new_name,recording=rec,frame=frame,camera_height_cm=int(parts[0][1:]),background=parts[1],object_set=parts[2],
                nominal_distance_m='' if parts[2]=='empty' else int(parts[-1])/100,empty_reference=not gt.any(),**{f'pixel_{k}':v for k,v in counts.items()},
                pixel_iou=counts['inter']/counts['union'] if counts['union'] else '',tp50=op['tp'],fn50=op['fn'],fp50=op['fp'],localization_recall50=op['recall'],component_f1_50=op['f1'],prediction_components=len(result['prediction_components'])))
            for obj in result['reference_objects']:
                attr=attrs[obj['id']];newobjects.append(dict(model=new_name,recording=rec,frame=frame,id=obj['id'],parent_item_id=attr['parent_item_id'],unit=attr['unit'],
                    **{k:obj[k] for k in ['area_pixels','covered_pixels','coverage','adjusted_iou']},localized50=obj['adjusted_iou']>=.5))
            result.pop('reference_objects');result.pop('prediction_components');evaluations.append(result)
        sources[new_name]=dict(predictions=str((new_dir/'gte').relative_to(ROOT)),frames=320,pooled=aggregate(evaluations),legacy_full_frame_pixel=dict(full,
            iou=full['inter']/full['union'],precision=full['inter']/full['pred'],recall=full['inter']/full['gt']))
        frames+=newframes;objects+=newobjects
    csvwrite(OUT/'frames.csv',frames);csvwrite(OUT/'objects.csv',objects)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    cohorts={}
    for mode in ['rgb','depth','rgbd']:
        cohorts[mode,'reference']=[f'rgbd_reference210_s{s}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{s}' for s in range(3)]
        cohorts[mode,'automatic']=[f'rgbd_sam210_s{s}' if mode=='rgbd' else NAME if mode=='rgb' and s==0 else f'{mode}_sam_v2_s{s}' for s in range(3)]
    object_lookup=defaultdict(list);frame_lookup=defaultdict(list)
    for r in objects:object_lookup[r['model']].append(r)
    for r in frames:frame_lookup[r['model']].append(r)
    domain={};condition=[];categories=[]
    for mode in ['rgb','depth','rgbd']:
        values=defaultdict(list)
        for seed,model in enumerate(cohorts[mode,'reference']):
            run=(ROOT/sources[model]['predictions']).parent
            candidates=[run/'scores.json',run.parent.parent/'scores'/f'{run.name}.json'];score=next((p for p in candidates if p.exists()),None);assert score,model
            isod=json.loads(score.read_text())['ISOD'];assert len(isod['frames'])==30
            test=px(isod['frames']);gte=px([{k:int(r['pixel_'+k]) for k in ['inter','union','pred','gt']} for r in frame_lookup[model]])
            gte['object_recall']=np.mean([float(r['coverage'])>=.5 for r in object_lookup[model]])
            empty=[r for r in frame_lookup[model] if r['object_set']=='empty'];assert len(empty)==20
            gte['empty_false_pixels_per_frame']=np.mean([int(r['pixel_pred']) for r in empty])
            for dataset,result in [('ISOD',test),('GTE',gte)]:
                for key,v in result.items():values[dataset,key].append(v)
            for cat in ['rags','tapes','random','chess','screws','cables','hexkeys']:
                rr=[r for r in frame_lookup[model] if r['object_set']==cat];oo=[r for r in object_lookup[model] if r['recording'].split('_')[2]==cat]
                categories.append(dict(modality=mode,seed=seed,object_set=cat,recordings=len(rr)//5,pixel_iou=sum(int(r['pixel_inter']) for r in rr)/sum(int(r['pixel_union']) for r in rr),object_recall=float(np.mean([float(r['coverage'])>=.5 for r in oo]))))
        domain[mode]={dataset:{key:stats(values[dataset,key]) for d,key in values if d==dataset} for dataset in ['ISOD','GTE']}
    for (mode,source),models in cohorts.items():
        for seed,model in enumerate(models):
            rr=frame_lookup[model];oo=object_lookup[model];assert len(rr)==320 and len(oo)==1310
            counts=px([{k:int(r['pixel_'+k]) for k in ['inter','union','pred','gt']} for r in rr]);counts['object_recall']=float(np.mean([float(r['coverage'])>=.5 for r in oo]))
            condition.append(dict(modality=mode,source=source,seed=seed,model=model,**counts))
    csvwrite(OUT/'category_scores.csv',categories);csvwrite(OUT/'supervision_scores.csv',condition)
    (OUT/'domain_comparison.json').write_text(json.dumps(domain,indent=2)+'\n')
    # Published SegmentMeIfYouCan component metrics: union of all accepted native boxes
    # gives a common obstacle-region output; no predicted/GT object-count equality.
    component_runs={}
    groups={**{('ESANet '+{'rgb':'RGB','depth':'D','rgbd':'RGB-D'}[mode],source):models for (mode,source),models in cohorts.items()},
        ('U-Net RGB','reference'):unet_names if updated_unet else [f'unet_full_reference_s{s}' for s in range(3)],
        ('YOLOv8n-seg RGB','reference'):seg_names if updated_seg else [f'yolo_full_reference_s{s}' for s in range(3)],
        ('SAM 3 direct','direct'):['sam_three_prompt_v2'],('Floor plane D','direct'):['floor_plane']}
    for (family,source),models in groups.items():
        for seed,model in enumerate(models):
            op=sources[model]['pooled'];oo=object_lookup[model]
            component_runs[model]=dict(family=family,source=source,seed=seed if source!='direct' else None,output='obstacle mask',
                object_recall=float(np.mean([float(r['coverage'])>=.5 for r in oo])),component_recall=op['operating_points']['50']['recall'],
                false_components_per_frame=op['operating_points']['50']['fp']/320,component_mean_f1=op['mean_f1'])
    native_root=extended if all((extended/'runs'/f'yolov8n_detect_{source}_e60_s{s}'/'VERIFIED').exists() for source in ['reference','automatic'] for s in range(3)) else ROOT/'results/instance_detection_20261122'
    native_frames=[]
    reference_rows=manifest['frames']
    for source in ['reference','automatic']:
        for seed in range(3):
            model=f'yolov8n_detect_{source}_e60_s{seed}';rr=json.loads((native_root/'runs'/model/'gte_boxes.json').read_text())['frames'];lookup={r['image']:r for r in rr};results=[];coverage=[]
            for r in reference_rows:
                ids=raster(ROOT/'results/gte_instance_proposals_20261121'/r['path']).copy();ignore=np.zeros(ids.shape,bool)
                for attr in r['objects']:
                    if attr['unit']=='uncertain_reference_fragment':ignore|=ids==attr['id']
                ids[ignore]=0;pred=lookup[f'{r["recording"]}_{r["frame"]}'];binary=np.zeros(ids.shape,bool);h,w=binary.shape
                for box,conf in zip(pred['boxes_xyxy'],pred['scores']):
                    if conf<.25:continue
                    x0,y0,x1,y1=box;x0=max(0,int(np.floor(x0)));y0=max(0,int(np.floor(y0)));x1=min(w,int(np.ceil(x1)));y1=min(h,int(np.ceil(y1)));binary[y0:y1,x0:x1]=True
                result=evaluate(ids,binary,ignore=ignore)
                for obj in result['reference_objects']:coverage.append(obj['coverage']>=.5)
                native_frames.append(dict(model=model,source=source,seed=seed,recording=r['recording'],frame=r['frame'],
                    covered=sum(obj['coverage']>=.5 for obj in result['reference_objects']),references=len(result['reference_objects']),
                    component_recall=result['operating_points']['50']['recall'],false_components=result['operating_points']['50']['fp']))
                result.pop('reference_objects');result.pop('prediction_components');results.append(result)
            op=aggregate(results);assert len(coverage)==1310
            component_runs[model]=dict(family='YOLOv8n RGB',source=source,seed=seed,output='union of native boxes',object_recall=float(np.mean(coverage)),
                component_recall=op['operating_points']['50']['recall'],false_components_per_frame=op['operating_points']['50']['fp']/320,component_mean_f1=op['mean_f1'])
    csvwrite(OUT/'native_component_frames.csv',native_frames)
    (OUT/'component_comparison.json').write_text(json.dumps(dict(standard='SegmentMeIfYouCan adjusted component overlap/precision and mean F1 over thresholds 0.25 to 0.75',
        primary_recovery='Fraction of supplied reference hazard-unit masks with at least 50% of pixels covered by the union of predictions; companion used in partner report, not a new official benchmark metric',
        adaptation='Supplied reference IDs instead of binary-GT connected components; no small-target filtering; full-frame GTE domain. Native boxes rasterized by floor/ceil edges and union before component extraction.',
        behavior='All touching prediction components are combined for each reference; other reference objects are excluded from its excess-background penalty. Splits/merges are not one-to-one count mismatches.',
        limits='Box unions give coarser occupied regions than masks. Component precision measures spatial extent, not box AP. Groups and unannotated recording props remain.',runs=component_runs),indent=2)+'\n')
    (OUT/'COMPLETE').write_text('Recovered third RGB/SAM semantic seed; domain, category, supervision and common box comparisons verified\n')
    print('Revision analysis complete')

if __name__=='__main__':main()
