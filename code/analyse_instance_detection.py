"""Fixed-threshold native-box evaluation and plots; never optimize on GTE."""
import argparse
import csv
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import tarfile

import numpy as np


def iou_matrix(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(2);aa=(a[:,2:]-a[:,:2]).prod(1);bb=(b[:,2:]-b[:,:2]).prod(1)
    return np.divide(inter,aa[:,None]+bb[None,:]-inter,out=np.zeros_like(inter),where=(aa[:,None]+bb[None,:]-inter)>0)


def match(gt,boxes,scores,conf=.25,overlap=.5):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4);scores=np.asarray(scores,dtype=float)
    selected=np.flatnonzero(scores>=conf);selected=selected[np.argsort(-scores[selected],kind='stable')]
    matrix=iou_matrix(boxes[selected],gt);used=set();matches=[];false=[]
    for j,pred in enumerate(selected):
        eligible=[i for i in range(len(gt)) if i not in used and matrix[j,i]>=overlap]
        if eligible:
            ident=max(eligible,key=lambda i:matrix[j,i]);used.add(ident);matches.append(dict(prediction=int(pred),reference=ident,iou=float(matrix[j,ident])))
        else:false.append(int(pred))
    return dict(tp=len(matches),fp=len(false),fn=len(gt)-len(used),matched=matches,false_predictions=false,missed_references=[i for i in range(len(gt)) if i not in used])


def write_csv(path,rows):
    assert rows
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--reference-manifest',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.reference_manifest:manifest=json.loads(a.reference_manifest.read_text())
    elif (a.base/'input/reference_manifest.json').exists():manifest=json.loads((a.base/'input/reference_manifest.json').read_text())
    else:
        with tarfile.open(a.base/'reference_labels.tar.gz') as tar:manifest=json.load(tar.extractfile('reference_manifest.json'))
    truth={Path(r['image']).parent.parent.name+'_'+str(r['frame']):r for r in manifest['gte']}
    assert len(truth)==320 and sum(len(r['boxes']) for r in truth.values())==1310
    summary={};frames=[];objects=[];completed=[]
    for source in ('reference','automatic'):
        for seed in range(3):
            run=a.base/'runs'/f'yolov8n_detect_{source}_s{seed}'
            if not (run/'VERIFIED').exists():
                assert a.allow_partial,f'Missing verified run: {run.name}'
                continue
            settings=json.loads((run/'settings.json').read_text());assert settings['task']=='detect' and settings['epochs']>=1 and not settings['gate']
            predictions=json.loads((run/'gte_boxes.json').read_text())['frames'];assert len(predictions)==320 and {r['image'] for r in predictions}==set(truth)
            total=defaultdict(int);empty=[]
            for pred in predictions:
                ref=truth[pred['image']];gt=[b['xyxy'] for b in ref['boxes']];r=match(gt,pred['boxes_xyxy'],pred['scores'])
                recording=ref['recording'];parts=recording.split('_');distance=None if parts[2]=='empty' else int(parts[3])/100
                row=dict(model=run.name,source=source,seed=seed,recording=recording,frame=ref['frame'],object_set=parts[2],
                    height_cm=int(parts[0][1:]),background=parts[1],nominal_floor_distance_m=distance,reference_units=len(gt),
                    tp=r['tp'],fp=r['fp'],fn=r['fn'],empty_reference=not gt)
                frames.append(row)
                for key in ('tp','fp','fn'):total[key]+=r[key]
                if not gt:empty.append(r['fp'])
                hits={m['reference']:m for m in r['matched']}
                for ident,box in enumerate(ref['boxes']):
                    objects.append(dict(model=run.name,source=source,seed=seed,recording=recording,frame=ref['frame'],id=box['id'],
                        mask_area_pixels=box['pixels'],box_detected50=ident in hits,matched_iou=hits[ident]['iou'] if ident in hits else None))
            assert total['tp']+total['fn']==1310 and len(empty)==20
            m=json.loads((run/'gte_metrics.json').read_text())['metrics']
            summary[run.name]=dict(source=source,seed=seed,frames=320,reference_unit_views=1310,
                box_AP50=m['metrics/mAP50(B)'],box_AP50_95=m['metrics/mAP50-95(B)'],
                fixed_confidence=.25,matching_iou=.5,precision=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None,
                recall=total['tp']/1310,false_boxes_per_frame=total['fp']/320,**total,
                empty_frames_with_detection=sum(v>0 for v in empty),empty_frames=20,empty_false_boxes_per_frame=float(np.mean(empty)))
            completed.append(run.name)
    write_csv(a.out/'frame_scores.csv',frames);write_csv(a.out/'object_scores.csv',objects)
    with (a.out/'summary.json').open('w') as f:json.dump(dict(complete=len(completed)==6,runs=summary,
        domain='Full-frame diagnostic; recording props not exhaustively annotated; some reference units are groups.',
        ap='Ultralytics box AP over annotated hazard units. P/R recomputed at fixed confidence 0.25; not the test-optimized operating point returned by val().',
        matching='Confidence-ordered one-to-one matching to reference boxes at IoU >=0.5; duplicate boxes count as FP.',
        references='Assisted instance boxes for training/validation; GTE manually corrected item/group masks converted to boxes.'),f,indent=2)
    if len(completed)==6:
        artifacts=[]
        for name in completed:
            for rel in ['gte_boxes.json','gte_metrics.json','results.csv','settings.json','args.yaml','weights/best.pt','weights/last.pt']:
                path=a.base/'runs'/name/rel
                if path.exists():artifacts.append(dict(path=str(path.relative_to(a.base)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (a.out/'artifact_registry.json').write_text(json.dumps(dict(complete=True,base=str(a.base),
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifacts=artifacts),indent=2)+'\n')
    print(json.dumps(dict(completed_runs=completed,complete=len(completed)==6)))


if __name__=='__main__':main()
