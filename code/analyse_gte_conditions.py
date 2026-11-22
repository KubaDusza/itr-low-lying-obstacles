"""Descriptive condition/appearance analysis of existing GTE predictions.

No model computation. Nearby RGB floor colors are appearance proxies from
uncalibrated camera RGB, not physical reflectance or a visibility threshold.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/results_report_20261122'
ANALYSIS=ROOT/'results/triton_20261121/production/analysis_final'


def read(path):return list(csv.DictReader(path.open()))


def write(path,rows):
    assert rows
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    cv2.setNumThreads(1)
    frames=read(ANALYSIS/'gte_full_diagnostic/frames.csv')
    observations=read(ROOT/'results/evaluation_v2_20261121/gte_reference_observations.csv')
    depth=read(ROOT/'results/frozen_v1/gte_depth_visibility.csv')
    depthmap={(r['recording'],r['frame'],r['item']):r for r in depth}
    inventory=read(ROOT/'data_collection/items.csv');byset=defaultdict(list)
    for r in inventory:byset[r['object_set']].append(r)
    condition=[]
    for mode in ('rgb','depth','rgbd'):
        for seed in range(3):
            name=f'rgbd_reference210_s{seed}' if mode=='rgbd' else f'esanet_full_{mode}_reference_s{seed}'
            selected=[r for r in frames if r['model']==name and r['empty_reference']=='False']
            for field in ('background','object_set','nominal_distance_m','camera_height_cm'):
                for value in sorted({r[field] for r in selected}):
                    g=[r for r in selected if r[field]==value];recordings=defaultdict(list)
                    for r in g:recordings[r['recording']].append(r)
                    assert all(len(v)==5 for v in recordings.values())
                    inter=sum(int(r['pixel_inter']) for r in g);union=sum(int(r['pixel_union']) for r in g)
                    tp=sum(int(r['tp50']) for r in g);fn=sum(int(r['fn50']) for r in g)
                    condition.append(dict(modality=mode,seed=seed,condition=field,value=value,recordings=len(recordings),frames=len(g),
                        pixel_iou=inter/union,micro_localization_recall=tp/(tp+fn),
                        macro_recording_localization_recall=float(np.mean([np.mean([float(r['localization_recall50']) for r in rr]) for rr in recordings.values()])),
                        false_components_per_frame=sum(int(r['fp50']) for r in g)/len(g)))
    write(OUT/'conditions.csv',condition)
    # One shared neighborhood rule, fixed in the November 21 analysis plan.
    cues=[];grouped=defaultdict(list)
    for r in observations:grouped[r['recording'],r['frame']].append(r)
    for (recording,frame),items in grouped.items():
        rgb=cv2.imread(str(ROOT/'data/OWN'/recording/'rgb'/f'{frame}.png'))
        assert rgb is not None
        lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        ids=cv2.imread(str(ROOT/'data/OWN_manual'/f'{recording}_{frame}_id.png'),0)
        foreground=cv2.imread(str(ROOT/'data/OWN_manual'/f'{recording}_{frame}.png'),0)>0
        floor=cv2.imread(str(ROOT/'data/OWN'/recording/'floor.png'),0)>0
        assert ids.shape==foreground.shape==floor.shape==rgb.shape[:2]
        all_distance=cv2.distanceTransform((~foreground).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        object_set=recording.split('_')[2]
        for r in items:
            ident=next(i for i,item in enumerate(byset[object_set],1) if item['item_id']==r['item_id'])
            obj=(ids==ident)&foreground
            assert int(obj.sum())==int(r['area_px'])
            distance=cv2.distanceTransform((~obj).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            ring=floor&(all_distance>=4)&(distance<=20)
            enough=obj.sum()>=10 and ring.sum()>=40
            difference=float(np.linalg.norm(np.median(lab[obj],axis=0)-np.median(lab[ring],axis=0))) if enough else None
            dep=depthmap[recording,frame,r['item_label']]
            cues.append(dict(recording=recording,frame=int(frame),item_id=r['item_id'],item=r['item_label'],
                area_pixels=int(obj.sum()),local_floor_pixels=int(ring.sum()),camera_RGB_deltaE76=difference,
                depth_valid_fraction=float(r['valid_depth_fraction']),median_mask_depth_mm=r['median_mask_depth_mm'],
                residual_difference_mm=dep['residual_difference_mm'],residual_auc=dep['residual_auc'],
                residual_interpretation='camera-axis floor-fit residual, not physical height'))
    write(OUT/'appearance_depth_cues.csv',cues)
    # Compare the same inventory item/mount/distance across backgrounds.
    units=read(OUT/'matched_units.csv');pairs=defaultdict(dict)
    for r in units:
        height,bg,objset,distance=r['recording'].split('_')
        if bg not in ('floor','poster'):continue
        if objset not in ('cables','chess','rags'):continue
        key=(height,objset,distance,r['parent_item_id'])
        assert bg not in pairs[key], 'Ambiguous multiple units for one inventory item'
        pairs[key][bg]=r
    paired=[]
    for (height,objset,distance,item),backgrounds in sorted(pairs.items()):
        if set(backgrounds)!={'floor','poster'}:continue
        g=dict(camera=height,object_set=objset,nominal_distance_m=int(distance)/100,item_id=item)
        for bg,r in backgrounds.items():
            for mode in ('rgb','depth','rgbd'):g[f'{bg}_{mode}_localization_rate']=float(r[mode+'_localization_rate'])
            c=[v for v in cues if v['recording']==r['recording'] and v['item_id']==item]
            assert len(c)==5
            for metric in ('camera_RGB_deltaE76','depth_valid_fraction','residual_difference_mm','residual_auc'):
                values=[float(v[metric]) for v in c if v[metric] is not None and v[metric]!='']
                g[f'{bg}_{metric}']=float(np.median(values)) if values else None
        paired.append(g)
    write(OUT/'paired_background_items.csv',paired)
    (OUT/'conditions_notes.json').write_text(json.dumps(dict(
        scene_domain='Full-frame diagnostic, recording props included; not cropped-input results.',
        appearance='Euclidean difference between median object/floor CIELAB vectors derived from uncalibrated camera RGB. Not a physical contrast measurement or detection threshold.',
        neighborhood='Reference floor, at least 4 px from any annotated foreground and within 20 px of the target; >=10 object pixels and >=40 floor pixels.',
        depth='Previously verified valid-depth and floor-residual measurements; camera-axis residual is not height.',
        confounds='One room; object type/size/material/distance/viewpoint and height/tilt can covary. Nearby floor may contain unannotated setup details.',
        item_observations=len(cues),matched_background_item_configuration_pairs=len(paired),
        static_repeats=5,model_seeds=3),indent=2)+'\n')
    print(f'{len(cues)} cue observations; {len(paired)} matched item/configuration/background pairs')


if __name__=='__main__':main()
