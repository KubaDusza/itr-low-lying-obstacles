"""Collect every number used in the revised results section into one JSON (three-seed mean/SD)."""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; REV=ROOT/'results/results_report_20261122/revision'
def ms(v): v=[x for x in v if x is not None]; return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else None,values=[float(x) for x in v])
summary=json.loads((REV/'summary.json').read_text())['source_runs']; comp=json.loads((REV/'component_comparison.json').read_text())['runs']
objs=defaultdict(list)
for r in csv.DictReader((REV/'objects.csv').open()): objs[r['model']].append(float(r['coverage'])>=.5)
GTE={'ESANet RGB / ref':[f'esanet_full_rgb_reference_s{s}' for s in range(3)],'ESANet D / ref':[f'esanet_full_depth_reference_s{s}' for s in range(3)],
 'ESANet RGB-D / ref':[f'rgbd_reference210_s{s}' for s in range(3)],'ESANet RGB / SAM':['rgb_sam_v2_s0_recovery_glaurung','rgb_sam_v2_s1','rgb_sam_v2_s2'],
 'ESANet D / SAM':[f'depth_sam_v2_s{s}' for s in range(3)],'ESANet RGB-D / SAM':[f'rgbd_sam210_s{s}' for s in range(3)],
 'U-Net / ref':[f'unet_logged_reference_s{s}' for s in range(3)],'YOLOv8n-seg / ref':[f'yoloseg_e60_reference_s{s}' for s in range(3)],
 'U-Net / SAM':[f'unet_logged_sam_s{s}' for s in range(3)],'YOLOv8n-seg / SAM':[f'yoloseg_e60_sam_s{s}' for s in range(3)],
 'YOLOv8n boxes / ref':[f'yolov8n_detect_reference_e60_s{s}' for s in range(3)],'YOLOv8n boxes / SAM':[f'yolov8n_detect_automatic_e60_s{s}' for s in range(3)],
 'SAM 3 direct':['sam_three_prompt_v2'],'Floor plane':['floor_plane'],
 'RGB-D / ref, depth removed':[f'rgbd_reference210_s{s}__missing_raw_depth' for s in range(3)],
 'RGB-D / ref, depth from other recording':[f'rgbd_reference210_s{s}__permuted_depth' for s in range(3)],
 'RGB-D / ref, normal (same job)':[f'rgbd_reference210_s{s}__normal' for s in range(3)]}
gte={}
for arm,models in GTE.items():
    row=defaultdict(list)
    for m in models:
        if m in comp:
            c=comp[m]; row['object_recall'].append(c['object_recall']); row['component_f1'].append(c['component_mean_f1']); row['false_regions_per_frame'].append(c['false_components_per_frame'])
        else:
            p=summary[m]['pooled']; row['object_recall'].append(float(np.mean(objs[m]))); row['component_f1'].append(p['mean_f1']); row['false_regions_per_frame'].append(p['operating_points']['50']['fp']/320)
        if m in summary: row['pixel_iou'].append(summary[m]['pooled']['pixel']['iou'])
    gte[arm]={k:ms(v) for k,v in row.items()}
iso=json.loads((ROOT/'results/test_references_20261122/isod_test300_scores.json').read_text())
ISOD={'ESANet RGB / ref':'esanet_rgb_reference','ESANet D / ref':'esanet_depth_reference','ESANet RGB-D / ref':'esanet_rgbd_reference','ESANet RGB / SAM':'esanet_rgb_sam',
 'ESANet D / SAM':'esanet_depth_sam','ESANet RGB-D / SAM':'esanet_rgbd_sam','U-Net / ref':'unet_reference','YOLOv8n-seg / ref':'yoloseg_reference','U-Net / SAM':'unet_sam','YOLOv8n-seg / SAM':'yoloseg_sam',
 'YOLOv8n boxes / ref':'boxes_reference','YOLOv8n boxes / SAM':'boxes_sam','SAM 3 direct':'sam_direct','Floor plane':'floor_plane'}
isod={}
for arm,p in ISOD.items():
    keys=[p] if p in iso else [f'{p}_s{s}' for s in range(3)]
    isod[arm]=dict(pixel_iou=ms([iso[k]['pooled']['pixel']['iou'] for k in keys]),object_recall=ms([iso[k]['object_recall'] for k in keys]),
                   component_f1=ms([iso[k]['pooled']['mean_f1'] for k in keys]),false_regions_per_frame=ms([iso[k]['false_regions_per_frame'] for k in keys]),
                   reviewed30_pixel_iou=ms([iso[k]['reviewed30']['iou'] for k in keys]))
out=dict(gte=gte,isod300=isod,bootstrap=json.loads((REV/'bootstrap_contrasts.json').read_text())['contrasts'],
         isod_reference_quality=json.loads((ROOT/'data/ISOD_reference_test_v2/manifest.json').read_text())['quality'] | {'per_frame':'see manifest'})
(ROOT/'results/results_report_20261122/report_numbers.json').write_text(json.dumps(out,indent=1)+'\n')
f=lambda d:f"{d['mean']:.3f}"+(f"±{d['sd']:.3f}" if d['sd'] else '')
print(f"{'arm':40s} | GTE IoU  objR  F1  FR/f | ISOD IoU objR F1 FR/f")
for arm in GTE:
    g=gte[arm]; i=isod.get(arm)
    print(f"{arm:40s} | {f(g['pixel_iou']) if 'pixel_iou' in g else '-':12s} {f(g['object_recall'])} {f(g['component_f1'])} {f(g['false_regions_per_frame'])} | "+(f"{f(i['pixel_iou'])} {f(i['object_recall'])} {f(i['component_f1'])} {f(i['false_regions_per_frame'])}" if i else ''))
