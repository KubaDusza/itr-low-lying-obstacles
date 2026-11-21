"""Audit and score cached predictions; full-frame diagnostic domain only.

No inference or training. Newly proposed fastener separation is identified as
provisional. Legacy pixel scores retain every accepted reference pixel; the
component domain neutralizes explicitly uncertain reference fragments only.
Recording props remain in this full-frame diagnostic and can cause alerts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hazard_metrics import evaluate, aggregate, ratio


def raster(path):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
            raise ValueError(f'Non-grayscale raster: {path}')
        a = a[..., 0]
    return a


def sources(root):
    runs = [(p.parent.name, p) for p in sorted((root/'results/frozen_v1').glob('*/runs/*/gte'))]
    runs += [(p.parent.name, p) for p in sorted((root/'results/sam_v2_20261121').glob('*/runs/*/gte'))
             if (p.parent/'VERIFIED').exists()]
    production=root/'results/triton_20261121/production'
    runs += [(p.parent.name,p) for p in sorted((production/'runs').glob('*/gte'))
             if (p.parent/'VERIFIED').exists() and p.parent.name!='real_data_gate']
    for run in sorted((production/'depth_ablations').glob('rgbd_reference210_s*')):
        if (run/'VERIFIED').exists():
            runs += [(run.name+'__'+variant,run/variant/'gte') for variant in
                     ('normal','missing_raw_depth','permuted_depth')]
    runs += [('floor_plane', root/'results/frozen_v1/floor_plane_v1/GTE-Floor'),
             ('sam_direct', root/'results/frozen_v1/sam_direct_v1/predictions/GTE-Floor'),
             ('sam_three_prompt_v2', root/'data/OWN_sam_v2')]
    return [(name, p) for name, p in runs if p.is_dir()]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    reference_base = a.root/'results/gte_instance_proposals_20261121'
    manifest = json.loads((reference_base/'manifest.json').read_text())
    records = manifest['frames']
    assert len(records) == 320 and len({(r['recording'], r['frame']) for r in records}) == 320
    legacy = {r['run']: r for r in csv.DictReader((a.root/'results/frozen_v1/scores.csv').open())}
    summary = dict(domain='full-frame diagnostic, including recording props',
                   component_reference_status=manifest['status'],
                   component_scores_provisional=True, crop_inference_completed=False,
                   metric='ID-aware SegmentMeIfYouCan adaptation; not instance AP',
                   reference_manifest_sha256=hashlib.sha256((reference_base/'manifest.json').read_bytes()).hexdigest(),
                   source_runs={}, missing_predictions_are_errors=True)
    frame_fields = ['model','recording','frame','camera_height_cm','background','object_set','nominal_distance_m',
                    'empty_reference','pixel_inter','pixel_union','pixel_pred','pixel_gt','pixel_iou',
                    'tp50','fn50','fp50','localization_recall50','component_f1_50','prediction_components']
    object_fields = ['model','recording','frame','id','parent_item_id','unit','area_pixels',
                     'covered_pixels','coverage','adjusted_iou','localized50']
    with (a.out/'frames.csv').open('w') as ff, (a.out/'objects.csv').open('w') as of:
        fw = csv.DictWriter(ff, fieldnames=frame_fields); fw.writeheader()
        ow = csv.DictWriter(of, fieldnames=object_fields); ow.writeheader()
        for name, predictions in sources(a.root):
            results = []; full_counts = dict(inter=0, union=0, pred=0, gt=0); ledger = []
            empty_frames = []; flagged_fragments = 0
            for row in records:
                run, frame = row['recording'], row['frame']
                original = a.root/'data/OWN_manual'/f'{run}_{frame}.png'
                if hashlib.sha256(original.read_bytes()).hexdigest() != row['source_binary_sha256']:
                    raise ValueError(f'Reference changed: {original}')
                gt = raster(original) > 0; ids = raster(reference_base/row['path']).copy()
                if not np.array_equal(ids > 0, gt): raise ValueError(f'Reference union mismatch: {run}/{frame}')
                path = predictions/run/f'{frame}.png'
                pred = raster(path) > 0  # Missing files fail, never become empty detections.
                if pred.shape != gt.shape: raise ValueError(f'Prediction shape mismatch: {path}')
                ignored = np.zeros(ids.shape, bool)
                attributes = {o['id']: o for o in row['objects']}
                for obj in row['objects']:
                    if obj['unit'] == 'uncertain_reference_fragment': ignored |= ids == obj['id']
                flagged_fragments += int(ignored.any()); ids[ignored] = 0
                result = evaluate(ids, pred, ignore=ignored)
                counts = dict(inter=int((gt & pred).sum()), union=int((gt | pred).sum()), pred=int(pred.sum()), gt=int(gt.sum()))
                for k, v in counts.items(): full_counts[k] += v
                parts = run.split('_'); nominal = None if parts[2] == 'empty' else int(parts[-1])/100
                op = result['operating_points']['50']
                fw.writerow(dict(model=name, recording=run, frame=frame, camera_height_cm=int(parts[0][1:]),
                    background=parts[1], object_set=parts[2], nominal_distance_m=nominal,
                    empty_reference=not gt.any(), **{f'pixel_{k}':v for k,v in counts.items()},
                    pixel_iou=ratio(counts['inter'], counts['union']),tp50=op['tp'],fn50=op['fn'],fp50=op['fp'],
                    localization_recall50=op['recall'],component_f1_50=op['f1'],
                    prediction_components=len(result['prediction_components'])))
                for obj in result['reference_objects']:
                    attr = attributes[obj['id']]
                    ow.writerow(dict(model=name,recording=run,frame=frame,id=obj['id'],parent_item_id=attr['parent_item_id'],
                        unit=attr['unit'],**{k:obj[k] for k in ['area_pixels','covered_pixels','coverage','adjusted_iou']},
                        localized50=obj['adjusted_iou']>=.5))
                if not gt.any(): empty_frames.append(dict(recording=run,frame=frame,
                    false_components=len(result['prediction_components']),false_pixels=counts['pred']))
                ledger.append(dict(recording=run,frame=frame,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                result.pop('reference_objects'); result.pop('prediction_components'); results.append(result)
            pooled = aggregate(results)
            full = dict(**full_counts,iou=ratio(full_counts['inter'],full_counts['union']),
                        precision=ratio(full_counts['inter'],full_counts['pred']),recall=ratio(full_counts['inter'],full_counts['gt']))
            compatibility = None
            if name in legacy:
                compatibility = abs(full['iou']-float(legacy[name]['gte_iou'])) < 1e-12
                if not compatibility: raise ValueError(f'Legacy pixel score changed for {name}')
            summary['source_runs'][name] = dict(predictions=str(predictions.relative_to(a.root)),
                frames=len(results), pooled=pooled,legacy_full_frame_pixel=full,
                legacy_pixel_iou_matches_saved=compatibility,uncertain_fragment_frames=flagged_fragments,
                empty_frames=empty_frames,prediction_ledger=ledger)
            print(json.dumps(dict(model=name,frames=len(results),pixel_iou=full['iou'],localization=pooled['operating_points']['50'])),flush=True)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__': main()
