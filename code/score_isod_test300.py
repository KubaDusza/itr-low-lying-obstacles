"""Score every reported method on all 300 ISOD test frames (three held-out locations).

Reference: data/ISOD_reference_test_v2 (assisted, 30 human-reviewed). Reference units for
component metrics are 8-connected components of the binary reference (ISOD has no IDs).
Boxes are filled at confidence 0.25 and united, as on GTE-Floor.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

import hazard_metrics as hm
from floor_plane import floor_gap

ROOT = Path(__file__).resolve().parents[1]
SITES = ['HDCL_HW_A', 'MEB_CM', 'MEB_HW_B']
T = ROOT / 'results/test_references_20261122/pred'
SAM2 = ROOT / 'results/sam_v2_20261121'
PROD = ROOT / 'results/triton_20261121/production/runs'


def sources():
    s = {}
    for k in range(3):
        s[f'esanet_rgb_reference_s{k}'] = T / f'esanet_full_rgb_reference_s{k}'
        s[f'esanet_depth_reference_s{k}'] = T / f'esanet_full_depth_reference_s{k}'
        s[f'esanet_rgbd_reference_s{k}'] = PROD / f'rgbd_reference210_s{k}/pred'
        s[f'esanet_rgbd_sam_s{k}'] = PROD / f'rgbd_sam210_s{k}/pred'
        s[f'unet_reference_s{k}'] = T / f'unet_logged_reference_s{k}'
        s[f'yoloseg_reference_s{k}'] = T / f'yoloseg_long_reference_s{k}'
        s[f'boxes_reference_s{k}'] = T / f'yolov8n_detect_reference_s{k}/isod_boxes.json'
        s[f'boxes_sam_s{k}'] = T / f'yolov8n_detect_automatic_s{k}/isod_boxes.json'
    s['esanet_rgb_sam_s0'] = SAM2 / 'glaurung/runs/rgb_sam_v2_s0_recovery_glaurung/pred'
    s['esanet_rgb_sam_s1'] = SAM2 / 'rivendell/runs/rgb_sam_v2_s1/pred'
    s['esanet_rgb_sam_s2'] = SAM2 / 'rivendell/runs/rgb_sam_v2_s2/pred'
    s['esanet_depth_sam_s0'] = SAM2 / 'rivendell/runs/depth_sam_v2_s0/pred'
    s['esanet_depth_sam_s1'] = SAM2 / 'glaurung/runs/depth_sam_v2_s1/pred'
    s['esanet_depth_sam_s2'] = SAM2 / 'glaurung/runs/depth_sam_v2_s2/pred'
    s['sam_direct'] = ROOT / 'data/ISOD_sam_all_v2'
    s['floor_plane'] = 'floor'
    return s


def loader(source, shape_of):
    if source == 'floor':
        st = json.loads((ROOT / 'results/frozen_v1/floor_plane_v1/settings.json').read_text())

        def floor(site, i):
            raw = cv2.imread(str(ROOT / f'data/ISOD/data/{site}/depth/{i}.png'), cv2.IMREAD_UNCHANGED)
            gap, valid = floor_gap(raw, st['depth_range_m'], seed_from=0)
            return valid & (gap >= st['threshold_m'])
        return floor
    if str(source).endswith('.json'):
        boxes = {(r['site'], r['frame']): r for r in json.loads(Path(source).read_text())['frames']}

        def filled(site, i):
            r = boxes[site, i]
            m = np.zeros(shape_of(site, i), bool)
            for (x0, y0, x1, y1), c in zip(r['boxes'], r['scores']):
                if c >= .25:
                    m[int(np.floor(y0)):int(np.ceil(y1)), int(np.floor(x0)):int(np.ceil(x1))] = True
            return m
        return filled

    def mask(site, i):
        m = cv2.imread(str(Path(source) / site / f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        assert m is not None, (source, site, i)
        return m > 0
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=ROOT / 'data/ISOD_reference_test_v2')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/test_references_20261122/isod_test300_scores.json')
    a = ap.parse_args()
    manifest = json.loads((a.reference / 'manifest.json').read_text())
    assert len(manifest['frames']) == 300
    refs = {}
    for r in manifest['frames']:
        g = cv2.imread(str(a.reference / r['path']), cv2.IMREAD_GRAYSCALE) > 0
        refs[r['site'], r['frame']] = (g, ndimage.label(g, structure=np.ones((3, 3)))[0], r['review_status'])
    shape_of = lambda s, i: refs[s, i][0].shape
    out = {}
    for name, source in sources().items():
        load = loader(source, shape_of)
        per, per_site = [], {s: [] for s in SITES}
        reviewed = []
        objects = covered = 0
        empty_fp = []
        for (site, i), (g, ids, status) in refs.items():
            p = load(site, i)
            if p.shape != g.shape:
                p = cv2.resize(p.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            r = hm.evaluate(ids, p)
            r.pop('prediction_components')
            objs = r.pop('reference_objects')
            objects += len(objs)
            covered += sum(o['coverage'] >= .5 for o in objs)
            per.append(r)
            per_site[site].append(r)
            if status == 'human_reviewed':
                reviewed.append(r)
            if not g.any():
                empty_fp.append(int(p.sum()))
        agg = hm.aggregate(per)
        false_regions = agg['operating_points']['50']['fp'] / 300
        out[name] = dict(pooled=agg, object_recall=covered / objects if objects else None, objects=objects,
                         false_regions_per_frame=false_regions,
                         per_site={s: hm.aggregate(v)['pixel'] | {'mean_f1': hm.aggregate(v)['mean_f1']}
                                   for s, v in per_site.items()},
                         reviewed30=hm.aggregate(reviewed)['pixel'] | {'mean_f1': hm.aggregate(reviewed)['mean_f1']},
                         empty_frames=len(empty_fp))
        px = agg['pixel']
        print(f"{name:28s} IoU={px['iou']:.3f} P={px['precision']:.3f} R={px['recall']:.3f} "
              f"objR={out[name]['object_recall']:.3f} mF1={agg['mean_f1']:.3f} FR/f={false_regions:.2f} "
              f"rev30IoU={out[name]['reviewed30']['iou']:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
