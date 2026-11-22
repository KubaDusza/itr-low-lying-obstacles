"""Paired recording bootstrap for the report's GTE-Floor contrasts.

Resamples the 64 recordings with replacement (10,000 draws). In each draw every arm pools its
pixel/object counts over the drawn recordings and its three seeds, so the interval reflects
scene sampling with training variation averaged; seed SD is reported separately.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'results/results_report_20261122/revision'
B = 10_000

ARMS = {
    'rgb_ref': [f'esanet_full_rgb_reference_s{s}' for s in range(3)],
    'd_ref': [f'esanet_full_depth_reference_s{s}' for s in range(3)],
    'rgbd_ref': [f'rgbd_reference210_s{s}' for s in range(3)],
    'rgb_sam': ['rgb_sam_v2_s0_recovery_glaurung', 'rgb_sam_v2_s1', 'rgb_sam_v2_s2'],
    'd_sam': [f'depth_sam_v2_s{s}' for s in range(3)],
    'rgbd_sam': [f'rgbd_sam210_s{s}' for s in range(3)],
    'unet_ref': [f'unet_logged_reference_s{s}' for s in range(3)],
    'unet_sam': [f'unet_logged_sam_s{s}' for s in range(3)],
    'seg_ref': [f'yoloseg_e60_reference_s{s}' for s in range(3)],
    'seg_sam': [f'yoloseg_e60_sam_s{s}' for s in range(3)],
    'boxes_ref': [f'yolov8n_detect_reference_e60_s{s}' for s in range(3)],
    'boxes_sam': [f'yolov8n_detect_automatic_e60_s{s}' for s in range(3)],
}
CONTRASTS = [('rgbd_ref', 'rgb_ref'), ('d_ref', 'rgb_ref'), ('rgb_sam', 'rgb_ref'), ('rgbd_sam', 'rgbd_ref'),
             ('d_sam', 'd_ref'), ('unet_sam', 'unet_ref'), ('seg_sam', 'seg_ref'), ('boxes_sam', 'boxes_ref')]


def per_recording():
    """arm -> recording -> summed counts over seeds and frames."""
    rec = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))  # inter, union, objects, covered, fp50, frames
    model_arm = {m: a for a, ms in ARMS.items() for m in ms}
    for r in csv.DictReader((REV / 'frames.csv').open()):
        a = model_arm.get(r['model'])
        if a:
            rec[a][r['recording']] += [int(r['pixel_inter']), int(r['pixel_union']), 0, 0, int(r['fp50']), 1]
    for r in csv.DictReader((REV / 'objects.csv').open()):
        a = model_arm.get(r['model'])
        if a:
            rec[a][r['recording']] += [0, 0, 1, float(r['coverage']) >= .5, 0, 0]
    for r in csv.DictReader((REV / 'native_component_frames.csv').open()):
        a = model_arm.get(r['model'])
        if a:
            rec[a][r['recording']] += [0, 0, int(r['references']), int(r['covered']), int(r['false_components']), 1]
    return rec


def metrics(c):
    with np.errstate(invalid='ignore', divide='ignore'):
        return dict(pixel_iou=c[..., 0] / c[..., 1], object_recall=c[..., 3] / c[..., 2],
                    false_regions_per_frame=c[..., 4] / c[..., 5])


def main():
    rec = per_recording()
    recordings = sorted(rec['rgb_ref'])
    assert len(recordings) == 64
    arr = {a: np.stack([rec[a][r] for r in recordings]) for a in ARMS}
    rng = np.random.default_rng(20261122)
    draws = rng.integers(0, len(recordings), size=(B, len(recordings)))
    counts = np.stack([np.bincount(d, minlength=len(recordings)) for d in draws])  # B x R weights
    out = dict(method=__doc__.strip(), draws=B, recordings=len(recordings), arms={}, contrasts={})
    boot = {}
    for a, x in arr.items():
        point = metrics(x.sum(0))
        boot[a] = metrics(counts @ x)
        out['arms'][a] = {k: float(v) for k, v in point.items()}
    for a, b in CONTRASTS:
        res = {}
        for k in ('pixel_iou', 'object_recall', 'false_regions_per_frame'):
            if a.startswith('boxes') and k == 'pixel_iou':
                continue
            d = boot[a][k] - boot[b][k]
            res[k] = dict(delta=out['arms'][a][k] - out['arms'][b][k], ci95=[float(np.nanpercentile(d, 2.5)),
                                                                          float(np.nanpercentile(d, 97.5))])
        out['contrasts'][f'{a} - {b}'] = res
    path = ROOT / 'results/results_report_20261122/revision/bootstrap_contrasts.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    for name, res in out['contrasts'].items():
        print(name, {k: f"{v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in res.items()})


if __name__ == '__main__':
    main()
