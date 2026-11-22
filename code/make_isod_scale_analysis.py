"""ISOD component recovery by image area and estimated depth, using saved predictions."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from score_isod_test300 import sources, loader
from make_report_revision_assets import ROOT, F, COLORS, MODES, save

cv2.setNumThreads(1)
OUT = ROOT / 'results/results_report_20261122/isod_scale'
OUT.mkdir(exist_ok=True)
manifest = json.loads((ROOT / 'data/ISOD_reference_test_v2/manifest.json').read_text())
features, masks = [], {}
for frame in manifest['frames']:
    site, number = frame['site'], frame['frame']
    gt = cv2.imread(str(ROOT / 'data/ISOD_reference_test_v2' / frame['path']), 0) > 0
    n, ids, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    raw = cv2.imread(str(ROOT / f'data/ISOD/data/{site}/depth/{number}.png'), cv2.IMREAD_UNCHANGED)
    assert raw.shape == gt.shape == (422, 640)
    start = len(features)
    for ident in range(1, n):
        x, y, w, h, area = stats[ident]
        z = raw[y:y+h, x:x+w][ids[y:y+h, x:x+w] == ident]
        valid = z[z > 0]
        fraction = len(valid) / int(area)
        features.append(dict(site=site, frame=number, component=ident, area_pixels=int(area),
                             valid_depth_fraction=fraction,
                             estimated_depth_m=float(np.median(valid) / 1000) if fraction >= .5 else None,
                             review_status=frame['review_status']))
    masks[site, number] = ids, start, n
assert len(masks) == 300
expected = json.loads((ROOT / 'results/test_references_20261122/isod_test300_scores.json').read_text())
all_sources = sources()
for mode, _ in MODES:
    for seed in range(3):
        name = f'esanet_{mode}_reference_s{seed}'
        load = loader(all_sources[name], lambda site, frame: masks[site, frame][0].shape)
        covered = []
        for (site, number), (ids, start, n) in masks.items():
            pred = load(site, number)
            assert pred.shape == ids.shape
            hits = np.bincount(ids.ravel(), weights=pred.ravel(), minlength=n)
            counts = np.bincount(ids.ravel(), minlength=n)
            values = hits[1:] / counts[1:]
            for offset, coverage in enumerate(values):
                features[start + offset][f'{mode}_s{seed}_coverage'] = float(coverage)
            covered.extend(values >= .5)
        assert len(covered) == expected[name]['objects']
        assert abs(float(np.mean(covered)) - expected[name]['object_recall']) < 1e-12

with (OUT / 'regions.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(features[0])); writer.writeheader(); writer.writerows(features)
bins = [
    ('area', [r['area_pixels'] < 100 for r in features],
     [100 <= r['area_pixels'] < 1000 for r in features], [r['area_pixels'] >= 1000 for r in features]),
    ('depth', [r['estimated_depth_m'] is not None and r['estimated_depth_m'] < 1 for r in features],
     [r['estimated_depth_m'] is not None and 1 <= r['estimated_depth_m'] < 2 for r in features],
     [r['estimated_depth_m'] is not None and r['estimated_depth_m'] >= 2 for r in features]),
]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9))
summary = dict(frames=300, sites=sorted({r['site'] for r in features}), components=len(features),
               grid=[640, 422], units='8-connected reference regions; not verified physical instances',
               depth_units='raw millimetres / 1000, as in the dataset loader',
               depth_filter='median positive depth; at least half of region pixels must have positive depth',
               missing_depth_regions=sum(r['estimated_depth_m'] is None for r in features), bins={})
for panel, (kind, *groups) in enumerate(bins):
    ax = axes[panel]
    counts = [sum(group) for group in groups]
    assert all(counts), (kind, counts)
    summary['bins'][kind] = dict(counts=counts, results={})
    for offset, (mode, label) in zip([-.06, 0, .06], MODES):
        points = [[np.mean([r[f'{mode}_s{seed}_coverage'] >= .5
                            for r, selected in zip(features, group) if selected]) for seed in range(3)]
                  for group in groups]
        a = np.array(points)
        summary['bins'][kind]['results'][mode] = dict(mean=a.mean(1).tolist(), sd=a.std(1, ddof=1).tolist(), values=a.tolist())
        ax.errorbar(np.arange(3) + offset, a.mean(1), yerr=a.std(1, ddof=1), fmt='o',
                    color=COLORS[mode], ms=3.5, lw=1, label=label)
    labels = ['<100 px', '100–999 px', '≥1000 px'] if kind == 'area' else ['<1 m', '1–2 m', '≥2 m']
    ax.set_xticks(range(3), [f'{label}\n(n={n})' for label, n in zip(labels, counts)])
    ax.set_xlim(-.35, 2.35); ax.set_ylim(0, 1); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title('(a) Image area' if kind == 'area' else '(b) Estimated depth', fontsize=8.5)
    ax.tick_params(labelsize=7); ax.grid(axis='y', color='.85', lw=.5); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Fraction of regions detected', fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(.5, 0))
fig.tight_layout(pad=.5, rect=(0, .14, 1, 1)); save(fig, 'isod_distance_size')
summary['reference_manifest_sha256'] = hashlib.sha256((ROOT / 'data/ISOD_reference_test_v2/manifest.json').read_bytes()).hexdigest()
summary['validated_against_scores_sha256'] = hashlib.sha256((ROOT / 'results/test_references_20261122/isod_test300_scores.json').read_bytes()).hexdigest()
summary['prediction_sources'] = {name: str(path.relative_to(ROOT)) for name, path in all_sources.items()
                                 if name.startswith(('esanet_rgb_reference_', 'esanet_depth_reference_', 'esanet_rgbd_reference_'))}
summary['regions_csv_sha256'] = hashlib.sha256((OUT / 'regions.csv').read_bytes()).hexdigest()
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
