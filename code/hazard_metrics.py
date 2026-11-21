"""ID-aware adaptation of SegmentMeIfYouCan component metrics.

Equations checked against the authors' segment_metrics implementation:
https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/metrics/instance_segmentation.py

Differences: supplied reference IDs (possibly disconnected), no small-target
filter, fixed prediction operating point, explicit valid/ignore masks. This is
localization/false-alert evaluation, not one-to-one instance AP or navigation.
"""
import numpy as np
from scipy import ndimage

THRESHOLDS = tuple(range(25, 76, 5))


def ratio(a, b):
    return float(a / b) if b else None


def evaluate(reference_ids, prediction, valid=None, ignore=None):
    ids = np.asarray(reference_ids)
    pred = np.asarray(prediction, dtype=bool)
    if ids.shape != pred.shape or ids.ndim != 2:
        raise ValueError('Expected aligned 2-D reference IDs and prediction')
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError('Reference IDs must be nonnegative integers')
    roi = np.ones(ids.shape, bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    if roi.shape != ids.shape:
        raise ValueError('Invalid scoring-domain shape')
    if ignore is not None:
        ignored = np.asarray(ignore, dtype=bool)
        if ignored.shape != ids.shape:
            raise ValueError('Invalid ignore-mask shape')
        # Targets take priority over nuisance annotations; outside the fixed
        # valid domain remains outside. A crop must separately pass retention.
        roi &= ~ignored | (ids > 0)
    gt = (ids > 0) & roi
    pred = pred & roi
    labels, count = ndimage.label(pred, structure=np.ones((3, 3), np.uint8))
    present = np.unique(ids[gt])
    dense = np.zeros(ids.shape, np.int32)
    dense[gt] = np.searchsorted(present, ids[gt]) + 1
    pairs = dense[roi].astype(np.int64) * (count + 1) + labels[roi]
    table = np.bincount(pairs, minlength=(len(present) + 1) * (count + 1)).reshape(len(present) + 1, count + 1)
    gt_rows, pred_rows = [], []
    for row, ident in enumerate(present, 1):
        area = int(table[row].sum())
        hits = np.flatnonzero(table[row, 1:]) + 1
        overlap = int(table[row, hits].sum())
        outside = int(table[0, hits].sum())
        gt_rows.append(dict(id=int(ident), area_pixels=area, covered_pixels=overlap,
                            coverage=ratio(overlap, area), adjusted_iou=ratio(overlap, area + outside),
                            touching_prediction_components=hits.tolist()))
    for ident in range(1, count + 1):
        area = int(table[:, ident].sum())
        overlap = int(table[1:, ident].sum())
        pred_rows.append(dict(id=ident, area_pixels=area, target_pixels=overlap,
                              precision=ratio(overlap, area),
                              touching_reference_ids=present[np.flatnonzero(table[1:, ident])].tolist()))
    operating = {}
    for percent in THRESHOLDS:
        threshold = percent / 100
        tp = sum(r['adjusted_iou'] >= threshold for r in gt_rows)
        fn = len(gt_rows) - tp
        fp = sum(r['precision'] < threshold for r in pred_rows)
        operating[str(percent)] = dict(tp=tp, fn=fn, fp=fp, recall=ratio(tp, tp + fn),
                                       f1=ratio(2 * tp, 2 * tp + fn + fp))
    inter = int((gt & pred).sum()); union = int((gt | pred).sum())
    return dict(reference_objects=gt_rows, prediction_components=pred_rows,
                operating_points=operating, evaluated_pixels=int(roi.sum()),
                pixel=dict(inter=inter, union=union, pred=int(pred.sum()), gt=int(gt.sum()),
                           iou=ratio(inter, union), precision=ratio(inter, int(pred.sum())),
                           recall=ratio(inter, int(gt.sum()))))


def aggregate(results):
    totals = {key: sum(r['pixel'][key] for r in results) for key in ('inter', 'union', 'pred', 'gt')}
    operating = {}
    for t in THRESHOLDS:
        counts = {key: sum(r['operating_points'][str(t)][key] for r in results) for key in ('tp', 'fn', 'fp')}
        tp, fn, fp = (counts[k] for k in ('tp', 'fn', 'fp'))
        operating[str(t)] = dict(**counts, recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fn + fp))
    f1s = [r['f1'] for r in operating.values() if r['f1'] is not None]
    return dict(frames=len(results), pixel=dict(**totals, iou=ratio(totals['inter'], totals['union']),
                precision=ratio(totals['inter'], totals['pred']), recall=ratio(totals['inter'], totals['gt'])),
                operating_points=operating, mean_f1=ratio(sum(f1s), len(f1s)),
                all_threshold_f1_defined=len(f1s) == len(THRESHOLDS))
