"""Safety cases and differential checks against the published metric function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from hazard_metrics import evaluate, aggregate


def safety_checks():
    ids = np.zeros((30, 40), np.uint16); ids[8:12, 8:12] = 7
    perfect = evaluate(ids, ids > 0)
    assert perfect['operating_points']['50'] == dict(tp=1, fn=0, fp=0, recall=1., f1=1.)
    missed = evaluate(ids, np.zeros_like(ids))
    assert missed['operating_points']['50']['fn'] == 1
    painted = evaluate(ids, np.ones_like(ids))
    assert painted['reference_objects'][0]['coverage'] == 1.
    assert painted['operating_points']['50']['tp'] == 0 and painted['operating_points']['50']['fp'] == 1
    # Multiple disconnected portions of the same cable remain one reference ID.
    cable = ids.copy(); cable[15:18, 15:18] = 7
    assert len(evaluate(cable, cable > 0)['reference_objects']) == 1
    tiny = np.zeros_like(ids); tiny[1, 1] = 1; tiny[1, 8] = 2
    assert evaluate(tiny, tiny > 0)['operating_points']['50']['tp'] == 2
    empty = evaluate(np.zeros_like(ids), np.zeros_like(ids))
    assert empty['pixel']['iou'] is None and empty['operating_points']['50']['f1'] is None
    false = evaluate(np.zeros_like(ids), ids > 0)
    assert false['operating_points']['50']['fp'] == 1
    ignored = np.zeros_like(ids, bool); ignored[20:25, 20:25] = True
    pred = (ids > 0) | ignored
    ignored[8:10, 8:10] = True
    checked = evaluate(ids, pred, ignore=ignored)
    assert checked['operating_points']['50']['tp'] == 1 and checked['operating_points']['50']['fp'] == 0
    merged = np.zeros_like(ids); merged[8:12, 18:22] = 9; merged |= ids
    pred = merged > 0; pred[9, 12:18] = True
    result = evaluate(merged, pred)
    assert len(result['prediction_components']) == 1 and len(result['reference_objects']) == 2
    assert result['prediction_components'][0]['touching_reference_ids'] == [7, 9]
    try:
        evaluate(ids, np.zeros((1, 1)))
        raise AssertionError('Shape mismatch accepted')
    except ValueError:
        pass
    total = aggregate([perfect, false])
    assert total['operating_points']['50']['f1'] == 2 / 3
    assert total['pixel']['iou'] == .5
    return 11


def differential_checks(source):
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'segment_metrics')
    namespace = dict(np=np, EasyDict=lambda **kw: dict(**kw))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    rng = np.random.default_rng(20261121)
    for _ in range(50):
        ids = np.zeros((24, 32), np.uint16)
        for ident in (3, 7, 15):
            y, x = rng.integers([0, 0], [19, 27]); ids[y:y+5, x:x+5] = ident
        pred = ndimage.binary_dilation(rng.random(ids.shape) < .04)
        valid = rng.random(ids.shape) > .1
        result = evaluate(ids, pred, valid=valid)
        cc, _ = ndimage.label(pred & valid, structure=np.ones((3, 3)))
        official = namespace['segment_metrics'](ids[valid], cc[valid])
        np.testing.assert_allclose([r['adjusted_iou'] for r in result['reference_objects']], official['sIoU_gt'])
        np.testing.assert_allclose([r['precision'] for r in result['prediction_components']], official['prec_pred'])
        for t in (25, 50, 75):
            for name in ('tp', 'fn', 'fp'):
                assert result['operating_points'][str(t)][name] == official[f'{name}_{t}']
    return dict(random_scenes=50, upstream_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_url='https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/metrics/instance_segmentation.py')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--upstream', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args(); report = dict(safety_cases_passed=safety_checks())
    if a.upstream: report['differential'] = differential_checks(a.upstream)
    if a.out: a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
