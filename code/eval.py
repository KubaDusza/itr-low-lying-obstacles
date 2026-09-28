"""Score obstacle predictions against the ground truth built by prep_gt.py.

Usage:
  python eval.py depth [split]              # dataset's ground-plane method
  python eval.py masks <dir> [split]        # any method that wrote <dir>/<site>/<i>.png
"""
import json
import sys

import numpy as np
from PIL import Image

import isod


def gt(site, i):
    m = np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0
    return m.astype(np.uint8), isod.instances(m.astype(np.uint8))


def run(method, split="test", pred_dir=None, n=100):
    sites = isod.load_splits()[split]
    rows = []
    for site in sites:
        for i in range(n):
            g, ins = gt(site, i)
            _, _, label, mask = isod.frame(site, i)
            if method == "depth":
                pred = isod.depth_baseline(mask, label)
            else:
                pred = (np.array(Image.open(f"{pred_dir}/{site}/{i}.png")) > 0).astype(np.uint8)
            rows.append(isod.score(pred, g, ins, ignore=isod.ignore_region(label, g)))
    return isod.totals(rows)


if __name__ == "__main__":
    method = sys.argv[1]
    pred_dir = sys.argv[2] if method == "masks" else None
    split = sys.argv[3 if method == "masks" else 2] if len(sys.argv) > (3 if method == "masks" else 2) else "test"
    r = run(method, split, pred_dir)
    print(f"{method} on {split}: " + "  ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(json.dumps(r))
