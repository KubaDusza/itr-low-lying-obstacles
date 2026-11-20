"""Constants for the depth channel, measured on the training sites only.

Clipping percentiles and the mean/std of the encoded channel have to come from training
data alone, or the test sites leak into the input normalisation. Also reports the share of
missing readings, which is the number to compare against our own camera.

  python depth_stats.py --root /data/cs_courses/jdusza_itr    # writes <root>/depth_stats.json
"""
import argparse
import json
import os

import cv2
import numpy as np

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--every", type=int, default=5, help="sample every Nth frame")
    a = ap.parse_args()

    with open(f"{a.root}/splits.json") as f:
        splits = json.load(f)

    vals, invalid, obst_invalid, floor_invalid = [], [], [], []
    for site in splits["train"]:
        for i in range(0, 100, a.every):
            raw = cv2.imread(f"{a.root}/ISOD/data/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
            label = cv2.imread(f"{a.root}/ISOD/data/{site}/label/{i}.png", 0)
            gt = cv2.imread(f"{a.root}/ISOD_gt/{site}/{i}.png", 0) > 0
            miss = raw == 0
            invalid.append(miss.mean())
            if gt.any():
                obst_invalid.append(miss[gt].mean())
            floor = label == 255
            if floor.any():
                floor_invalid.append(miss[floor].mean())
            vals.append(raw[~miss].astype(np.float32).ravel()[::37] / 1000.0)

    v = np.concatenate(vals)
    st = {"near_m": float(np.percentile(v, 1)), "far_m": float(np.percentile(v, 99))}
    # mean and std of the encoded channel itself, over the same sample
    enc = st["near_m"] / np.clip(v, st["near_m"], st["far_m"])
    st["mean"], st["std"] = float(enc.mean()), float(enc.std())
    st["invalid_frac"] = float(np.mean(invalid))
    st["invalid_frac_obstacle"] = float(np.mean(obst_invalid))
    st["invalid_frac_floor"] = float(np.mean(floor_invalid))
    st["sites"] = splits["train"]

    # second pass: mean and std of the three geometric channels, same training frames
    g = [[], [], []]
    for site in splits["train"][::2]:
        for i in range(0, 100, 4 * a.every):
            raw = cv2.imread(f"{a.root}/ISOD/data/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
            ch = modal.encode_geo(raw, st)          # unstandardised on the first pass
            for j in range(3):
                g[j].append(ch[j].ravel()[::53])
    st["geo"] = {"mean": [float(np.concatenate(x).mean()) for x in g],
                 "std": [float(np.concatenate(x).std()) for x in g]}

    # sanity: the encoding must survive a round trip through the real range
    probe = modal.encode_depth(np.array([[0, 500, 2000, 9000]], np.uint16), st)
    st["probe_0_500_2000_9000mm"] = [round(float(x), 3) for x in probe[0]]

    # written through a temporary file: a training run may be starting while this runs
    tmp = f"{a.root}/depth_stats.json.tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, f"{a.root}/depth_stats.json")
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
