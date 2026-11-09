"""Export ISOD into the directory layout ESANet's dataset loader expects.

  ITR_DATA=/data/cs_courses/jdusza_itr python export_isod.py --out <root>/esanet_isod

Writes, per split (train / valid / test):

  <out>/<split>/rgb/<site>_<i>.png        symlink to the ISOD frame
  <out>/<split>/depth/<site>_<i>.png      symlink, uint16 millimetres, 0 = no reading
  <out>/<split>/labels_2/<site>_<i>.png   0 = void, 1 = floor, 2 = obstacle
  <out>/<split>.txt                       the file list
  <out>/depth_stats.json                  depth mean and std in mm, training split only

RGB and depth are symlinked rather than copied: the same bytes go through both pipelines,
so nothing can drift between the two and no disk is wasted. The labels are new files
because ESANet needs the three-way class image, built from the same obstacle masks and the
same ignore region that eval.py uses.
"""
import argparse
import json
import os

import cv2
import numpy as np

import isod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    splits = isod.load_splits()
    # ESANet's loader calls the middle split "valid"; ours is called "val"
    NAME = {"train": "train", "val": "valid", "test": "test"}
    for key, sites in splits.items():
        split = NAME[key]
        names = []
        for d in ("rgb", "depth", "labels_2"):
            os.makedirs(f"{a.out}/{split}/{d}", exist_ok=True)
        for site in sites:
            for i in range(100):
                name = f"{site}_{i}"
                names.append(name)
                for d, src in (("rgb", f"{isod.DATA}/{site}/rgb/{i}.png"),
                               ("depth", f"{isod.DATA}/{site}/depth/{i}.png")):
                    dst = f"{a.out}/{split}/{d}/{name}.png"
                    if not os.path.islink(dst) and not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
                label = cv2.imread(f"{isod.DATA}/{site}/label/{i}.png", 0)
                gt = cv2.imread(f"{isod.GT}/{site}/{i}.png", 0) > 0
                out = np.zeros(label.shape, np.uint8)      # 0 = void
                out[label == 255] = 1                      # drivable floor
                out[gt] = 2                                # obstacle
                cv2.imwrite(f"{a.out}/{split}/labels_2/{name}.png", out)
        with open(f"{a.out}/{split}.txt", "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"{split}: {len(names)} frames")

    # depth statistics in millimetres, over the training split only, ignoring no-reading
    vals = []
    for site in splits["train"]:
        for i in range(0, 100, 5):
            raw = cv2.imread(f"{isod.DATA}/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
            v = raw[raw > 0].astype(np.float32).ravel()[::37]
            vals.append(v)
    v = np.concatenate(vals)
    st = {"mean_mm": float(v.mean()), "std_mm": float(v.std()),
          "note": "training split only, zero (no reading) excluded, as ESANet's raw mode does"}
    with open(f"{a.out}/depth_stats.json", "w") as f:
        json.dump(st, f, indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
