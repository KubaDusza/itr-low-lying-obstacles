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
from pathlib import Path

import cv2
import numpy as np

import isod


def versioned_frames(root, split, sites, limit=None):
    manifest = json.loads((root / "manifest.json").read_text())
    frames = [(row["site"], row["frame"], root / row["path"]) for row in manifest["frames"]]
    expected = {(site, i) for site in sites for i in (range(4, 100, 10) if split == "test" else range(100))}
    if manifest["split"] != split or len(frames) != len(expected) or {(s, i) for s, i, _ in frames} != expected:
        raise ValueError(f"Invalid {split} manifest in {root}")
    if limit:
        frames = sorted(frames, key=lambda row: (row[1], row[0]))[:limit]
    return frames


def export_gte(source, output, isod_export):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("train", "train.txt", "valid", "valid.txt", "depth_stats.json"):
        (output / name).symlink_to((isod_export / name).resolve())
    frames = sorted(path for path in source.glob("*/rgb/*.png") if path.stem.isdigit())
    if len(frames) != 320:
        raise ValueError(f"Expected 320 GTE-Floor RGB frames, got {len(frames)}")
    for folder in ("rgb", "depth", "labels_2"):
        (output / "test" / folder).mkdir(parents=True)
    names = []
    for rgb in frames:
        run = rgb.parent.parent
        name = f"{run.name}_{rgb.stem}"
        raw = run / "depth" / rgb.name
        depth = cv2.imread(str(raw), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise ValueError(f"Invalid depth for {name}")
        (output / "test" / "rgb" / f"{name}.png").symlink_to(rgb.resolve())
        (output / "test" / "depth" / f"{name}.png").symlink_to(raw.resolve())
        # ESANet's dataset returns the label shape to restore prediction resolution.
        placeholder = np.ones(depth.shape, dtype=np.uint8)
        if not cv2.imwrite(str(output / "test" / "labels_2" / f"{name}.png"), placeholder):
            raise OSError(f"Could not save GTE-Floor input dimensions for {name}")
        names.append(name)
    (output / "test.txt").write_text("\n".join(names) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-masks", type=Path)
    ap.add_argument("--val-masks", type=Path)
    ap.add_argument("--test-masks", type=Path)
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--val-limit", type=int)
    ap.add_argument("--depth-stats", type=Path)
    ap.add_argument("--gte", type=Path)
    ap.add_argument("--gte-out", type=Path)
    a = ap.parse_args()
    roots = {"train": a.train_masks, "val": a.val_masks, "test": a.test_masks}
    versioned = all(roots.values())
    if any(roots.values()) and not versioned:
        ap.error("Provide all three mask directories")
    if (a.gte is None) != (a.gte_out is None):
        ap.error("Provide --gte and --gte-out together")
    if versioned:
        Path(a.out).mkdir(parents=True, exist_ok=False)
    limits = {"train": a.train_limit, "val": a.val_limit}
    cv2.setNumThreads(1)

    splits = isod.load_splits()
    # ESANet's loader calls the middle split "valid"; ours is called "val"
    NAME = {"train": "train", "val": "valid", "test": "test"}
    for key, sites in splits.items():
        split = NAME[key]
        names = []
        for d in ("rgb", "depth", "labels_2"):
            os.makedirs(f"{a.out}/{split}/{d}", exist_ok=True)
        items = (versioned_frames(roots[key], key, sites, limits.get(key)) if versioned else
                 [(site, i, Path(isod.GT) / site / f"{i}.png") for site in sites for i in range(100)])
        for site, i, mask in items:
            name = f"{site}_{i}"
            names.append(name)
            for d in ("rgb", "depth"):
                src = Path(isod.DATA) / site / d / f"{i}.png"
                dst = Path(a.out) / split / d / f"{name}.png"
                if versioned and not src.is_file():
                    raise FileNotFoundError(src)
                if not dst.is_symlink() and not dst.exists():
                    dst.symlink_to(src.resolve())
            gt = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise FileNotFoundError(mask)
            if versioned:
                out = np.ones(gt.shape, np.uint8)
            else:
                label = cv2.imread(f"{isod.DATA}/{site}/label/{i}.png", 0)
                out = (label == 255).astype(np.uint8)
            out[gt > 0] = 2
            if not cv2.imwrite(f"{a.out}/{split}/labels_2/{name}.png", out):
                raise OSError(f"Could not save {name} in {split}")
        with open(f"{a.out}/{split}.txt", "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"{split}: {len(names)} frames")

    # depth statistics in millimetres, over the training split only, ignoring no-reading
    if a.depth_stats:
        st = json.loads(a.depth_stats.read_text())
    else:
        vals = []
        for site in splits["train"]:
            for i in range(0, 100, 5):
                raw = cv2.imread(f"{isod.DATA}/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
                v = raw[raw > 0].astype(np.float32).ravel()[::37]
                vals.append(v)
        v = np.concatenate(vals)
        st = {"mean_mm": float(v.mean()), "std_mm": float(v.std()),
              "note": "training split only, zero readings excluded"}
    with open(f"{a.out}/depth_stats.json", "w") as f:
        json.dump(st, f, indent=1)
    print(json.dumps(st, indent=1))
    if a.gte:
        export_gte(a.gte, a.gte_out, Path(a.out))


if __name__ == "__main__":
    main()
