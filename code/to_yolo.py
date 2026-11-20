"""Export the obstacle ground truth as a YOLO segmentation dataset (one class)."""
import argparse
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

import isod

OUT = os.path.join(isod.ROOT, "data", "yolo")
MIN_POLY_AREA = 20


def polygons(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_POLY_AREA or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        c[:, 0] /= w
        c[:, 1] /= h
        out.append(c.clip(0, 1))
    return out


def export():
    for split, sites in isod.load_splits().items():
        for sub in ("images", "labels"):
            os.makedirs(f"{OUT}/{sub}/{split}", exist_ok=True)
        n_obj = 0
        for site in sites:
            for i in range(100):
                stem = f"{site}_{i}"
                shutil.copyfile(f"{isod.DATA}/{site}/rgb/{i}.png", f"{OUT}/images/{split}/{stem}.png")
                m = (np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0).astype(np.uint8)
                polys = polygons(m)
                n_obj += len(polys)
                with open(f"{OUT}/labels/{split}/{stem}.txt", "w") as f:
                    for p in polys:
                        f.write("0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) + "\n")
        print(f"{split}: {len(sites)} sites, {len(sites) * 100} images, {n_obj} objects")

    with open(f"{OUT}/data.yaml", "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n  0: obstacle\n")
    print("wrote", OUT)


def export_versioned(out, masks, train_limit, val_limit):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stats = {}
    split_sites = isod.load_splits()
    for split, root in masks.items():
        manifest = json.loads((root / "manifest.json").read_text())
        rows = manifest["frames"]
        ids = {(row["site"], row["frame"]) for row in rows}
        expected = {(site, i) for site in split_sites[split]
                    for i in (range(4, 100, 10) if split == "test" else range(100))}
        if manifest["split"] != split or ids != expected or len(ids) != len(rows):
            raise ValueError(f"Invalid {split} manifest in {root}")
        limit = {"train": train_limit, "val": val_limit, "test": None}[split]
        if limit:
            rows = sorted(rows, key=lambda row: (row["frame"], row["site"]))[:limit]
        images = out / "images" / split
        labels = out / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        preserved = original = polies = 0
        for row in rows:
            site, i = row["site"], row["frame"]
            stem = f"{site}_{i}"
            image = Path(isod.DATA) / site / "rgb" / f"{i}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            (images / f"{stem}.png").symlink_to(image.resolve())
            raw = cv2.imread(str(root / row["path"]), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(root / row["path"])
            mask = (raw > 0).astype(np.uint8)
            polys = polygons(mask)
            (labels / f"{stem}.txt").write_text(
                "".join("0 " + " ".join(f"{v:.5f}" for v in p.ravel()) + "\n" for p in polys))
            recon = np.zeros_like(mask)
            scale = np.array([mask.shape[1], mask.shape[0]])
            for poly in polys:
                cv2.fillPoly(recon, [np.rint(poly * scale).astype(np.int32)], 1)
            polies += len(polys)
            original += int(mask.sum())
            preserved += int((mask & recon).sum())
        stats[split] = dict(frames=len(rows), polygons=polies, foreground_pixels=original,
                            preserved_pixels=preserved)
    (out / "data.yaml").write_text(
        f"path: {json.dumps(str(out))}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: obstacle\n")
    (out / "polygon_audit.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--train-masks", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--test-masks", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()
    explicit = (args.out, args.train_masks, args.val_masks, args.test_masks)
    if any(explicit):
        if not all(explicit):
            parser.error("Set --out and all three mask directories")
        cv2.setNumThreads(1)
        export_versioned(args.out, dict(train=args.train_masks, val=args.val_masks, test=args.test_masks),
                         args.train_limit, args.val_limit)
    else:
        export()
