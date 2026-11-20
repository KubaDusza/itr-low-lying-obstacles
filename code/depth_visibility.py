"""Compare measured depth on annotated GTE-Floor objects and adjacent floor."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from floor_plane import floor_gap
from own_items import item_sets
from run_floor_baseline import depth


def read_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError(f"Missing or misaligned mask {path}")
    return mask


def measure(own, manual, settings):
    legend = item_sets()
    for path in sorted(manual.glob("*_id.png")):
        run, frame = path.stem[:-3].rsplit("_", 1)
        if run.endswith("_empty"):
            continue
        height, background, object_set, distance = run.split("_")
        raw = depth(own / run / "depth" / f"{frame}.png")
        ids = read_mask(path, raw.shape)
        foreground = read_mask(manual / f"{run}_{frame}.png", raw.shape) > 0
        floor = read_mask(own / run / "floor.png", raw.shape) > 0
        gap, valid_gap = floor_gap(raw, settings, seed_from=0)
        dist_from_foreground = cv2.distanceTransform(
            (~foreground).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        floor = valid_gap & floor & (dist_from_foreground >= 4)
        for item in legend[object_set]:
            obj = (ids == item["idx"]) & foreground
            if not obj.any():
                continue
            dist_from_item = cv2.distanceTransform(
                (~obj).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            object_gap = gap[obj & valid_gap]
            floor_gap_near_item = gap[floor & (dist_from_item <= 20)]
            row = dict(recording=run, frame=int(frame), height_cm=int(height[1:]),
                       background=background, distance_cm=int(distance),
                       item=item["label"], shape=item["shape"], pixels=int(obj.sum()),
                       depth_valid_fraction=float((raw[obj] != 0).mean()),
                       plane_valid_fraction=float(object_gap.size / obj.sum()),
                       floor_pixels=int(floor_gap_near_item.size),
                       residual_difference_mm=None, residual_auc=None)
            if object_gap.size >= 10 and floor_gap_near_item.size >= 40:
                row["residual_difference_mm"] = float(
                    1000 * (np.median(object_gap) - np.median(floor_gap_near_item))
                )
                ordered = np.sort(floor_gap_near_item)
                below = np.searchsorted(ordered, object_gap, side="left")
                at_or_below = np.searchsorted(ordered, object_gap, side="right")
                row["residual_auc"] = float(((below + at_or_below) / (2 * ordered.size)).mean())
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--floor-settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    settings = json.loads(args.floor_settings.read_text())["depth_range_m"]
    rows = list(measure(args.own, args.manual, settings))
    if not rows:
        raise ValueError("No annotated objects found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    examples = defaultdict(list)
    for row in rows:
        if row["height_cm"] == 52 and row["background"] == "floor" and row["item"] in (
            "flat paper towel", "crumpled tissue"
        ):
            examples[(row["distance_cm"], row["item"])].append(row)
    print(f"{len(rows)} object observations in {len({r['recording'] for r in rows})} recordings")
    for key, values in sorted(examples.items()):
        valid = [v for v in values if v["residual_auc"] is not None]
        residuals = [v["residual_difference_mm"] for v in valid]
        aucs = [v["residual_auc"] for v in valid]
        print(*key, f"{len(valid)}/{len(values)}", "valid-depth fraction",
              round(float(np.median([v["depth_valid_fraction"] for v in values])), 3),
              "residual difference (mm)", round(float(np.median(residuals)), 1) if valid else "NA",
              "AUC", round(float(np.median(aucs)), 3) if valid else "NA")


if __name__ == "__main__":
    main()
