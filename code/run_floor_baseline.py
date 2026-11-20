"""Calibrate a blind depth-only floor fit and export frozen predictions."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from floor_plane import floor_gap


THRESHOLDS_M = np.array([0.002, 0.004, 0.008, 0.012, 0.02, 0.035, 0.05, 0.08, 0.12, 0.20, 0.35, 0.49])


def depth(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.ndim != 2:
        raise ValueError(f"Expected a 16-bit depth image at {path}")
    return image


def settings(root):
    splits = json.loads((root / "splits.json").read_text())
    samples = []
    for site in splits["train"]:
        for i in range(0, 100, 5):
            raw = depth(root / "ISOD" / "data" / site / "depth" / f"{i}.png")
            samples.append(raw[raw != 0][::37])
    near, far = np.percentile(np.concatenate(samples), [1, 99]) / 1000
    return dict(near_m=float(near), far_m=float(far))


def reference_entries(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for row in manifest["frames"]:
        yield row["site"], row["frame"], directory / row["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--gte", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    st = settings(args.root)
    isod_data = args.root / "ISOD" / "data"

    counts = np.zeros((len(THRESHOLDS_M), 3), dtype=np.int64)
    n = 0
    for site, i, reference in reference_entries(args.validation):
        gap, valid = floor_gap(depth(isod_data / site / "depth" / f"{i}.png"), st, seed_from=0)
        mask = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != gap.shape:
            raise ValueError(f"Missing or misaligned validation mask {reference}")
        gt = mask > 0
        for j, threshold in enumerate(THRESHOLDS_M):
            prediction = valid & (gap >= threshold)
            counts[j] += ((prediction & gt).sum(), (prediction & ~gt).sum(), (~prediction & gt).sum())
        n += 1
    if n != 300:
        raise ValueError(f"Expected 300 validation frames, got {n}")
    ious = counts[:, 0] / np.maximum(counts.sum(axis=1), 1)
    best = int(ious.argmax())
    threshold = float(THRESHOLDS_M[best])

    args.out.mkdir(parents=True, exist_ok=False)
    for group, rows in (
        ("ISOD", [(site, i, isod_data / site / "depth" / f"{i}.png")
                  for site, i, _ in reference_entries(args.test)]),
        ("GTE-Floor", [(path.parent.parent.name, int(path.stem), path)
                       for path in sorted(args.gte.glob("*/depth/*.png")) if path.stem.isdigit()]),
    ):
        expected = 30 if group == "ISOD" else 320
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {group} depth frames, got {len(rows)}")
        for scene, i, path in rows:
            gap, valid = floor_gap(depth(path), st, seed_from=0)
            prediction = valid & (gap >= threshold)
            output = args.out / group / scene / f"{i}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), prediction.astype(np.uint8) * 255):
                raise OSError(f"Could not save {output}")
    metadata = dict(depth_range_m=st, threshold_m=threshold,
                    validation_iou=float(ious[best]), validation_frames=n)
    (args.out / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
