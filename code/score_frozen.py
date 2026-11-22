"""Score predictions against the frozen ISOD subset and GTE-Floor manual masks."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import cv2

import isod


def score_frames(entries):
    rows = []
    by_group = defaultdict(list)
    for group, frame, reference, prediction in entries:
        gt = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(prediction), cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None or gt.shape != pred.shape:
            raise ValueError(f"Missing or misaligned masks for {group}/{frame}: {prediction}")
        counts = isod.score(pred > 0, gt > 0)
        rows.append(dict(group=group, frame=frame, **counts))
        by_group[group].append(counts)
    return dict(pooled=isod.totals(rows), by_group={g: isod.totals(v) for g, v in by_group.items()},
                frames=rows)


def isod_frames(reference, predictions):
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest["split"] != "test":
        raise ValueError("Expected the frozen ISOD test manifest")
    for row in manifest["frames"]:
        yield row["site"], row["frame"], reference / row["path"], predictions / row["path"]


def gte_frames(reference, predictions):
    for path in sorted(reference.glob("*.png")):
        if path.stem.endswith("_id"):
            continue
        run, i = path.stem.rsplit("_", 1)
        yield run, int(i), path, predictions / run / f"{i}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isod-reference", type=Path)
    parser.add_argument("--isod-predictions", type=Path)
    parser.add_argument("--gte-reference", type=Path)
    parser.add_argument("--gte-predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for name, reference, predictions, entries, expected in (
        ("ISOD", args.isod_reference, args.isod_predictions, isod_frames, 30),
        ("GTE-Floor", args.gte_reference, args.gte_predictions, gte_frames, 320),
    ):
        if reference is None and predictions is None:
            continue
        if reference is None or predictions is None:
            parser.error(f"Provide both paths for {name}")
        result = score_frames(entries(reference, predictions))
        if len(result["frames"]) != expected:
            raise ValueError(f"Expected {expected} {name} frames")
        results[name] = result
        print(name, result["pooled"])
    if not results:
        parser.error("Provide reference and prediction paths")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.out.parent, delete=False) as output:
        output.write(json.dumps(results, indent=2) + "\n")
    Path(output.name).replace(args.out)


if __name__ == "__main__":
    main()
