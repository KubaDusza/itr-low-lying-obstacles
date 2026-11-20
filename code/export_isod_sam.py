"""Calibrate RGB-only SAM on reviewed validation masks and export training labels."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

import isod
from isod_sam_review import EXTRA_PROMPTS, PROMPT_GROUPS, read_cache


STRUCTURE = set(PROMPT_GROUPS["structure"]) | set(EXTRA_PROMPTS["structure"])
BROAD = {"small box", "toy", "fabric on the floor", "plastic bag", "backpack"}
THRESHOLDS = np.array([0.15, 0.30, 0.45, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


def calibrate(cache, validation):
    scores = np.zeros((2, len(THRESHOLDS), 3), dtype=np.int64)
    samples = 0
    for site in isod.load_splits()["val"]:
        for file in sorted((cache / site).glob("*.npz")):
            reference = cv2.imread(str(validation / site / f"{file.stem}.png"), 0) > 0
            score_maps = np.zeros((2, *reference.shape), dtype=np.float32)
            for prompt, confidence, mask in read_cache(file, reference.shape):
                if prompt in STRUCTURE or confidence < THRESHOLDS[0]:
                    continue
                np.maximum(score_maps[0], confidence, out=score_maps[0], where=mask)
                if prompt not in BROAD:
                    np.maximum(score_maps[1], confidence, out=score_maps[1], where=mask)
            for variant, score_map in enumerate(score_maps):
                hits = np.histogram(score_map[reference], bins=[*THRESHOLDS, np.inf])[0]
                extras = np.histogram(score_map[~reference], bins=[*THRESHOLDS, np.inf])[0]
                tp, fp = hits[::-1].cumsum()[::-1], extras[::-1].cumsum()[::-1]
                scores[variant] += np.stack((tp, fp, reference.sum() - tp), axis=1)
            samples += 1
    if not samples:
        raise ValueError("No cached validation predictions")
    iou = scores[..., 0] / np.maximum(scores.sum(axis=-1), 1)
    variant, threshold_index = np.unravel_index(iou.argmax(), iou.shape)
    threshold = float(THRESHOLDS[threshold_index])
    return threshold, bool(variant), dict(validation_frames=samples,
                                          pooled_iou=float(iou[variant, threshold_index]))


def export(cache, destination, validation):
    if destination.exists():
        raise FileExistsError(destination)
    inference = json.loads((cache / "manifest.json").read_text())
    if inference.get("source_mask", {}).get("drivable_floor_rgb_scale", 1) != 1 or "crop" in inference:
        raise ValueError("Expected full-image RGB inference")
    inputs = [(site, i, cache / site / f"{i}.npz")
              for site in isod.load_splits()["train"] for i in range(100)]
    missing = [file for _, _, file in inputs if not file.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} training frames are missing SAM proposals")

    threshold, narrow, metrics = calibrate(cache, validation)
    excluded = STRUCTURE | BROAD if narrow else STRUCTURE
    prompts = set(inference["prompts"]) - excluded
    print(dict(confidence=threshold, prompts=sorted(prompts), **metrics), flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".isod-sam-") as tmp:
        stage = Path(tmp)
        records = []
        for site, frame, file in inputs:
            with np.load(file) as candidate_file:
                shape = tuple(candidate_file["shape"])
            mask = np.zeros(shape, dtype=bool)
            for prompt, confidence, detection in read_cache(file, shape):
                if prompt in prompts and confidence >= threshold:
                    mask |= detection
            path = Path(site) / f"{frame}.png"
            output = stage / path
            output.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(output), mask.astype(np.uint8) * 255):
                raise OSError(f"Could not save {output}")
            records.append(dict(site=site, frame=frame, path=path.as_posix(),
                                shape=list(map(int, shape)),
                                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                review_status="unattended_sam", foreground_pixels=int(mask.sum())))
        manifest = dict(split="train", complete=True, confidence=threshold,
                        prompts=sorted(prompts), calibration=metrics, inference=inference, frames=records)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    export(args.cache, args.destination, args.validation)


if __name__ == "__main__":
    main()
