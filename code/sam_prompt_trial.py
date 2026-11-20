"""Run RGB-only SAM prompt candidates and score the fixed development subset."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import cv2
import numpy as np


def infer(args):
    os.environ["ITR_DATA"] = str(args.root.resolve())
    from isod_sam_review import cache_predictions
    trial = json.loads(args.trial.read_text())
    frames = args.trial.with_name("trial_frames.json")
    frames.write_text(json.dumps(trial["frames"]) + "\n")
    cache_predictions(argparse.Namespace(
        frames=frames, cache=args.cache, model=trial["model"], revision=trial["revision"],
        batch=2, lower_crop=False, source_context=None, prompt=trial["inference_prompts"],
    ))


def metrics(counts):
    tp, fp, fn = np.asarray(counts, dtype=np.int64).sum(axis=0)
    frame_ious = [a / max(1, a + b + c) for a, b, c in counts]
    return dict(pooled_iou=float(tp / max(1, tp + fp + fn)), mean_frame_iou=float(np.mean(frame_ious)),
                precision=float(tp / max(1, tp + fp)), recall=float(tp / max(1, tp + fn)))


def score(args):
    from isod_sam_review import read_cache
    trial = json.loads(args.trial.read_text())
    source = args.root / "data"
    frames = trial["frames"]
    masks = {}
    references = {}
    for site, frame in frames:
        gt = cv2.imread(str(source / "ISOD_reference_train_v1" / site / f"{frame}.png"), 0)
        if gt is None:
            raise FileNotFoundError(f"Missing reference for {site}/{frame}")
        references[site, frame] = gt > 0
        by_phrase = defaultdict(lambda: np.zeros(gt.shape, dtype=bool))
        for phrase, confidence, mask in read_cache(args.cache / site / f"{frame}.npz", gt.shape):
            if confidence >= trial["fixed_confidence"]:
                by_phrase[phrase] |= mask
        masks[site, frame] = by_phrase

    def evaluate(phrases, save_to=None, frozen_baseline=False):
        counts = []
        for site, frame in frames:
            gt = references[site, frame]
            if frozen_baseline:
                raw = cv2.imread(str(source / "ISOD_sam_train_v1" / site / f"{frame}.png"), 0)
                if raw is None:
                    raise FileNotFoundError(f"Missing frozen SAM mask for {site}/{frame}")
                pred = raw > 0
            else:
                pred = np.zeros(gt.shape, dtype=bool)
                for phrase in phrases:
                    pred |= masks[site, frame][phrase]
            counts.append((int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())))
            if save_to:
                target = save_to / site / f"{frame}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), pred.astype(np.uint8) * 255):
                    raise OSError(target)
        return metrics(counts)

    singleton = {phrase: evaluate([phrase]) for phrase in trial["inference_prompts"]}
    best_generic = max(trial["generic_prompts"], key=lambda phrase: (singleton[phrase]["mean_frame_iou"],
                                                                   singleton[phrase]["pooled_iou"]))
    profiles = {
        "frozen_baseline": trial["baseline_prompts"],
        "best_generic": [best_generic],
        "user_generic": ["something small on the floor"],
        "specific": trial["specific_prompts"],
        "specific_on_floor": trial["specific_floor_prompts"],
        "hybrid": [best_generic] + trial["specific_floor_prompts"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    scores = {}
    for name, phrases in profiles.items():
        scores[name] = dict(phrases=phrases, **evaluate(
            phrases, args.out / name, frozen_baseline=name == "frozen_baseline"))
    result = dict(development_frames=frames, selection_metric="mean frame IoU; pooled IoU breaks ties",
                  fixed_confidence=trial["fixed_confidence"], best_generic=best_generic,
                  profiles=scores, single_phrase_scores=singleton,
                  note="Development results on eight reviewed training frames. Held-out rooms have not been evaluated.")
    (args.out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, value in scores.items():
        print(name, {key: round(value[key], 3) for key in ("pooled_iou", "mean_frame_iou", "precision", "recall")})
    print("Best generic phrase", best_generic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer", "score"])
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.command == "infer":
        infer(args)
    else:
        if args.out is None:
            parser.error("Provide --out for scoring")
        score(args)


if __name__ == "__main__":
    main()
