"""Compare prompt unions using cached RGB-only SAM predictions on development frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from isod_sam_review import read_cache
from sam_prompt_trial import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.directory
    trial = json.loads((directory / "prompt_trial.json").read_text())
    previous = json.loads((directory / "trial1/scores.json").read_text())
    phrases = trial["inference_prompts"]
    predictions, references = [], []
    for site, frame in trial["frames"]:
        raw = cv2.imread(str(root / "data/ISOD_reference_train_v1" / site / f"{frame}.png"), 0)
        if raw is None:
            raise FileNotFoundError(f"Missing reference {site}/{frame}")
        gt = np.packbits((raw > 0).reshape(-1))
        by_phrase = {phrase: np.zeros_like(gt) for phrase in phrases}
        for phrase, confidence, mask in read_cache(directory / "cache" / site / f"{frame}.npz", raw.shape):
            if confidence >= trial["fixed_confidence"]:
                by_phrase[phrase] |= np.packbits(mask.reshape(-1))
        predictions.append(by_phrase)
        references.append(gt)

    def evaluate(selected):
        counts = []
        for by_phrase, gt in zip(predictions, references, strict=True):
            pred = np.zeros_like(gt)
            for phrase in selected:
                pred |= by_phrase[phrase]
            tp = int(np.bitwise_count(pred & gt).sum())
            fp = int(np.bitwise_count(pred & ~gt).sum())
            fn = int(np.bitwise_count(~pred & gt).sum())
            counts.append((tp, fp, fn))
        return dict(phrases=selected, **metrics(counts), per_frame=[
            dict(site=site, frame=frame, tp=tp, fp=fp, fn=fn, iou=tp / max(1, tp + fp + fn))
            for (site, frame), (tp, fp, fn) in zip(trial["frames"], counts, strict=True)
        ])

    broad = trial["generic_prompts"]
    names = trial["specific_prompts"]
    floor_names = trial["specific_floor_prompts"]
    best = previous["best_generic"]
    recipes = {
        "best_single": [best],
        "all_broad": broad,
        "all_floor_broad": [phrase for phrase in broad if "on the floor" in phrase],
        "all_names_on_floor": floor_names,
        "best_plus_names_on_floor": [best] + floor_names,
        "all_broad_plus_names_on_floor": broad + floor_names,
        "all_broad_plus_names_and_floor_names": broad + names + floor_names,
        "all_71_phrases": phrases,
    }
    results = {name: evaluate(selected) for name, selected in recipes.items()}
    selected = [best]
    current = results["best_single"]
    steps = []
    # Bounded exploratory search; this does not establish a globally optimal set.
    while len(selected) < 6:
        candidates = [(phrase, evaluate(selected + [phrase])) for phrase in phrases if phrase not in selected]
        phrase, candidate = max(candidates, key=lambda pair: (pair[1]["mean_frame_iou"], pair[1]["pooled_iou"]))
        if candidate["mean_frame_iou"] <= current["mean_frame_iou"] + 1e-10:
            break
        selected.append(phrase)
        current = candidate
        steps.append(dict(added=phrase, mean_frame_iou=current["mean_frame_iou"], pooled_iou=current["pooled_iou"]))
    results["greedy_up_to_six"] = current
    output = dict(
        note="Eight development frames only. Exploratory greedy additions, at most six phrases. No global-optimum claim; reserved frames untouched.",
        fixed_confidence=trial["fixed_confidence"], frames=trial["frames"],
        recipes=results, greedy_steps=steps,
    )
    (directory / "trial1/union_scores.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, record in results.items():
        print(name, {key: round(record[key], 4) for key in ("mean_frame_iou", "pooled_iou", "precision", "recall")})
    print("Selected", selected)
    print("Additions", steps)


if __name__ == "__main__":
    main()
