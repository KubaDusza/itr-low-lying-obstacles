"""Export assisted obstacle references for all 300 ISOD test frames.

Same assistance pipeline as the training references (review_isod_draft.frame_proposal on the
source-full-crayon SAM cache). The 30 human-reviewed test frames override their proposals and
also measure proposal quality on the test split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import review_isod_draft as draft

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "data/ISOD_reference_test_v1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    if a.destination.exists():
        raise FileExistsError(a.destination)
    review = draft.review
    source = Path(review.isod.DATA)
    cache = review.SAM_CACHE / "source-full-crayon"
    reviewed = {(f["site"], f["frame"]) for f in json.loads((REVIEWED / "manifest.json").read_text())["frames"]}
    assert len(reviewed) == 30
    a.destination.mkdir(parents=True)
    records, quality = [], []
    for site in review.isod.load_splits()["test"]:
        for i in range(review.N_FRAMES):
            assert (cache / site / f"{i}.npz").is_file(), f"missing proposal cache {site}/{i}"
            comp, rows, _ = draft.frame_proposal(site, i)
            proposal = np.isin(comp, [r["id"] for r in rows if r["keep"]]).astype(np.uint8) * 255
            label = cv2.imread(str(source / site / "label" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            if (site, i) in reviewed:
                mask = cv2.imread(str(REVIEWED / site / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
                state, origin = "human_reviewed", f"ISOD_reference_test_v1/{site}/{i}.png"
                inter = int(((mask > 0) & (proposal > 0)).sum())
                union = int(((mask > 0) | (proposal > 0)).sum())
                quality.append(dict(site=site, frame=i, inter=inter, union=union,
                                    iou=inter / union if union else 1.0))
            else:
                mask = proposal
                state, origin = "isod_sam_prelabel", f"ISOD_sam_review/source-full-crayon/{site}/{i}.npz"
            if (mask is None or label is None or mask.shape != label.shape or
                    not np.isin(mask, (0, 255)).all() or np.any((mask > 0) & (label != 0))):
                raise ValueError(f"Invalid reference mask for {site}/{i}")
            out = a.destination / site / f"{i}.png"
            out.parent.mkdir(exist_ok=True)
            assert cv2.imwrite(str(out), mask)
            records.append(dict(site=site, frame=i, path=f"{site}/{i}.png",
                                sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                                shape=list(mask.shape), review_status=state, source=origin))
    pooled = sum(q["inter"] for q in quality) / sum(q["union"] for q in quality)
    summary = dict(frames=len(records), reviewed=len(quality),
                   assisted=len(records) - len(quality),
                   proposal_vs_reviewed=dict(mean_image_iou=float(np.mean([q["iou"] for q in quality])),
                                             pooled_iou=pooled, per_frame=quality))
    manifest = dict(split="test", complete=True, frames=records,
                    note="Assisted references as for training; 30 human-reviewed frames override proposals.",
                    quality=summary)
    (a.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "proposal_vs_reviewed"} |
                     {"mean_image_iou": summary["proposal_vs_reviewed"]["mean_image_iou"], "pooled_iou": pooled}))


if __name__ == "__main__":
    main()
