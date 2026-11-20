"""Export versioned ISOD reference training masks using reviewed masks first."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

import review_isod_draft as draft


def export_training_masks(destination, allow_partial):
    review = draft.review
    data_root = Path(review.isod.DATA_ROOT)
    source = Path(review.isod.DATA)
    cache = review.SAM_CACHE / "source-full-crayon"
    if destination.exists():
        raise FileExistsError(destination)

    inputs = []
    missing = []
    for site in review.isod.load_splits()["train"]:
        for i in range(review.N_FRAMES):
            name = f"{i}.png"
            manual = next((root / site / name for root in (Path(review.OUT), draft.REFERENCE)
                           if (root / site / f"{i}.json").is_file()), None)
            proposal = cache / site / f"{i}.npz"
            if manual is not None or proposal.is_file():
                inputs.append((site, i, manual, proposal))
            else:
                missing.append([site, i])
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} training frames are missing cached proposals")

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".isod-train-") as tmp:
        stage = Path(tmp)
        records = []
        for site, i, manual, proposal in inputs:
            if manual is None:
                comp, rows, _ = draft.frame_proposal(site, i)
                mask = np.isin(comp, [r["id"] for r in rows if r["keep"]]).astype(np.uint8) * 255
                state = "isod_sam_prelabel"
            else:
                mask = cv2.imread(str(manual), cv2.IMREAD_GRAYSCALE)
                state = "human_reviewed"

            label = cv2.imread(str(source / site / "label" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            if (mask is None or label is None or mask.shape != label.shape or
                    not np.isin(mask, (0, 255)).all() or np.any((mask > 0) & (label != 0))):
                raise ValueError(f"Invalid reference mask for {site}/{i}")
            relative = Path(site) / f"{i}.png"
            output = stage / relative
            output.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(output), mask):
                raise OSError(f"Could not save {output}")
            origin = manual if manual is not None else proposal
            records.append(dict(site=site, frame=i, path=relative.as_posix(),
                                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                shape=list(mask.shape), review_status=state,
                                source=str(origin.relative_to(data_root) if origin.is_relative_to(data_root) else origin)))
        manifest = dict(split="train", complete=not missing, frames=records, missing=missing)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(destination)
    return Counter(record["review_status"] for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    counts = export_training_masks(args.destination, args.allow_partial)
    print(dict(counts))


if __name__ == "__main__":
    main()
