"""Ask a pretrained SAM 2.1 model for object masks at points/boxes that a PERSON chose.

Why: our v0 masks miss objects that touch walls or furniture, because the rule works on which
pixels are connected. SAM segments by appearance, so a click on such an object may give its
outline even when it touches a wall. This script only produces PROPOSALS. A person decides
whether each proposal is right and whether the object counts as an obstacle.

Input
  --isod      ISOD 'data' folder (read only)
  --prompts   CSV, one row per object: site, frame, object, kind (point|box), x1, y1, x2, y2, why
              point: (x1, y1) is a pixel on the object; box: (x1, y1)-(x2, y2) surrounds it
  --out       output folder, outside the dataset (same folder as compare_annotations.py)
Output
  <out>/sam_masks/<site>/<i>.png          union of all proposals in that frame (255 = proposed)
  <out>/sam_objects/<site>_<i>_<object>.png   one mask per prompt
  <out>/sam_log.csv                       SAM's own quality score and area per prompt

Model: facebook/sam2.1-hiera-small from Hugging Face (about 180 MB, downloaded once on first
use). Runs on the Apple GPU ("mps") if PyTorch sees it, else on the CPU.
  python sam_propose.py --isod ~/Downloads/ISOD/data --prompts sam_prompts_pilot.csv --out <folder>
Testing only: --random_weights builds the same network with random weights (no download), to
check that the code runs. Its masks are meaningless.
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Config, Sam2ImageProcessor, Sam2Model, Sam2Processor

MODEL_NAME = "facebook/sam2.1-hiera-small"


def load_sam(random_weights):
    if random_weights:
        return Sam2Model(Sam2Config()), Sam2Processor(image_processor=Sam2ImageProcessor())
    return Sam2Model.from_pretrained(MODEL_NAME), Sam2Processor.from_pretrained(MODEL_NAME)


def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_prompts(path):
    """Group the prompt rows by frame: {(site, frame): [row, row, ...]}."""
    prompts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            prompts.setdefault((row["site"], row["frame"]), []).append(row)
    return prompts


def segment_one_object(model, processor, image, row, device):
    """One prompt -> one mask. With a single click SAM is unsure how big the object is
    (a shoelace? the shoe?), so it returns 3 candidate masks with a predicted quality score;
    we keep the candidate with the highest score."""
    if row["kind"] == "point":
        x, y = float(row["x1"]), float(row["y1"])
        inputs = processor(images=image, input_points=[[[[x, y]]]], input_labels=[[[1]]],
                           return_tensors="pt")
    else:  # box: x1, y1, x2, y2
        box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    # Back to the original 640x422 image size; result: [objects, candidates, height, width]
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    scores = outputs.iou_scores[0, 0].float().cpu().numpy()  # one score per candidate
    best = int(np.argmax(scores))
    return masks[0, best].numpy().astype(bool), float(scores[best])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isod", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random_weights", action="store_true", help="code test only, no download")
    args = parser.parse_args()
    assert not os.path.abspath(args.out).startswith(os.path.abspath(args.isod)), \
        "--out must not be inside the dataset folder"

    device = choose_device()
    model, processor = load_sam(args.random_weights)
    model = model.to(device).eval()
    print(f"SAM 2.1 on {device}" + (" (RANDOM WEIGHTS: masks are meaningless)" if args.random_weights else ""))

    os.makedirs(f"{args.out}/sam_objects", exist_ok=True)
    log_rows = []
    for (site, frame), rows in read_prompts(args.prompts).items():
        image = Image.open(f"{args.isod}/{site}/rgb/{frame}.png").convert("RGB")
        union = np.zeros((image.height, image.width), dtype=bool)
        for row in rows:
            mask, score = segment_one_object(model, processor, image, row, device)
            union |= mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                f"{args.out}/sam_objects/{site}_{frame}_{row['object']}.png")
            log_rows.append({"site": site, "frame": frame, "object": row["object"],
                             "kind": row["kind"], "sam_score": round(score, 3),
                             "area_px": int(mask.sum()), "why": row["why"]})
            print(f"{site}/{frame} {row['object']}: score {score:.2f}, {int(mask.sum())} px")
        os.makedirs(f"{args.out}/sam_masks/{site}", exist_ok=True)
        Image.fromarray(union.astype(np.uint8) * 255).save(f"{args.out}/sam_masks/{site}/{frame}.png")

    with open(f"{args.out}/sam_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)


if __name__ == "__main__":
    main()
