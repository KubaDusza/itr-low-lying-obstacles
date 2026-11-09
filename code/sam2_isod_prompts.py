"""SAM 2.1 proposals for chosen ISOD objects, next to Jakub's v0 obstacle mask.

What it does, for every row of the prompt CSV (one object, chosen by a person):
  1. gives SAM the RGB image and ONE prompt: a point on the object, or a box around it;
  2. keeps ALL THREE candidate masks SAM returns, with SAM's own quality score for each;
  3. draws a close-up: RGB with the prompt | candidate 1 | candidate 2 | candidate 3 | v0 mask.
For every frame it also draws RGB | ISOD floor label | v0 | union of the highest-scored candidates.

Nothing in the dataset is changed. SAM's masks are PROPOSALS: a person decides if they are right.

Input (all read-only)
  --sam2     your SAM 2 folder (code + checkpoints/sam2.1_hiera_small.pt)
  --isod     ISOD 'data' folder with <site>/rgb and <site>/label
  --prompts  CSV: site, frame, object, kind (point|box), x1, y1, x2, y2, why
Output (in --out, which must be outside the dataset)
  rgb_original/<site>_<frame>.png        exact copy of the input photo
  v0_masks/<site>_<frame>.png            isod.obstacles() of the floor label (255 = obstacle)
  sam_candidates/<site>_<frame>_<object>_c<k>.png   candidate k = 1, 2, 3 (255 = proposed)
  sam_best_union/<site>_<frame>.png      union of each prompt's highest-scored candidate
  sam_scores.csv                         score, area and "highest" flag for every candidate
  figures/prompt_<site>_<frame>_<object>.png and figures/frame_<site>_<frame>.png

  PYTORCH_ENABLE_MPS_FALLBACK=1 python sam2_isod_prompts.py --sam2 ~/sam2 \
      --isod ~/Downloads/ISOD/data --prompts sam_prompts.csv --out <folder> --only HB_LM/0
"""
import argparse
import csv
import os
import shutil

import cv2
import numpy as np
import torch

import isod  # code/isod.py: isod.obstacles() is the rule that made the v0 masks

CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"   # SAM 2.1 "small" architecture
CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
GREEN, MAGENTA, RED = (0, 255, 0), (255, 0, 255), (255, 0, 0)


# --------------------------------------------------------------------------------- SAM
def load_predictor(sam2_folder, device, random_weights):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    checkpoint = None if random_weights else os.path.join(sam2_folder, CHECKPOINT)
    model = build_sam2(CONFIG, checkpoint, device=device, apply_postprocessing=False)
    return SAM2ImagePredictor(model)


def masks_to_boolean(masks, image_shape):
    """SAM2ImagePredictor.predict(return_logits=False) thresholds the masks itself
    (mask logit > 0) and then returns them as float32 numbers 0.0 / 1.0, shape (3, H, W).
    NumPy can only use True/False (or integer) arrays to select pixels, which is why
    `colour[mask] = ...` failed. We check that the values really are only 0 and 1 (and not
    raw logits, which would need a different treatment) before turning them into True/False."""
    assert masks.ndim == 3 and masks.shape[1:] == image_shape[:2], f"unexpected shape {masks.shape}"
    values = set(np.unique(masks).tolist())
    assert values <= {0.0, 1.0}, f"masks are not 0/1 (values {sorted(values)[:5]}...): logits?"
    return masks > 0.5


def ask_sam(predictor, row):
    """One prompt -> 3 candidate masks + SAM's predicted quality score (IoU estimate) for each.
    Three candidates because one click is ambiguous: part of the object, the object, or the
    object plus its surroundings."""
    if row["kind"] == "point":
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[float(row["x1"]), float(row["y1"])]]),  # (x, y) in pixels
            point_labels=np.array([1]),                                      # 1 = "this is the object"
            multimask_output=True)
    else:
        box = np.array([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks, scores


# --------------------------------------------------------------------------------- drawing
def overlay(rgb, mask, colour):
    picture = rgb.copy()
    picture[mask] = (0.5 * picture[mask] + 0.5 * np.array(colour)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(picture, contours, -1, colour, 1)
    return picture


def draw_prompt(picture, row, scale=1, x0=0, y0=0):
    def to_view(x, y):
        return int((float(x) - x0) * scale), int((float(y) - y0) * scale)
    if row["kind"] == "point":
        cv2.drawMarker(picture, to_view(row["x1"], row["y1"]), RED, cv2.MARKER_CROSS, 16, 2)
    else:
        cv2.rectangle(picture, to_view(row["x1"], row["y1"]), to_view(row["x2"], row["y2"]), RED, 2)


def titled(picture, title):
    picture = picture.copy()
    cv2.rectangle(picture, (0, 0), (picture.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(picture, title, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return picture


def close_up_window(row, width=160, height=120):
    """A 160x120 window around the prompt (inside the 640x422 image)."""
    if row["kind"] == "point":
        cx, cy = float(row["x1"]), float(row["y1"])
    else:
        cx = (float(row["x1"]) + float(row["x2"])) / 2
        cy = (float(row["y1"]) + float(row["y2"])) / 2
    x0 = int(min(max(cx - width / 2, 0), 640 - width))
    y0 = int(min(max(cy - height / 2, 0), 422 - height))
    return x0, y0, x0 + width, y0 + height


def prompt_figure(rgb, v0, candidates, scores, row, scale=2):
    """Close-up: RGB+prompt | candidate 1 | candidate 2 | candidate 3 | v0 (all enlarged 2x)."""
    x0, y0, x1, y1 = close_up_window(row)

    def zoom(picture):
        return cv2.resize(picture[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    first = zoom(rgb)
    draw_prompt(first, row, scale, x0, y0)
    panels = [titled(first, f"{row['site']}/{row['frame']} {row['object']}"[:44])]
    best = int(np.argmax(scores))
    for k in range(3):
        mark = " HIGHEST" if k == best else ""
        picture = zoom(overlay(rgb, candidates[k], MAGENTA))
        panels.append(titled(picture, f"SAM c{k + 1}: score {scores[k]:.2f}, {int(candidates[k].sum())} px{mark}"))
    panels.append(titled(zoom(overlay(rgb, v0, GREEN)), "v0 (Jakub's rule)"))
    return np.hstack(panels)


def frame_figure(rgb, floor_label, v0, sam_union, rows):
    with_prompts = rgb.copy()
    for row in rows:
        draw_prompt(with_prompts, row)
    return np.hstack([titled(with_prompts, "RGB + prompts"),
                      titled(np.stack([floor_label] * 3, axis=2), "ISOD floor label (white = drivable)"),
                      titled(overlay(rgb, v0, GREEN), "v0 obstacles (Jakub's rule)"),
                      titled(overlay(rgb, sam_union, MAGENTA), "SAM: highest-scored candidates (proposal)")])


def save_rgb(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2", required=True, help="SAM 2 folder, e.g. ~/sam2")
    parser.add_argument("--isod", required=True, help="ISOD 'data' folder")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="limit to frames, e.g. HB_LM/0")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--random_weights", action="store_true",
                        help="code test only: SAM with random weights, masks are meaningless")
    args = parser.parse_args()
    isod_folder = os.path.abspath(os.path.expanduser(args.isod))
    out = os.path.abspath(os.path.expanduser(args.out))
    assert not out.startswith(isod_folder), "--out must not be inside the dataset folder"
    for sub in ("rgb_original", "v0_masks", "sam_candidates", "sam_best_union", "figures"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)

    prompts = {}
    with open(args.prompts) as f:
        for row in csv.DictReader(f):
            name = f"{row['site']}/{row['frame']}"
            if args.only is None or name in args.only:
                prompts.setdefault(name, []).append(row)
    assert prompts, "no prompts selected"

    predictor = load_predictor(os.path.expanduser(args.sam2), args.device, args.random_weights)
    print(f"SAM 2.1 small on {args.device}" + ("  (RANDOM WEIGHTS: test only)" if args.random_weights else ""))

    score_rows = []
    for name, rows in prompts.items():
        site, frame = name.split("/")
        rgb_path = f"{isod_folder}/{site}/rgb/{frame}.png"
        rgb = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # SAM wants RGB
        floor_label = cv2.imread(f"{isod_folder}/{site}/label/{frame}.png", cv2.IMREAD_UNCHANGED)
        v0 = isod.obstacles(floor_label) > 0
        shutil.copyfile(rgb_path, f"{out}/rgb_original/{site}_{frame}.png")  # unmodified copy
        cv2.imwrite(f"{out}/v0_masks/{site}_{frame}.png", v0.astype(np.uint8) * 255)

        predictor.set_image(rgb)  # computes the image features once; every prompt reuses them
        union = np.zeros(v0.shape, dtype=bool)
        for row in rows:
            raw_masks, scores = ask_sam(predictor, row)
            raw_info = f"{raw_masks.dtype}, shape {raw_masks.shape}, values {sorted(set(np.unique(raw_masks).tolist()))}"
            candidates = masks_to_boolean(raw_masks, rgb.shape)
            best = int(np.argmax(scores))
            union |= candidates[best]
            for k in range(3):
                cv2.imwrite(f"{out}/sam_candidates/{site}_{frame}_{row['object']}_c{k + 1}.png",
                            candidates[k].astype(np.uint8) * 255)
                score_rows.append({"site": site, "frame": frame, "object": row["object"],
                                   "kind": row["kind"], "candidate": k + 1,
                                   "sam_score": round(float(scores[k]), 3),
                                   "area_px": int(candidates[k].sum()),
                                   "highest_score": "yes" if k == best else "",
                                   "pixels_also_in_v0": int((candidates[k] & v0).sum()),
                                   "why": row["why"]})
            figure = prompt_figure(rgb, v0, candidates, scores, row)
            save_rgb(f"{out}/figures/prompt_{site}_{frame}_{row['object']}.png", figure)
            print(f"{name} {row['object']}: raw masks {raw_info}; scores "
                  + ", ".join(f"{s:.2f}" for s in scores) + f"; highest = c{best + 1}")
        cv2.imwrite(f"{out}/sam_best_union/{site}_{frame}.png", union.astype(np.uint8) * 255)
        save_rgb(f"{out}/figures/frame_{site}_{frame}.png", frame_figure(rgb, floor_label, v0, union, rows))

    with open(f"{out}/sam_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"done: {len(score_rows) // 3} prompts, results in {out}")


if __name__ == "__main__":
    main()
