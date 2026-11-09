"""Prepare a human review of the ISOD obstacle masks. Reads the dataset, never changes it.

Why
---
Our obstacle masks (ISOD_gt) were made by a rule, not drawn by hand (isod.obstacles): in ISOD's
"not drivable" region, the largest connected piece is assumed to be walls + furniture and is
removed, blobs under 20 px are removed, and what is left counts as obstacles. Nobody has yet
checked that this worked in every image. This script does NOT fix anything. It points a human
reviewer to the places where the rule is most likely to have failed, and also picks an unbiased
sample of frames so that we can estimate how often it fails overall.

What it looks for (each is a hint, not a verdict: a person decides)
--------------------------------------------------------------------
  wall_bump       a small part sticking out of the big wall region into the floor. An object
                  lying against a wall or a chair leg joins the wall region and is removed with
                  it, so it is MISSING from the masks. Found by "opening" the wall region (see
                  find_wall_bumps); ordinary walls and corners also trigger it sometimes.
  border_obstacle an obstacle touching the image edge. Can be a real object cut by the frame,
                  or a piece of wall/furniture that was not connected to the big wall region and
                  was wrongly KEPT as an obstacle.
  large_obstacle  an obstacle bigger than LARGE_OBSTACLE_PX: more likely furniture than a floor item.
  small_obstacle  an obstacle smaller than SMALL_OBSTACLE_PX: real, or label noise?
  dropped_blob    a not-drivable blob under 20 px that the rule threw away: noise, or a screw?

Depth is deliberately NOT used to find suspects. Depth is one of the inputs we are comparing; if
it decided which masks get reviewed (and maybe corrected), the corrected masks would favour what
depth can see, and the RGB vs depth comparison would no longer be fair.

Two review samples
------------------
  systematic  frames 0, 20, 40, 60, 80 of every site = 100 frames, chosen before looking at
              anything. Only this sample can estimate the overall error rate.
  flagged     the frames that score highest on each hint (a few per site at most). Good for
              finding the TYPES of error, but biased towards problems: never use it for rates.

Outputs (in --out)
------------------
  frames.csv              all 2,000 frames: split, hint counts and pixel areas
  summary.json            totals per split, thresholds used, mask fingerprint
  review_images/*.jpg     one picture per frame to review (see draw_review_image for colours)
  review_log.csv          the sheet the reviewer fills in (verdict columns empty)

  python review_isod_masks.py --root <isod_work> --out <folder>
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import isod
from train_esanet import ground_truth_fingerprint, read_frame

# Thresholds. They only decide which frames a human looks at, never what counts as an error.
WALL_BUMP_OPENING_PX = 31    # parts of the wall region narrower than ~31 px count as "bumps"
LARGE_OBSTACLE_PX = 5000     # about 2 % of a 640x422 frame
SMALL_OBSTACLE_PX = 144      # the "small" tercile boundary of the earlier object analysis
SYSTEMATIC_FRAMES = [0, 20, 40, 60, 80]
FLAGGED_PER_HINT = 30        # frames per hint in the flagged sample
MAX_FLAGGED_PER_SITE = 2     # per hint, so that the flagged sample covers many scenes


def find_wall_region_and_dropped_blobs(floor_label):
    """Repeat the rule's first step: split "not drivable" into connected pieces.
    Returns the largest piece (what the rule calls walls + furniture) and the pieces under
    isod.MIN_AREA pixels (what the rule throws away as noise)."""
    not_drivable = (floor_label == 0).astype(np.uint8)
    n, pieces, stats, _ = cv2.connectedComponentsWithStats(not_drivable, connectivity=8)
    wall = np.zeros(floor_label.shape, dtype=bool)
    dropped = np.zeros(floor_label.shape, dtype=bool)
    if n <= 1:  # nothing is "not drivable" in this frame
        return wall, dropped
    areas = stats[1:, cv2.CC_STAT_AREA]       # piece 0 is the drivable floor itself
    wall_id = 1 + int(np.argmax(areas))
    wall = pieces == wall_id
    for k in range(1, n):
        if k != wall_id and stats[k, cv2.CC_STAT_AREA] < isod.MIN_AREA:
            dropped |= pieces == k
    return wall, dropped


def find_wall_bumps(wall):
    """Morphological "opening" = shrink the region, then grow it back by the same amount.
    Parts narrower than the brush (WALL_BUMP_OPENING_PX) cannot survive the shrinking, so they
    do not come back: wall minus opened wall = the thin parts that stick out. A real wall is
    wide and survives. Bumps touching the image border are ignored (usually wall edges)."""
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (WALL_BUMP_OPENING_PX, WALL_BUMP_OPENING_PX))
    opened = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, brush) > 0
    thin_parts = (wall & ~opened).astype(np.uint8)

    bumps = np.zeros(wall.shape, dtype=bool)
    n, pieces, stats, _ = cv2.connectedComponentsWithStats(thin_parts, connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= isod.MIN_AREA and not touches_border(stats[k], *wall.shape):
            bumps |= pieces == k
    return bumps


def touches_border(component_stats, height=422, width=640):
    x, y = component_stats[cv2.CC_STAT_LEFT], component_stats[cv2.CC_STAT_TOP]
    w, h = component_stats[cv2.CC_STAT_WIDTH], component_stats[cv2.CC_STAT_HEIGHT]
    return x == 0 or y == 0 or x + w >= width or y + h >= height


def analyse_frame(floor_label, obstacle):
    """All hints for one frame. Returns a dict of numbers (for frames.csv) and of masks
    (for the picture)."""
    wall, dropped = find_wall_region_and_dropped_blobs(floor_label)
    bumps = find_wall_bumps(wall)

    border = np.zeros(obstacle.shape, dtype=bool)
    large = np.zeros(obstacle.shape, dtype=bool)
    small = np.zeros(obstacle.shape, dtype=bool)
    n, pieces, stats, _ = cv2.connectedComponentsWithStats(obstacle.astype(np.uint8), connectivity=8)
    n_border = n_large = n_small = 0
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if touches_border(stats[k], *obstacle.shape):
            border |= pieces == k
            n_border += 1
        if area >= LARGE_OBSTACLE_PX:
            large |= pieces == k
            n_large += 1
        if area < SMALL_OBSTACLE_PX:
            small |= pieces == k
            n_small += 1

    n_bumps = cv2.connectedComponents(bumps.astype(np.uint8), connectivity=8)[0] - 1
    n_dropped = cv2.connectedComponents(dropped.astype(np.uint8), connectivity=8)[0] - 1
    numbers = {
        "n_obstacles": n - 1,
        "obstacle_px": int(obstacle.sum()),
        "wall_bump": n_bumps, "wall_bump_px": int(bumps.sum()),
        "border_obstacle": n_border, "border_obstacle_px": int(border.sum()),
        "large_obstacle": n_large, "large_obstacle_px": int(large.sum()),
        "small_obstacle": n_small,
        "dropped_blob": n_dropped, "dropped_blob_px": int(dropped.sum()),
        "mask_equals_rule": bool(np.array_equal(obstacle, isod.obstacles(floor_label) > 0)),
    }
    masks = {"wall": wall, "bumps": bumps, "border": border, "large": large, "dropped": dropped}
    return numbers, masks


def draw_review_image(rgb, obstacle, masks, title):
    """Left: the plain photo (to judge what is really there). Right: the same photo with
      green fill + outline  = obstacle in the current masks
      darkened             = the removed wall/furniture region (ignored in training)
      red circle           = wall_bump (maybe an object lost inside the wall region)
      orange outline       = border_obstacle or large_obstacle (maybe wall/furniture kept)
      magenta circle       = dropped_blob (tiny piece the rule removed)
    Circles are drawn around small hints so the eye finds them quickly."""
    overlay = rgb.copy()
    overlay[masks["wall"]] = (overlay[masks["wall"]] * 0.5).astype(np.uint8)
    overlay[obstacle] = (0.5 * overlay[obstacle] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    draw_outline(overlay, obstacle, (0, 255, 0), thickness=1)
    draw_outline(overlay, masks["border"] | masks["large"], (255, 140, 0), thickness=2)
    draw_circles(overlay, masks["bumps"], (255, 0, 0))
    draw_circles(overlay, masks["dropped"], (255, 0, 255))

    picture = np.hstack([rgb, overlay])
    cv2.rectangle(picture, (0, 0), (picture.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(picture, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return picture


def draw_outline(image, mask, colour, thickness):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, thickness)


def draw_circles(image, mask, colour):
    """A circle around every separate blob of the mask, a bit larger than the blob."""
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        radius = 8 + max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) // 2
        centre = (int(centres[k][0]), int(centres[k][1]))
        cv2.circle(image, centre, radius, colour, 2)


def choose_flagged_frames(rows, already_chosen):
    """For each hint, the frames with the largest flagged area, at most MAX_FLAGGED_PER_SITE
    per site, FLAGGED_PER_HINT in total. Sorting by area (ties by name) makes it repeatable."""
    chosen = []
    for hint, key in (("wall_bump", "wall_bump_px"), ("border_obstacle", "border_obstacle_px"),
                      ("large_obstacle", "large_obstacle_px"), ("dropped_blob", "dropped_blob_px"),
                      ("small_obstacle", "small_obstacle")):
        candidates = sorted((r for r in rows if r[key] > 0),
                            key=lambda r: (-r[key], r["site"], r["frame"]))
        per_site = {}
        for r in candidates:
            name = (r["site"], r["frame"])
            if name in already_chosen or per_site.get(r["site"], 0) >= MAX_FLAGGED_PER_SITE:
                continue
            chosen.append((f"flagged:{hint}", r))
            already_chosen.add(name)
            per_site[r["site"]] = per_site.get(r["site"], 0) + 1
            if len([c for c in chosen if c[0] == f"flagged:{hint}"]) >= FLAGGED_PER_HINT:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="isod_work folder")
    parser.add_argument("--out", required=True, help="folder for the review package")
    parser.add_argument("--gt_folder", default="ISOD_gt", help="mask folder inside --root")
    args = parser.parse_args()
    os.makedirs(f"{args.out}/review_images", exist_ok=True)

    with open(f"{args.root}/splits.json") as f:
        splits = json.load(f)
    split_of = {}
    for part, sites in splits.items():
        for site in sites:
            split_of[site] = part
    all_sites = sorted(split_of)

    # 1. Hints for all 2,000 frames.
    rows = []
    for site in all_sites:
        for i in range(100):
            _, _, floor_label, obstacle = read_frame(args.root, site, i, args.gt_folder)
            numbers, _ = analyse_frame(floor_label, obstacle)
            rows.append({"site": site, "split": split_of[site], "frame": i, **numbers})
        print("analysed", site, flush=True)
    with open(f"{args.out}/frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 2. The two review samples.
    systematic = [("systematic", r) for r in rows if r["frame"] in SYSTEMATIC_FRAMES]
    already_chosen = {(r["site"], r["frame"]) for _, r in systematic}
    review = systematic + choose_flagged_frames(rows, already_chosen)

    # 3. One picture per reviewed frame, and the empty review sheet.
    hint_names = ["wall_bump", "border_obstacle", "large_obstacle", "small_obstacle", "dropped_blob"]
    log_rows = []
    for number, (sample, r) in enumerate(review, start=1):
        rgb, _, floor_label, obstacle = read_frame(args.root, r["site"], r["frame"], args.gt_folder)
        _, masks = analyse_frame(floor_label, obstacle)
        hints = ";".join(h for h in hint_names if r[h] > 0) or "none"
        image_name = f"{number:03d}_{r['site']}_{r['frame']}.jpg"
        title = f"#{number}  {r['site']} frame {r['frame']}  ({r['split']})  {sample}  hints: {hints}"
        picture = draw_review_image(rgb, obstacle, masks, title)
        cv2.imwrite(f"{args.out}/review_images/{image_name}", cv2.cvtColor(picture, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        log_rows.append({"review_id": number, "sample": sample, "site": r["site"],
                         "split": r["split"], "frame": r["frame"], "auto_hints": hints,
                         "image": image_name, "verdict": "", "severity": "", "cause": "",
                         "where": "", "comment": "", "reviewer": "", "date": ""})
    with open(f"{args.out}/review_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)

    # 4. Summary: totals per split, the settings, and which masks were reviewed.
    summary = {"gt_folder": args.gt_folder,
               "gt_fingerprint": ground_truth_fingerprint(args.root, args.gt_folder, all_sites),
               "thresholds": {"WALL_BUMP_OPENING_PX": WALL_BUMP_OPENING_PX,
                              "LARGE_OBSTACLE_PX": LARGE_OBSTACLE_PX,
                              "SMALL_OBSTACLE_PX": SMALL_OBSTACLE_PX,
                              "MIN_AREA (isod.py)": isod.MIN_AREA},
               "review_frames": {"systematic": len(systematic), "flagged": len(review) - len(systematic)},
               "per_split": {}}
    for part in ("train", "val", "test"):
        part_rows = [r for r in rows if r["split"] == part]
        summary["per_split"][part] = {
            "frames": len(part_rows),
            "frames_masks_differ_from_rule": sum(not r["mask_equals_rule"] for r in part_rows),
            **{f"frames_with_{h}": sum(r[h] > 0 for r in part_rows) for h in hint_names},
        }
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"{len(review)} review images in {args.out}/review_images; fill in {args.out}/review_log.csv")


if __name__ == "__main__":
    main()
