"""Check the ISOD working copy before any ESANet training. Reads the dataset, never changes it.

Why: every later number depends on the data being what we believe it is. This script turns
those beliefs into checks, over all 2,000 frames:

  1. the split: 14 / 3 / 3 sites, no site in two parts, every site with frames 0..99;
  2. the files: 8-bit colour RGB, 16-bit depth, single-channel labels, all the same size
     (read_frame in train_esanet.py raises an error otherwise), and the mask/ folder eval uses;
  3. the label encoding: which values ISOD's floor label and our obstacle mask really contain;
  4. where the obstacle masks come from: are they exactly isod.obstacles(label), i.e. the
     automatic rule, with no manual corrections?
  5. per split: share of obstacle, ignored (void) and background pixels, and of missing depth;
  6. optionally (--esanet) one training and one validation frame through ESANet's own
     preprocessing, to check tensor sizes, label values and that missing depth stays 0;
  7. a picture (isod_audit.png) to check by eye that depth lines up with the RGB image.

  python check_isod_esanet.py --root <isod_work> --out <folder for the report> [--esanet <ESANet folder>]

Writes <out>/isod_audit.json and <out>/isod_audit.png.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import isod
from train_esanet import BACKGROUND, OBSTACLE, VOID, make_esanet_label, read_frame


def check_split(root, splits, problems):
    """Part 1: sizes of the split, overlaps, and the number of files per site."""
    sizes = {part: len(sites) for part, sites in splits.items()}
    all_sites = splits["train"] + splits["val"] + splits["test"]
    if sizes != {"train": 14, "val": 3, "test": 3} or len(set(all_sites)) != 20:
        problems.append(f"the split is not 14/3/3 different sites: {sizes}")

    for site in all_sites:
        for folder in ("rgb", "depth", "label", "mask"):  # mask/ is read by eval.py
            files = [f for f in os.listdir(f"{root}/ISOD/data/{site}/{folder}") if f.endswith(".png")]
            if len(files) != 100:
                problems.append(f"{site}/{folder} has {len(files)} png files, expected 100")
        gt_files = [f for f in os.listdir(f"{root}/ISOD_gt/{site}") if f.endswith(".png")]
        if len(gt_files) != 100:
            problems.append(f"ISOD_gt/{site} has {len(gt_files)} png files, expected 100")

    sites_on_disk = os.listdir(f"{root}/ISOD/data")
    unused = sorted(set(sites_on_disk) - set(all_sites))
    return sizes, unused


def check_frames(root, sites):
    """Parts 2-5 for the frames of one split. read_frame() stops with a clear error if a file
    is missing or has the wrong type or size, so reaching the end already means part 2 passed."""
    frame_sizes = set()
    floor_label_values = set()
    obstacle_mask_values = set()
    frames_not_matching_rule = []
    n_frames = n_pixels = 0
    n_obstacle = n_void = n_background = 0
    n_depth_missing = n_depth_missing_on_obstacles = 0
    max_depth = 0

    for site in sites:
        for i in range(100):
            rgb, depth, floor_label, obstacle = read_frame(root, site, i)
            obstacle_png = cv2.imread(f"{root}/ISOD_gt/{site}/{i}.png", cv2.IMREAD_UNCHANGED)
            label = make_esanet_label(floor_label, obstacle)

            frame_sizes.add(depth.shape)
            floor_label_values.update(np.unique(floor_label).tolist())
            obstacle_mask_values.update(np.unique(obstacle_png).tolist())

            # Recompute the obstacle mask with the automatic rule and compare pixel by pixel.
            recomputed = isod.obstacles(floor_label) > 0
            if not np.array_equal(obstacle, recomputed):
                frames_not_matching_rule.append(f"{site}/{i}")

            n_frames += 1
            n_pixels += label.size
            n_obstacle += int((label == OBSTACLE).sum())
            n_void += int((label == VOID).sum())
            n_background += int((label == BACKGROUND).sum())
            n_depth_missing += int((depth == 0).sum())
            n_depth_missing_on_obstacles += int(((depth == 0) & obstacle).sum())
            max_depth = max(max_depth, int(depth.max()))

    summary = {
        "frames": n_frames,
        "obstacle_share": n_obstacle / n_pixels,
        "void_share_ignored": n_void / n_pixels,
        "background_share": n_background / n_pixels,
        "depth_missing_share": n_depth_missing / n_pixels,
        "depth_missing_share_on_obstacles": n_depth_missing_on_obstacles / max(1, n_obstacle),
        "depth_max_mm": max_depth,
    }
    return summary, frame_sizes, floor_label_values, obstacle_mask_values, frames_not_matching_rule


def check_esanet_preprocessing(esanet_folder, root, splits, problems):
    """Part 6: push one training and one validation frame through ESANet's transforms.
    The depth mean/std here are placeholders: we only check sizes and values, not training."""
    sys.path.insert(0, esanet_folder)  # make ESANet's "src" package importable
    from src.preprocessing import get_preprocessor

    results = {}
    for phase, site in (("train", splits["train"][0]), ("test", splits["val"][0])):
        rgb, depth, floor_label, obstacle = read_frame(root, site, 0)
        preprocess = get_preprocessor(depth_mean=2000.0, depth_std=1000.0, depth_mode="raw",
                                      height=448, width=640, phase=phase)
        np.random.seed(0)  # the training transforms are random; make this check repeatable
        sample = preprocess({"image": rgb, "depth": depth.astype(np.float32),
                             "label": make_esanet_label(floor_label, obstacle)})

        info = {
            "image_shape": list(sample["image"].shape),
            "depth_shape": list(sample["depth"].shape),
            "label_shape": list(sample["label"].shape),
            "label_values": sorted(int(v) for v in np.unique(sample["label"].numpy())),
            "share_of_depth_exactly_zero": float((sample["depth"] == 0).float().mean()),
        }
        if "label_down" in sample:  # only in training: the smaller labels for the side outputs
            info["small_label_shapes"] = {rate: list(small.shape)
                                          for rate, small in sample["label_down"].items()}
        if info["image_shape"] != [3, 448, 640] or info["depth_shape"] != [1, 448, 640]:
            problems.append(f"ESANet preprocessing ({phase}) gave unexpected shapes: {info}")
        if not set(info["label_values"]) <= {VOID, BACKGROUND, OBSTACLE}:
            problems.append(f"ESANet preprocessing ({phase}) gave label values {info['label_values']}")
        print("ESANet preprocessing,", phase, info, flush=True)
        results[phase] = info
    return results


def save_contact_sheet(root, splits, path):
    """Part 7: one row per site, frame 50: RGB | depth in colour | label.
    Label colours: black = ignored, grey = background, red = obstacle. The obstacle outline is
    drawn in yellow on the RGB and on the depth image: if depth is registered to RGB, the
    outline must sit on the objects in both."""
    label_colours = np.array([[0, 0, 0], [128, 128, 128], [255, 0, 0]], dtype=np.uint8)
    rows = []
    for site in (splits["train"][0], splits["train"][7], splits["val"][0], splits["test"][0]):
        rgb, depth, floor_label, obstacle = read_frame(root, site, 50)
        label = make_esanet_label(floor_label, obstacle)

        # Depth to colour: near = warm, far (5 m and more) = cold, missing = black.
        depth_8bit = 255 - np.clip(depth / 5000 * 255, 0, 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
        depth_colour = cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB)
        depth_colour[depth == 0] = 0

        label_colour = label_colours[label]  # each pixel value 0/1/2 picks a colour row

        outline = cv2.Canny(obstacle.astype(np.uint8) * 255, 100, 200) > 0
        rgb[outline] = (255, 255, 0)
        depth_colour[outline] = (255, 255, 0)
        rows.append(np.hstack([rgb, depth_colour, label_colour]))
    sheet = np.vstack(rows)
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="isod_work folder")
    parser.add_argument("--out", required=True, help="folder for isod_audit.json and .png")
    parser.add_argument("--esanet", default=None, help="ESANet folder, to check its preprocessing")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    problems = []  # every check that fails adds a sentence here
    with open(f"{args.root}/splits.json") as f:
        splits = json.load(f)
    split_sizes, unused_sites = check_split(args.root, splits, problems)
    report = {"split_sizes": split_sizes, "sites_on_disk_not_in_split": unused_sites,
              "per_split": {}}

    all_sizes, all_floor_values, all_obstacle_values, all_mismatches = set(), set(), set(), []
    for part in ("train", "val", "test"):
        summary, sizes, floor_values, obstacle_values, mismatches = check_frames(args.root, splits[part])
        print(part, json.dumps(summary), flush=True)
        report["per_split"][part] = summary
        all_sizes |= sizes
        all_floor_values |= floor_values
        all_obstacle_values |= obstacle_values
        all_mismatches += mismatches

    report["frame_sizes"] = sorted(all_sizes)
    report["floor_label_values"] = sorted(all_floor_values)
    report["obstacle_mask_values"] = sorted(all_obstacle_values)
    report["frames_where_mask_differs_from_automatic_rule"] = len(all_mismatches)
    report["examples_of_differing_frames"] = all_mismatches[:10]

    if all_sizes != {(422, 640)}:
        problems.append(f"frame sizes {sorted(all_sizes)}, expected only 422 x 640")
    if not all_floor_values <= {0, 255}:
        problems.append(f"floor label contains {sorted(all_floor_values)}; make_esanet_label "
                        "treats every non-zero value as floor")
    if not all_obstacle_values <= {0, 255}:
        problems.append(f"obstacle masks contain {sorted(all_obstacle_values)}, expected 0 and 255")

    # The U-Net runs cached their own depth statistics; check they used the same train sites.
    if os.path.exists(f"{args.root}/depth_stats.json"):
        with open(f"{args.root}/depth_stats.json") as f:
            unet_stats = json.load(f)
        report["unet_depth_stats_used_same_train_sites"] = \
            unet_stats.get("train_sites") == sorted(splits["train"])

    if args.esanet:
        report["esanet_preprocessing"] = check_esanet_preprocessing(args.esanet, args.root,
                                                                    splits, problems)
    save_contact_sheet(args.root, splits, f"{args.out}/isod_audit.png")

    report["problems"] = problems
    with open(f"{args.out}/isod_audit.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print("  -", p)
    else:
        print("No problems found.")


if __name__ == "__main__":
    main()
