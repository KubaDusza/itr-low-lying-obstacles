"""Read-only audit of isod_work before any ESANet run. Writes only into --out.

Checks, for all 2,000 frames:
  * the split: 14/3/3 sites, disjoint, every site present with frames 0..99,
  * files: RGB uint8 3-channel, depth uint16, floor label and obstacle mask, all the same size,
  * label encoding: which values the floor label and the obstacle mask actually contain,
  * provenance: is ISOD_gt exactly isod.obstacles(label)? (i.e. derived, not hand-corrected),
  * per split: obstacle / void (ignored) / background shares and invalid-depth rates,
  * the cached depth_stats.json of the U-Net runs was built from the same train sites,
and, with --esanet, pushes a train and a val frame through ESANet's own preprocessing to check
tensor shapes, depth zeros and label values. Saves a contact sheet to look at alignment.

  python check_isod_esanet.py --root <isod_work> --out <dir> [--esanet <ESANet checkout>]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import isod
from train_esanet import BACKGROUND, OBSTACLE, VOID, load_frame, make_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--esanet", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report, problems = {}, []

    with open(f"{a.root}/splits.json") as f:
        splits = json.load(f)
    sizes = {k: len(v) for k, v in splits.items()}
    report["split_sizes"] = sizes
    all_sites = [s for v in splits.values() for s in v]
    if sizes != {"train": 14, "val": 3, "test": 3} or len(set(all_sites)) != 20:
        problems.append(f"split is not 14/3/3 disjoint sites: {sizes}")
    on_disk = sorted(os.listdir(f"{a.root}/ISOD/data"))
    report["sites_on_disk_not_in_split"] = sorted(set(on_disk) - set(all_sites))

    label_values, gt_values, shapes = set(), set(), set()
    gt_differs, per_split = [], {}
    for split, sites in splits.items():
        c = {"frames": 0, "obstacle": 0, "void": 0, "background": 0, "pixels": 0,
             "depth_invalid": 0, "depth_invalid_on_obstacle": 0, "depth_max_mm": 0}
        for site in sites:
            for sub in ("rgb", "depth", "label", "mask"):  # mask/ is read by eval.py
                n = len([f for f in os.listdir(f"{a.root}/ISOD/data/{site}/{sub}") if f.endswith(".png")])
                if n != 100:
                    problems.append(f"{site}/{sub}: {n} png files, expected 100")
            for i in range(100):
                bgr = cv2.imread(f"{a.root}/ISOD/data/{site}/rgb/{i}.png", cv2.IMREAD_UNCHANGED)
                floor = cv2.imread(f"{a.root}/ISOD/data/{site}/label/{i}.png", cv2.IMREAD_UNCHANGED)
                gt = cv2.imread(f"{a.root}/ISOD_gt/{site}/{i}.png", cv2.IMREAD_UNCHANGED)
                if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                    problems.append(f"{site}/{i}: rgb {bgr.dtype} {bgr.shape}")
                if floor.ndim != 2 or gt.ndim != 2:
                    problems.append(f"{site}/{i}: label {floor.shape} / gt {gt.shape} not single-channel")
                    continue
                _, depth, label = load_frame(a.root, site, i)  # asserts uint16 depth, equal shapes
                shapes.add(depth.shape)
                label_values.update(np.unique(floor).tolist())
                gt_values.update(np.unique(gt).tolist())
                if not np.array_equal(gt > 0, isod.obstacles(floor) > 0):
                    gt_differs.append(f"{site}/{i}")
                c["frames"] += 1
                c["pixels"] += label.size
                c["obstacle"] += int((label == OBSTACLE).sum())
                c["void"] += int((label == VOID).sum())
                c["background"] += int((label == BACKGROUND).sum())
                c["depth_invalid"] += int((depth == 0).sum())
                c["depth_invalid_on_obstacle"] += int(((depth == 0) & (label == OBSTACLE)).sum())
                c["depth_max_mm"] = max(c["depth_max_mm"], int(depth.max()))
        per_split[split] = {
            "frames": c["frames"],
            "obstacle_share": c["obstacle"] / c["pixels"],
            "void_share_ignored": c["void"] / c["pixels"],
            "background_share": c["background"] / c["pixels"],
            "depth_invalid_share": c["depth_invalid"] / c["pixels"],
            "depth_invalid_share_on_obstacles": c["depth_invalid_on_obstacle"] / max(1, c["obstacle"]),
            "depth_max_mm": c["depth_max_mm"],
        }
        print(split, json.dumps(per_split[split]), flush=True)

    report.update(per_split=per_split, frame_shapes=sorted(shapes),
                  floor_label_values=sorted(label_values), gt_values=sorted(gt_values),
                  gt_frames_not_equal_to_derived_rule=len(gt_differs),
                  gt_frames_not_equal_examples=gt_differs[:10])
    if shapes != {(422, 640)}:
        problems.append(f"frame shapes {shapes}, expected only (422, 640)")
    if not set(label_values) <= {0, 255}:
        problems.append(f"floor label has values {sorted(label_values)}; make_label treats every "
                        "non-zero value as floor")
    if not set(gt_values) <= {0, 255}:
        problems.append(f"ISOD_gt has values {sorted(gt_values)}")

    if os.path.exists(f"{a.root}/depth_stats.json"):
        with open(f"{a.root}/depth_stats.json") as f:
            old = json.load(f)
        report["unet_depth_stats_same_train_sites"] = old.get("train_sites") == sorted(splits["train"])

    if a.esanet:
        report["esanet_preprocessing"] = check_preprocessing(a.esanet, a.root, splits, problems)

    contact_sheet(a.root, splits, f"{a.out}/isod_audit.png")
    report["problems"] = problems
    with open(f"{a.out}/isod_audit.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("PROBLEMS:" if problems else "no problems found", *problems, sep="\n  ")


def check_preprocessing(esanet_dir, root, splits, problems):
    """One train and one val frame through ESANet's own transforms (dummy depth stats are
    enough here: only shapes, zero-preservation and label values are checked)."""
    sys.path.insert(0, esanet_dir)
    from src.preprocessing import get_preprocessor
    out = {}
    for phase, site in (("train", splits["train"][0]), ("test", splits["val"][0])):
        rgb, depth, label = load_frame(root, site, 0)
        pre = get_preprocessor(depth_mean=2000.0, depth_std=1000.0, depth_mode="raw",
                               height=448, width=640, phase=phase)
        np.random.seed(0)
        s = pre({"image": rgb, "depth": depth.astype(np.float32), "label": label})
        info = {"image": list(s["image"].shape), "depth": list(s["depth"].shape),
                "label": list(s["label"].shape),
                "label_values": sorted(np.unique(s["label"].numpy()).astype(int).tolist()),
                "depth_exact_zeros_share": float((s["depth"] == 0).float().mean())}
        if "label_down" in s:
            info["label_down"] = {r: list(v.shape) for r, v in s["label_down"].items()}
        if info["image"] != [3, 448, 640] or info["depth"] != [1, 448, 640]:
            problems.append(f"ESANet preprocessing ({phase}) shapes {info}")
        if not set(info["label_values"]) <= {VOID, BACKGROUND, OBSTACLE}:
            problems.append(f"ESANet preprocessing ({phase}) label values {info['label_values']}")
        out[phase] = info
        print("preprocessing", phase, info, flush=True)
    return out


def contact_sheet(root, splits, path):
    """RGB | depth (colour) | ESANet label (black = void, grey = background, red = obstacle),
    one frame from each of 4 sites. Obstacles should sit on the floor in both RGB and depth."""
    rows = []
    for site in (splits["train"][0], splits["train"][7], splits["val"][0], splits["test"][0]):
        rgb, depth, label = load_frame(root, site, 50)
        d = depth.astype(np.float32)
        d8 = np.where(depth > 0, 255 - np.clip(d / 5000 * 255, 0, 255), 0).astype(np.uint8)
        dcol = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        dcol[depth == 0] = 0
        lab = np.stack([np.choose(label, [0, 128, 255]), np.choose(label, [0, 128, 0]),
                        np.choose(label, [0, 128, 0])], 2).astype(np.uint8)
        edge = cv2.dilate(cv2.Canny((label == OBSTACLE).astype(np.uint8) * 255, 0, 1), None)
        for img in (rgb, dcol):  # obstacle outline on RGB and depth to judge registration
            img[edge > 0] = (255, 255, 0)
        rows.append(np.hstack([rgb, dcol, lab]))
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("wrote", path)


if __name__ == "__main__":
    main()
