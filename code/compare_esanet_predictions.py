"""Compare RGB and RGB-D predictions on the ISOD test set."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isod


# 1. Paths
BASE = Path("/data/cs_courses/jgomezel/esanet_pilot")
DATA = Path.home() / "Flat Obstacle Detection/datasets/isod_work"

RGB = BASE / "runs_50_rgb/esanet_r34nbt1d_rgb_s0/pred_test"
RGBD = BASE / "runs_50_rgbd/esanet_r34nbt1d_rgbd_s0/pred_test"

OUTPUT = BASE / "comparison_50"
OUTPUT.mkdir(exist_ok=True)


def read_mask(path):
    """Read a binary mask: 0 = background, 1 = obstacle."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(path)

    return (image > 0).astype(np.uint8)


def calculate_scores(prediction, ground_truth, ignore):
    """Use the project's original evaluation function."""
    return isod.score(prediction, ground_truth, ignore=ignore)


def iou(row):
    """Calculate IoU for one image."""
    return row["inter"] / max(1, row["union"])


def make_panel(row):
    """Draw RGB, reference mask and both predictions."""
    site = row["site"]
    filename = row["filename"]

    image_path = DATA / "ISOD/data" / site / "rgb" / filename

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt = read_mask(DATA / "ISOD_gt" / site / filename)
    rgb = read_mask(RGB / site / filename)
    rgbd = read_mask(RGBD / site / filename)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = [
        "Original RGB",
        "Reference v0",
        f"RGB — IoU {row['rgb_iou']:.3f}",
        f"RGB-D — IoU {row['rgbd_iou']:.3f}",
    ]

    images = [image, gt, rgb, rgbd]

    for ax, picture, title in zip(axes, images, titles):
        ax.imshow(picture, cmap="gray" if picture.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"{site}/{filename} — ISOD v0")
    fig.tight_layout()

    name = f"{site}_{Path(filename).stem}.png"
    fig.savefig(OUTPUT / name, dpi=150)
    plt.close(fig)


# 2. Evaluate all test images
results = []
rgb_scores = []
rgbd_scores = []

for prediction_path in sorted(RGB.glob("*/*.png")):
    site = prediction_path.parent.name
    filename = prediction_path.name

    rgb = read_mask(prediction_path)
    rgbd = read_mask(RGBD / site / filename)
    gt = read_mask(DATA / "ISOD_gt" / site / filename)

    floor_path = DATA / "ISOD/data" / site / "label" / filename
    floor_label = cv2.imread(str(floor_path), cv2.IMREAD_UNCHANGED)

    if floor_label is None:
        raise FileNotFoundError(floor_path)

    if not (rgb.shape == rgbd.shape == gt.shape == floor_label.shape):
        raise ValueError(f"Different image sizes: {site}/{filename}")

    # Ignore walls and other non-evaluable regions.
    ignore = isod.ignore_region(floor_label, gt)

    score_rgb = calculate_scores(rgb, gt, ignore)
    score_rgbd = calculate_scores(rgbd, gt, ignore)

    rgb_scores.append(score_rgb)
    rgbd_scores.append(score_rgbd)

    results.append({
        "site": site,
        "filename": filename,
        "rgb_iou": iou(score_rgb),
        "rgbd_iou": iou(score_rgbd),
        "rgb_union": score_rgb["union"],
        "rgbd_union": score_rgbd["union"],
        "gt_pixels": score_rgb["gt"],
    })


# 3. Verify the global results
total_rgb = isod.totals(rgb_scores)
total_rgbd = isod.totals(rgbd_scores)

print("Images evaluated:", len(results))
print("Global RGB IoU:", total_rgb["IoU"])
print("Global RGB-D IoU:", total_rgbd["IoU"])

if len(results) != 300:
    raise ValueError("Expected exactly 300 test images.")

if abs(total_rgb["IoU"] - 0.7416340646658787) > 1e-5:
    raise ValueError("RGB IoU does not match the training evaluation.")

if abs(total_rgbd["IoU"] - 0.7206495587259634) > 1e-5:
    raise ValueError("RGB-D IoU does not match the training evaluation.")


# 4. Save and rank the results
df = pd.DataFrame(results)

df["iou_difference"] = df["rgbd_iou"] - df["rgb_iou"]

df.to_csv(OUTPUT / "per_image_results.csv", index=False)

# Only rank images containing labeled obstacles.
with_obstacles = df[df["gt_pixels"] > 0]

best_rgbd = with_obstacles.nlargest(5, "iou_difference")
best_rgb = with_obstacles.nsmallest(5, "iou_difference")

print("\nImages where RGB-D improves most:")
print(best_rgbd[["site", "filename", "rgb_iou", "rgbd_iou"]])

print("\nImages where RGB improves most:")
print(best_rgb[["site", "filename", "rgb_iou", "rgbd_iou"]])

# Aggregate inter/union counts by site, as in the original evaluation.
site_rows = []
for site in sorted(df["site"].unique()):
    indices = df.index[df["site"] == site]
    site_rgb = isod.totals([rgb_scores[i] for i in indices])
    site_rgbd = isod.totals([rgbd_scores[i] for i in indices])

    site_rows.append({
        "site": site,
        "rgb_iou": site_rgb["IoU"],
        "rgbd_iou": site_rgbd["IoU"],
    })

pd.DataFrame(site_rows).to_csv(
    OUTPUT / "per_site_results.csv", index=False
)

# Save visual comparisons for the most different cases.
selected = pd.concat([best_rgbd, best_rgb]).drop_duplicates(
    subset=["site", "filename"]
)

for _, row in selected.iterrows():
    make_panel(row)

print("\nResults saved in:", OUTPUT)
