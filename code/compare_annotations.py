"""Side-by-side look at ISOD annotations for a few frames. Reads the dataset, never changes it.

For each frame it writes, into --out (a folder OUTSIDE the dataset):
  v0_masks/<site>/<i>.png      Jakub's obstacle mask, made with isod.obstacles() (255 = obstacle)
  panels/<site>_<i>.png        4 columns: RGB | ISOD floor label | v0 obstacles | SAM proposal
                               (the SAM column says "no SAM proposal yet" until a mask exists in
                               <out>/sam_masks/<site>/<i>.png)

Every v0 obstacle gets a number in the third column, so a written review can point at "v0 #7".
These are pictures for a human to judge; nothing here decides what is correct.

  python compare_annotations.py --isod ~/Downloads/ISOD/data --out <folder> HB_LM/0 HL_LM/40
"""
import argparse
import os

import cv2
import numpy as np

import isod  # our project's helper; isod.obstacles() is the rule that made the v0 masks


def make_v0_mask(floor_label):
    """Exactly the rule used for ISOD_gt: largest not-drivable piece removed, blobs < 20 px removed."""
    return isod.obstacles(floor_label) > 0


def outline(image, mask, colour, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, colour, thickness)


def overlay_mask(rgb, mask, colour, number_the_objects=False):
    """Tint the mask's pixels on top of the photo and outline them; optionally number each blob."""
    picture = rgb.copy()
    picture[mask] = (0.5 * picture[mask] + 0.5 * np.array(colour)).astype(np.uint8)
    outline(picture, mask, colour)
    if number_the_objects:
        n, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x = int(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH])
            y = int(stats[k, cv2.CC_STAT_TOP])
            cv2.putText(picture, str(k), (min(x + 2, 620), max(y + 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return picture


def with_title(picture, title):
    picture = picture.copy()
    cv2.rectangle(picture, (0, 0), (picture.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(picture, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return picture


def make_panel(rgb, floor_label, v0, sam):
    """RGB | floor label (white = drivable) | v0 in green, numbered | SAM in magenta (if any)."""
    floor_picture = np.stack([floor_label] * 3, axis=2)
    v0_picture = overlay_mask(rgb, v0, (0, 255, 0), number_the_objects=True)
    if sam is None:
        sam_picture = (rgb * 0.3).astype(np.uint8)
        cv2.putText(sam_picture, "no SAM proposal yet", (200, 215), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    else:
        sam_picture = overlay_mask(rgb, sam, (255, 0, 255), number_the_objects=True)
    return np.hstack([with_title(rgb, "RGB"),
                      with_title(floor_picture, "ISOD floor label (white = drivable)"),
                      with_title(v0_picture, f"v0 obstacles (Jakub's rule): {count(v0)}"),
                      with_title(sam_picture, "SAM proposal (not ground truth)")])


def count(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isod", required=True, help="ISOD 'data' folder with <site>/rgb, label")
    parser.add_argument("--out", required=True, help="output folder, outside the dataset")
    parser.add_argument("frames", nargs="+", help="frames as SITE/NUMBER, e.g. HB_LM/0")
    args = parser.parse_args()
    assert not os.path.abspath(args.out).startswith(os.path.abspath(args.isod)), \
        "--out must not be inside the dataset folder"

    for name in args.frames:
        site, i = name.split("/")
        bgr = cv2.imread(f"{args.isod}/{site}/rgb/{i}.png", cv2.IMREAD_UNCHANGED)
        floor_label = cv2.imread(f"{args.isod}/{site}/label/{i}.png", cv2.IMREAD_UNCHANGED)
        assert bgr is not None and floor_label is not None, f"missing files for {name}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        v0 = make_v0_mask(floor_label)
        os.makedirs(f"{args.out}/v0_masks/{site}", exist_ok=True)
        cv2.imwrite(f"{args.out}/v0_masks/{site}/{i}.png", v0.astype(np.uint8) * 255)

        sam = None
        sam_path = f"{args.out}/sam_masks/{site}/{i}.png"
        if os.path.exists(sam_path):
            sam = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE) > 0

        os.makedirs(f"{args.out}/panels", exist_ok=True)
        panel = make_panel(rgb, floor_label, v0, sam)
        cv2.imwrite(f"{args.out}/panels/{site}_{i}.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"{name}: {count(v0)} v0 obstacles, {int(v0.sum())} px; SAM: "
              f"{'none yet' if sam is None else str(count(sam)) + ' regions'}")


if __name__ == "__main__":
    main()
