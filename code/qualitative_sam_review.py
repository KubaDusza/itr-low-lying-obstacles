"""Render frozen SAM masks beside human references for qualitative inspection."""

import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from isod_sam_review import read_cache
from floor_plane import floor_gap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results/frozen_v1"
OUT = RESULTS / "qualitative"
SETTINGS = json.loads((DATA / "ISOD_sam_train_v1/manifest.json").read_text())
CACHE = DATA / "ISOD_sam_review/source-full-crayon"
PREDICTIONS = RESULTS / "sam_direct_v1/predictions"


def read(path, rgb=False):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB) if rgb else value > 0


def validation_prediction(site, frame, shape):
    result = np.zeros(shape, dtype=bool)
    for phrase, confidence, mask in read_cache(CACHE / site / f"{frame}.npz", shape):
        if phrase in SETTINGS["prompts"] and confidence >= SETTINGS["confidence"]:
            result |= mask
    return result


def overlay(rgb, mask, color):
    output = rgb.astype(np.float32)
    output[mask] = output[mask] * 0.45 + np.asarray(color) * 0.55
    return output.astype(np.uint8)


def sheet(stem, title, samples, selection, prediction_name="Automatic SAM"):
    fig, axes = plt.subplots(len(samples), 4, figsize=(15, len(samples) * 2.65 + 1.2))
    axes = np.atleast_2d(axes)
    compact = len(samples) == 2
    fig.subplots_adjust(left=0.02, right=0.995, top=0.80 if compact else 0.86,
                        bottom=0.045, wspace=0.025, hspace=0.16)
    fig.suptitle(title, x=0.02, y=0.987, ha="left", weight="bold", fontsize=18)
    fig.text(0.02, 0.925 if compact else 0.942, selection, fontsize=10, color="#414141")
    colors = {"Correct obstacle": "#22B77D", "Missed obstacle": "#EFAE27", "Extra prediction": "#D23FA6"}
    fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()],
               loc="upper left", bbox_to_anchor=(0.015, 0.90 if compact else 0.93),
               ncols=3, frameon=False, fontsize=10)
    for i, sample in enumerate(samples):
        rgb, reference, prediction = sample["rgb"], sample["reference"], sample["prediction"]
        assert rgb.shape[:2] == reference.shape == prediction.shape
        intersection = int((reference & prediction).sum())
        union = int((reference | prediction).sum())
        error = rgb.astype(np.float32) * 0.35
        error[reference & prediction] = (34, 183, 125)
        error[reference & ~prediction] = (239, 174, 39)
        error[~reference & prediction] = (210, 63, 166)
        views = [rgb, overlay(rgb, prediction, (75, 150, 240)),
                 overlay(rgb, reference, (75, 150, 240)), error.astype(np.uint8)]
        labels = [sample["label"], f"{prediction_name}  /  IoU {intersection / max(1, union):.3f}",
                  "Human reference", "Agreement and errors"]
        for axis, view, label in zip(axes[i], views, labels):
            axis.imshow(view)
            axis.set_title(label, loc="left", fontsize=10, pad=5)
            axis.set_axis_off()
    output = OUT / f"{stem}.png"
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return dict(image=output.name, title=title, selection=selection,
                frames=[s["label"] for s in samples])


def isod_sample(site, frame, validation=False):
    split = "val" if validation else "test"
    reference = read(DATA / f"ISOD_reference_{split}_v1" / site / f"{frame}.png")
    prediction = (validation_prediction(site, frame, reference.shape) if validation else
                  read(PREDICTIONS / "ISOD" / site / f"{frame}.png"))
    return dict(label=f"{site} / {frame}", rgb=read(DATA / "ISOD/data" / site / "rgb" / f"{frame}.png", rgb=True),
                reference=reference, prediction=prediction)


def depth_example():
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.7))
    fig.subplots_adjust(left=0.025, right=0.99, top=0.84, bottom=0.06, wspace=0.035, hspace=0.3)
    fig.suptitle("RGB and RGB-D on the white poster", x=0.025, y=0.98,
                 ha="left", weight="bold", fontsize=18)
    fig.text(0.025, 0.93, "ISOD-assisted ESANet, seed 0. Flat paper towel on the left, crumpled tissue on the right.", fontsize=11)
    fig.text(0.025, 0.895, "Matched object crops. Green = detected reference; amber = missed; magenta = extra prediction.", fontsize=10)
    settings = json.loads((RESULTS / "floor_plane_v1/settings.json").read_text())["depth_range_m"]
    depth_plot = None
    for row, distance in enumerate(["075", "150", "250"]):
        scene = f"h52_poster_rags_{distance}"
        rgb = read(DATA / "OWN" / scene / "rgb/2.png", rgb=True)
        ids = cv2.imread(str(DATA / "OWN_manual" / f"{scene}_2_id.png"), 0)
        reference = read(DATA / "OWN_manual" / f"{scene}_2.png")
        focus = reference & ((ids == 1) | (ids == 2))
        y, x = np.where(focus)
        left, right = max(0, int(x.min()) - 12), min(rgb.shape[1], int(x.max()) + 13)
        center = (int(y.min()) + int(y.max())) // 2
        half = max(int((right - left) / 3.3), int((y.max() - y.min()) / 2) + 12)
        top, bottom = max(0, center - half), min(rgb.shape[0], center + half)
        crop = np.s_[top:bottom, left:right]
        axes[row, 0].imshow(rgb[crop])
        axes[row, 0].contour(focus[crop], levels=[0.5], colors=["#1BB9C8"], linewidths=1)
        axes[row, 0].set_title(f"{int(distance) / 100:g} m floor mark / RGB + reference", loc="left", fontsize=10)
        raw = cv2.imread(str(DATA / "OWN" / scene / "depth/2.png"), cv2.IMREAD_UNCHANGED)
        residual, valid = floor_gap(raw, settings, seed_from=0)
        depth_plot = axes[row, 1].imshow(np.ma.array(residual[crop] * 1000, mask=~valid[crop]),
                                         vmin=-50, vmax=150, cmap="coolwarm")
        axes[row, 1].set_title("Measured depth / floor-fit residual", loc="left", fontsize=10)
        for col, (model, directory) in enumerate([
            ("RGB", RESULTS / "glaurung/runs/esanet_full_rgb_reference_s0/gte"),
            ("RGB-D", RESULTS / "shire/runs/esanet_full_rgbd_reference_s0/gte"),
        ], start=2):
            prediction = read(directory / scene / "2.png")
            error = rgb.astype(np.float32) * 0.35
            error[reference & prediction] = (34, 183, 125)
            error[reference & ~prediction] = (239, 174, 39)
            error[~reference & prediction] = (210, 63, 166)
            axes[row, col].imshow(error[crop].astype(np.uint8))
            axes[row, col].set_title(model, loc="left", fontsize=10)
        for axis in axes[row]:
            axis.set_axis_off()
    colorbar = fig.colorbar(depth_plot, ax=list(axes[:, 1]), orientation="horizontal",
                            fraction=0.035, pad=0.03, aspect=30)
    colorbar.set_label("Predicted floor depth minus measured depth (mm)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    output = OUT / "rgb_depth_white_objects.png"
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return dict(image=output.name, title="RGB and RGB-D on the white poster",
                selection="Same two white objects, 52 cm mount, frame 2 at each distance. A descriptive example from seed 0.",
                frames=["h52_poster_rags_075/2", "h52_poster_rags_150/2", "h52_poster_rags_250/2"])


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    audit = json.loads((OUT / "validation_sam_audit.json").read_text())
    galleries = []
    sites = ["ABEC_CM", "HL_BM", "MEB_HW_A"]
    for quantile, name in [(0.2, "lower"), (0.8, "higher")]:
        selected = []
        for site in sites:
            rows = sorted([r for r in audit if r["site"] == site], key=lambda r: r["iou"])
            row = rows[round(quantile * (len(rows) - 1))]
            selected.append(isod_sample(site, row["frame"], validation=True))
        galleries.append(sheet(f"sam_validation_{name}", f"SAM on the validation calibration set ({name} overlap)", selected,
                               f"One frame per validation location, nearest its {int(quantile * 100)}th percentile of frame IoU."))

    metrics = json.loads((RESULTS / "sam_direct_metrics.json").read_text())
    selected = []
    for site in ["HDCL_HW_A", "MEB_CM", "MEB_HW_B"]:
        rows = sorted([r for r in metrics["ISOD"]["frames"] if r["group"] == site],
                      key=lambda r: r["inter"] / max(1, r["union"]))
        row = rows[(len(rows) - 1) // 2]
        selected.append(isod_sample(site, row["frame"]))
    galleries.append(sheet("sam_heldout_isod", "SAM on held-out ISOD locations", selected,
                           "Median frame IoU at each test location (10 reviewed frames per location). Frozen prompts and threshold."))

    for background in ["floor", "poster"]:
        selected = []
        for distance in ["075", "150", "250"]:
            scene = f"h52_{background}_rags_{distance}"
            selected.append(dict(label=f"{background}, {int(distance) / 100:g} m floor mark, frame 2",
                                 rgb=read(DATA / "OWN" / scene / "rgb/2.png", rgb=True),
                                 reference=read(DATA / "OWN_manual" / f"{scene}_2.png"),
                                 prediction=read(PREDICTIONS / "GTE-Floor" / scene / "2.png")))
        galleries.append(sheet(f"sam_gte_{background}", f"SAM on GTE-Floor ({background} background)", selected,
                               "52 cm camera mount. Same cloth/tissue set and frame index at three marked floor distances."))

    galleries.append(depth_example())
    prompt_text = "\n".join(SETTINGS["prompts"]) + "\n"
    (OUT / "sam_prompts.txt").write_text(prompt_text)
    (OUT / "example_selection.json").write_text(json.dumps(galleries, indent=2) + "\n")
    sections = "\n".join(
        f'<section><h2>{html.escape(g["title"])}</h2><p>{html.escape(g["selection"])}</p>'
        f'<a href="{g["image"]}"><img src="{g["image"]}" alt="{html.escape(g["title"])}"></a></section>'
        for g in galleries
    )
    page = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Automatic SAM label review</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#171717;background:#fff;max-width:1500px;margin:45px auto;padding:0 25px}}
h1{{font-size:34px}}h2{{font-size:25px;margin-top:55px}}p{{max-width:1000px}}pre{{white-space:pre-wrap;font:16px/1.8 monospace}}
img{{display:block;width:100%;height:auto}}section{{border-top:1px solid #ddd;margin-top:40px}}a{{color:#124c8c}}
</style><h1>Automatic SAM label review</h1>
<p>Full-image RGB input. Each phrase runs separately. We union detections with confidence at least 0.60 and mask probability at least 0.50. The same fixed settings apply to every displayed image.</p>
<p>Prompt selection and confidence tuning used 231 of the 300 human-reviewed validation frames. Replaying the saved candidates reproduces 0.307 pooled calibration IoU. Held-out ISOD and GTE-Floor images follow the same settings.</p>
<h2>Exact retained prompts</h2><pre>{html.escape(', '.join(SETTINGS['prompts']))}</pre>
<p>Human ground-truth creation also used SAM proposals, the original ISOD annotation, and human corrections. The unattended masks shown here use RGB and the fixed phrase list. The model weights stayed fixed during calibration.</p>
{sections}
<p>Blue overlays indicate foreground. Green marks correct obstacle pixels, amber marks missed reference pixels, and magenta marks extra predicted pixels. Click any figure for its full-resolution image. Selection rules are stated above each group.</p>
<p>GTE-Floor examples show one frame per static recording. Compare all five frames and matched recordings before making claims about distance, appearance, or shape. Pale/dark objects can differ in size, material, and shape.</p>
'''
    (OUT / "SAM_LABEL_REVIEW.html").write_text(page)
    print(OUT / "SAM_LABEL_REVIEW.html")


if __name__ == "__main__":
    main()
