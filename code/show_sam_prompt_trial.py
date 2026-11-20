"""Create an RGB-and-mask review for the eight-frame SAM prompt trial."""

import argparse
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory
    scores = json.loads((output / "scores.json").read_text())
    root = Path(__file__).resolve().parents[1]
    frames = scores["development_frames"]
    profiles = ["frozen_baseline", "user_generic", "best_generic", "hybrid"]
    labels = ["Existing 16 phrases", "Something small on the floor", "Best broad phrase", "Broad + specific floor phrases"]
    figures = []
    for offset in range(0, len(frames), 4):
        selected = frames[offset:offset + 4]
        fig, axes = plt.subplots(len(selected), 5, figsize=(16, 10.5))
        fig.subplots_adjust(left=0.015, right=0.995, top=0.87, bottom=0.025, wspace=0.035, hspace=0.18)
        fig.suptitle("SAM prompt comparison on the eight development frames", x=0.015, y=0.98,
                     ha="left", fontsize=18, weight="bold")
        fig.text(0.015, 0.943, f"Best broad phrase by mean frame IoU: {scores['best_generic']}", fontsize=11)
        fig.text(0.015, 0.911, "Cyan = reference boundary. Green = correct obstacle; amber = missed; magenta = extra.", fontsize=10)
        for row, (site, frame) in enumerate(selected):
            rgb = cv2.imread(str(root / "data/ISOD/data" / site / "rgb" / f"{frame}.png"))
            raw = cv2.imread(str(root / "data/ISOD_reference_train_v1" / site / f"{frame}.png"), 0)
            if rgb is None or raw is None:
                raise FileNotFoundError(f"Missing inputs for {site}/{frame}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            gt = raw > 0
            axes[row, 0].imshow(rgb)
            axes[row, 0].contour(gt, levels=[0.5], colors=["#04D7E4"], linewidths=0.7)
            axes[row, 0].set_title(f"{site} / {frame}", loc="left", fontsize=9)
            for col, (profile, label) in enumerate(zip(profiles, labels), start=1):
                mask = cv2.imread(str(output / profile / site / f"{frame}.png"), 0)
                if mask is None:
                    raise FileNotFoundError(f"Missing {profile} prediction for {site}/{frame}")
                pred = mask > 0
                image = rgb.astype(np.float32) * 0.4
                image[gt & pred] = (34, 183, 125)
                image[gt & ~pred] = (239, 174, 39)
                image[~gt & pred] = (210, 63, 166)
                axes[row, col].imshow(image.astype(np.uint8))
                iou = (gt & pred).sum() / max(1, (gt | pred).sum())
                axes[row, col].set_title(f"{label}\nIoU {iou:.3f}", loc="left", fontsize=8)
            for axis in axes[row]:
                axis.set_axis_off()
        file = output / f"comparison_{offset // 4 + 1}.png"
        fig.savefig(file, dpi=155, facecolor="white")
        plt.close(fig)
        figures.append(file.name)

    names = {**dict(zip(profiles, labels)), "specific": "Observed object names", "specific_on_floor": "Observed names + on the floor"}
    rows = []
    for name, record in scores["profiles"].items():
        values = "".join(f"<td>{record[key]:.3f}</td>" for key in ("mean_frame_iou", "pooled_iou", "precision", "recall"))
        rows.append(f"<tr><th>{html.escape(names[name])}</th>{values}</tr>")
    generic = json.loads((output.parent / "prompt_trial.json").read_text())["generic_prompts"]
    ranking = sorted(generic, key=lambda phrase: scores["single_phrase_scores"][phrase]["mean_frame_iou"], reverse=True)
    phrase_rows = "".join(f"<tr><th>{html.escape(phrase)}</th><td>{scores['single_phrase_scores'][phrase]['mean_frame_iou']:.3f}</td>"
                          f"<td>{scores['single_phrase_scores'][phrase]['pooled_iou']:.3f}</td></tr>" for phrase in ranking)
    pictures = "".join(f'<a href="{file}"><img src="{file}" alt="Development frame mask comparison"></a>' for file in figures)
    page = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM prompt trial</title><style>
body{{font:17px/1.5 Inter,system-ui,sans-serif;max-width:1550px;padding:24px;margin:auto;color:#171717;background:#fff}}
h1{{font-size:32px}}p{{max-width:90ch}}table{{border-collapse:collapse;margin:25px 0;width:min(1050px,100%)}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccc}}td{{font-variant-numeric:tabular-nums}}img{{width:100%;height:auto;margin:25px 0}}a{{color:#174e88}}li{{margin:5px 0}}
</style><h1>SAM prompt development results</h1>
<p>Eight RGB development images. Model, confidence 0.60, mask probability 0.50, and mask-union logic stayed fixed. Only the text prompts changed. The 14 reserved frames have not been evaluated.</p>
<p>Human masks provide the scoring references. The existing 16-phrase row uses the frozen baseline masks from the previous experiment. Other rows use the new SAM inference cache.</p>
<table><thead><tr><th>Prompt recipe</th><th>Mean frame IoU</th><th>Pooled IoU</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Mean frame IoU gives every image equal weight. Pooled IoU combines pixels across the eight images. These are development comparisons used for prompt selection.</p>
{pictures}<h2>Broad phrases</h2><table><thead><tr><th>Phrase</th><th>Mean frame IoU</th><th>Pooled IoU</th></tr></thead><tbody>{phrase_rows}</tbody></table>
<h2>Exact recipes</h2><ul>{''.join('<li><strong>'+html.escape(names[name])+'</strong><br>'+html.escape(', '.join(record['phrases']))+'</li>' for name,record in scores['profiles'].items())}</ul>
'''
    (output / "REVIEW.html").write_text(page)
    print(output / "REVIEW.html")


if __name__ == "__main__":
    main()
