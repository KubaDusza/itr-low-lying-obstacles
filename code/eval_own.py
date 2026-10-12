"""Score the frozen ISOD models on our own recordings.

Same metric as ISOD: mask overlap on the obstacle class, with everything outside the floor
plane ignored. Runs whose ground truth failed the reliability check in own_gt.py are reported
but excluded from the aggregates, and the empty runs are scored separately as a pure
false-positive test, where any predicted pixel is wrong by construction.

  python eval_own.py            # writes results/own/*.csv and prints the tables
"""
import csv
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN, PRED = f"{ROOT}/data/OWN", f"{ROOT}/data/OWN_pred"
RES = f"{ROOT}/results/own"
MODELS = [("YOLOv8n-seg", "yolo"), ("U-Net", "unet_mobilenet_v2"),
          ("DeepLabV3+", "deeplabv3plus_mobilenet_v2")]
DIST = {"075": "0.87 m", "150": "1.56 m", "250": "2.60 m"}


def counts(pred, gt, floor):
    pred = (pred > 0) & (floor > 0)
    gt = (gt > 0) & (floor > 0)
    return dict(inter=int((pred & gt).sum()), union=int((pred | gt).sum()),
                pred=int(pred.sum()), gt=int(gt.sum()))


def totals(rows):
    t = {k: sum(r[k] for r in rows) for k in ("inter", "union", "pred", "gt")}
    return dict(IoU=t["inter"] / max(1, t["union"]),
                precision=t["inter"] / max(1, t["pred"]),
                recall=t["inter"] / max(1, t["gt"]), **t)


def main():
    os.makedirs(RES, exist_ok=True)
    meta = {r["run"]: r for r in json.load(open(f"{OWN}/gt_summary.json"))}
    rows = []
    for run, m in sorted(meta.items()):
        floor = cv2.imread(f"{OWN}/{run}/floor.png", 0)
        n = len(glob.glob(f"{OWN}/{run}/gt/*.png"))
        for name, key in MODELS:
            per_frame = []
            for i in range(n):
                p = cv2.imread(f"{PRED}/{key}/{run}/{i}.png", 0)
                g = cv2.imread(f"{OWN}/{run}/gt/{i}.png", 0)
                if p is None or g is None:
                    continue
                per_frame.append(counts(p, g, floor))
            if not per_frame:
                continue
            t = totals(per_frame)
            ious = [c["inter"] / max(1, c["union"]) for c in per_frame]
            rows.append(dict(run=run, model=name, block=m["block"], distance=m["distance"],
                             object_set=run.split("_")[2], reliable=m["reliable"],
                             is_empty=run.endswith("_empty"),
                             frames=len(per_frame), IoU=t["IoU"], precision=t["precision"],
                             recall=t["recall"], gt=t["gt"], pred=t["pred"],
                             IoU_frame_std=float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
                             **{k: t[k] for k in ("inter", "union")}))
    with open(f"{RES}/per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    good = [r for r in rows if r["reliable"] and not r["is_empty"]]

    def agg(sel):
        return totals([{k: r[k] for k in ("inter", "union", "pred", "gt")}
                       for r in sel if True] ) if sel else None

    def table(title, keyfn, order=None):
        print(f"\n{title}\n")
        keys = order or sorted({keyfn(r) for r in good})
        print("| Method | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for name, _ in MODELS:
            cells = []
            for k in keys:
                sel = [r for r in good if r["model"] == name and keyfn(r) == k]
                t = totals([{kk: r[kk] for kk in ("inter", "union", "pred", "gt")} for r in sel]) if sel else None
                cells.append(f"{t['IoU']:.3f}" if t else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## Overall (IoU / precision / recall), reliable runs only\n")
    print("| Method | IoU | Precision | Recall | runs | frames |")
    print("|---|---|---|---|---|---|")
    overall = {}
    for name, _ in MODELS:
        sel = [r for r in good if r["model"] == name]
        t = totals([{k: r[k] for k in ("inter", "union", "pred", "gt")} for r in sel])
        overall[name] = t
        print(f"| {name} | {t['IoU']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
              f"{len(sel)} | {sum(r['frames'] for r in sel)} |")

    table("### IoU by distance", lambda r: DIST[r["distance"]], [DIST[k] for k in ("075", "150", "250")])
    table("### IoU by camera height", lambda r: "52 cm" if r["block"].startswith("h52") else "25 cm",
          ["52 cm", "25 cm"])
    table("### IoU by background", lambda r: "bare floor" if "floor" in r["block"] else "white poster",
          ["bare floor", "white poster"])
    table("### IoU by object set", lambda r: r["object_set"])

    print("\n### False positives on the empty-floor runs (no obstacle present)\n")
    print("| Method | predicted px / frame | frames with any prediction |")
    print("|---|---|---|")
    for name, _ in MODELS:
        sel = [r for r in rows if r["model"] == name and r["is_empty"]]
        if not sel:
            continue
        px = sum(r["pred"] for r in sel) / max(1, sum(r["frames"] for r in sel))
        print(f"| {name} | {px:.0f} | {sum(1 for r in sel if r["pred"] > 0)}/{len(sel)} runs |")

    with open(f"{RES}/overall.json", "w") as f:
        json.dump({k: {m: v[m] for m in ("IoU", "precision", "recall")} for k, v in overall.items()}, f, indent=1)
    print(f"\nwrote {RES}/per_run.csv and overall.json")


if __name__ == "__main__":
    main()
