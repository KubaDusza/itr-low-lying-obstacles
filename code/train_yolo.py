"""Fine-tune YOLO-seg on the ISOD obstacle dataset, then write test-split masks.

Run on a GPU server:
  CUDA_VISIBLE_DEVICES=0 /opt/venv/jdusza_venv/bin/python train_yolo.py --root /data/cs_courses/jdusza_itr
Masks land in <root>/pred/yolo/<site>/<i>.png, ready for eval.py.
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def mask_predictions(model, paths, args):
    for offset in range(0, len(paths), args.batch):
        batch = [str(path) for path in paths[offset:offset + args.batch]]
        results = model.predict(batch, conf=args.conf, imgsz=args.imgsz,
                                retina_masks=True, device=0, stream=True, verbose=False)
        for result in results:
            if result.masks is None:
                yield np.zeros(result.orig_shape, dtype=bool)
            else:
                mask = result.masks.data.any(dim=0).cpu().numpy()
                if mask.shape != result.orig_shape:
                    raise ValueError(f"Prediction is not aligned to {result.path}")
                yield mask


def manifest_frames(root, limit=None):
    rows = json.loads((root / "manifest.json").read_text())["frames"]
    if limit:
        rows = sorted(rows, key=lambda row: (row["frame"], row["site"]))[:limit]
    return [Path(row["path"]) for row in rows]


def run_versioned(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    model = YOLO(args.model)
    optimization = {}
    if args.optimizer:
        optimization.update(optimizer=args.optimizer, lr0=args.lr0, momentum=0.9, warmup_bias_lr=0.0)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=str(args.output.resolve()), name="fit", device=0,
                seed=args.seed, workers=args.workers, patience=args.patience, save_period=5,
                **optimization)
    weights = args.output / "fit" / "weights"
    candidates = sorted(weights.glob("epoch*.pt")) + [weights / "last.pt"]
    root = Path(args.root) / "ISOD" / "data"
    val_frames = manifest_frames(args.val_masks, args.val_limit)
    val_imgs = [root / path.parent / "rgb" / path.name for path in val_frames]
    scores = {}
    for checkpoint in candidates:
        model = YOLO(str(checkpoint))
        intersection = union = 0
        for path, mask in zip(val_frames, mask_predictions(model, val_imgs, args), strict=True):
            gt = cv2.imread(str(args.val_masks / path), cv2.IMREAD_GRAYSCALE)
            if gt is None or gt.shape != mask.shape:
                raise ValueError(f"Missing or misaligned validation mask {path}")
            intersection += int((mask & (gt > 0)).sum())
            union += int((mask | (gt > 0)).sum())
        scores[checkpoint.name] = intersection / max(1, union)
    best = max(scores, key=scores.get)
    (args.output / "selection.json").write_text(json.dumps(dict(best=best, val_iou=scores), indent=2) + "\n")
    model = YOLO(str(weights / best))

    frames = val_frames if args.val_limit else manifest_frames(args.test_masks)
    if args.predict_all_test and not args.val_limit:
        with open(Path(args.root) / "splits.json") as file:
            frames = [Path(site) / f"{frame}.png" for site in json.load(file)["test"] for frame in range(100)]
    groups = [("pred", [(root / path.parent / "rgb" / path.name, path) for path in frames])]
    if args.gte and not args.val_limit:
        images = sorted(path for path in args.gte.glob("*/rgb/*.png") if path.stem.isdigit())
        if len(images) != 320:
            raise ValueError(f"Expected 320 GTE-Floor images, got {len(images)}")
        groups.append(("gte", [(path, Path(path.parent.parent.name) / path.name) for path in images]))
    for group, records in groups:
        masks = mask_predictions(model, [path for path, _ in records], args)
        for (_, relative), mask in zip(records, masks, strict=True):
            output = args.output / group / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), mask.astype(np.uint8) * 255):
                raise OSError(f"Could not save {output}")
        print(group, len(records), "predictions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default=None, help="skip training, predict with these weights")
    ap.add_argument("--name", default="yolov8n_seg", help="run name under runs/ and pred/")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--val-masks", type=Path)
    ap.add_argument("--test-masks", type=Path)
    ap.add_argument("--gte", type=Path)
    ap.add_argument("--val-limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--predict-all-test", action="store_true")
    a = ap.parse_args()
    versioned = (a.data, a.output, a.val_masks, a.test_masks)
    if any(versioned):
        if not all(versioned) or a.weights:
            ap.error("Provide --data, --output, --val-masks, and --test-masks for versioned training")
        cv2.setNumThreads(1)
        return run_versioned(a)

    if a.weights:
        model = YOLO(a.weights)
    else:
        model = YOLO(a.model)
        model.train(data=f"{a.root}/yolo/data.yaml", epochs=a.epochs, imgsz=a.imgsz,
                    batch=a.batch, project=f"{a.root}/runs", name=a.name, exist_ok=True)

    img_dir = f"{a.root}/yolo/images/test"
    files = sorted(os.listdir(img_dir))
    t0 = time.time()
    for f in files:
        site, i = f[:-4].rsplit("_", 1)
        r = model.predict(f"{img_dir}/{f}", conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        h, w = r.orig_shape
        # r.masks.xy is already in original-image coordinates; r.masks.data is not
        # (it still carries the letterbox padding), so use the polygons.
        out = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(out, [poly.astype(np.int32)], 255)
        d = f"{a.root}/pred/{a.name}/{site}"
        os.makedirs(d, exist_ok=True)
        Image.fromarray(out).save(f"{d}/{i}.png")
    dt = time.time() - t0
    print(f"predicted {len(files)} images in {dt:.1f}s -> {len(files) / dt:.1f} FPS (incl. I/O)")


if __name__ == "__main__":
    main()
