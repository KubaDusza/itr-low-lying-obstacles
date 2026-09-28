"""Fine-tune YOLO-seg on the ISOD obstacle dataset, then write test-split masks.

Run on a GPU server:
  CUDA_VISIBLE_DEVICES=0 /opt/venv/jdusza_venv/bin/python train_yolo.py --root /data/cs_courses/jdusza_itr
Masks land in <root>/pred/yolo/<site>/<i>.png, ready for eval.py.
"""
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default=None, help="skip training, predict with these weights")
    a = ap.parse_args()

    if a.weights:
        model = YOLO(a.weights)
    else:
        model = YOLO(a.model)
        model.train(data=f"{a.root}/yolo/data.yaml", epochs=a.epochs, imgsz=a.imgsz,
                    batch=a.batch, project=f"{a.root}/runs", name="yolov8n_seg", exist_ok=True)

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
        d = f"{a.root}/pred/yolo/{site}"
        os.makedirs(d, exist_ok=True)
        Image.fromarray(out).save(f"{d}/{i}.png")
    dt = time.time() - t0
    print(f"predicted {len(files)} images in {dt:.1f}s -> {len(files) / dt:.1f} FPS (incl. I/O)")


if __name__ == "__main__":
    main()
