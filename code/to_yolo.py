"""Export the obstacle ground truth as a YOLO segmentation dataset (one class)."""
import os
import shutil

import cv2
import numpy as np
from PIL import Image

import isod

OUT = os.path.join(isod.ROOT, "data", "yolo")
MIN_POLY_AREA = 20


def polygons(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_POLY_AREA or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        c[:, 0] /= w
        c[:, 1] /= h
        out.append(c.clip(0, 1))
    return out


def export():
    for split, sites in isod.load_splits().items():
        for sub in ("images", "labels"):
            os.makedirs(f"{OUT}/{sub}/{split}", exist_ok=True)
        n_obj = 0
        for site in sites:
            for i in range(100):
                stem = f"{site}_{i}"
                shutil.copyfile(f"{isod.DATA}/{site}/rgb/{i}.png", f"{OUT}/images/{split}/{stem}.png")
                m = (np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0).astype(np.uint8)
                polys = polygons(m)
                n_obj += len(polys)
                with open(f"{OUT}/labels/{split}/{stem}.txt", "w") as f:
                    for p in polys:
                        f.write("0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) + "\n")
        print(f"{split}: {len(sites)} sites, {len(sites) * 100} images, {n_obj} objects")

    with open(f"{OUT}/data.yaml", "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n  0: obstacle\n")
    print("wrote", OUT)


if __name__ == "__main__":
    export()
