"""Build obstacle ground truth from the ISOD floor labels, plus the site splits.

Writes data/ISOD_gt/<site>/<i>.png (0/255 obstacle mask), data/splits.json and a
contact sheet of overlays for eyeballing the result.
"""
import json
import os
import random

import cv2
import numpy as np
from PIL import Image

import isod

N_FRAMES = 100


def build():
    os.makedirs(isod.GT, exist_ok=True)
    stats = {}
    for site in isod.sites():
        os.makedirs(f"{isod.GT}/{site}", exist_ok=True)
        n_obj = n_px = 0
        for i in range(N_FRAMES):
            _, _, label, _ = isod.frame(site, i)
            obst = isod.obstacles(label)
            Image.fromarray(obst * 255).save(f"{isod.GT}/{site}/{i}.png")
            n_obj += len(isod.instances(obst))
            n_px += int(obst.sum())
        stats[site] = {"objects": n_obj, "obstacle_px_per_frame": n_px // N_FRAMES}
        print(f"{site:12s} objects={n_obj:5d}  obstacle px/frame={n_px // N_FRAMES}")
    return stats


def splits(seed=0):
    s = isod.sites()
    random.Random(seed).shuffle(s)
    out = {"test": sorted(s[:3]), "val": sorted(s[3:6]), "train": sorted(s[6:])}
    with open(isod.SPLITS, "w") as f:
        json.dump(out, f, indent=1)
    print({k: len(v) for k, v in out.items()}, out)


def contact_sheet(path, n=8, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        site = rng.choice(isod.sites())
        i = rng.randrange(N_FRAMES)
        rgb, _, label, _ = isod.frame(site, i)
        obst = np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0
        ov = rgb.copy()
        ov[obst] = (0.35 * ov[obst] + np.array([166, 0, 0])).astype(np.uint8)
        ov = cv2.drawContours(
            ov, cv2.findContours(obst.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0],
            -1, (255, 255, 0), 1)
        rows.append(np.hstack([rgb, np.stack([label] * 3, 2), ov]))
    sheet = np.vstack(rows)
    Image.fromarray(sheet).save(path)
    print("wrote", path)


if __name__ == "__main__":
    st = build()
    with open(f"{isod.GT}/stats.json", "w") as f:
        json.dump(st, f, indent=1)
    splits()
    contact_sheet(f"{isod.ROOT}/explore/gt_contact_sheet.png")
