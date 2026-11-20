"""Write test-split masks for a trained arm, optionally with the depth input broken.

  python predict_modal.py --modality rgbd --seed 0
  python predict_modal.py --modality rgbd --seed 0 --depth const     # ablation
  python predict_modal.py --modality rgbd --seed 0 --depth shuffle   # ablation

The ablations are the cheap way to find out whether a measured RGB-D difference is really
about depth. `const` feeds the training mean everywhere, so the channel carries no
information; `shuffle` feeds a real depth map from another scene, so the channel keeps its
statistics but no longer matches the RGB image. A difference that survives `const` was
never depth: it came from the extra parameters or from a perturbed optimisation. One that
survives `shuffle` but not `const` is depth statistics rather than depth-RGB
correspondence.

Masks go to <root>/pred/modal/<name>[_<ablation>]/<site>/<i>.png at the original
resolution, so eval.py and summarize.py score them like any other method.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

import modal

OUT_SIZE = (640, 422)       # ISOD frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--modality", required=True, choices=modal.MODALITIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--depth", default="real", choices=["real", "const", "shuffle"])
    ap.add_argument("--encoder", default="mobilenet_v2")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = modal.load_depth_stats(a.root)
    name = modal.run_name(a.modality, a.seed, a.scratch, a.blind, a.tag)
    net = modal.load(a.root, name, a.modality, dev, encoder=a.encoder)
    ds = modal.ISOD(a.root, a.split, stats)

    out = f"{a.root}/pred/modal/{name}" + (f"_{a.depth}" if a.depth != "real" else "")
    # shuffle: pair each frame with the depth of one 137 positions later, which crosses site
    # boundaries, so the depth never belongs to the scene it is shown with
    offset, n = 137, len(ds)
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for k in range(n):
            rgb, dep, geo, _, _ = ds[k]
            if a.depth == "shuffle":
                _, dep, geo, _, _ = ds[(k + offset) % n]
            blind = a.blind or a.depth == "const"
            logit = modal.forward(net, a.modality, rgb[None].to(dev), dep[None].to(dev),
                                  geo[None].to(dev), blind)
            p = (torch.sigmoid(logit[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
            site, i = ds.items[k]
            os.makedirs(f"{out}/{site}", exist_ok=True)
            Image.fromarray(cv2.resize(p, OUT_SIZE, interpolation=cv2.INTER_NEAREST)).save(
                f"{out}/{site}/{i}.png")
    dt = time.time() - t0
    print(f"{os.path.basename(out)}: {n} frames in {dt:.1f}s ({n / dt:.1f} FPS incl. I/O)")


if __name__ == "__main__":
    main()
