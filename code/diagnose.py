"""Checks that decide whether the modality comparison means anything.

  python diagnose.py check                 # before training: is the comparison really paired?
  python diagnose.py depth --root ...      # after training: did the depth path move off zero?

`check` is also the smoke test: it builds every arm and pushes a batch through each.

Why `check` exists: the claim "RGB-D minus RGB measures the depth input" only holds if the
two arms start from the same function and see the same data. That is a property of the code,
so it is checked in the code rather than asserted in the README. It compares every tensor
the arms share and then verifies that at step 0 each fusion arm produces bit-identical
output to the RGB arm for arbitrary depth input.
"""
import argparse
import glob
import json
import sys

import torch

import modal

FUSION = [k for k, v in modal.ARMS.items() if "expand" in v or "two_stream" in v]


def check(encoder="mobilenet_v2", seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {}
    for m in modal.MODALITIES:
        modal.seed_all(seed)
        nets[m] = modal.build(m, encoder).to(dev).eval()
        print(f"{m:8s} {modal.ARMS[m]['ch']} ch  "
              f"{sum(p.numel() for p in nets[m].parameters()) / 1e6:6.2f}M params")

    rgb = torch.randn(2, 3, 64, 64, device=dev)
    dep = torch.randn(2, 1, 64, 64, device=dev)
    geo = torch.randn(2, 3, 64, 64, device=dev)

    print("\nshared weights, RGB arm vs. each fusion arm (max abs difference):")
    ok = True
    for m in FUSION:
        a = dict(nets["rgb"].named_parameters())
        b = dict(nets[m].named_parameters())
        worst, worst_k, extra = 0.0, "-", []
        for k, v in b.items():
            ka = k.replace("net.", "", 1) if "two_stream" in modal.ARMS[m] else k
            if ka in a and a[ka].shape == v.shape:
                d = float((a[ka] - v).detach().abs().max())
                if d >= worst:
                    worst, worst_k = d, k
            else:
                extra.append(k)
        print(f"  {m:8s} {len(b) - len(extra):3d} shared, max diff {worst:.2e} ({worst_k}); "
              f"{len(extra)} not shared")
        ok &= worst == 0.0

    print("\nfunctional equivalence at step 0 (same RGB, random depth):")
    with torch.no_grad():
        ref = modal.forward(nets["rgb"], "rgb", rgb, dep, geo)
        for m in FUSION:
            d = float((modal.forward(nets[m], m, rgb, dep, geo) - ref).abs().max())
            print(f"  {m:8s} max |out - rgb_out| = {d:.2e}")
            ok &= d == 0.0

    print("\nforward and backward, every arm:")
    y = torch.zeros(2, 64, 64, device=dev)
    w = torch.ones(2, 64, 64, device=dev)
    for m in modal.MODALITIES:
        nets[m].train()
        loss = modal.masked_loss(modal.forward(nets[m], m, rgb, dep, geo), y, w)
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in nets[m].parameters() if p.grad is not None)
        print(f"  {m:8s} loss {float(loss):.4f}  total |grad| {g:.3e}")

    print("\nPAIRED" if ok else "\nNOT PAIRED - fix before running the experiment")
    return 0 if ok else 1


def depth_path(root):
    """Did the zero-initialised depth path learn anything, in every trained fusion run?"""
    rows = []
    for p in sorted(glob.glob(f"{root}/runs/modal/*.json")):
        info = json.load(open(p))
        if info.get("depth_path"):
            rows.append((info["name"], info["val_IoU_final"], info["depth_path"]))
    if not rows:
        print("no trained fusion runs found")
        return 1
    for name, iou, d in rows:
        if "gates" in d:
            print(f"{name:22s} val_IoU {iou:.4f}  gates {d['gates']}")
        else:
            print(f"{name:22s} val_IoU {iou:.4f}  |conv1 rgb| {d['conv1_rgb_norm']:.3f}  "
                  f"|conv1 depth| {d['conv1_depth_norm']:.3f}  "
                  f"per-channel ratio {d['per_channel_ratio']:.2f}")
    print("\nA depth norm or a gate that stayed at zero would make 'fusion = RGB' an\n"
          "artefact of the initialisation rather than a result about depth.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "depth"])
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.exit(check(a.encoder, a.seed) if a.cmd == "check" else depth_path(a.root))
