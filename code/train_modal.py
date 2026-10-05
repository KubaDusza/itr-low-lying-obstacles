"""Train the SAME segmentation model on RGB, depth or RGB-D (early fusion) input.

Jakub's train_seg.py is left untouched; the loss, image size and RGB normalisation are
imported from it, and the training loop below is the same loop. The RGB arm is run through
this script too, so all three modalities share one code path, one seed and one naming scheme.

  CUDA_VISIBLE_DEVICES=0 python train_modal.py --modality rgb   --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_modal.py --modality depth --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_modal.py --modality rgbd  --seed 0

Outputs (never collide with the original RGB run unet_mobilenet_v2):
  runs/<arch>_<encoder>_<modality>_s<seed>{.pt,_history.csv,_config.json}
  pred/<arch>_<encoder>_<modality>_s<seed>/<site>/<i>.png   -> python eval.py masks <dir>
  
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_seg import MEAN, SIZE, STD, masked_loss  # Jakub's code, reused as-is

IN_CH = {"rgb": 3, "depth": 1, "rgbd": 4}
ARCH = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}


# --------------------------------------------------------------------------- depth
def read_depth(root, site, i):
    d = cv2.imread(f"{root}/ISOD/data/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
    # without IMREAD_UNCHANGED OpenCV silently converts to 8-bit; fail loudly instead
    assert d is not None and d.dtype == np.uint16 and d.ndim == 2, \
        f"depth {site}/{i}: {None if d is None else (d.dtype, d.shape)}"
    return d


def encode_depth(d, st):
    """uint16 mm -> clipped inverse depth, standardised with train-set statistics.

    near/d keeps a fixed, monotonic relation to metric depth inside [near, far] (values
    outside saturate at the clip limits, the same for every run), spends resolution on
    the near range where small obstacles are, and is close to stereo disparity, where
    the sensor noise is roughly uniform. Invalid (0) maps to 0 = "no return / far",
    instead of being aliased with the nearest possible distance.
    """
    d = d.astype(np.float32)
    x = np.where(d > 0, st["near"] / np.clip(d, st["near"], st["far"]), 0.0)
    return ((x - st["mean"]) / st["std"]).astype(np.float32)


def depth_stats(root, sites, step=5):
    """Clip range and mean/std from the TRAIN sites only, cached in <root>/depth_stats.json
    so every depth and RGB-D run (and every seed) uses identical constants."""
    path = f"{root}/depth_stats.json"
    if os.path.exists(path):
        with open(path) as f:
            st = json.load(f)
        assert st["train_sites"] == sorted(sites), f"{path} was built from a different split"
        return st
    sample, inv = [], {k: [0, 0] for k in ("obstacle", "floor", "ignore")}
    for site in sites:
        for i in range(0, 100, step):
            d = read_depth(root, site, i)
            label = cv2.imread(f"{root}/ISOD/data/{site}/label/{i}.png", 0)
            gt = cv2.imread(f"{root}/ISOD_gt/{site}/{i}.png", 0) > 0
            sample.append(d.ravel()[::7])
            for k, m in (("obstacle", gt), ("floor", label == 255), ("ignore", (label == 0) & ~gt)):
                inv[k][0] += int(((d == 0) & m).sum())
                inv[k][1] += int(m.sum())
    sample = np.concatenate(sample)
    near, far = np.percentile(sample[sample > 0], [1, 99])
    st = {"near": float(near), "far": float(far), "mean": 0.0, "std": 1.0}
    x = encode_depth(sample, st)  # all pixels, invalid included: that is what the net sees
    st.update(mean=float(x.mean()), std=float(x.std()), train_sites=sorted(sites),
              invalid_rate={k: v[0] / max(1, v[1]) for k, v in inv.items()})
    with open(path, "w") as f:
        json.dump(st, f, indent=1)
    print("depth stats:", json.dumps(st))
    return st


# --------------------------------------------------------------------------- data
class ISODModal(Dataset):
    """Same as train_seg.ISOD, plus depth. Every geometric op hits all arrays together."""

    def __init__(self, root, split, modality, dstats=None, augment=False):
        with open(f"{root}/splits.json") as f:
            self.sites = json.load(f)[split]
        self.root, self.modality, self.dstats, self.augment = root, modality, dstats, augment
        self.items = [(s, i) for s in self.sites for i in range(100)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        rgb = cv2.cvtColor(cv2.imread(f"{self.root}/ISOD/data/{site}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
        label = cv2.imread(f"{self.root}/ISOD/data/{site}/label/{i}.png", 0)
        gt = cv2.imread(f"{self.root}/ISOD_gt/{site}/{i}.png", 0) > 0
        ignore = (label == 0) & ~gt  # walls and furniture
        depth = None
        if self.modality != "rgb":
            depth = read_depth(self.root, site, i)
            assert depth.shape == rgb.shape[:2], f"{site}/{i}: depth {depth.shape} vs rgb {rgb.shape}"
        if self.augment and np.random.rand() < 0.5:
            rgb, gt, ignore = rgb[:, ::-1], gt[:, ::-1], ignore[:, ::-1]
            if depth is not None:
                depth = depth[:, ::-1]
        gt = cv2.resize(gt.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        ignore = cv2.resize(ignore.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        chans = []
        if self.modality in ("rgb", "rgbd"):
            rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_LINEAR)
            chans.append((rgb.astype(np.float32) / 255 - MEAN) / STD)
        if depth is not None:
            # NEAREST: bilinear would blend valid depth with 0 (invalid) and invent
            # intermediate depths exactly on obstacle boundaries
            depth = cv2.resize(np.ascontiguousarray(depth), SIZE, interpolation=cv2.INTER_NEAREST)
            chans.append(encode_depth(depth, self.dstats)[..., None])
        x = np.concatenate(chans, 2).transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)).float(),
                torch.from_numpy(np.ascontiguousarray(gt)).float(),
                torch.from_numpy(np.ascontiguousarray(1 - ignore)).float())


# --------------------------------------------------------------------------- model
def first_conv(encoder):
    return next(m for m in encoder.modules() if isinstance(m, torch.nn.Conv2d))


def build_net(arch, encoder, modality, weights="imagenet"):
    net = ARCH[arch](encoder_name=encoder, encoder_weights=weights,
                     in_channels=IN_CH[modality], classes=1)
    if modality == "rgbd":
        # smp copies the pretrained R filter onto the depth channel and scales all four
        # filters by 3/4 (exact behaviour depends on the smp version). Put the pretrained
        # RGB filters back and start the depth filter at zero: the RGB-D net starts from
        # exactly the RGB net's pretrained features, and anything it gains is learned from depth.
        ref = ARCH[arch](encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
        conv, ref_conv = first_conv(net.encoder), first_conv(ref.encoder)
        assert conv.in_channels == 4 and ref_conv.in_channels == 3
        with torch.no_grad():
            conv.weight[:, :3] = ref_conv.weight
            conv.weight[:, 3:] = 0.0
    # depth (1 ch): smp sums the three RGB filters -> a grayscale filter. Kept as is,
    # but it means "depth with ImageNet init", not depth from scratch; say so in the report.
    return net


# --------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--arch", default="unet", choices=list(ARCH))
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--modality", required=True, choices=list(IN_CH))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    name = f"{a.arch}_{a.encoder}_{a.modality}_s{a.seed}"
    os.makedirs(f"{a.root}/runs", exist_ok=True)
    if os.path.exists(f"{a.root}/runs/{name}.pt") and not a.overwrite:
        raise SystemExit(f"{name} already exists (use --overwrite or another --seed)")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    with open(f"{a.root}/splits.json") as f:
        train_sites = json.load(f)["train"]
    dstats = depth_stats(a.root, train_sites) if a.modality != "rgb" else None

    dev = "cuda"
    net = build_net(a.arch, a.encoder, a.modality).to(dev)
    # own generator: shuffle order and worker seeds depend on --seed only, not on how
    # many random numbers model construction consumed (it differs between modalities)
    tr = DataLoader(ISODModal(a.root, "train", a.modality, dstats, augment=True),
                    batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True,
                    generator=torch.Generator().manual_seed(a.seed))
    va = DataLoader(ISODModal(a.root, "val", a.modality, dstats), batch_size=a.batch, num_workers=8)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tr))
    scaler = torch.amp.GradScaler()

    with open(f"{a.root}/runs/{name}_config.json", "w") as f:
        json.dump({**vars(a), "in_channels": IN_CH[a.modality], "depth_stats": dstats,
                   "smp": smp.__version__, "torch": torch.__version__}, f, indent=1)

    # ---- from here on: identical to train_seg.py ----
    hist = open(f"{a.root}/runs/{name}_history.csv", "w")
    hist.write("epoch,train_loss,val_IoU,seconds\n")
    t_start = time.time()
    for ep in range(a.epochs):
        net.train()
        run = 0.0
        for x, y, w in tr:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = masked_loss(net(x), y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        net.eval()
        inter = union = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, w in va:
                p = (torch.sigmoid(net(x.to(dev))[:, 0]) > 0.5).float().cpu() * w
                inter += (p * y).sum().item()
                union += ((p + y) > 0).float().sum().item()
        print(f"epoch {ep + 1}/{a.epochs} loss={run / len(tr):.4f} val_IoU={inter / max(1, union):.4f}",
              flush=True)
        hist.write(f"{ep + 1},{run / len(tr):.5f},{inter / max(1, union):.5f},{time.time() - t_start:.1f}\n")
        hist.flush()

    hist.close()
    torch.save(net.state_dict(), f"{a.root}/runs/{name}.pt")

    out_root = f"{a.root}/pred/{name}"
    ds = ISODModal(a.root, "test", a.modality, dstats)
    net.eval()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for k, (site, i) in enumerate(ds.items):
            x, _, _ = ds[k]
            p = torch.sigmoid(net(x[None].to(dev))[0, 0]) > 0.5
            m = cv2.resize(p.cpu().numpy().astype(np.uint8) * 255, (640, 422),
                           interpolation=cv2.INTER_NEAREST)
            os.makedirs(f"{out_root}/{site}", exist_ok=True)
            Image.fromarray(m).save(f"{out_root}/{site}/{i}.png")
    dt = time.time() - t0
    print(f"{name}: {len(ds)} test images in {dt:.1f}s -> {len(ds) / dt:.1f} FPS (incl. I/O)")


if __name__ == "__main__":
    main()