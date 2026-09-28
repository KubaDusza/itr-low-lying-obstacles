"""Train a semantic segmentation model (U-Net / DeepLabV3+) on the ISOD obstacle masks.

Mask in, mask out: no polygons, unlike the YOLO instance model. Walls and furniture
are ignored in the loss the same way eval.py ignores them.

  CUDA_VISIBLE_DEVICES=0 python train_seg.py --arch unet --encoder mobilenet_v3_large
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (640, 448)  # multiple of 32 for the encoder


class ISOD(Dataset):
    def __init__(self, root, split, augment=False):
        with open(f"{root}/splits.json") as f:
            self.sites = json.load(f)[split]
        self.root, self.augment = root, augment
        self.items = [(s, i) for s in self.sites for i in range(100)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        rgb = cv2.cvtColor(cv2.imread(f"{self.root}/ISOD/data/{site}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
        label = cv2.imread(f"{self.root}/ISOD/data/{site}/label/{i}.png", 0)
        gt = cv2.imread(f"{self.root}/ISOD_gt/{site}/{i}.png", 0) > 0
        ignore = (label == 0) & ~gt  # walls and furniture
        if self.augment and np.random.rand() < 0.5:
            rgb, gt, ignore = rgb[:, ::-1], gt[:, ::-1], ignore[:, ::-1]
        rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        ignore = cv2.resize(ignore.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(gt)).float(),
                torch.from_numpy(np.ascontiguousarray(1 - ignore)).float())


def masked_loss(logits, y, w):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], y, reduction="none")
    bce = (bce * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logits[:, 0]) * w
    t = y * w
    dice = 1 - (2 * (p * t).sum() + 1) / ((p + t).sum() + 1)
    return bce + dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--arch", default="unet", choices=["unet", "deeplabv3plus"])
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    dev = "cuda"
    net = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[a.arch](
        encoder_name=a.encoder, encoder_weights="imagenet", classes=1).to(dev)
    tr = DataLoader(ISOD(a.root, "train", augment=True), batch_size=a.batch, shuffle=True,
                    num_workers=8, drop_last=True)
    va = DataLoader(ISOD(a.root, "val"), batch_size=a.batch, num_workers=8)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tr))
    scaler = torch.amp.GradScaler()

    name = f"{a.arch}_{a.encoder}"
    os.makedirs(f"{a.root}/runs", exist_ok=True)
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

    # test-split masks at the original resolution, plus inference speed
    out_root = f"{a.root}/pred/{name}"
    ds = ISOD(a.root, "test")
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
