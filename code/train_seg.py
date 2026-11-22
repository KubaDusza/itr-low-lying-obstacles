"""Train a semantic segmentation model (U-Net / DeepLabV3+) on the ISOD obstacle masks.

Versioned masks use full-image binary targets. Legacy runs use the original ignore region.

  CUDA_VISIBLE_DEVICES=0 python train_seg.py --arch unet --encoder mobilenet_v3_large
"""
import argparse
import json
import os
from pathlib import Path
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


def rgb_tensor(rgb):
    rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_LINEAR)
    x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(x))


class ISOD(Dataset):
    def __init__(self, root, split, augment=False, mask_dir=None, limit=None):
        with open(f"{root}/splits.json") as f:
            self.sites = json.load(f)[split]
        self.root, self.augment = root, augment
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.items = [(s, i) for s in self.sites for i in range(100)]
        if self.mask_dir:
            manifest = json.loads((self.mask_dir / "manifest.json").read_text())
            entries = [(row["site"], row["frame"]) for row in manifest["frames"]]
            if (manifest["split"] != split or len(set(entries)) != len(entries) or
                    not set(entries).issubset(self.items)):
                raise ValueError(f"Invalid {split} manifest in {self.mask_dir}")
            if split in ("train", "val") and len(entries) != len(self.items):
                raise ValueError(f"Incomplete {split} masks in {self.mask_dir}")
            self.items = entries
        if limit:
            self.items = sorted(self.items, key=lambda item: (item[1], item[0]))[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        rgb = cv2.cvtColor(cv2.imread(f"{self.root}/ISOD/data/{site}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
        mask_root = self.mask_dir or Path(self.root) / "ISOD_gt"
        mask = cv2.imread(str(mask_root / site / f"{i}.png"), 0)
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError(f"Missing or misaligned mask for {site}/{i} in {mask_root}")
        gt = mask > 0
        if self.mask_dir:
            ignore = np.zeros_like(gt)
        else:
            label = cv2.imread(f"{self.root}/ISOD/data/{site}/label/{i}.png", 0)
            ignore = (label == 0) & ~gt
        if self.augment and np.random.rand() < 0.5:
            rgb, gt, ignore = rgb[:, ::-1], gt[:, ::-1], ignore[:, ::-1]
        gt = cv2.resize(gt.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        ignore = cv2.resize(ignore.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
        return (rgb_tensor(rgb),
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
    ap.add_argument("--train-masks", type=Path)
    ap.add_argument("--val-masks", type=Path)
    ap.add_argument("--test-masks", type=Path)
    ap.add_argument("--output", type=Path, help="Directory for this paired run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-limit", type=int, help="Frame limit for a smoke test")
    ap.add_argument("--val-limit", type=int, help="Frame limit for a smoke test")
    ap.add_argument("--predict-all-test", action="store_true", help="Export all test RGB predictions without needing their reference masks")
    a = ap.parse_args()
    explicit = (a.train_masks, a.val_masks, a.test_masks, a.output)
    if any(explicit) and not all(explicit):
        ap.error("Set --train-masks, --val-masks, --test-masks, and --output together")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cv2.setNumThreads(1)

    dev = "cuda"
    net = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[a.arch](
        encoder_name=a.encoder, encoder_weights="imagenet", classes=1).to(dev)
    tr = DataLoader(ISOD(a.root, "train", augment=True, mask_dir=a.train_masks, limit=a.train_limit),
                    batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    va = DataLoader(ISOD(a.root, "val", mask_dir=a.val_masks, limit=a.val_limit),
                    batch_size=a.batch, num_workers=a.workers)
    if not len(tr) or not len(va):
        raise ValueError("Training and validation need at least one batch")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tr))
    scaler = torch.amp.GradScaler()

    name = f"{a.arch}_{a.encoder}"
    if a.output:
        run_dir = a.output
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "settings.json").write_text(json.dumps(vars(a), default=str, indent=2) + "\n")
        checkpoint = run_dir / "best.pt"
        out_root = run_dir / "pred"
    else:
        run_dir = Path(a.root) / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / f"{name}.pt"
        out_root = Path(a.root) / "pred" / name
    hist = (run_dir / f"{name}_history.csv").open("w")
    hist.write("epoch,train_loss,val_loss,val_IoU,seconds\n")
    t_start = time.time()
    best = -1.0
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
        validation_loss = 0.0
        validation_samples = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, w in va:
                logits = net(x.to(dev))
                validation_loss += masked_loss(logits, y.to(dev), w.to(dev)).item() * len(x)
                validation_samples += len(x)
                p = (torch.sigmoid(logits[:, 0]) > 0.5).float().cpu() * w
                inter += (p * y).sum().item()
                union += ((p + y) > 0).float().sum().item()
        val_loss = validation_loss / validation_samples
        print(f"epoch {ep + 1}/{a.epochs} loss={run / len(tr):.4f} val_loss={val_loss:.4f} val_IoU={inter / max(1, union):.4f}",
              flush=True)
        hist.write(f"{ep + 1},{run / len(tr):.5f},{val_loss:.5f},{inter / max(1, union):.5f},{time.time() - t_start:.1f}\n")
        hist.flush()
        if a.output:
            state = dict(epoch=ep + 1, model=net.state_dict(), optimizer=opt.state_dict(),
                         scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                         torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state())
            torch.save(state, run_dir / "last_checkpoint.pt")
            if (ep + 1) % 5 == 0:
                torch.save(state, run_dir / f"checkpoint_epoch_{ep + 1}.pt")
        if a.output and inter / max(1, union) > best:
            best = inter / max(1, union)
            torch.save(net.state_dict(), checkpoint)
            (run_dir / "selection.json").write_text(json.dumps(dict(
                criterion="maximum pooled validation obstacle IoU",
                epoch=ep + 1, validation_iou=best, checkpoint="best.pt"), indent=2) + "\n")

    hist.close()
    if a.output:
        net.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    else:
        torch.save(net.state_dict(), checkpoint)

    # Smoke runs verify prediction export on validation inputs.
    if a.train_limit or a.val_limit:
        ds = ISOD(a.root, "val", mask_dir=a.val_masks, limit=a.val_limit)
    else:
        ds = ISOD(a.root, "test", mask_dir=a.test_masks)
    items = ds.items
    if a.predict_all_test and not (a.train_limit or a.val_limit):
        with open(f"{a.root}/splits.json") as file:
            items = [(site, frame) for site in json.load(file)["test"] for frame in range(100)]
    net.eval()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for site, i in items:
            rgb = cv2.cvtColor(cv2.imread(f"{a.root}/ISOD/data/{site}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
            x = rgb_tensor(rgb)
            p = torch.sigmoid(net(x[None].to(dev))[0, 0]) > 0.5
            m = cv2.resize(p.cpu().numpy().astype(np.uint8) * 255, (640, 422),
                           interpolation=cv2.INTER_NEAREST)
            os.makedirs(f"{out_root}/{site}", exist_ok=True)
            Image.fromarray(m).save(f"{out_root}/{site}/{i}.png")
    dt = time.time() - t0
    print(f"{name}: {len(items)} predictions in {dt:.1f}s -> {len(items) / dt:.1f} FPS (incl. I/O)")


if __name__ == "__main__":
    main()
