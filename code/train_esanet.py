"""Train the official ESANet (R34-NBt1D) on ISOD obstacle masks, for RGB, depth or RGB-D input.

The model, augmentation, normalisation and loss come unchanged from the authors' code
(https://github.com/TUI-NICR/ESANet), imported from a separate checkout given by --esanet.
This file only adds what ESANet does not have for our task:

  * reading ISOD with our existing split (splits.json) and obstacle masks (ISOD_gt/),
  * the three-value label ESANet expects: 0 = void (walls/furniture, ignored),
    1 = background (everything else that is not an obstacle), 2 = obstacle,
  * depth mean/std and median-frequency class weights computed on the TRAIN sites only,
  * checkpoint selection by validation OBSTACLE IoU (ESANet selects by mIoU over all classes),
    keeping one best checkpoint (weights only) instead of one file per improvement,
  * test masks written at the native 640x422 resolution, scored by the unchanged eval.py.

  python train_esanet.py --esanet <ESANet checkout> --root <isod_work> --out <run dir> \
      --modality rgbd --seed 0 --epochs 500
  python train_esanet.py ... --smoke      # 2 epochs, a few batches: checks the pipeline only

Outputs in <out>/<name>/ (name = esanet_r34nbt1d_<modality>_s<seed>):
  config.json, history.csv, best.pt (model weights), pred_test/<site>/<i>.png, test_eval.json
"""
import argparse
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HEIGHT, WIDTH = 448, 640       # network input (report: 640x448; same as the U-Net runs)
NATIVE_W, NATIVE_H = 640, 422  # ISOD frame size; test masks are written at this size
VOID, BACKGROUND, OBSTACLE = 0, 1, 2


# --------------------------------------------------------------------------- data
def load_frame(root, site, i):
    """RGB uint8 (H,W,3), depth uint16 mm (H,W, 0 = no reading), ESANet label uint8 (H,W)."""
    base = f"{root}/ISOD/data/{site}"
    bgr = cv2.imread(f"{base}/rgb/{i}.png")
    depth = cv2.imread(f"{base}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
    floor_label = cv2.imread(f"{base}/label/{i}.png", cv2.IMREAD_GRAYSCALE)  # 255 = drivable
    gt = cv2.imread(f"{root}/ISOD_gt/{site}/{i}.png", cv2.IMREAD_GRAYSCALE)  # 255 = obstacle
    assert bgr is not None and depth is not None and floor_label is not None and gt is not None, \
        f"missing file for {site}/{i}"
    assert depth.dtype == np.uint16, f"{site}/{i}: depth is {depth.dtype}, expected uint16 mm"
    assert depth.shape == bgr.shape[:2] == floor_label.shape == gt.shape, f"{site}/{i}: shape mismatch"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, make_label(floor_label, gt > 0)


def make_label(floor_label, obstacle):
    """Same convention as the U-Net runs (train_modal.py): ignore = not drivable and not obstacle."""
    label = np.full(floor_label.shape, BACKGROUND, np.uint8)
    label[(floor_label == 0) & ~obstacle] = VOID
    label[obstacle] = OBSTACLE
    return label


class ISODForESANet(Dataset):
    def __init__(self, root, sites, preprocessor):
        self.root, self.preprocessor = root, preprocessor
        self.items = [(s, i) for s in sites for i in range(100)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        rgb, depth, label = load_frame(self.root, site, i)
        # ESANet's ToTensor casts depth to float32; values stay in millimetres
        sample = {"image": rgb, "depth": depth.astype(np.float32), "label": label}
        return self.preprocessor(sample)


def train_statistics(root, train_sites):
    """Depth mean/std over VALID (non-zero) train pixels, as ESANet does for raw depth, and
    ESANet's median-frequency class weights. Train sites only, full native resolution."""
    s = ss = n = 0.0
    pixels = np.zeros(3)             # pixels per class (void, background, obstacle)
    image_pixels_with = np.zeros(3)  # summed image size of frames in which the class occurs
    for site in train_sites:
        for i in range(100):
            _, depth, label = load_frame(root, site, i)
            d = depth[depth > 0].astype(np.float64)
            s, ss, n = s + d.sum(), ss + (d * d).sum(), n + d.size
            counts = np.bincount(label.ravel(), minlength=3)
            pixels += counts
            image_pixels_with += (counts > 0) * label.size
    mean = s / n
    frequency = pixels[1:] / image_pixels_with[1:]  # without void, as in ESANet
    weights = np.median(frequency) / frequency
    return {"depth_mean": mean, "depth_std": float(np.sqrt(ss / n - mean * mean)),
            "class_weights": weights.tolist(), "class_pixels": pixels.tolist(),
            "train_sites": sorted(train_sites)}


def git_revision(path):
    """Commit of a checkout, '+dirty' if it has uncommitted changes (for config.json)."""
    run = lambda *cmd: subprocess.run(["git", "-C", path, *cmd], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD") + ("+dirty" if run("status", "--porcelain") else "")


def seed_worker(worker_id):
    """ESANet's augmentation draws from np.random; without this every worker would draw the
    same numbers. torch.initial_seed() differs per worker and follows the loader's generator."""
    np.random.seed(torch.initial_seed() % 2**32)


# --------------------------------------------------------------------------- model
def esanet_args(esanet_dir, modality, epochs, pretrained_dir):
    """The authors' own argument parser, so every default is theirs, not retyped here."""
    sys.path.insert(0, esanet_dir)
    from src.args import ArgumentParserRGBDSegmentation
    p = ArgumentParserRGBDSegmentation()
    p.set_common_args()
    flags = ["--modality", modality, "--height", str(HEIGHT), "--width", str(WIDTH),
             "--epochs", str(epochs), "--raw_depth"]
    flags += ["--pretrained_dir", pretrained_dir] if pretrained_dir else ["--no_imagenet_pretraining"]
    return p.parse_args(flags)


def forward(model, batch, modality, device):
    if modality == "rgbd":
        return model(batch["image"].to(device), batch["depth"].to(device))
    return model((batch["image"] if modality == "rgb" else batch["depth"]).to(device))


def obstacle_counts(logits, label):
    """Pixel counts for the obstacle class, void pixels excluded (eval.py does the same)."""
    pred = logits.argmax(1) == (OBSTACLE - 1)  # network classes: 0 = background, 1 = obstacle
    true = label == OBSTACLE
    valid = label != VOID
    pred = pred & valid
    return [int((pred & true).sum()), int((pred | true).sum()), int(pred.sum()), int(true.sum())]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esanet", required=True, help="path to the TUI-NICR/ESANet checkout")
    ap.add_argument("--pretrained_dir", default=None,
                    help="folder with r34_NBt1D.pth (ImageNet); omit only for --smoke")
    ap.add_argument("--root", required=True, help="isod_work: ISOD/data, ISOD_gt, splits.json")
    ap.add_argument("--out", required=True, help="approved storage for runs (NOT the home NFS)")
    ap.add_argument("--modality", required=True, choices=["rgb", "depth", "rgbd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=None,
                    help="override ESANet's batch size (8); only for CPU smoke tests")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 3 batches, 10 test frames")
    a = ap.parse_args()
    if not a.pretrained_dir and not a.smoke:
        raise SystemExit("--pretrained_dir is required for real runs (ImageNet encoder weights)")
    if a.smoke:
        a.epochs = 2

    args = esanet_args(a.esanet, a.modality, a.epochs, a.pretrained_dir)
    if a.batch:
        args.batch_size = a.batch
    from src.build_model import build_model
    from src.preprocessing import get_preprocessor
    from src.utils import CrossEntropyLoss2d

    name = f"esanet_r34nbt1d_{a.modality}_s{a.seed}" + ("_smoke" if a.smoke else "")
    run_dir = f"{a.out}/{name}"
    if os.path.exists(f"{run_dir}/history.csv") and not a.smoke:
        raise SystemExit(f"{run_dir} already has a history.csv; choose another --out or --seed")
    os.makedirs(run_dir, exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    with open(f"{a.root}/splits.json") as f:
        splits = json.load(f)
    assert not set(splits["train"]) & (set(splits["val"]) | set(splits["test"])), "split overlap"

    stats_path = f"{a.out}/isod_train_stats.json"  # shared by every run in --out
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            st = json.load(f)
        assert st["train_sites"] == sorted(splits["train"]), f"{stats_path}: different split"
    else:
        print("computing depth mean/std and class weights on the train sites ...", flush=True)
        st = train_statistics(a.root, splits["train"])
        with open(stats_path, "w") as f:
            json.dump(st, f, indent=1)
    print("train statistics:", {k: v for k, v in st.items() if k != "train_sites"}, flush=True)

    def preprocessor(phase):
        return get_preprocessor(depth_mean=st["depth_mean"], depth_std=st["depth_std"],
                                depth_mode="raw", height=HEIGHT, width=WIDTH, phase=phase,
                                train_random_rescale=(args.aug_scale_min, args.aug_scale_max))

    train_set = ISODForESANet(a.root, splits["train"], preprocessor("train"))
    val_set = ISODForESANet(a.root, splits["val"], preprocessor("test"))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=a.workers, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=a.workers)

    with contextlib.redirect_stdout(io.StringIO()):  # build_model prints the whole network
        model, device = build_model(args, n_classes=2)  # background, obstacle (+ void, ignored)
    loss_fn = CrossEntropyLoss2d(weight=np.array(st["class_weights"], np.float32), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)  # as ESANet
    # as ESANet: one scheduler step per EPOCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs, div_factor=25, pct_start=0.1,
        anneal_strategy="cos", final_div_factor=1e4)

    n_params = sum(p.numel() for p in model.parameters())
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump({"our_args": vars(a), "esanet_args": vars(args), "train_stats": st,
                   "n_params": n_params, "torch": torch.__version__,
                   "our_git": git_revision(os.path.dirname(os.path.abspath(__file__))),
                   "esanet_git": git_revision(a.esanet),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "selection": "best validation obstacle IoU (void excluded), 448x640"},
                  f, indent=1, default=str)

    history = open(f"{run_dir}/history.csv", "w")
    history.write("epoch,lr,train_loss,val_loss,val_IoU,val_precision,val_recall,seconds\n")
    best_iou, t_start = -1.0, time.time()
    max_batches = 3 if a.smoke else None
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        train_losses = []
        for b, batch in enumerate(train_loader):
            if max_batches and b >= max_batches:
                break
            targets = [batch["label"].to(device)] + \
                      [batch["label_down"][r].to(device) for r in sorted(batch["label_down"])]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(loss_fn(forward(model, batch, a.modality, device), targets))
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch + 1}, batch {b}")
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses, counts = [], np.zeros(4)
        with torch.no_grad():
            for b, batch in enumerate(val_loader):
                if max_batches and b >= max_batches:
                    break
                label = batch["label"].to(device)
                logits = forward(model, batch, a.modality, device)  # eval mode: full-res only
                val_losses.append(loss_fn([logits], [label])[0].item())  # same weighting as train
                counts += obstacle_counts(logits, label)
        inter, union, pred, true = counts
        iou, prec, rec = inter / max(1, union), inter / max(1, pred), inter / max(1, true)
        history.write(f"{epoch + 1},{lr:.6g},{np.mean(train_losses):.5f},{np.mean(val_losses):.5f},"
                      f"{iou:.5f},{prec:.5f},{rec:.5f},{time.time() - t_start:.1f}\n")
        history.flush()
        print(f"epoch {epoch + 1}/{args.epochs} lr={lr:.4g} train_loss={np.mean(train_losses):.4f} "
              f"val_loss={np.mean(val_losses):.4f} val_IoU={iou:.4f}", flush=True)
        if iou > best_iou:  # overwrite: one checkpoint per run, weights only
            best_iou = iou
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "val_IoU": float(iou)},
                       f"{run_dir}/best.pt")
    history.close()

    # test masks at native resolution from the selected checkpoint, then the shared scorer
    ckpt = torch.load(f"{run_dir}/best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_set = ISODForESANet(a.root, splits["test"], preprocessor("test"))
    n_test = 10 if a.smoke else len(test_set)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for k in range(n_test):
            site, i = test_set.items[k]
            batch = {key: v[None] for key, v in test_set[k].items() if key in ("image", "depth")}
            logits = forward(model, batch, a.modality, device)
            logits = F.interpolate(logits, (NATIVE_H, NATIVE_W), mode="bilinear", align_corners=False)
            mask = (logits.argmax(1)[0] == OBSTACLE - 1).cpu().numpy().astype(np.uint8) * 255
            os.makedirs(f"{run_dir}/pred_test/{site}", exist_ok=True)
            Image.fromarray(mask).save(f"{run_dir}/pred_test/{site}/{i}.png")
    dt = time.time() - t0
    result = {"best_epoch": ckpt["epoch"], "best_val_IoU": ckpt["val_IoU"], "n_params": n_params,
              "test_frames_written": n_test, "fps_incl_io": n_test / dt,
              "train_hours": (time.time() - t_start) / 3600,
              "peak_gpu_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
    if not a.smoke:  # scored by eval.py, the same code that scored every other method
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import eval as isod_eval
        import isod
        isod.GT, isod.DATA, isod.SPLITS = (f"{a.root}/ISOD_gt", f"{a.root}/ISOD/data",
                                           f"{a.root}/splits.json")
        result["test"] = isod_eval.run("masks", "test", f"{run_dir}/pred_test")
    with open(f"{run_dir}/test_eval.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
