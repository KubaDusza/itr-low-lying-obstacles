"""Train one U-Net + MobileNetV2 on one input, for the RGB / depth / RGB-D comparison.

Everything except the input is held fixed: splits, loss, augmentation, batch size, learning
rate, 15-epoch OneCycle schedule, final-epoch weights. Each arm is repeated over seeds, and
the comparison is paired: at a given seed every arm starts from identical weights and sees
identical batches in identical order (see modal.py, checked by `diagnose.py check`).

  CUDA_VISIBLE_DEVICES=0 python train_modal.py --modality rgbd --seed 0

The flags exist so that "depth is weak" cannot be dismissed as our mistake:

  --scratch          no ImageNet weights, for either arm. The published fairness device
                     when two arms live in different function spaces (Wang et al., ECCV
                     2018, Depth-aware CNN). If depth still loses with no pretraining
                     anywhere, the RGB-pretrained encoder was not the reason.
  --blind            train a fusion arm with the depth input replaced by a constant. Same
                     architecture, same parameter count, no depth information: the control
                     for "RGB-D only won because it is a bigger model".
  --limit / --epochs can the arm fit a handful of frames at all? A depth model that reaches
                     a high score on 50 training frames has the capacity to represent these
                     obstacles, so a low test score is about information, not capacity.
  --target floor     the same depth arm on a task depth is known to suit. If it scores well
                     there on this schedule, the schedule is not what fails on obstacles.

Writes <root>/runs/modal/<name>.pt, <name>_history.csv and <name>.json (config, validation
curve, and how large the depth path ended up).
"""
import argparse
import json
import os
import time

import torch

import modal


def depth_path_size(net, modality):
    """How much of the trained model is actually reading depth.

    Both fusion arms start with their depth path at zero, so they *could* stay there and
    make "fusion = RGB" true by construction while saying nothing about depth. These
    numbers say whether the path moved.
    """
    spec = modal.ARMS[modality]
    with torch.no_grad():
        if "two_stream" in spec:
            return {"gates": [round(float(g), 4) for g in net.gate]}
        if "expand" in spec:
            k = spec["expand"]
            for m in net.encoder.modules():
                for c in m.children():
                    if isinstance(c, torch.nn.Conv2d) and c.in_channels == 3 + k:
                        rgb, dep = c.weight[:, :3], c.weight[:, 3:]
                        return {"conv1_rgb_norm": float(rgb.norm()),
                                "conv1_depth_norm": float(dep.norm()),
                                "per_channel_ratio": float((dep.norm() / k ** 0.5) /
                                                           (rgb.norm() / 3 ** 0.5))}
    return {}


def validate(net, modality, va, dev, blind=False):
    net.eval()
    inter = union = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for rgb, dep, geo, y, w in va:
            out = modal.forward(net, modality, rgb.to(dev), dep.to(dev), geo.to(dev), blind)
            p = (torch.sigmoid(out[:, 0]) > 0.5).float().cpu() * w
            inter += (p * y).sum().item()
            union += ((p + y) > 0).float().sum().item()
    return inter / max(1, union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--modality", required=True, choices=modal.MODALITIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--target", default="obstacle", choices=["obstacle", "floor"])
    ap.add_argument("--limit", type=int, default=0, help="train on N frames only")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    ap.add_argument("--encoder", default="mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    dev = "cuda"
    name = modal.run_name(a.modality, a.seed, a.scratch, a.blind, a.tag)
    out = f"{a.root}/runs/modal"
    os.makedirs(out, exist_ok=True)

    stats = modal.load_depth_stats(a.root)
    modal.seed_all(a.seed)
    net = modal.build(a.modality, a.encoder, pretrained=not a.scratch).to(dev)
    tr, va = modal.loaders(a.root, stats, a.batch, a.seed, a.target, a.limit)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(tr))
    scaler = torch.amp.GradScaler()

    hist = open(f"{out}/{name}_history.csv", "w")
    hist.write("epoch,train_loss,val_IoU,seconds\n")
    curve, t0 = [], time.time()
    for ep in range(a.epochs):
        net.train()
        run = 0.0
        for rgb, dep, geo, y, w in tr:
            rgb, dep, geo = rgb.to(dev), dep.to(dev), geo.to(dev)
            y, w = y.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = modal.forward(net, a.modality, rgb, dep, geo, a.blind)
                loss = modal.masked_loss(logits, y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        iou = validate(net, a.modality, va, dev, a.blind)
        curve.append(iou)
        print(f"{name} epoch {ep + 1}/{a.epochs} loss={run / len(tr):.4f} val_IoU={iou:.4f}",
              flush=True)
        hist.write(f"{ep + 1},{run / len(tr):.5f},{iou:.5f},{time.time() - t0:.1f}\n")
        hist.flush()
    hist.close()

    torch.save(net.state_dict(), f"{out}/{name}.pt")
    info = dict(vars(a), name=name, val_curve=curve, val_IoU_final=curve[-1],
                val_IoU_best=max(curve), minutes=round((time.time() - t0) / 60, 2),
                params=sum(p.numel() for p in net.parameters()),
                train_frames=len(tr.dataset), depth_stats=stats,
                depth_path=depth_path_size(net, a.modality))
    with open(f"{out}/{name}.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in
                      ("name", "val_IoU_final", "minutes", "params", "depth_path")}, indent=1))


if __name__ == "__main__":
    main()
