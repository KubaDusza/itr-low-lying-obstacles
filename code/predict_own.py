"""Run the frozen ISOD-trained models on our own recordings.

No fine-tuning, no threshold change, the same depth constants: the question is whether what
was learned on ISOD transfers to a different camera, room and mounting height.

  python predict_own.py --root /data/cs_courses/jdusza_itr            # the modality arms
  python predict_own.py --root ... --legacy                           # yolo / unet / deeplab

Reads <root>/OWN/<run>/{rgb,depth}/*.png, writes <root>/OWN_pred/<name>/<run>/<i>.png.

Our frames are 640x362 against ISOD's 640x422, so they are resized to 640x384: the same
1.064 vertical stretch the training pipeline applied when it resized 422 to 448. Without
that, part of any measured drop would just be a change of aspect ratio.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

import modal

SIZE = (640, 384)
OUT = (640, 362)
LEGACY = (("unet_mobilenet_v2", "unet"), ("deeplabv3plus_mobilenet_v2", "deeplabv3plus"))


def frames(own, run):
    return sorted(glob.glob(f"{own}/{run}/rgb/*.png"),
                  key=lambda p: int(os.path.basename(p)[:-4]))


def inputs(own, run, i, stats):
    rgb = cv2.cvtColor(cv2.imread(f"{own}/{run}/rgb/{i}.png"), cv2.COLOR_BGR2RGB)
    raw = cv2.imread(f"{own}/{run}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
    return modal.to_tensors(rgb, raw, stats, SIZE)


def write(masks, out):
    os.makedirs(out, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(f"{out}/{i}.png", m)


def run_arm(root, own, runs, name, modality, stats, dev, blind=False):
    net = modal.load(root, name, modality, dev)
    n = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for run in runs:
            masks = []
            for i in range(len(frames(own, run))):
                rgb, dep, geo = inputs(own, run, i, stats)
                logit = modal.forward(net, modality, rgb[None].to(dev), dep[None].to(dev),
                                      geo[None].to(dev), blind)
                m = (torch.sigmoid(logit[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
                masks.append(cv2.resize(m, OUT, interpolation=cv2.INTER_NEAREST))
            write(masks, f"{root}/OWN_pred/{name}/{run}")
            n += len(masks)
    print(f"  {name}: {n} masks")


def run_yolo(root, own, runs, weights, conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    n = 0
    for run in runs:
        masks = []
        for p in frames(own, run):
            r = model.predict(p, conf=conf, imgsz=640, verbose=False)[0]
            h, w = r.orig_shape
            m = np.zeros((h, w), np.uint8)
            if r.masks is not None:
                # xy is in original-image coordinates; masks.data still has letterbox padding
                for poly in r.masks.xy:
                    if len(poly) >= 3:
                        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            masks.append(cv2.resize(m, OUT, interpolation=cv2.INTER_NEAREST))
        write(masks, f"{root}/OWN_pred/yolo/{run}")
        n += len(masks)
    print(f"  yolo: {n} masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--own", default=None, help="defaults to <root>/OWN")
    ap.add_argument("--legacy", action="store_true",
                    help="the three original RGB models instead of the modality arms")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="checkpoint names to run; default: every modality arm found")
    a = ap.parse_args()

    own = a.own or f"{a.root}/OWN"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = sorted(d for d in os.listdir(own) if os.path.isdir(f"{own}/{d}/rgb"))
    stats = modal.load_depth_stats(a.root)
    print(f"{len(runs)} runs, device {dev}")

    if a.legacy:
        import segmentation_models_pytorch as smp
        for name, arch in LEGACY:
            ck = f"{a.root}/runs/{name}.pt"
            if not os.path.exists(ck):
                print(f"  missing {ck}")
                continue
            net = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[arch](
                encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
            net.load_state_dict(torch.load(ck, map_location=dev))
            net = net.to(dev).eval()
            n = 0
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for run in runs:
                    masks = []
                    for i in range(len(frames(own, run))):
                        rgb, _, _ = inputs(own, run, i, stats)
                        p = (torch.sigmoid(net(rgb[None].to(dev))[0, 0]) > 0.5)
                        masks.append(cv2.resize(p.cpu().numpy().astype(np.uint8) * 255, OUT,
                                                interpolation=cv2.INTER_NEAREST))
                    write(masks, f"{a.root}/OWN_pred/{name}/{run}")
                    n += len(masks)
            print(f"  {name}: {n} masks")
        w = f"{a.root}/runs/yolov8n_seg/weights/best.pt"
        if os.path.exists(w):
            run_yolo(a.root, own, runs, w)
        return

    names = a.arms
    if names is None:
        names = [os.path.basename(p)[:-3] for p in
                 sorted(glob.glob(f"{a.root}/runs/modal/*.pt"))]
        # the tuning and capacity runs are about ISOD only, not about transfer
        names = [n for n in names if not any(t in n for t in
                                             ("overfit", "floor", "long", "lr1e3", "lr3e3"))]
    for name in names:
        modality = name.split("_s")[0].replace("_scratch", "").replace("_blind", "")
        if modality not in modal.ARMS:
            print(f"  cannot tell the modality of {name}, skipped")
            continue
        run_arm(a.root, own, runs, name, modality, stats, dev, blind="_blind" in name)
    print(f"done -> {a.root}/OWN_pred/")


if __name__ == "__main__":
    main()
