"""Run the frozen ISOD-trained models on our own recordings.

No fine-tuning and no threshold changes: the point is whether what we learned on ISOD
transfers to a different camera, room and mounting height.

  /opt/venv/jdusza_venv/bin/python predict_own.py --root /data/cs_courses/jdusza_itr

Reads <root>/OWN/<run>/rgb/*.png, writes <root>/OWN_pred/<model>/<run>/<i>.png (255 = obstacle).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (640, 384)        # multiple of 32, same 1.06 vertical stretch train_seg.py applied
OUT = (640, 362)         # our frames


def seg_models(root, device):
    import segmentation_models_pytorch as smp
    out = {}
    for name, arch in (("unet_mobilenet_v2", smp.Unet), ("deeplabv3plus_mobilenet_v2", smp.DeepLabV3Plus)):
        ck = f"{root}/runs/{name}.pt"
        if not os.path.exists(ck):
            print(f"  missing {ck}"); continue
        net = arch(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
        net.load_state_dict(torch.load(ck, map_location=device))
        out[name] = net.to(device).eval()
    return out


def run_seg(net, paths, device):
    masks = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for p in paths:
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            x = (cv2.resize(rgb, SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - MEAN) / STD
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].float().to(device)
            m = (torch.sigmoid(net(x)[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
            masks.append(cv2.resize(m, OUT, interpolation=cv2.INTER_NEAREST))
    return masks


def run_yolo(weights, paths, conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    masks = []
    for p in paths:
        r = model.predict(p, conf=conf, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        m = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            # xy is in original-image coordinates; masks.data still carries letterbox padding
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        masks.append(cv2.resize(m, OUT, interpolation=cv2.INTER_NEAREST))
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/cs_courses/jdusza_itr")
    ap.add_argument("--own", default=None, help="defaults to <root>/OWN")
    a = ap.parse_args()
    own = a.own or f"{a.root}/OWN"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = sorted(d for d in os.listdir(own) if os.path.isdir(f"{own}/{d}/rgb"))
    print(f"{len(runs)} runs, device {device}")

    nets = seg_models(a.root, device)
    yolo_w = f"{a.root}/runs/yolov8n_seg/weights/best.pt"
    jobs = [(n, None) for n in nets] + ([("yolo", yolo_w)] if os.path.exists(yolo_w) else [])

    for name, w in jobs:
        n_img = 0
        for run in runs:
            paths = sorted(glob.glob(f"{own}/{run}/rgb/*.png"),
                           key=lambda p: int(os.path.basename(p)[:-4]))
            masks = run_yolo(w, paths) if name == "yolo" else run_seg(nets[name], paths, device)
            d = f"{a.root}/OWN_pred/{name}/{run}"
            os.makedirs(d, exist_ok=True)
            for i, m in enumerate(masks):
                cv2.imwrite(f"{d}/{i}.png", m)
            n_img += len(masks)
        print(f"  {name}: {n_img} masks")
    print(f"done -> {a.root}/OWN_pred/")


if __name__ == "__main__":
    main()
