# -*- coding: utf-8 -*-
"""Write ESANet's test-split obstacle masks in the format eval.py scores.

Run from inside the ESANet clone, which is where its `src` package lives:

  cd <root>/esanet
  python ../code/esanet/predict_esanet.py --ckpt_path results/isod/<run>/ckpt_best.pth \
      --dataset isod --dataset_dir <root>/esanet_isod --modality rgbd --raw_depth \
      --out <root>/pred/esanet/<run>

ESANet predicts three classes (void / floor / obstacle); the obstacle channel is taken as
the prediction and written as a 0/255 PNG at the original resolution. Scoring then goes
through exactly the same eval.py as every other arm, so the numbers sit in one table and
no metric difference can be mistaken for a model difference.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# run from inside the ESANet clone: its `src` package sits in the working directory, which
# is not on the path when a script is called by absolute path
sys.path.insert(0, os.getcwd())

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

OBSTACLE = 2            # class index in the exported labels: 0 void, 1 floor, 2 obstacle


def main():
    parser = ArgumentParserRGBDSegmentation(
        description="ESANet obstacle masks for the ISOD test split")
    parser.set_common_args()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--pred_split", type=str, default="test")
    args = parser.parse_args()

    args.pretrained_on_imagenet = False      # the checkpoint carries the weights
    _, data_loader, *_ = prepare_data(args, with_input_orig=True,
                                      split=args.pred_split)
    n_classes = data_loader.dataset.n_classes_without_void
    model, device = build_model(args, n_classes=n_classes)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"loaded {args.ckpt_path}: epoch {ckpt.get('epoch')} "
          f"validation checkpoint score {ckpt.get('best_miou')}")

    names = list(data_loader.dataset._filenames)
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    k = 0
    for sample in data_loader:
        image, depth = sample["image"].to(device), sample["depth"].to(device)
        label_orig = sample["label_orig"]
        _, h, w = label_orig.shape
        inputs = {"rgbd": (image, depth), "rgb": (image,),
                  "depth": (depth,)}[args.modality]
        logits = model(*inputs)
        logits = F.interpolate(logits, (h, w), mode="bilinear",
                               align_corners=False)
        # +1 because the void class is not predicted: channel 0 is class 1
        pred = torch.argmax(logits, dim=1).cpu().numpy() + 1
        for b in range(pred.shape[0]):
            site, frame = names[k].rsplit("_", 1)
            d = f"{args.out}/{site}"
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(f"{d}/{frame}.png",
                        ((pred[b] == OBSTACLE) * 255).astype(np.uint8))
            k += 1
    print(f"{k} masks -> {args.out}")


if __name__ == "__main__":
    main()
