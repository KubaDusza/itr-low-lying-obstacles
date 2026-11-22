"""Apply a validation-selected U-Net checkpoint to GTE-Floor RGB images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from train_seg import rgb_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gte", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frames = sorted(path for path in args.gte.glob("*/rgb/*.png") if path.stem.isdigit())
    if len(frames) != 320:
        raise ValueError(f"Expected 320 RGB frames, got {len(frames)}")
    net = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, classes=1)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net = net.cuda().eval()
    args.out.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for frame in frames:
            rgb = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
            prediction = torch.sigmoid(net(rgb_tensor(rgb)[None].cuda())[0, 0]) > 0.5
            mask = cv2.resize(prediction.cpu().numpy().astype(np.uint8) * 255,
                              (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            path = args.out / frame.parent.parent.name / frame.name
            path.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(path), mask):
                raise OSError(f"Could not save {path}")
    print(f"Exported {len(frames)} masks to {args.out}")


if __name__ == "__main__":
    main()
