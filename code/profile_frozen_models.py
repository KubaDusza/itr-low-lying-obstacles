"""Measure synchronized single-frame forward latency for the learned models."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


@torch.inference_mode()
def profile(model, inputs, warmup, repeats):
    model.eval().cuda()
    for _ in range(warmup):
        model(*inputs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        model(*inputs)
        end.record()
    torch.cuda.synchronize()
    milliseconds = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                input_shapes=[list(image.shape) for image in inputs],
                forward_median_ms=float(np.median(milliseconds)),
                forward_p90_ms=float(np.percentile(milliseconds, 90)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esanet", type=Path, required=True)
    parser.add_argument("--yolo-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("Use positive warmup and repeat counts")

    sys.path.insert(0, str(args.esanet.resolve()))
    import segmentation_models_pytorch as smp
    from ultralytics import YOLO
    from src.args import ArgumentParserRGBDSegmentation
    from src.build_model import build_model

    esa_args = ArgumentParserRGBDSegmentation()
    esa_args.set_common_args()
    esa = esa_args.parse_args([
        "--dataset", "isod", "--raw_depth", "--encoder", "resnet34",
        "--encoder_block", "NonBottleneck1D", "--nr_decoder_blocks", "3",
        "--context_module", "ppm", "--decoder_channels_mode", "decreasing",
        "--encoder_decoder_fusion", "add", "--fuse_depth_in_rgb_encoder", "SE-add",
        "--upsampling", "learned-3x3-zeropad", "--height", "448", "--width", "640",
    ])
    esa.pretrained_on_imagenet = False
    rgb = torch.zeros(1, 3, 448, 640, device="cuda")
    depth = torch.zeros(1, 1, 448, 640, device="cuda")
    scores = {}
    for modality, inputs in (("rgb", (rgb,)), ("depth", (depth,)), ("rgbd", (rgb, depth))):
        esa.modality = modality
        model, _ = build_model(esa, n_classes=2)
        scores[f"esanet_{modality}"] = profile(model, inputs, args.warmup, args.repeats)
        del model
    unet = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, classes=1)
    scores["unet_rgb"] = profile(unet, (rgb,), args.warmup, args.repeats)
    del unet
    selection = json.loads((args.yolo_run / "selection.json").read_text())
    weights = args.yolo_run / "fit" / "weights" / selection["best"]
    yolo = YOLO(str(weights)).model.float()
    scores["yolov8n_seg_rgb"] = profile(yolo, (rgb,), args.warmup, args.repeats)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, dtype="float32",
                  batch=1, warmup=args.warmup, repeats=args.repeats,
                  timing="CUDA events around model forward with inputs already on GPU",
                  models=scores)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
