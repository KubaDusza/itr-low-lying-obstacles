"""Short GPU compute benchmark; synthetic inputs, no dataset or accuracy claims.

Uses the published ESANet architecture and the current experiment's batch/grid.
FP32 is the existing baseline; AMP is a separate throughput experiment.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from src.models.model import ESANet
    from src.models.model_one_modality import ESANetOneModality
    from src.utils import CrossEntropyLoss2d

    assert torch.cuda.is_available(), "Run inside a GPU allocation, not on a login node"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    results = {
        "kind": "synthetic_training_compute_only",
        "limitations": "Excludes input loading, augmentation, validation and checkpoint I/O; random initialization.",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "batch": 8, "height": 448, "width": 640,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "measurements": [],
    }
    for modality in ("rgb", "rgbd"):
        for precision in ("fp32", "amp_fp16"):
            torch.manual_seed(0)
            common = dict(height=448, width=640, num_classes=2,
                          pretrained_on_imagenet=False, encoder_block="NonBottleneck1D",
                          channels_decoder=[512, 256, 128], nr_decoder_blocks=[3, 3, 3],
                          context_module="ppm", encoder_decoder_fusion="add",
                          upsampling="learned-3x3-zeropad")
            if modality == "rgbd":
                model = ESANet(encoder_rgb="resnet34", encoder_depth="resnet34",
                               fuse_depth_in_rgb_encoder="SE-add", **common)
            else:
                model = ESANetOneModality(encoder="resnet34", input_channels=3,
                                         weighting_in_encoder="SE-add", **common)
            model = model.cuda().train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
            criterion = CrossEntropyLoss2d(torch.device("cuda"), [1., 1.])
            scaler = torch.amp.GradScaler("cuda", enabled=precision != "fp32")
            rgb = torch.randn(8, 3, 448, 640, device="cuda")
            depth = torch.randn(8, 1, 448, 640, device="cuda")
            targets = [torch.randint(1, 3, (8, 448 // scale, 640 // scale), device="cuda")
                       for scale in (1, 8, 16, 32)]
            durations = []
            for step in range(args.warmup + args.steps):
                if step == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=precision != "fp32"):
                    output = model(rgb, depth) if modality == "rgbd" else model(rgb)
                    loss = sum(criterion(output, targets))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if step >= args.warmup:
                    durations.append(elapsed)
            assert torch.isfinite(loss).item(), "Non-finite loss"
            median = statistics.median(durations)
            row = dict(modality=modality, precision=precision, steps=args.steps,
                       median_step_s=median, mean_step_s=statistics.mean(durations),
                       images_per_s=8 / median, train_compute_s_per_1400_images=175 * median,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       final_loss=float(loss.detach()))
            results["measurements"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del output, loss, model, optimizer, criterion, scaler, rgb, depth, targets
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
