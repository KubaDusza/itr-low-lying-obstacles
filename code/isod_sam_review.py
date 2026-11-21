"""Test SAM 3 as a reference-annotation aid for reviewed ISOD development frames.

Run cache on a reserved GPU with ISOD RGB available. Pass the resulting cache to
score on the machine holding the review masks and proposal maps. Neither step
changes the reference masks. This cache is separate from unattended SAM labels.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import isod


PROMPT_GROUPS = {
    "cables": ("cable", "power cord", "electrical wire"),
    "soft": ("cloth", "rag", "paper towel", "doormat"),
    "objects": ("shoe", "slipper", "toy", "tennis racket", "small hand tool"),
    "structure": ("desk", "chair leg", "cabinet", "table leg"),
}
CACHE_THRESHOLD = 0.10
SCORE_THRESHOLDS = (0.15, 0.30, 0.45, 0.60)
FURNITURE_THRESHOLDS = (0.30, 0.45, 0.60)
FLOOR_MARGIN = 15
CROP_PROMPTS = ("cable bundle", "power cables on the floor", "tangled electrical cables",
                "loose electrical cord", "extension cord", "power strip")
EXTRA_PROMPTS = {
    "obstacles": ("cable bundle", "extension cord", "power strip", "crayon", "fabric on the floor",
                  "piece of paper", "plastic bag", "backpack", "small box"),
    "structure": ("wall", "door", "baseboard", "desk leg", "table", "chair", "trash can",
                  "radiator", "shelving unit"),
}


def prompt_union(candidates, phrases, threshold, shape):
    union = np.zeros(shape, dtype=bool)
    for phrase, confidence, mask in candidates:
        if phrase in phrases and confidence >= threshold:
            union |= mask
    return union


def proposals(comp, base, nd, masks, groups):
    """Keep the baseline and vary prompt groups, cut geometry, and furniture exclusion."""
    yield "baseline", 0, "component", base
    furniture = {t: prompt_union(masks, PROMPT_GROUPS["structure"], t, nd.shape)
                 for t in FURNITURE_THRESHOLDS}
    cleaned = {t: base & ~veto for t, veto in furniture.items()}
    area = np.bincount(comp.ravel())
    floor_gap = cv2.distanceTransform(nd.astype(np.uint8), cv2.DIST_L2, 3)
    near_floor = floor_gap <= FLOOR_MARGIN
    for group, phrases in groups.items():
        for threshold in SCORE_THRESHOLDS:
            sam = prompt_union(masks, phrases, threshold, nd.shape) & nd
            yield group, threshold, "add", base | sam
            overlap = np.bincount(comp[sam], minlength=len(area))
            # Snap detections covering at least half of a source region to its original boundary.
            matched = (2 * overlap >= area) & (area > 0)
            matched[0] = False
            yield group, threshold, "add_snapped", base | sam | matched[comp]
            yield group, threshold, "add_near_floor", base | (sam & near_floor)
            for t, veto in furniture.items():
                yield group, threshold, f"add_with_veto_{t:.2f}", base | (sam & ~veto)
                yield group, threshold, f"veto_and_add_{t:.2f}", cleaned[t] | sam
            if group in ("cable bundle", "cropped cables"):
                yield group, threshold, "veto_0.45_and_add_near_floor", cleaned[0.45] | (sam & near_floor)
    for t, prediction in cleaned.items():
        yield "structure", t, "veto_only", prediction


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_cache(path, shape):
    with np.load(path) as data:
        if tuple(data["shape"]) != tuple(shape):
            raise ValueError(f"shape mismatch in {path}")
        prompts, scores, masks = data["prompts"], data["scores"], data["masks"]
        if not (len(prompts) == len(scores) == len(masks)):
            raise ValueError(f"candidate count mismatch in {path}")
        return [(str(p), float(s), np.unpackbits(m, count=np.prod(shape)).reshape(shape) > 0)
                for p, s, m in zip(prompts, scores, masks)]


def reviews():
    root = Path(isod.DATA_ROOT) / "ISOD_gt2"
    test = set(isod.load_splits()["test"])
    return [(p.parent.name, int(p.stem)) for p in sorted(root.glob("*/*.json"))
            if p.stem.isdigit() and p.parent.name not in test]


def cache_predictions(args):
    import torch
    import transformers
    from transformers import Sam3Model, Sam3Processor

    from sam3_prelabel import segment

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for SAM 3 inference")
    frames = json.loads(args.frames.read_text())
    prompts = (CROP_PROMPTS if args.lower_crop else
               tuple(p for group in PROMPT_GROUPS.values() for p in group))
    if args.source_context is not None:
        prompts += EXTRA_PROMPTS["obstacles"] + EXTRA_PROMPTS["structure"]
    if args.prompt:
        prompts = tuple(args.prompt)
    args.cache.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    provenance = dict(model=args.model, revision=args.revision, transformers=transformers.__version__,
                      prompts=prompts, detection_threshold=CACHE_THRESHOLD, mask_threshold=0.5)
    if args.lower_crop:
        provenance["crop"] = "bottom 55 percent"
    if args.source_context is not None:
        provenance["source_mask"] = dict(
            drivable_floor_rgb_scale=args.source_context,
            note="The original ISOD label dims drivable-floor pixels in the model input.")
    manifest = args.cache / "manifest.json"
    manifest_text = json.dumps(provenance, indent=2)
    if manifest.exists() and manifest.read_text() != manifest_text:
        raise SystemExit("Choose a new cache directory for a different model or prompt list")
    model = Sam3Model.from_pretrained(args.model, revision=args.revision,
                                     local_files_only=True, dtype=dtype).cuda().eval()
    processor = Sam3Processor.from_pretrained(args.model, revision=args.revision,
                                             local_files_only=True)
    manifest.write_text(manifest_text)

    for start in range(0, len(frames), args.batch):
        pending = [(s, i, args.cache / s / f"{i}.npz") for s, i in frames[start:start + args.batch]
                   if not (args.cache / s / f"{i}.npz").is_file()]
        if not pending:
            continue
        images = []
        for site, frame, _ in pending:
            rgb = cv2.imread(str(Path(isod.DATA) / site / "rgb" / f"{frame}.png"))
            if rgb is None:
                raise FileNotFoundError(f"{site}/{frame} RGB")
            if args.source_context is not None and args.source_context < 1:
                label = read_gray(Path(isod.DATA) / site / "label" / f"{frame}.png")
                rgb[label == 255] = np.rint(rgb[label == 255] * args.source_context).astype(np.uint8)
            images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        candidates = [[] for _ in pending]
        for prompt in prompts:
            offsets = [int(rgb.shape[0] * 0.45) if args.lower_crop else 0 for rgb in images]
            crops = [rgb[y:] for rgb, y in zip(images, offsets, strict=True)]
            found = segment(model, processor, crops, prompt, CACHE_THRESHOLD, "cuda")
            for dest, result, y in zip(candidates, found, offsets, strict=True):
                for mask, score, _ in result:
                    full = np.pad(mask, ((y, 0), (0, 0))) if y else mask
                    dest.append((prompt, score, np.packbits(full.reshape(-1) > 0)))
        for (site, frame, path), rgb, found in zip(pending, images, candidates, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, shape=rgb.shape[:2],
                                prompts=np.array([p for p, _, _ in found], dtype=str),
                                scores=np.array([s for _, s, _ in found], dtype=np.float32),
                                masks=np.array([m for _, _, m in found], dtype=np.uint8))
            os.replace(temporary, path)
        print(f"Cached through frame {min(start + args.batch, len(frames))}/{len(frames)}",
              flush=True)


def score(args):
    import isod_learn

    fields = ("site", "frame", "prompt_group", "threshold", "mode", "cut", "tp", "fp", "fn")
    positive_groups = {k: set(v) for k, v in PROMPT_GROUPS.items() if k != "structure"}
    positive_groups["all"] = set().union(*positive_groups.values())
    positive_groups.update({p: {p} for p in sorted(positive_groups["all"])})
    if args.crop_cache:
        positive_groups["cropped cables"] = set(CROP_PROMPTS)
        positive_groups.update({p: {p} for p in CROP_PROMPTS})
    ref_dir = Path(isod.DATA_ROOT) / "ISOD_gt2"
    samples = reviews()
    caches = [cache for cache in (args.cache, args.crop_cache) if cache is not None]
    for cache in caches:
        missing = [f"{s}/{i}" for s, i in samples if not (cache / s / f"{i}.npz").is_file()]
        if missing:
            raise SystemExit(f"{cache} is missing {len(missing)} frames: {', '.join(missing[:5])}")
    counts = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=np.int64)))
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for site, frame in samples:
            ref = read_gray(ref_dir / site / f"{frame}.png") > 0
            nd = read_gray(Path(isod.DATA) / site / "label" / f"{frame}.png") == 0
            comp, rows = isod_learn.load_comp(site, frame)
            base = np.isin(comp, [r["id"] for r in rows if r["keep"]])
            masks = []
            for cache in caches:
                masks.extend(read_cache(cache / site / f"{frame}.npz", ref.shape))
            cut = (ref_dir / site / f"{frame}_cut.png").exists()
            for group, threshold, mode, prediction in proposals(comp, base, nd, masks, positive_groups):
                tp = int((prediction & ref).sum())
                fp = int((prediction & ~ref).sum())
                fn = int((ref & ~prediction).sum())
                counts[site][(group, threshold, mode)] += tp, fp, fn
                writer.writerow(dict(site=site, frame=frame, prompt_group=group,
                                     threshold=threshold, mode=mode, cut=cut, tp=tp, fp=fp, fn=fn))

    def iou(values):
        return float(values[0] / max(1, values.sum()))

    baseline = ("baseline", 0, "component")
    if len(counts) < 2:
        return
    for site, held in sorted(counts.items()):
        development = [other for name, other in counts.items() if name != site]
        best = max(held, key=lambda setting: iou(sum(d[setting] for d in development)))
        print(f"{site}: baseline {iou(held[baseline]):.3f} → {iou(held[best]):.3f} IoU "
              f"using {best} selected on the other sites; FP/FN {held[baseline][1:].tolist()} "
              f"→ {held[best][1:].tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listed = commands.add_parser("frames")
    listed.add_argument("output", type=Path)
    cached = commands.add_parser("cache")
    cached.add_argument("--frames", required=True, type=Path)
    cached.add_argument("--cache", required=True, type=Path)
    cached.add_argument("--model", required=True)
    cached.add_argument("--revision", required=True)
    cached.add_argument("--batch", type=int, default=2)
    cached.add_argument("--lower-crop", action="store_true")
    cached.add_argument("--prompt", action="append", help="Run only this phrase. Repeat for more phrases.")
    cached.add_argument("--source-context", type=float,
                        help="Use the expanded vocabulary and scale RGB on drivable-floor pixels "
                             "by this factor. Use 1 for full-RGB control.")
    scored = commands.add_parser("score")
    scored.add_argument("--cache", required=True, type=Path)
    scored.add_argument("--crop-cache", type=Path)
    scored.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "frames":
        args.output.write_text(json.dumps(reviews(), indent=2))
    elif args.command == "cache":
        if args.batch < 1:
            parser.error("--batch must be positive")
        if args.source_context is not None:
            if args.lower_crop or not 0 <= args.source_context <= 1:
                parser.error("--source-context requires full frames and a scale between 0 and 1")
        cache_predictions(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
