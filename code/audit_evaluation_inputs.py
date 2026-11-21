"""Read-only audit of references and metadata before object-level evaluation.

No predictions enter this audit. Requires numpy and Pillow, not a GPU.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    with Image.open(path) as image:
        values = np.array(image)
    # The web annotator saved grayscale masks as opaque RGBA PNGs.
    # Alpha is not foreground and must not be interpreted as an item ID.
    if values.ndim == 3:
        if not (np.array_equal(values[..., 0], values[..., 1]) and
                np.array_equal(values[..., 0], values[..., 2])):
            raise ValueError(f"Expected grayscale channels: {path}")
        if values.shape[-1] == 4 and not np.all(values[..., 3] == 255):
            raise ValueError(f"Unexpected mask transparency: {path}")
        values = values[..., 0]
    return values


def audit(root):
    data = root / "data"
    splits = json.loads((data / "splits.json").read_text())
    problems = []
    result = {"scope": "read-only input integrity and analysis eligibility; not prediction quality",
              "splits": splits, "references": {}, "problems": problems}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[a]) & set(splits[b]):
            problems.append(f"Site leakage: {a}/{b}")
    raw_catalog = {}
    for split, sites in splits.items():
        for site in sites:
            by_kind = {}
            for kind in ("rgb", "depth"):
                paths = list((data / "ISOD" / "data" / site / kind).glob("*.png"))
                frames = {int(p.stem) for p in paths if p.stem.isdigit()}
                by_kind[kind] = len(frames)
                if frames != set(range(100)):
                    problems.append(f"Incomplete raw frame catalogue: {site}/{kind}")
            raw_catalog[site] = dict(split=split, **by_kind)
    result["isod_raw_catalogue"] = raw_catalog

    sources = [("ISOD_reference_train_v1", "train", 1400),
               ("ISOD_reference_val_v1", "val", 300),
               ("ISOD_reference_test_v1", "test", 30),
               ("ISOD_sam_train_v2", "train", 1400)]
    for name, split, count in sources:
        base = data / name
        manifest = json.loads((base / "manifest.json").read_text())
        records = manifest["frames"]
        statuses, shapes = Counter(), Counter()
        empty, identities = [], set()
        for row in records:
            key = (row["site"], row["frame"])
            if key in identities:
                problems.append(f"Duplicate frame: {name}/{key}")
            identities.add(key)
            if row["site"] not in splits[split]:
                problems.append(f"Wrong site split: {name}/{key}")
            path = base / row["path"]
            mask = load_image(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                problems.append(f"Hash mismatch: {path}")
            if list(mask.shape) != row["shape"] or not set(np.unique(mask)).issubset({0, 255}):
                problems.append(f"Invalid mask shape/encoding: {path}")
            for kind in ("rgb", "depth"):
                raw = data / "ISOD" / "data" / row["site"] / kind / f"{row['frame']}.png"
                if not raw.exists():
                    problems.append(f"Missing input: {raw}")
                else:
                    with Image.open(raw) as image:
                        if image.size != (mask.shape[1], mask.shape[0]):
                            problems.append(f"Input dimensions mismatch: {raw}")
            statuses[row.get("review_status", "unspecified")] += 1
            shapes[str(list(mask.shape))] += 1
            if not mask.any():
                empty.append(dict(site=row["site"], frame=row["frame"]))
        if len(records) != count:
            problems.append(f"Wrong frame count: {name}: {len(records)} != {count}")
        if split != "test":
            expected = {(site, i) for site in splits[split] for i in range(100)}
            if identities != expected:
                problems.append(f"Incomplete or unexpected frame IDs: {name}")
        result["references"][name] = dict(frames=len(records), status_counts=dict(statuses),
                                              shapes=dict(shapes), empty_frames=len(empty),
                                              empty_frame_ids=empty)
    result["isod_test_coverage"] = dict(reviewed_frames=30, total_split_frames=300,
                                          status="provisional until coverage is expanded")

    with (root / "data_collection" / "items.csv").open() as stream:
        items = list(csv.DictReader(stream))
    legend = defaultdict(list)
    for item in items:
        legend[item["object_set"]].append(item)
    grouped = [dict(item_id=r["item_id"], label=r["label"], notes=r["notes"])
               for r in items if r["multi"].strip()]
    missing_dimensions = {key: sum(not r[key].strip() for r in items)
                          for key in ("height_mm", "long_mm", "short_mm")}
    result["gte_inventory"] = dict(authority="data_collection/items.csv",
        inventory_sha256=hashlib.sha256((root / "data_collection" / "items.csv").read_bytes()).hexdigest(),
        obsolete_for_measurements="data_collection/objects.csv is the earlier planned inventory",
        inventory_entries=len(items), sets={k: len(v) for k, v in legend.items()},
        multi_enabled_entries=grouped, missing_dimension_fields=missing_dimensions,
        warning="Some IDs designate multiple physical objects; dimensions are not implied by item names.")

    manual = data / "OWN_manual"
    masks = sorted(p for p in manual.glob("*.png") if not p.stem.endswith("_id"))
    observations, unmatched_foreground, stray_ids = [], 0, 0
    runs, empty_frames = defaultdict(list), []
    reference_fingerprint = hashlib.sha256()
    available_pairings = defaultdict(set)
    for path in masks:
        recording, frame_text = path.stem.rsplit("_", 1)
        frame = int(frame_text)
        parts = recording.split("_")
        camera, background, object_set = parts[:3]
        distance = int(parts[3]) if object_set != "empty" else None
        mask = load_image(path) > 0
        id_path = manual / f"{path.stem}_id.png"
        ids = load_image(id_path)
        for member in (path, id_path):
            reference_fingerprint.update(member.name.encode())
            reference_fingerprint.update(hashlib.sha256(member.read_bytes()).digest())
        raw = load_image(data / "OWN" / recording / "depth" / f"{frame}.png")
        with Image.open(data / "OWN" / recording / "rgb" / f"{frame}.png") as image:
            rgb_size = image.size
        if mask.shape != ids.shape or mask.shape != raw.shape or rgb_size != (mask.shape[1], mask.shape[0]):
            problems.append(f"GTE dimensions mismatch: {recording}/{frame}")
            continue
        unmatched_foreground += int((mask & (ids == 0)).sum())
        stray_ids += int(((ids > 0) & ~mask).sum())
        values = set(int(x) for x in np.unique(ids)) - {0}
        known = set(range(1, len(legend.get(object_set, [])) + 1))
        if values - known:
            problems.append(f"Unknown item IDs: {recording}/{frame}: {sorted(values - known)}")
        if object_set == "empty":
            empty_frames.append(path.stem)
            if mask.any():
                problems.append(f"Nonempty reference for empty scene: {path.stem}")
        for idx, item in enumerate(legend.get(object_set, []), 1):
            selected = mask & (ids == idx)
            if not selected.any():
                continue
            depth = raw[selected]
            valid = depth[depth > 0]
            ys, xs = np.nonzero(selected)
            observations.append(dict(recording=recording, frame=frame,
                item_id=item["item_id"], item_label=item["label"],
                annotation_unit="multi_enabled_id" if item["multi"].strip() else "item",
                camera=camera, background=background, nominal_ground_distance_cm=distance,
                area_px=int(selected.sum()), bbox_width_px=int(xs.max()-xs.min()+1),
                bbox_height_px=int(ys.max()-ys.min()+1),
                valid_depth_fraction=float(len(valid) / len(depth)),
                median_mask_depth_mm=float(np.median(valid)) if len(valid) else None))
        runs[recording].append(frame)
        available_pairings[(camera, object_set, distance)].add(background)
    if len(masks) != 320 or len(runs) != 64:
        problems.append(f"GTE counts: {len(masks)} frames, {len(runs)} recordings")
    if unmatched_foreground or stray_ids:
        problems.append(f"GTE ID/union disagreement: {unmatched_foreground} unassigned foreground, {stray_ids} stray ID pixels")
    for recording, frames in runs.items():
        if sorted(frames) != list(range(5)):
            problems.append(f"Incomplete recording: {recording}")
    result["gte_references"] = dict(frames=len(masks), recordings=len(runs),
        reference_fingerprint_sha256=reference_fingerprint.hexdigest(),
        empty_frames=len(empty_frames), empty_recordings=len({x.rsplit('_',1)[0] for x in empty_frames}),
        item_or_group_observations=len(observations),
        distinct_item_or_group_recordings=len({(r["recording"],r["item_id"]) for r in observations}),
        multi_enabled_observations=sum(r["annotation_unit"] == "multi_enabled_id" for r in observations),
        observations_under_20_pixels=sum(r["area_px"] < 20 for r in observations),
        foreground_pixels_without_id=unmatched_foreground, id_pixels_outside_foreground=stray_ids,
        both_backgrounds_available_for=[list(k) for k,v in available_pairings.items()
                                        if {"floor","poster"}.issubset(v)],
        notes=["Five frames are repeats of a static scene, not five independent scenes.",
               "Median mask depth can be floor leakage; it is not validated object distance.",
               "Capture configurations change camera height and tilt together."])
    return result, observations


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result, rows = audit(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "input_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "gte_reference_observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["recording", "frame", "item_id"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"reference_counts": {k: v["frames"] for k,v in result["references"].items()},
                      "gte": result["gte_references"], "inventory": result["gte_inventory"],
                      "problems": result["problems"]}, indent=2))
    raise SystemExit(bool(result["problems"]))


if __name__ == "__main__":
    main()
