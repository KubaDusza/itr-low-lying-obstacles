"""Pre-label our own recordings with SAM 3, so hand annotation starts from something.

SAM 3 does promptable concept segmentation: you give it a short noun phrase and it returns a
mask for *every* instance of that concept in the image. We know exactly what is on the floor in
each run -- the object set was chosen before the shutter opened and is recorded in
data_collection/dataset_index.csv -- so we can tell it, rather than asking an open-vocabulary
model to guess what counts as an obstacle.

Prompting, one phrase at a time
-------------------------------
PCS takes one concept per call and returns all of its instances, so there is no "describe the
whole scene in a sentence" mode, and a long sentence would be the wrong thing anyway: the model
was trained on short noun phrases. Each object set therefore gets a handful of phrases, written
from the photographs in figures/dataset/, and every phrase is run separately and the results
merged. That is better than one generic prompt for three reasons:

  * recall -- "cable" and "electrical plug" fire on different parts of the same object, and the
    union covers the whole thing where either alone clips it;
  * precision -- a phrase that names the actual object beats "obstacle", which SAM 3 has no
    concept of;
  * traceability -- every kept instance carries the phrase that produced it, so a bad phrase can
    be spotted and dropped instead of quietly polluting every mask.

Phrases are batched across frames in one forward pass, so "one at a time" costs nothing.

What gets thrown away
---------------------
Text prompts alone would also return the measuring tape, the cardboard distance tubes and
whoever was standing in shot. Three filters run over the raw instances:

  * NEGATIVE phrases are segmented too and their union becomes a veto mask; an instance covered
    by it is dropped. These are the fixtures that lie *on the floor*, where geometry cannot help.
  * the floor mask from own_gt.py (floor.png, the fitted ground plane) must contain a reasonable
    share of the instance, which removes the wall, the desks and anything above knee height.
  * an area window, the same one own_gt.py uses, drops single-pixel noise and whole-floor blobs.

Nothing here is ground truth. It is a starting mask for code/annotate.py, which is where the
masks are actually decided.

  python code/sam3_prelabel.py --list                     # print the prompts, touch nothing
  python code/sam3_prelabel.py --root /data/cs_courses/jdusza_itr   # on a GPU server

Writes <root>/OWN/<run>/sam/<i>.png (255 = obstacle, the union),
       <root>/OWN/<run>/sam/<i>_id.png (per-instance index, 1..N, for later unpicking),
       <root>/OWN/<run>/sam/<i>.json (phrase, score and box per instance) and
       <root>/OWN/<run>/sam/preview_<i>.jpg.
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "facebook/sam3"

# The objects actually on the floor, read off the photographs in figures/dataset/.
# Short noun phrases: that is what SAM 3's concept head was trained on.
PROMPTS = {
    # 1_cables.jpg: white extension lead with a moulded plug, thin white USB-C lead,
    # black mains lead with a European plug, thin black USB lead.
    "cables": ["cable", "power cord", "white cable", "black cable",
               "electrical plug", "extension cord"],
    # 2_tapes.jpg: five rolls, red/black insulating, yellow, wide beige masking, small yellow,
    # and a very small red one.
    "tapes": ["roll of tape", "adhesive tape roll", "masking tape roll", "insulating tape"],
    # 3_hexkeys.jpg: four bare steel allen keys of decreasing size plus an aluminium
    # carpenter's square. Thin, specular, and the hard case for colour.
    "hexkeys": ["hex key", "allen key", "metal rod", "carpenter square", "metal ruler"],
    # 4_random.jpg: Rubik's cube, wire keycap puller, green ballpoint pen, yellow screwdriver.
    "random": ["screwdriver", "rubik's cube", "ballpoint pen", "small metal tool"],
    # 5_screws.jpg: eye bolt, two machine screws, a wood screw, a small bolt, and a loose pile.
    "screws": ["screw", "bolt", "eye bolt", "pile of screws", "metal fastener"],
    # 6_chess.jpg: cream queen and pawn, black queen and pawn. The positive control for depth.
    "chess": ["chess piece", "chess pawn", "chess queen"],
    # 7_rags.jpg: flat white paper towel, crumpled white tissue, folded pink microfibre cloth,
    # bundled purple microfibre cloth.
    # The two white items are the point of the set and the hardest thing in it: white paper on
    # a white poster. They need phrases of their own -- "cloth" alone finds only the coloured two.
    "rags": ["cloth", "microfiber cloth", "folded cloth", "crumpled cloth",
             "paper towel", "crumpled tissue", "white tissue", "crumpled white paper",
             "wrinkled paper", "white rag"],
    # the noise floor: nothing was placed, so nothing should be found.
    "empty": [],
}

# Fixtures that lie on the floor, inside the scored region, in every run. The floor mask cannot
# reject these, so SAM 3 is asked to find them and they are subtracted.
NEGATIVE = ["tape measure", "measuring tape", "shoe", "foot", "cardboard tube",
            "cardboard roll", "trash can"]

# A negative phrase that overlaps the set's own concept deletes the objects we came for. The
# tapes set collides with two of them: prompted with "tape measure" or "cardboard roll" on a
# tapes run, SAM 3 returns the tape rolls themselves -- each covers all five of them at ~100% --
# and the veto then removes every one. The first full pass found 0.3 instances per frame on a
# set with five rolls in it. The rolls are compact and the measuring tape is a long thin strip,
# so the ground-plane and area filters still carry the load. "cardboard tube" is kept: it finds
# the distance tubes and leaves the rolls alone.
NEGATIVE_SKIP = {"tapes": {"tape measure", "measuring tape", "cardboard roll"}}

MIN_AREA, MAX_AREA = 20, 60000      # the window own_gt.py scores with
FLOOR_FRAC = 0.25                   # a kept instance must sit this far inside the ground plane
VETO_FRAC = 0.5                     # covered this much by a negative instance -> drop
DEDUP_IOU = 0.6                     # same object found by two phrases


def run_items(index_csv):
    """run -> (object_set, the human description of what was placed)."""
    out = {}
    if os.path.exists(index_csv):
        for r in csv.DictReader(open(index_csv)):
            out[r["run"]] = (r["object_set"], r["items"])
    return out


def frames(d):
    return sorted(glob.glob(f"{d}/*.png"), key=lambda p: int(os.path.basename(p)[:-4]))


def segment(model, processor, images, phrase, thr, device):
    """One phrase against a batch of frames. Returns a list of (mask, score, box) per frame."""
    import torch
    inputs = processor(images=images, text=[phrase] * len(images),
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=thr, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())
    out = []
    for r in res:
        masks = r["masks"].cpu().numpy().astype(np.uint8)
        scores = r["scores"].float().cpu().numpy()   # bf16 does not convert to numpy
        boxes = r["boxes"].float().cpu().numpy()
        out.append([(masks[i], float(scores[i]), boxes[i].tolist()) for i in range(len(scores))])
    return out


def keep(inst, floor, veto):
    """Area window, ground-plane overlap and the negative-phrase veto."""
    m, score, box = inst
    a = int(m.sum())
    if not (MIN_AREA <= a <= MAX_AREA):
        return None
    if float((m & (floor > 0)).sum()) / a < FLOOR_FRAC:
        return None
    if veto is not None and float((m & veto).sum()) / a > VETO_FRAC:
        return None
    return dict(area=a, score=score, box=[round(v, 1) for v in box])


def dedup(kept):
    """Two phrases naming the same object: keep the higher-scoring one."""
    kept = sorted(kept, key=lambda k: -k["score"])
    out = []
    for k in kept:
        if any(float((k["mask"] & o["mask"]).sum()) /
               max(1, int((k["mask"] | o["mask"]).sum())) > DEDUP_IOU for o in out):
            continue
        out.append(k)
    return out


def preview(rgb, insts):
    out = rgb.copy()
    for i, k in enumerate(insts):
        colour = [(60, 220, 60), (60, 160, 255), (240, 120, 60), (220, 60, 220)][i % 4]
        out[k["mask"] > 0] = (0.45 * np.array(colour) + 0.55 * out[k["mask"] > 0]).astype(np.uint8)
        cs, _ = cv2.findContours(k["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, colour, 1)
        x1, y1 = int(k["box"][0]), int(k["box"][1])
        cv2.putText(out, f"{k['phrase']} {k['score']:.2f}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT, help="holds OWN/; the GPU server uses /data/cs_courses/jdusza_itr")
    ap.add_argument("--index", default=f"{ROOT}/data_collection/dataset_index.csv")
    ap.add_argument("--model", default=MODEL,
                    help="facebook/sam3 needs an approved access request; a byte-identical "
                         "re-upload works the same while that is pending")
    ap.add_argument("--runs", default="", help="comma-separated substrings; default every run")
    ap.add_argument("--score", type=float, default=0.45, help="SAM 3 confidence threshold")
    ap.add_argument("--neg-score", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=5, help="frames per forward pass")
    ap.add_argument("--list", action="store_true", help="print the prompt table and exit")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    items = run_items(a.index)
    if a.list:
        for s, ps in PROMPTS.items():
            print(f"{s:8s} {', '.join(ps) if ps else '(nothing placed)'}")
        print(f"\nnegative {', '.join(NEGATIVE)}")
        for st, skip in NEGATIVE_SKIP.items():
            print(f"  minus {', '.join(sorted(skip))} for '{st}', which they would delete")
        return

    own = f"{a.root}/OWN"
    runs = sorted(d for d in os.listdir(own) if os.path.isdir(f"{own}/{d}/rgb"))
    if a.runs:
        want = [w.strip() for w in a.runs.split(",") if w.strip()]
        runs = [r for r in runs if any(w in r for w in want)]
    print(f"{len(runs)} runs")

    import torch
    from transformers import Sam3Model, Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()
    processor = Sam3Processor.from_pretrained(a.model)
    print(f"{a.model} on {device}")

    for run in runs:
        out_dir = f"{own}/{run}/sam"
        if os.path.isdir(out_dir) and not a.overwrite:
            print(f"  {run:30s} already done")
            continue
        oset = items.get(run, (run.split("_")[2] if len(run.split("_")) > 2 else "", ""))[0]
        phrases = PROMPTS.get(oset)
        if phrases is None:
            print(f"  {run:30s} no prompts for set '{oset}' -- skipped")
            continue

        paths = frames(f"{own}/{run}/rgb")
        bgr = [cv2.imread(p) for p in paths]
        images = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr]
        fl = cv2.imread(f"{own}/{run}/floor.png", cv2.IMREAD_GRAYSCALE)
        floor = fl if fl is not None else np.full(bgr[0].shape[:2], 255, np.uint8)

        # the fixtures, so they can be subtracted from whatever the positive phrases return
        veto = [np.zeros(bgr[0].shape[:2], np.uint8) for _ in images]
        for phrase in [n for n in NEGATIVE if n not in NEGATIVE_SKIP.get(oset, ())]:
            for i0 in range(0, len(images), a.batch):
                chunk = images[i0:i0 + a.batch]
                for j, insts in enumerate(segment(model, processor, chunk, phrase,
                                                  a.neg_score, device)):
                    for m, _, _ in insts:
                        veto[i0 + j] |= m

        found = [[] for _ in images]
        for phrase in phrases:
            for i0 in range(0, len(images), a.batch):
                chunk = images[i0:i0 + a.batch]
                for j, insts in enumerate(segment(model, processor, chunk, phrase,
                                                  a.score, device)):
                    f = i0 + j
                    for inst in insts:
                        k = keep(inst, floor, veto[f])
                        if k:
                            k.update(phrase=phrase, mask=inst[0])
                            found[f].append(k)

        os.makedirs(out_dir, exist_ok=True)
        n_total = 0
        for f, insts in enumerate(found):
            insts = dedup(insts)
            n_total += len(insts)
            union = np.zeros(bgr[f].shape[:2], np.uint8)
            ids = np.zeros(bgr[f].shape[:2], np.uint8)
            for i, k in enumerate(insts):
                union[k["mask"] > 0] = 255
                ids[k["mask"] > 0] = i + 1
            cv2.imwrite(f"{out_dir}/{f}.png", union)
            cv2.imwrite(f"{out_dir}/{f}_id.png", ids)
            cv2.imwrite(f"{out_dir}/preview_{f}.jpg", preview(bgr[f], insts),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            json.dump(dict(run=run, frame=f, object_set=oset,
                           items=items.get(run, ("", ""))[1], prompts=phrases,
                           instances=[{x: k[x] for x in ("phrase", "score", "area", "box")}
                                      for k in insts]),
                      open(f"{out_dir}/{f}.json", "w"), indent=1)
        print(f"  {run:30s} {oset:8s} {n_total:3d} instances over {len(found)} frames")

    print(f"done -> {own}/<run>/sam/")


if __name__ == "__main__":
    main()
