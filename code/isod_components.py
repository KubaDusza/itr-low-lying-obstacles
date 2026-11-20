"""Split the ISOD not-drivable mask into components and propose which are floor obstacles.

ISOD labels every pixel drivable or not. "Not drivable" holds three different things: the
walls, the furniture standing on the floor, and the objects lying on it. Only the last are
what this project detects, and the dataset does not separate them.

The mask itself is trusted. Nothing here adds a pixel that ISOD did not mark, and nothing
redraws a boundary: the only question asked of each pixel is which of the three it belongs
to, and that question is answered per connected component, not per pixel.

Two things the earlier `isod.obstacles()` got wrong, which this replaces:

  * it dropped the single largest component, so a second wall or cabinet that happens not to
    touch the first survived as an "obstacle". 46% of frames have more than one component
    over 5,000 px;
  * anything touching the wall-and-furniture region joined it and was thrown away with it. A
    garment brushing a cabinet leg is lost over a few pixels of contact.

The second is fixed before components are taken: the mask is eroded, which parts those thin
threads, components are found on the eroded mask, and each is then grown back inside the
original mask. An object connected to the furniture by a hair becomes its own component; an
object genuinely continuous with it does not, and the reviewer cuts that one by hand.

The proposal is deliberately geometric and semantic, never depth-based. Deciding with depth
what counts as an obstacle and then asking whether depth detects obstacles would answer
itself.

    python code/isod_components.py                 # all sites
    python code/isod_components.py --sites HL_BM   # one

Writes, per frame, to data/ISOD_comp/<site>/:
    <i>_comp.png   component index per pixel, 0 = drivable, 1..N = a component
    <i>.json       per component: area, box, the cues, and the proposed keep/drop
"""
import argparse
import json
import os

import cv2
import numpy as np

import isod

OUT = os.path.join(os.path.dirname(isod.GT), "ISOD_comp")
MIN_AREA = 20          # px, as in isod.py: below this a blob is label noise
ERODE = 2              # px of erosion before components, to part thin contacts
TOP_BAND = 0.12        # a component reaching this far up the frame is structure
ENCLOSED = 0.80        # share of a component's border that must face drivable floor


def components(label):
    """Component index per pixel of the not-drivable mask, thin contacts parted."""
    nd = (label == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1, 2 * ERODE + 1))
    core = cv2.erode(nd, k)
    n, seeds = cv2.connectedComponents(core, 8)
    if n <= 1:                                  # nothing survived the erosion
        n, seeds = cv2.connectedComponents(nd, 8)
        return n, seeds * nd
    # grow every seed back through the mask it came from, so no labelled pixel is lost and
    # the boundaries stay exactly where ISOD drew them
    out = seeds.astype(np.uint16)          # cv2 morphology takes uint16, not int32
    kernel = np.ones((3, 3), np.uint8)
    while True:
        grown = cv2.dilate(out, kernel)
        grown[nd == 0] = 0
        take = (out == 0) & (grown > 0)
        if not take.any():
            break
        out[take] = grown[take]
    return n, out.astype(np.int32)


def cues(comp, idx, nd, floor):
    """What is visible about one component without looking at depth."""
    m = comp == idx
    ys, xs = np.where(m)
    H, W = comp.shape
    area = int(m.sum())
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # the ring one pixel outside the component: what does it border on?
    ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) & ~m
    n_ring = int(ring.sum())
    on_floor = float((ring & (floor > 0)).sum()) / n_ring if n_ring else 0.0
    return dict(area=area, box=[x0, y0, x1, y1],
                top=round(float(y0) / H, 3),
                enclosed=round(on_floor, 3),
                touches_top=bool(y0 <= TOP_BAND * H),
                touches_side=bool(x0 == 0 or x1 >= W - 1),
                touches_bottom=bool(y1 >= H - 1))


def propose(c):
    """Keep means: lying on the floor. Drop means: wall, furniture, or structure.

    The rule the reviewer is checking, in order:
      1. anything reaching the top of the frame is a wall, a door or something standing;
      2. anything whose border faces floor nearly all the way round is lying on it;
      3. what is left is judged on how much of its border faces floor.
    """
    if c["touches_top"]:
        return False, "reaches the top of the frame"
    if c["enclosed"] >= ENCLOSED:
        return True, "surrounded by floor"
    if c["touches_side"] and c["enclosed"] < 0.5:
        return False, "runs off the side with little floor around it"
    return c["enclosed"] >= 0.5, "mostly floor around it" if c["enclosed"] >= 0.5 else "little floor around it"


def frame_components(site, i):
    _, _, label, _ = isod.frame(site, i)
    return frame_components_from_label(np.array(label))


def frame_components_from_label(label):
    """Components and their proposals for one label image. Taking the label rather than a
    frame id is what lets the review tool re-split a frame after a hand cut."""
    nd = (label == 0).astype(np.uint8)
    floor = (label > 0).astype(np.uint8)
    n, comp = components(label)
    out = np.zeros_like(comp, np.int32)
    rows, next_id = [], 1
    for k in range(1, n):
        if int((comp == k).sum()) < MIN_AREA:
            continue
        c = cues(comp, k, nd, floor)
        keep, why = propose(c)
        c.update(id=next_id, keep=keep, why=why)
        out[comp == k] = next_id
        rows.append(c)
        next_id += 1
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="", help="comma-separated; default every site")
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    sites = [s.strip() for s in a.sites.split(",") if s.strip()] or isod.sites()
    for s in sites:
        d = f"{OUT}/{s}"
        os.makedirs(d, exist_ok=True)
        kept = total = 0
        for i in range(a.frames):
            comp, rows = frame_components(s, i)
            cv2.imwrite(f"{d}/{i}_comp.png", comp.astype(np.uint8))
            json.dump(rows, open(f"{d}/{i}.json", "w"))
            kept += sum(r["keep"] for r in rows)
            total += len(rows)
        print(f"  {s:12s} {total:5d} components, {kept:5d} proposed as obstacles")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
