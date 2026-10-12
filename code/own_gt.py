"""Ground truth for our own recordings, from a median background built across runs.

Every run in a group was shot from the same pose at the same tape line with a different set of
objects, so the per-pixel median over the group is the empty scene: an obstacle occupies a
pixel in one set out of seven, never in the majority. That beats differencing against the
dedicated empty run, because the tripod was nudged during the session, and it is the only
thing that works on the poster runs — the poster moves with the objects, but it is present in
every run of a distance group, so the median puts it in the background where it belongs.

Three corrections keep it honest:
  * every frame is registered to the group's anchor by phase correlation on the top third of
    the image, which is wall and furniture and never contains an obstacle;
  * frames are gain-matched to the background before differencing, since auto-exposure was on;
  * the scored region is the ground plane fitted to the group's median depth, so walls,
    furniture and anyone walking past fall outside it (ISOD's ignore convention).

  python own_gt.py

Writes data/OWN/<run>/gt/<i>.png (255 = obstacle), data/OWN/<run>/floor.png (255 = scored) and
gt_summary.json with a per-run reliability flag.
"""
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN = f"{ROOT}/data/OWN"
MIN_AREA = 20          # same as the ISOD ground truth
MAX_AREA = 60000       # bigger than this is the background moving, not an obstacle
LO, HI = 18, 45        # hysteresis on the intensity difference, after gain matching
PLANE_K = 3 * 0.0045   # 3 sigma of stereo noise, which grows as z^2
PLANE_TOL0 = 0.03
TOP = 0.35             # fraction of the image used for registration: wall, never floor

# intrinsics of the stored 640x422 frames: colour intrinsics, cropped by 60 px and scaled
S = 640 / 728
INTR = dict(fx=608.1510620117188 * S, fy=606.9262084960938 * S,
            ppx=(432.7842102050781 - 60) * S, ppy=242.90274047851562 * S)


def frames(run, sub):
    return sorted(glob.glob(f"{OWN}/{run}/{sub}/*.png"),
                  key=lambda p: int(os.path.basename(p)[:-4]))


def group_key(run):
    parts = run.split("_")
    return "_".join(parts[:2]), parts[-1]


def shift_to(img, anchor):
    """Translation of img relative to anchor, measured on the static top band."""
    h = int(img.shape[0] * TOP)
    a = cv2.cvtColor(anchor[:h].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(img[:h].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return dx, dy


def warp(img, dx, dy, nearest=False):
    if abs(dx) < 0.15 and abs(dy) < 0.15:
        return img
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, M, img.shape[1::-1],
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def floor_mask(z):
    """Fit the ground plane to the group's median depth; keep the pixels lying on it."""
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - INTR["ppx"]) / INTR["fx"] * z
    Y = (v - INTR["ppy"]) / INTR["fy"] * z
    ok = z > 0
    seed = ok & (v > h * 0.55)
    A = np.stack([X[seed], Y[seed], np.ones(int(seed.sum()))], 1)
    zz = z[seed]
    coef = np.zeros(3)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
        res = np.abs(A @ coef - zz)
        keep = res < max(0.02, 2.5 * np.median(res))
        A, zz = A[keep], zz[keep]
    pred = coef[0] * X + coef[1] * Y + coef[2]
    m = (ok & (np.abs(z - pred) < PLANE_TOL0 + PLANE_K * z ** 2)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.dilate(m, np.ones((15, 15), np.uint8))


def build_group(runs):
    anchor = cv2.imread(frames(runs[0], "rgb")[0]).astype(np.float32)
    rgb, depth, per_run = [], [], {}
    for r in runs:
        imgs, shifts = [], []
        for i, p in enumerate(frames(r, "rgb")):
            img = cv2.imread(p).astype(np.float32)
            dx, dy = shift_to(img, anchor)
            imgs.append(warp(img, dx, dy))
            d = cv2.imread(frames(r, "depth")[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth.append(warp(d, dx, dy, nearest=True))
            shifts.append(np.hypot(dx, dy))
        per_run[r] = (imgs, shifts)
        rgb += imgs
    bg = np.median(np.stack(rgb), 0)
    z = np.median(np.stack(depth), 0) / 1000.0
    return bg, floor_mask(z), per_run


def obstacles(img, bg, floor):
    on = floor > 0
    img = np.clip(img * (np.median(bg[on]) / max(1e-6, np.median(img[on]))), 0, 255)
    d = cv2.GaussianBlur(np.abs(img - bg).max(2), (3, 3), 0)
    m = ((d > LO) & on).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    strong = d > HI
    out, objs = np.zeros_like(m), 0
    for k in range(1, n):
        a, sel = stats[k, cv2.CC_STAT_AREA], cc == k
        if MIN_AREA <= a <= MAX_AREA and strong[sel].any():
            out[sel] = 255
            objs += 1
    return out, objs


def main():
    runs = sorted(os.path.basename(p) for p in glob.glob(f"{OWN}/h*") if os.path.isdir(p))
    groups = {}
    for r in runs:
        block, dist = group_key(r)
        if dist != "empty":
            groups.setdefault((block, dist), []).append(r)
    for (block, _), rs in groups.items():
        e = f"{block}_empty"
        if os.path.isdir(f"{OWN}/{e}") and e not in rs:
            rs.append(e)

    rows = []
    for (block, dist), rs in sorted(groups.items()):
        bg, floor, per_run = build_group(rs)
        print(f"[{block} {dist}] {len(rs)} runs, floor {float((floor > 0).mean()):.0%}")
        for r in rs:
            imgs, shifts = per_run[r]
            os.makedirs(f"{OWN}/{r}/gt", exist_ok=True)
            cv2.imwrite(f"{OWN}/{r}/floor.png", floor * 255)
            res = [obstacles(im, bg, floor) for im in imgs]
            for i, (m, _) in enumerate(res):
                cv2.imwrite(f"{OWN}/{r}/gt/{i}.png", m)
            area = float(np.mean([(m > 0).sum() for m, _ in res]))
            flagged = area / max(1, int((floor > 0).sum()))
            row = dict(run=r, block=block, distance=dist,
                       objects=float(np.mean([o for _, o in res])), area_px=area,
                       shift_px=float(np.mean(shifts)), floor_frac=float((floor > 0).mean()),
                       floor_flagged=flagged,
                       reliable=bool(flagged < (0.02 if r.endswith("_empty") else 0.12)))
            rows.append(row)
            print(f"   {r:30s} objects {row['objects']:5.1f}  area {int(area):6d} px  "
                  f"shift {row['shift_px']:5.1f} px  flagged {flagged:5.1%}"
                  f"{'' if row['reliable'] else '   <-- UNRELIABLE'}")

    seen = {}
    for r in rows:                     # empty runs appear once per distance group
        if r["run"] not in seen or r["area_px"] > seen[r["run"]]["area_px"]:
            seen[r["run"]] = r
    with open(f"{OWN}/gt_summary.json", "w") as f:
        json.dump(sorted(seen.values(), key=lambda r: r["run"]), f, indent=1)
    bad = [r for r in seen.values() if not r["reliable"]]
    print(f"\n{len(seen)} runs, {len(bad)} unreliable -> {OWN}/gt_summary.json")


if __name__ == "__main__":
    main()
