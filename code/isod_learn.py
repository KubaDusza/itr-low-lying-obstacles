"""Learn the keep/drop decision from the frames already reviewed, and propose the rest.

The hand-written rule in isod_components.py gets the easy components right and one kind of
component consistently wrong: a chair or table leg. A leg is not drivable, it is not part of
the big wall component once the erosion has parted it, and the floor wraps most of the way
around it, so every cue the rule has says "object lying on the floor". On the first 17
reviewed frames the rule was wrong on 34% of components and every single error was of that
one kind -- it kept something that should go.

A leg is separable, but on features the rule does not have: it is tall and narrow rather
than flat and wide, what sits directly above it is more furniture, and it was joined to the
structure before the erosion cut it loose. So instead of writing more rules, this fits a
small model to the decisions already made and applies it to the rest.

    python code/isod_learn.py --fit         # fit on the reviewed frames, report accuracy
    python code/isod_learn.py --apply --sites MEB_HW_A  # try the fitted model on one site

The logistic regression uses fourteen features from the original component proposals.
Validation holds out whole sites when possible. A saved review supplies decisions for any
components it leaves intact, including intact components on frames with a hand cut.
"""
import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np

import isod
import isod_components as comps

MODEL = os.path.join(os.path.dirname(isod.GT), "ISOD_comp", "keep_model.pkl")
REVIEWED = os.path.join(os.path.dirname(isod.GT), "ISOD_gt2")

NAMES = ["area_log", "width", "height", "aspect", "fill", "enclosed", "top", "bottom",
         "centre_y", "touches_top", "touches_side", "touches_bottom",
         "joined_to_structure", "structure_above"]


def features(site, i, comp=None, rows=None):
    """Fourteen numbers per component, none of them from depth.

    The two that the hand-written rule lacks, and that a leg is caught by:

    joined_to_structure -- whether this component was one body with the largest component
        before the erosion parted them. A leg is; a shoe lying on the floor beside a chair
        usually is too, which is why it is a feature and not a rule.
    structure_above -- what lies directly above the component's top edge. Above a leg is the
        rest of the chair. Above a cable is floor.
    """
    _, _, label, _ = isod.frame(site, i)
    label = np.array(label)
    if comp is None:
        comp, rows = comps.frame_components(site, i)
    H, W = label.shape
    nd = (label == 0).astype(np.uint8)
    # bodies of the untouched mask, and which of them is the structure
    n_raw, raw, st, _ = cv2.connectedComponentsWithStats(nd, 8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n_raw > 1 else 0

    out = []
    for r in rows:
        m = comp == r["id"]
        x0, y0, x1, y1 = r["box"]
        w, h = x1 - x0 + 1, y1 - y0 + 1
        # which raw body this component came from
        hit = np.bincount(raw[m], minlength=n_raw)
        hit[0] = 0
        joined = float(int(hit.argmax()) == big) if hit.any() else 0.0
        # the band just above the component: structure, or floor?
        band = slice(max(0, y0 - 8), y0), slice(x0, x1 + 1)
        above = nd[band]
        structure_above = float(above.mean()) if above.size else 0.0
        out.append([np.log1p(r["area"]), w / W, h / H, h / w,
                    r["area"] / (w * h), r["enclosed"], r["top"], y1 / H,
                    (y0 + y1) / 2 / H, float(r["touches_top"]), float(r["touches_side"]),
                    float(r["touches_bottom"]), joined, structure_above])
    return np.array(out, float) if out else np.zeros((0, len(NAMES)))


def reviewed():
    """Use original proposals whose labels are unambiguous in the saved mask."""
    test_sites = set(isod.load_splits()["test"])
    for p in sorted(glob.glob(f"{REVIEWED}/*/[0-9]*.json")):
        site = p.split(os.sep)[-2]
        i = int(os.path.basename(p)[:-5])
        if site in test_sites:
            continue
        comp, rows = load_comp(site, i)
        mask = cv2.imread(f"{REVIEWED}/{site}/{i}.png", cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != comp.shape:
            continue
        cut_path = f"{REVIEWED}/{site}/{i}_cut.png"
        cut = cv2.imread(cut_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(cut_path) else None
        clean, keep = [], []
        for row in rows:
            region = comp == row["id"]
            if cut is not None and np.any(cut[region]):
                continue
            selected = mask[region] > 0
            if not selected.size or (selected.any() and not selected.all()):
                continue
            clean.append(row)
            keep.append(selected.all())
        if clean:
            yield site, i, features(site, i, comp, clean), np.array(keep), clean


def load_comp(site, i):
    comp = cv2.imread(f"{comps.OUT}/{site}/{i}_comp.png", cv2.IMREAD_GRAYSCALE)
    rows = json.load(open(f"{comps.OUT}/{site}/{i}.json"))
    return comp, rows


def fit(verbose=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y, frames, sites, base = [], [], [], [], []
    for site, i, f, lab, rows in reviewed():
        X.append(f)
        y.append(lab)
        frames.extend([f"{site}/{i}"] * len(lab))
        sites.extend([site] * len(lab))
        base.extend(comps.propose(r)[0] for r in rows)
    if not X:
        raise SystemExit("no reviewed frames yet")
    X = np.vstack(X)
    y = np.concatenate(y)
    base = np.array(base)
    groups = sites if len(set(sites)) > 1 else frames
    held_out = "site by site" if groups is sites else "frame by frame within one site"
    if verbose:
        print(f"{len(y)} components from {len(set(frames))} reviewed development frames "
              f"({y.sum()} keep, {(~y).sum()} drop)")
        print(f"the hand-written rule agrees with you on {100 * (base == y).mean():.1f}%")

    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                               class_weight="balanced"))
    if verbose:
        pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        print(f"held out, {held_out}: {100 * (pred == y).mean():.1f}% agree  "
              f"(kept wrongly {fp}, dropped wrongly {fn})")
        print(f"  of the {int((base != y).sum())} the rule got wrong, "
              f"the model gets {int(((pred == y) & (base != y)).sum())} right")
    model.fit(X, y)
    if verbose:
        w = model[-1].coef_[0]
        order = np.argsort(-np.abs(w))
        print("  what it leans on: " +
              ", ".join(f"{NAMES[j]} {w[j]:+.2f}" for j in order[:6]))
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    pickle.dump(model, open(MODEL, "wb"))
    return model


def apply_all(model=None, sites=None):
    """Rewrite the proposal in each frame's json. Reviewed frames are left alone."""
    model = model or pickle.load(open(MODEL, "rb"))
    done = {p[len(REVIEWED) + 1:-5].replace(os.sep, "/")
            for p in glob.glob(f"{REVIEWED}/*/[0-9]*.json")}
    changed = total = 0
    for site in sites or isod.sites():
        for i in range(100):
            if f"{site}/{i}" in done:
                continue
            comp, rows = load_comp(site, i)
            if not rows:
                continue
            p = model.predict_proba(features(site, i, comp, rows))[:, 1]
            for r, prob in zip(rows, p):
                was = r["keep"]
                r["keep"] = bool(prob >= 0.5)
                r["prob"] = round(float(prob), 3)
                r["fresh"] = (was != r["keep"] or bool(0.25 <= prob <= 0.75)
                              or (r["keep"] and r["area"] > 20000))
                r["why"] = f"model {prob:.2f}"
                changed += was != r["keep"]
                total += 1
            json.dump(rows, open(f"{comps.OUT}/{site}/{i}.json", "w"))
    print(f"{total} components re-proposed, {changed} changed, "
          f"{len(done)} reviewed frames left untouched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sites", default="")
    a = ap.parse_args()
    model = fit() if a.fit or not os.path.exists(MODEL) else pickle.load(open(MODEL, "rb"))
    if a.apply:
        apply_all(model, [s.strip() for s in a.sites.split(",") if s.strip()] or None)


if __name__ == "__main__":
    main()
