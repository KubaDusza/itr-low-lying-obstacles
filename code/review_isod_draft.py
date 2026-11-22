"""Serve selected ISOD frames for SAM annotation with drafts stored separately."""

import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import review_isod as review
from isod_sam_review import EXTRA_PROMPTS, PROMPT_GROUPS, read_cache

EXAMPLES = {("HB_LM", i) for i in (8, 12, 16, 20, 24, 28, 40, 48, 52, 64, 72, 84, 88)} | {("MEB_HW_A", 19)}
EXAMPLES.update((site, i) for site in ("FL_LM", "BT_LM", "HDCL_HW_B") for i in (16, 28, 40, 64, 76, 88))
EXAMPLES.update((site, i) for site in ("ABEC_RM", "BT_BM", "HB_MM") for i in (12, 36, 60, 84))
EXAMPLES.update((site, i) for site, frames in {
    "ABEC_CM": (32, 35, 41, 44, 50, 56, 62, 68, 74, 80, 86, 92, 95, 99),
    "HL_BM": (1, 8, 16, 22, 29, 38, 46, 52, 59, 68, 78, 89, 92, 99),
    "MEB_HW_A": (21, 22, 27, 31, 38, 46, 52, 57, 68, 73, 82, 88, 96, 99),
}.items() for i in frames)
# One frame per training site, using a rotating offset through the fixed GPU batch.
TRAINING_SPOT_CHECK = (
    ("ABEC_MM", 4), ("ABEC_RM", 14), ("BT_BM", 24), ("BT_LM", 34),
    ("FL_BM", 44), ("FL_LM", 54), ("HB_G", 64), ("HB_LM", 74),
    ("HB_MM", 94), ("HDCL_HW_B", 94), ("HDCL_RM_A", 4), ("HDCL_RM_B", 14),
    ("HL_LM", 24), ("MEB_HW_C", 34),
)
STRUCTURE = set(PROMPT_GROUPS["structure"]) | set(EXTRA_PROMPTS["structure"])
FLOOR_MARGIN = 12
MIN_NEAR_FLOOR_PIXELS = 5
ENCLOSED_FLOOR_BORDER = 0.90
MIN_SAM_COVERAGE = 0.15
SAM_SCORE = 0.30
WEAK_SAM_SCORE = 0.20
MIN_WEAK_SAM_COVERAGE = 0.70
LARGE_REGION_AREA = 1000
MIN_FLOOR_SHARE = 0.10
STRUCTURE_FLOOR_SHARE = 0.35
AMBIGUOUS_PROMPTS = {"small box", "fabric on the floor"}
REFERENCE = Path(review.OUT)
review.OUT = os.environ.get("ITR_REVIEW_DRAFTS", str(review.SAM_CACHE / "review_drafts"))
review.FLAGS = str(Path(review.OUT) / "flags.json")
review.PAGE = review.PAGE.replace("const samPartition = false;", "const samPartition = true;")


def available_frames(site):
    return {i for s, i in EXAMPLES if s == site} | {
        int(path.stem) for path in (review.SAM_CACHE / "source-full-crayon" / site).glob("*.npz")
        if path.stem.isdigit()
    }


def partition(site, frame, *, complete_regions=True, recover_weak=True, include_crayon=True,
              check_structure=True, require_structure_floor=True):
    path = next((review.SAM_CACHE / folder / site / f"{frame}.npz"
                 for folder in ("source-full-crayon", "source-scale-1", "cache")
                 if (review.SAM_CACHE / folder / site / f"{frame}.npz").is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No cached SAM masks for {site}/{frame}")
    label = cv2.imread(f"{review.isod.DATA}/{site}/label/{frame}.png", cv2.IMREAD_GRAYSCALE)
    source = label == 0
    near_floor = cv2.distanceTransform(source.astype(np.uint8), cv2.DIST_L2, 5) <= FLOOR_MARGIN
    comp, base_rows = review.base_frame(site, frame)
    regions = np.zeros(label.shape, dtype=np.uint16)
    kept = set()
    check_floor = set()
    masks = {True: [], False: []}
    weak_support = np.zeros(label.shape, dtype=bool)
    candidates = read_cache(path, label.shape)
    crayon_path = review.SAM_CACHE / "crayon-pilot" / site / f"{frame}.npz"
    if include_crayon and crayon_path.is_file() and not any(p == "crayon" for p, _, _ in candidates):
        candidates.extend(read_cache(crayon_path, label.shape))
    for prompt, score, mask in sorted(candidates, key=lambda item: -item[1]):
        keep = prompt not in STRUCTURE
        if score < SAM_SCORE:
            if keep and score >= WEAK_SAM_SCORE:
                weak_support |= mask & source
            continue
        mask &= source
        area = int(mask.sum())
        if area < 20:
            continue
        if any(np.count_nonzero(mask & previous) >= 0.85 * np.count_nonzero(mask | previous)
               for _, previous in masks[keep]):
            continue
        masks[keep].append((area, mask))

    next_region = 1

    def assign(mask, keep, *, inherits_support=False):
        nonlocal next_region
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for component in range(1, n):
            if stats[component, cv2.CC_STAT_AREA] < review.comps.MIN_AREA:
                continue
            if next_region > np.iinfo(regions.dtype).max:
                raise ValueError("Too many regions to display")
            region = cc == component
            regions[region] = next_region
            if keep:
                floor_pixels = np.count_nonzero(region & near_floor)
                if inherits_support or floor_pixels >= MIN_NEAR_FLOOR_PIXELS:
                    kept.add(next_region)
                    if (complete_regions and not inherits_support and
                            stats[component, cv2.CC_STAT_AREA] >= LARGE_REGION_AREA and
                            floor_pixels < MIN_FLOOR_SHARE * stats[component, cv2.CC_STAT_AREA]):
                        check_floor.add(next_region)
                else:
                    check_floor.add(next_region)
            next_region += 1

    # Preserve tighter SAM boundaries so broad masks can be corrected a piece at a time.
    for keep in (True, False):
        for _, mask in sorted(masks[keep], key=lambda item: item[0]):
            assign(mask & (regions == 0), keep)
    if complete_regions:
        sam_kept = np.isin(regions, list(kept))
        for row in base_rows:
            if row["touches_top"] or row["enclosed"] < ENCLOSED_FLOOR_BORDER:
                continue
            component = comp == row["id"]
            if sam_kept[component].mean() >= MIN_SAM_COVERAGE:
                # Complete the part SAM left unassigned and preserve explicit structure regions.
                assign(component & (regions == 0), True, inherits_support=True)
        if recover_weak:
            for row in base_rows:
                if row["touches_top"] or row["enclosed"] < ENCLOSED_FLOOR_BORDER:
                    continue
                remainder = (comp == row["id"]) & (regions == 0)
                if remainder.any() and weak_support[remainder].mean() >= MIN_WEAK_SAM_COVERAGE:
                    assign(remainder, True, inherits_support=True)
    # Use ISOD components for the leftover background, including source pixels omitted there.
    assign(source & (comp == 0) & (regions == 0), False)
    if check_structure:
        structure_ids = [row["id"] for row in base_rows if row["touches_top"]]
        structure = np.isin(comp, structure_ids)
        areas = np.bincount(regions.ravel())
        cover = 0.70 * areas
        cover[0] = np.inf
        broad = np.zeros(len(areas), dtype=float)
        specific = broad.copy()
        negative = broad.copy()
        for prompt, score, mask in candidates:
            if score < SAM_SCORE:
                continue
            matches = np.bincount(regions[mask], minlength=len(areas)) >= cover
            scores = negative if prompt in STRUCTURE else broad if prompt in AMBIGUOUS_PROMPTS else specific
            scores[matches] = np.maximum(scores[matches], score)
        foreground = np.maximum(broad, specific)
        mostly_structure = np.bincount(regions[structure], minlength=len(areas)) >= 0.80 * areas
        ambiguous = (broad > 0) & (specific == 0) & mostly_structure
        conflict = (foreground > 0) & (negative >= foreground)
        if require_structure_floor:
            floor_pixels = np.bincount(regions[near_floor], minlength=len(areas))
            weak_floor = (mostly_structure & (areas >= LARGE_REGION_AREA) &
                          (floor_pixels < STRUCTURE_FLOOR_SHARE * areas))
            ambiguous |= weak_floor
        uncertain = kept.intersection(np.flatnonzero(ambiguous | conflict))
        kept.difference_update(uncertain)
        check_floor.update(uncertain)
    return regions, kept, check_floor


def flag_unmatched_floor(site, frame, rows):
    _, original_rows = review.base_frame(site, frame)
    enclosed_ids = {row["id"] for row in original_rows
                    if not row["touches_top"] and row["enclosed"] >= ENCLOSED_FLOOR_BORDER}
    for row in rows:
        if "sam" not in row and row["id"] in enclosed_ids:
            row["fresh"] = True


def frame_proposal(site, frame):
    sam, keep, check_floor = partition(site, frame)
    comp, rows = review.edit_components(site, frame, None, sam)
    for row in rows:
        row["keep"] = row.get("sam") in keep
        row["fresh"] = row.get("sam") in check_floor
    flag_unmatched_floor(site, frame, rows)
    comp, rows = review.separate_islands(comp, rows)
    for row in rows:
        if "sam" not in row and row["fresh"] and row["area"] >= review.comps.MIN_AREA:
            row["keep"] = True
    return comp, rows, sam


class Demo(review.Handler):
    split_islands = True

    def do_GET(self):
        route = urlparse(self.path)
        q = parse_qs(route.query)
        parts = route.path.strip("/").split("/")
        if route.path == "/training":
            return self._send(200, review.PAGE.encode(), "text/html; charset=utf-8")
        if route.path == "/api/training-queue":
            return self._json([
                dict(site=site, i=i, done=all((Path(review.OUT) / site / f"{i}{suffix}").is_file()
                                             for suffix in (".json", ".png")))
                for site, i in TRAINING_SPOT_CHECK
            ])
        if route.path == "/validation":
            site = q.get("site", ["ABEC_CM"])[0]
            if site not in review.isod.load_splits()["val"]:
                return self._send(404, b"Unknown validation site")
            pending = [i for i in sorted(available_frames(site))
                       if not any((root / site / f"{i}.json").is_file()
                                  for root in (REFERENCE, Path(review.OUT)))]
            if not pending:
                return self._send(200, b"No additional cached frames awaiting review.")
            self.send_response(302)
            self.send_header("Location", f"/?site={site}&sample=" + ",".join(map(str, pending[:10])))
            self.end_headers()
            return
        if len(parts) == 3 and parts[0] in ("reference", "source"):
            kind, site, name = parts
            if site not in review.isod.sites() or name not in {f"{i}.png" for i in available_frames(site)}:
                return self._send(404, b"Unknown example")
            directory = REFERENCE / site if kind == "reference" else Path(review.isod.DATA) / site / "label"
            image = directory / name
            if not image.is_file():
                return self._send(404, b"No saved correction")
            return self._send(200, image.read_bytes(), "image/png")
        if route.path == "/api/frame":
            site, frame = q["site"][0], int(q["i"][0])
            if site not in review.isod.sites() or frame not in available_frames(site):
                return self._send(404, b"Unknown example")
            if (Path(review.OUT) / site / f"{frame}.json").is_file():
                comp, rows, cut, sam = review.frame_state(site, frame, split_islands=True)
            else:
                cut = None
                comp, rows, sam = frame_proposal(site, frame)
            return self._json(dict(w=comp.shape[1], h=comp.shape[0], rows=rows,
                                   comp=review.pack_image(comp), comp16=True, cut=review.pack_image(cut),
                                   sam=review.pack_image(sam), sam16=sam is not None and sam.itemsize == 2,
                                   suggestions=[],
                                   reference=(REFERENCE / site / f"{frame}.png").is_file()))
        if route.path == "/" and not q.get("sample"):
            self.send_response(302)
            self.send_header("Location", "/validation?site=ABEC_CM")
            self.end_headers()
            return
        return super().do_GET()

    def handle_edit(self, path, q, site, frame, body):
        if site not in review.isod.sites() or frame not in available_frames(site):
            return self._send(403, b"Draft examples only")
        if path == "/api/recut" and q.get("preview") != ["1"]:
            return self._send(403, b"Preview only")
        return super().handle_edit(path, q, site, frame, body)

if __name__ == "__main__":
    port = int(os.environ.get("REVIEW_PORT", 8813))
    with review.http.server.ThreadingHTTPServer(("127.0.0.1", port), Demo) as server:
        print(f"Draft SAM reviewer running at http://127.0.0.1:{port}/")
        server.serve_forever()
