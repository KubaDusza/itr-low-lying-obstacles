# ISOD obstacle-mask review protocol (draft v1, 2026-11-09)

Status: **proposal for Juan and Jakub to agree before anyone looks at the review images.**
Nothing in this protocol changes `ISOD_gt`. SAM is not part of it.

## 1. Why
Our obstacle masks are produced by a rule (`code/isod.py: obstacles`), not drawn by hand. In
ISOD's "not drivable" region, the rule removes the largest connected piece (assumed to be walls
and furniture) and every blob under 20 px; what remains counts as obstacles. The rule can fail in
two directions:

| Failure | How it happens | Effect on our masks |
|---|---|---|
| **Missed obstacle** | an object touches a wall or furniture, so it is part of the largest piece and is removed with it; or it is under 20 px | obstacle labelled as ignored or background |
| **False obstacle** | a piece of furniture or wall is not connected to the largest piece (e.g. a table in the middle of the room) | furniture labelled as obstacle |
| **Floor-label error** | ISOD's own hand label is wrong (e.g. an object painted as floor, a coiled cable filled in) | both directions; not caused by our rule |

Every result we publish depends on these masks, so we must know how often each failure occurs
before the definitive experiments.

## 2. Annotation versions (never overwrite, always record)
| Version | Folder | Status | Allowed use |
|---|---|---|---|
| v0 | `ISOD_gt/` | automatic rule, unreviewed, **frozen** | technical pilot only; reported as provisional |
| v1 | `ISOD_gt_v1/` (only if needed) | v0 + documented corrections | definitive experiments, once agreed |

- v0 is never edited. Any correction goes into a **new folder**, with one row per changed frame
  in a change log (`site, frame, what changed, why, who, date`).
- Every training run stores the mask folder and a **fingerprint** (a hash over all 2,000 masks) in
  `config.json` (`train_esanet.py --gt_folder`). If a single pixel changes, the fingerprint changes.
  `review_isod_masks.py` writes the fingerprint of the version it reviewed into `summary.json`.
- The definitive table in the paper states the version and the fingerprint.

## 3. Generating the review package (read-only, CPU, a few minutes)
```bash
python code/review_isod_masks.py --root <isod_work> --out <review folder>
```
It writes `frames.csv` (automatic hints for all 2,000 frames), `summary.json`, one picture per
frame to review in `review_images/`, and `review_log.csv`, the sheet to fill in.

**Two samples, used for different questions:**
- **Systematic sample**: frames 0, 20, 40, 60 and 80 of every site (100 frames). Chosen by a fixed
  rule *before* looking at any image, and covering all 20 scenes equally, so it is the **only**
  sample used to estimate error rates.
- **Flagged sample**: the frames that score highest on each automatic hint (at most 2 per site per
  hint, up to 30 per hint). It shows *what kinds* of errors exist, but it is chosen because it
  looks suspicious, so it must **never** be used to estimate how often errors happen.

The automatic hints (`wall_bump`, `border_obstacle`, `large_obstacle`, `small_obstacle`,
`thin_obstacle`, `dropped_blob`) only decide which pictures a person looks at. **A hint is not an error.** The
person decides. Depth is not used to choose suspects: depth is one of the inputs we compare, and
choosing masks to check by depth would bias any later corrections in its favour.

## 4. How to review one picture
Left half: the plain photo. Right half: green = obstacle in the current masks; darkened = ignored
(walls/furniture); red circle = a thin part of the wall region (possible missed object); orange
outline = obstacle at the image border or very large (possible wall/furniture); cyan outline =
thin, cable-like obstacle (check its boundary); magenta circle = tiny blob removed by the rule.

Ask, for the **whole floor area** (not only where the hints are):
1. Is there any object on the floor that is **not** green? → `MISSED`
2. Is anything green that is **not** an object lying on the floor (wall, furniture leg, shadow,
   reflection)? → `FALSE`
3. Are the green shapes roughly right, or clearly too big/small/filled? → `BOUNDARY`
4. Are two or more separate objects joined into one green shape? → `MERGED`
   (fine for pixel IoU, matters only for per-object counts)

Fill in one row of `review_log.csv` per picture. If several errors are present, use the most
severe as `verdict` and list the others in `comment`.

| Column | Values |
|---|---|
| `verdict` | `OK`, `MISSED`, `FALSE`, `BOUNDARY`, `MERGED`, `UNSURE` |
| `severity` | `major`: a whole object the robot must avoid is missing, or a wall/furniture region is labelled obstacle. `minor`: boundary or shape errors only. Empty for `OK`. |
| `cause` | `RULE` (wall removal or 20 px limit), `FLOOR_LABEL` (ISOD's hand label), `UNKNOWN` |
| `where` | short location, e.g. "bottom left, cable against wall" |
| `comment`, `reviewer`, `date` | free text, initials, YYYY-MM-DD |

**Rules that keep the review fair**
- Do not look at any model's predictions while reviewing (otherwise we would "correct" towards
  one model).
- Review all splits with the same procedure. Never correct only the test set.
- Write down a verdict for every frame, including `OK` ones; skipping makes the rates meaningless.
- `UNSURE` is a valid answer; those frames are discussed by both reviewers.

**Agreement check (cheap, recommended):** Juan and Jakub both review the same 20 systematic frames
independently, then compare verdicts. If they agree on fewer than ~16 of 20, clarify the
definitions above before reviewing the rest.

## 5. From review to decision (agree the thresholds BEFORE reviewing)
Report from the systematic sample: frames with ≥ 1 major error, as k / 100 with a 95 % Wilson
interval, overall and per split, plus the error types and causes seen in the flagged sample.

| k / 100 frames with a major error | 95 % interval |
|---|---|
| 0 | 0.0 – 3.7 % |
| 2 | 0.6 – 7.0 % |
| 5 | 2.2 – 11.2 % |
| 10 | 5.5 – 17.4 % |
| 20 | 13.3 – 28.9 % |

**Proposed rule (to be agreed):**
- k ≤ 5 → keep v0 for the definitive experiments; report the error rate and examples as a limitation.
- k > 5, or errors concentrated in one split or one type → decide with Jakub how to build v1
  (manual correction of all frames, of flagged frames only, or a SAM-assisted proposal that a
  person verifies). Whatever the method, v1 is checked with this same protocol on a fresh
  systematic sample (frames 10, 30, 50, 70, 90 of every site).

Deciding the threshold after seeing the numbers would let the result steer the decision.

The same script also fixes the 20 frames of the annotation pilot (`pilot_frames.csv`) that
compares heuristic, manual and SAM-assisted masks: see `ANNOTATION_PILOT.md`.

## 6. Relation to the experiments
- **Technical pilot** (ESANet smoke test and 10-epoch timing run): may use v0. Its numbers are
  labelled *provisional* and are never reported as results.
- **Definitive experiments** (ESANet RGB / D / RGB-D × seeds): only on the agreed version, with its
  folder and fingerprint in every `config.json`.
- **GTE-Floor**: evaluation only, never training. Its manual annotations are not ready (Jakub);
  no GTE-Floor experiment is run until they are.
