# Annotation pilot: heuristic vs manual correction vs SAM-assisted correction (draft v1, 2026-11-09)

Status: **proposal, not approved.** Nothing here changes `ISOD_gt` (v0), creates a v1, or replaces
the methodology. SAM is evaluated as a possible *annotation aid*, not adopted.

## 1. Question
On ISOD, which way of producing obstacle masks gives the most correct masks for the least
human effort?

| Method | Starting point | Human work |
|---|---|---|
| **A** heuristic (v0) | `isod.obstacles(floor label)` | none |
| **B** manual correction | the v0 mask | fix it in the mask annotator |
| **C** SAM-assisted correction | SAM proposals (see §5) | fix them in the same annotator |

B and C use the same tool and the same obstacle definition; only the starting mask differs.
Correctness is judged against an **independent reference R** (§4), never against v0, because v0
is what is being tested.

## 2. Obstacle definition (applies to R, B and C, and to all splits)
An **obstacle** is a physically separate object lying on the drivable floor that a small robot
should not drive over: cables, gloves, cloths, toys, tools, packaging, small debris.
- Include the visible on-floor part of an object that touches a wall or furniture (e.g. the
  floor run of a cable plugged into a wall).
- Exclude walls, skirting boards, doors, fixed furniture and their legs, stair edges (they are
  "not drivable" structure, ignored in training, not obstacles).
- Exclude shadows, reflections, stains, floor texture and floor markings.
- Rugs or mats lying flat: **uncertain** unless clearly a loose small object.
- When unsure, paint the region as **uncertain** (excluded from scoring) instead of guessing.

SAM can *propose* a mask for anything that looks like an object; whether that object *is an
obstacle* is decided by this definition, by a person.

## 3. Images (chosen by `review_isod_masks.py` before anyone looks)
`pilot_frames.csv`, 20 frames:
- **Objective (10):** frame 40 of every second site in alphabetical order. Ordinary frames:
  they show what *typical* quality looks like.
- **Difficult (10):** the strongest examples of each automatic hint, each from a different site:
  2 `wall_bump` (object against wall), 2 `border_obstacle`, 2 `large_obstacle` (furniture?),
  2 `thin_obstacle` (cable-like), 1 `small_obstacle`, 1 `dropped_blob`. If fewer exist, fewer are
  taken (the number is in `summary.json`), never padded.

Results are always reported **separately** for the two halves. The difficult half is selected
for being suspicious; its error rate says nothing about ISOD as a whole. The whole-dataset rate
comes only from the 100-frame systematic review (`PROTOCOL.md`).

No model predictions (RGB, depth, RGB-D, U-Net, ESANet, YOLO) are shown at any step.

## 4. Reference R (independent of v0 and of SAM)
1. Annotator 1 draws obstacles **from scratch** in the annotator, seeing only the RGB image (not
   v0, not SAM, not depth), following §2, with an "uncertain" brush for doubtful regions.
2. Annotator 2 reviews each R mask against the RGB image and lists disagreements.
3. Both settle disagreements; unresolved regions become "uncertain". Every disagreement is
   logged (`pilot_disagreements.csv`: pilot_id, region, A1 view, A2 view, outcome).
R is a reference for this pilot only. It is not a new ground-truth version.

## 5. How B and C are produced
- **Tool:** Jakub's existing browser annotator (`code/annotate.py` on `main`: Python standard
  library only, opens with a starting mask, brush and wand), pointed at the 20 pilot frames.
  Only its input/output folders need adapting. No new annotation framework.
- **SAM proposals for C:** SAM 3 run on shire with Jakub's existing `sam3_prelabel.py` approach,
  using **one fixed list of generic noun phrases** for every frame (ISOD has no per-frame list of
  placed objects), e.g. *cable, power cord, glove, cloth, toy, small object on the floor*, and
  keeping only instances that overlap the drivable floor in ISOD's hand label (not depth). The
  phrase list is written down before the pilot and not tuned on the 20 pilot frames.
  Fallback if SAM 3 is unavailable: SAM 2.1 (point/box prompts only, no text) clicked inside
  the annotator — a larger code change, only if approved.
- **Order and crossover (to limit memory effects):** the 20 frames are split into two halves,
  each with 5 objective + 5 difficult frames. The person who did *not* draw R for a frame does
  B on one half and C on the other, with at least one day after R. Each frame therefore gets
  exactly one B or one C by the same person, never both.
- **Effort:** the annotator writes start and end time per frame (or a stopwatch) and counts
  "major edits" (adding or deleting a whole object).

## 6. Measurements (each method vs R; "uncertain" pixels excluded)
| Measure | Meaning | How |
|---|---|---|
| pixel IoU, precision, recall | overall mask agreement | `isod.score` / `isod.totals`, summed over frames |
| missed objects | R objects < 50 % covered by the method's mask | connected components of R |
| false regions | method components < 50 % covered by R | connected components of the method |
| boundary quality | visual: good / acceptable / poor, per frame | annotator notes |
| effort | minutes per frame, major edits per frame | timer, notes |

Reported per method, separately for the objective and difficult halves, plus a picture of every
frame (RGB | R | A | B or C). With 10 frames per half these are descriptive numbers, not
statistical tests.

## 7. Success criteria (agree before starting)
- **v0 is adequate for the objective frames** if A has no missed object and no false region in
  ≥ 9 of the 10 objective frames. (The whole-dataset rate still comes from the 100-frame review.)
- **Manual correction (B) is worth doing** if B fixes the major errors of A (missed/false
  objects vs R) at a median effort ≤ 2 minutes per frame. That would make correcting all
  ~2,000 frames, or only the flagged ones, feasible to cost.
- **SAM assistance (C) is worth adopting** only if C is at least as correct as B (no more
  missed/false objects, pixel IoU vs R within 0.02) **and** saves ≥ 30 % of the time per frame.
  Otherwise SAM adds a large dependency for no measured benefit.
- Any outcome is reported as found, including "v0 is fine" or "SAM does not help".

## 8. Cost
- Human: R ~5 min/frame × 20 + review ~1 min/frame; B or C ~2–4 min/frame → about 3–4 hours
  for two people in total.
- Compute: SAM 3 on 20 frames on one shire GPU, minutes (the checkpoint Jakub used is 3.4 GB and
  gated on Hugging Face; reuse his copy and environment if he agrees). No training, no fine-tuning.
- Mac: no SAM needed. Python with OpenCV/NumPy to view the package; the annotator runs in the browser.

## 9. Getting the exact v0 data onto the Mac (no SSH from the Mac)
1. On shire (Claude Code, `REMOTE_RUN.md` Phase C2): run `review_isod_masks.py`; it writes
   `pilot_files.txt` and `pilot_files.sha256`, then pack:
   `cd <isod_work> && tar czf ~/isod_v0_pilot.tgz -T <review>/pilot_files.txt`
   (20 frames + `splits.json` + all of `ISOD_gt`: roughly 15–40 MB).
2. Move it to the Mac by an approved route: VS Code on gtlpc103 → right-click → Download, then
   your university OneDrive/Google Drive or a USB stick. ISOD is CC BY-NC-SA: keep the copy
   private, research use only.
3. On the Mac, after unpacking, in Terminal:
   `shasum -a 256 -c pilot_files.sha256` → every line must say `OK`. This proves the files are
   byte-identical to the ones on shire, so the masks are the exact v0 used in the experiments.
   Never regenerate the masks locally.
