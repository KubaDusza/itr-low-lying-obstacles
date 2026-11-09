# Preliminary visual review of ISOD v0 masks — 8 frames (2026-11-09)

**What this is:** an independent, preliminary visual inspection by the coordinating assistant
(Claude), looking at the real RGB images, ISOD's floor labels and the v0 obstacle masks.
It is **not ground truth**, it changes no label, and it cannot judge pixel accuracy.
**SAM has not been run yet** (no SAM column below): model weights could not be downloaded from
this environment, and downloading needs your approval.

**Data:** original ISOD on Juan's Mac (`~/Downloads/ISOD/data`). v0 masks were regenerated
locally with the repository's `isod.obstacles()` (the same rule that built `ISOD_gt`) into a
separate folder; they have **not yet been checked byte-for-byte** against shire's `ISOD_gt`
(use `pilot_files.sha256` from shire for that).

**Frames:** HB_LM/0 (requested) and HL_LM/40 (typical, no strong hints) + 6 frames chosen
because the automatic hints were strongest (ABEC_MM/0, BT_LM/0, HL_BM/20, HDCL_RM_B/0,
MEB_HW_B/60, HDCL_RM_A/0). **6 of 8 frames were selected for being suspicious, so no error rate
for ISOD can be derived from this review.**

Evidence pictures: `findings/<site>_<frame>_findings.png` (left RGB, right v0 in green; red box =
object missing from v0, yellow = uncertain, cyan = boundary issue). Full panels: `panels/`.

## Findings
Confidence = how sure I am of the observation itself, from the image at 640×422.
"Confirmed" = clearly visible object, and clearly no v0 mask on it.

| ID | Frame | Location | Observation | More accurate | Confidence | Reason |
|---|---|---|---|---|---|---|
| F1 | HB_LM/0 | far left edge | blue-white striped object not in v0 | neither marks it as obstacle; v0 misses it | high (object present), medium (on floor, cut by edge) | in the floor label it joins the top-left not-drivable region, which the rule removes |
| F2 | HB_LM/0 | back, left black flip-flop | flip-flop not in v0 (its neighbour is) | v0 misses it | high | its blob touches the far not-drivable boundary |
| F3 | HB_LM/0 | red can at the base of the right wall | not in v0 | v0 misses it | medium-high | lies at the wall/floor junction; ISOD label puts it in the wall region |
| F4 | HB_LM/0 | bottom centre, white rectangle | labelled **drivable** by ISOD | undecidable | low | flat paper/cloth, or a floor mark? not resolvable from RGB |
| — | HB_LM/0 | v0 #1, thin dark vertical item at the far corner | in v0 | uncertain | low | could be a post/leg rather than a floor object |
| F5 | HL_LM/40 | v0 #1: sneaker + small blocks | one merged region that also covers floor between objects | v0 acceptable for pixels, wrong as "one object" | medium | coarse ISOD floor label (not the rule) |
| F6 | HL_LM/40 | v0 #6: pencil at bottom | region several times wider than the pencil | object found, boundary poor | medium-high | coarse ISOD floor label |
| F7 | ABEC_MM/0 | back centre, blue crumpled object | not in v0 | v0 misses it | high | touches chair legs → part of the removed furniture region |
| F8 | ABEC_MM/0 | back centre, beige cloth | not in v0 | v0 misses it | high | same: touches chair legs |
| F9 | ABEC_MM/0 | right, black cable loop under chair | not in v0 | v0 misses it | high | the loop touches a chair leg |
| — | ABEC_MM/0 | chair legs everywhere | correctly not obstacles | v0 correct | high | the large "wall bump" hint here comes from legs: false alarm |
| F10 | BT_LM/0 | back right, black remote | in the floor label, but not in v0 | v0 misses it | high | its blob touches the large cabinet region on the right |
| — | BT_LM/0 | v0 #14, large region bottom right | a real flat object cut by the frame, not furniture | v0 correct | medium | the "large obstacle" hint was a false alarm here |
| F11 | HL_BM/20 | left, blue cylinder | not in v0 | v0 misses it | medium-high (motion blur) | touches the plush toy, which joins the left not-drivable region |
| F12 | HL_BM/20 | left, grey bowl behind the plush toy | not in v0 | v0 misses it | medium | partly occluded; same cause |
| F13 | HL_BM/20 | large yellow plush toy | not an obstacle in v0 | undecidable | low | is a large soft toy a "low-lying obstacle"? definition question |
| F14 | HL_BM/20 | v0 #4, blue cable coil | filled solid blob, coarse outline | object found, boundary poor | medium | coarse ISOD floor label |
| F15 | HDCL_RM_B/0 | left, blue cable | not in v0; ISOD draws it as a filled blob joined to the desk | v0 misses it | high | joined to the desk region |
| F16 | HDCL_RM_B/0 | orange block at the chair column | not in v0 | v0 misses it | high | touches the chair column |
| F17 | MEB_HW_B/60 | thin pencils and pencil pile | captured (#4, #5, #6), outline coarse but reasonable | v0 adequate | medium | thin objects can survive the rule when isolated on open floor |
| F18 | HDCL_RM_A/0 | bottom left dark spot | not in v0 | undecidable | low | object or floor mark? Otherwise empty floor: v0's 0 obstacles look correct (high) |

## Summary
- **Confirmed pattern:** 11 objects (F1–F3, F7–F12, F15, F16) are clearly visible but absent
  from v0 in 5 of the 8 frames. **Every one of them touches a wall, a piece of furniture or a
  large region that the rule removes.** This is the failure the rule's design predicts.
- **Not found:** no case of furniture wrongly kept as an obstacle. The two "large obstacle"
  hints I checked were real objects.
- **Boundary quality** is limited by ISOD's own floor labels (merged objects, wide pencils,
  filled cable coils). This is not caused by Jakub's rule.
- **Isolated small and thin objects** on open floor are captured well (HB_LM/0, MEB_HW_B/60).
- **Definition questions** (F4, F13, F18) need a decision from Juan and Jakub, not an annotator.
- **Automatic hints:** useful for finding frames to check, but noisy. "wall bump" fired on chair
  legs and a chair base; "large obstacle" fired on real objects.

## What this does not show
- An error rate for ISOD (biased sample; use the 100-frame systematic review).
- Whether SAM would do better: not run yet. Prompts for exactly these objects are in
  `sam_prompts_pilot.csv`.
- Pixel accuracy of anything.
