# Dataset construction and label provenance

This records the data actually used by the November 2026 experiments. The original annotations, human edits and frozen snapshots remain separate.

## ISOD obstacle references

ISOD supplies aligned RGB/depth and drivable/not-drivable annotations. Not-drivable includes walls and furniture as well as floor objects. Its complement is therefore not directly an obstacle reference for this project.

The assistance pipeline combines SAM object and structure proposals with the original not-drivable mask. Floor proximity, structure conflicts and connected-region completion help retain floor items and reject background structure. Additions are constrained to the original source annotation. The reviewer can change region decisions and make cuts where a region mixes a target with furniture. Saved human corrections take priority over automatic proposals. Clickable regions are editing units, not necessarily physical object instances.

| Frozen reference snapshot | Frames | Review status | Role |
|---|---:|---|---|
| ISOD_reference_train_v1 | 1,400 | 70 human-reviewed; 1,330 assisted | Training targets for the reported reference-supervised semantic runs |
| ISOD_reference_val_v1 | 300 | All human-reviewed | Checkpoint selection; also used during labeling development |
| ISOD_reference_test_v1 | 30 | All human-reviewed; ten frames per test location | Provisional ISOD test results; the 300-frame split is not fully evaluated against reviewed references |

Locations are disjoint across the 14/3/3 train/validation/test partition. Human review was assisted by prelabels, so it can retain anchoring bias and imperfect boundaries. The reference training set must not be called 1,400 fully manual masks.

A retrospective replay on 370 reviewed training/validation frames gave mean image IoU 0.878 for the assisted proposals. These frames informed development, so this is a tuning-informed quality audit, not independent test accuracy. A later stratified random check of 14 previously unreviewed training frames gave mean image IoU 0.837 and pooled IoU 0.912; two frames had major errors. Two additional targeted frames brought this review to 16 frames.

The 16 corrections produce a separate 1,400-frame candidate with 85 unique reviewed masks and 1,315 assisted masks. One reviewed frame overlapped the previous 70. The frozen semantic training runs retain the original version. The new YOLOv8n box experiment uses this updated candidate through its corresponding reference-instance export. Two human-confirmed fragment groups (sunglasses and a loose floor loop) were also saved as reference-instance corrections. Assisted instance proposals elsewhere remain unverified as complete individual-object ground truth.

## RGB-only automatic training labels

This is a separate experiment. The automatic teacher receives the full RGB image and these fixed prompts:

- small object on the floor
- cable on the floor
- things on the floor

SAM 3 (`bodhicitta/sam3`, revision `cba430d22f6fdc3f06ad3841274ec7bb55885f2f`) runs with confidence threshold 0.6 and mask-probability threshold 0.5. Accepted masks are combined by union. There are no original ISOD masks, reference-mask intersections, negative masks or per-image corrections during generation. Empty predictions and errors remain in the 1,400 automatic training targets.

Prompt development used 30 reviewed training images, followed by ablation informed by all 300 reviewed validation images. Generation is RGB-only, but the complete development procedure is not annotation-free and must not be presented as using only 30 labels in total. No SAM model weights were retrained for this prompt study.

## GTE-Floor external references

The 320 frames are five repeats of each of 64 static recordings in one laboratory room. They are used for evaluation, not training or prompt/threshold selection. SAM-assisted masks were manually corrected and assigned inventory item/group IDs. Seven object sets cover thin cables/fasteners, cloths/tissues and taller controls. Some IDs intentionally describe groups, so ID counts are not counts of independent physical objects.

Clearly separated fasteners were proposed as separate reference units without changing the binary foreground union. A small uncertain fragment remains explicitly marked. Metrics retain small targets, including the 19-pixel hex key. Object dimensions were not measured; pixel area, nominal floor distance and valid depth are reported without inventing centimetre dimensions.

The proposed separation gives 262 scored reference-unit/recording pairs, or 1,310 observations across the five repeats. The original item/group maps had 244 pairs (1,220 observations). Neither number counts independent physical objects.

Current scoring uses full frames and includes recording props in its diagnostic background. The poster is not an ignore region containing credited positive targets. Cropped-input inference and verified prop-neutral scoring have not been completed, so full-frame precision/false-component counts are not operational false-alarm estimates.

Machine-readable version registry: `data/label_versions.json`. The original label snapshots and human-save histories are preserved for reproduction.

## Image area and physical dimensions

The main size graph uses annotated foreground pixels on the 640×362 GTE evaluation images. This measures apparent image detail, which changes with range even for the same physical object. It is not a physical footprint. A cable's length, thickness and visible area describe different aspects of the navigation hazard and should not be collapsed into a single centimetre size.

GTE calibration is recorded in `code/db3_to_frames.py`, including color/depth intrinsics and their alignment transform. Metric conversion would also need the 848×480 to 640×362 image resize, valid surface readings and the object's orientation/geometry. Thin-object pixels may contain floor depth; camera-axis floor residual is not object height. Exact recording calibration remains unverified for ISOD. Physical measurements or validated metric estimates are useful complementary attributes, but are not substituted for image area in this draft.

## Completed native box-detector experiment (November 22)

The new YOLOv8n detection experiment uses the saved instance IDs, rather than polygons from binary semantic masks. Each ID becomes one enclosing axis-aligned box; disconnected pieces of the same ID remain one target. Both user-confirmed grouping corrections passed the export checks. No target was removed from the export for being small. Other units remain assisted proposals and can represent unresolved groups.

All 1,400 training RGB images receive separate automatic instance labels from the pinned SAM checkpoint and the same three prompts, without original ISOD masks or reference annotations as generation inputs. Mask-overlap deduplication and fixed parent/child rules resolve repeated proposals; unresolved containment becomes an explicit cluster. The new labels contain 21,224 proposed individuals and 344 unresolved clusters. This is fresh native instance inference, not a claim of pixel-identical reproduction of the earlier semantic teacher cache. All 1,400 label hashes and image identities were verified.

Reference and automatic detectors each use seeds 0, 1 and 2, 20 epochs, the same pretrained YOLOv8n checkpoint, image size, optimizer and shared 300-frame reference-box validation set. Validation AP50:95 selects checkpoints. The reference source contains 85 reviewed binary masks and 1,315 assisted masks; review of binary inclusion does not certify every instance boundary or group.

Evaluation covers 320 GTE frames and 1,310 annotated unit/frame observations. Native box AP is separate from semantic-mask coverage/localization. Fixed operating-point precision/recall uses confidence 0.25 and confidence-ordered one-to-one box matching at IoU 0.5, with duplicate predictions counted as false positives. Recording props are not exhaustively annotated, so unmatched predictions are diagnostic rather than certified operational false alarms. The short detector experiment is not a convergence-matched architecture ranking.

Dataset manifests, exported labels, six training histories, shared-configuration audit, result summaries and remote model hashes are under `results/instance_detection_20261122`. Model checkpoints and full prediction exports remain on Triton at `/scratch/work/duszaj1/itr_instances_20261122/runs`.
