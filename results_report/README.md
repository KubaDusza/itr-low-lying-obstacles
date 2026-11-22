# Results report

`REPORT.pdf` (18 pages) combines the approved survey, the adapted methodology, results and conclusion (pages 5–6, four columns), references (page 7) and appendices A–H (pages 8–18). Rubric: `RUBRIC.md`. Build with `zsh build.sh`. Deadline: 2026-11-22 23:59. Nothing has been submitted.

## State (2026-11-22 evening)

- ISOD test: all 300 frames are scored. References are `data/ISOD_reference_test_v2`: the same assistance pipeline as training, with the 30 reviewed frames overriding their proposals. On those 30, proposals agree with the reviewed masks at pooled IoU 0.929 (`code/export_isod_test_references.py`).
- All 30 runs, plus direct SAM 3 and the floor-plane fit, have 300-frame ISOD and 320-frame GTE-Floor predictions. Scores: `code/score_isod_test300.py` → `results/test_references_20261122/isod_test300_scores.json`.
- YOLOv8n-seg uses the validation-stopped runs (91/101/108 epochs; selected epochs 35/80/75).
- Every number in the results comes from `results/results_report_20261122/report_numbers.json` (`code/collect_report_numbers.py`). GTE paired bootstrap: `code/bootstrap_gte_contrasts.py`.
- Tables and figures: `code/make_final_report_assets.py` (main table, full-score table, cloth figure with 15-observation counts, RGB-baseline curves).
- False regions: `results/false_regions_20261122` (pixel-weighted samples; Appendix G).

## Weights

The 12 GTE-only ESANet checkpoints (RGB/D, reference and SAM) are on Triton at `~/itr_weights/esanet_gte` (with local copies in `results/gte_export_20261122/weights`, gitignored). They reproduce the saved 30-frame predictions to ≤0.07% of pixels. All other checkpoints are on Triton scratch. GTE servers are no longer needed.
