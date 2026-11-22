# Results report

`REPORT.pdf` (16 pages) combines the approved survey (verbatim), adapted methodology, results/conclusion, references and appendices in IEEE format. Results and conclusion end within page6; references are on page6. Supporting appendices contain seed variation, per-item scores, ISOD size/depth analysis, label construction, training curves, examples, seed consistency, false-region inspection and inference cost. Nothing has been submitted automatically.

All **36 learned runs** are complete: ESANetRGB/D/RGB-D, U-NetRGB, YOLOv8n-segRGB and YOLOv8n box-onlyRGB, each trained on reference and automaticSAM labels with three seeds. Schedules:500 for ESANet, 15 for U-Net and 60 for YOLO. All are evaluated on300 ISOD test frames and320 GTE frames. Test references are270 assisted masks plus30 reviewed overrides; their quality is documented in the report.

The six final SAM-label U-Net/YOLOseg runs are included in the main table, text and curves. Frozen-reference versus automatic-label effects, mask agreement versus object recovery, and modality comparisons are kept distinct. The raw run-level U-Net `scores.json` files retain a30frame diagnostic; the canonical full300test evaluation is `results/test_references_20261122/isod_test300_scores.json`.

Build from root with `zsh results_report/build.sh`. Refresh current evidence/assets with `.venv-ml/bin/python code/refresh_results_evaluation.py`; update handwritten numbers from `results/results_report_20261122/report_numbers.json`, then rebuild/render/audit. `rubric_check.json` records the actual PDF SHA and verification results. `source_assets.json` records input/figure hashes. Review rules: `REVIEW_NOTES.md`; assignment: `RUBRIC.md`; active handoff: `../HANDOFF_CURRENT.md`.

GTE machines are no longer needed. All selected models are available on Triton; newSAM baseline outputs use home-space symlinks after the scratch quota failure. All evaluation predictions needed to rebuild this report are local. Model binaries/raw data/prediction caches are not committed; annotations, code, histories and numeric evidence are curated separately.
