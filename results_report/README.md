# Results report

`REPORT.pdf` combines the approved bibliography survey, updated methodology, results and conclusion in the IEEE conference template. Introduction and State of the Art remain verbatim from the approved methodology source. The original methodology report is unchanged. The results and conclusion occupy pages 5–6 (four columns); references and the original collection appendix follow. The user-supplied deadline is November 22 at 17:59 Europe/Paris. Nothing has been submitted automatically.

Build with `zsh build.sh`. The directory includes its own class, bibliography and image assets.

The main figures show semantic object coverage/localization, native box detection, distance/apparent-size effects, and all four cloths on bare floor versus poster. They use current predictions and seed variation. Matched qualitative views explicitly include fusion failures. The lab partner's emphasis on graphical comparisons and matched examples informed presentation; their earlier model results and statistics were not substituted for these experiments.

The six new YOLOv8n detection runs are complete: reference/automatic instance labels x seeds 0/1/2, 20 epochs. They are separate from the older YOLOv8n-seg runs and the ESANet semantic comparison. Native box AP and one-to-one box recall are not interchangeable with semantic-mask coverage/localization. Automatic RGB semantic results retain their honest two-seed count; the missing Georgia Tech run has not been retrieved.

Evidence:

- `results/instance_detection_20261122`: box labels, native detector configurations/histories, fixed-threshold scores, training curves, matched-configuration checks and remote artifact hashes.
- `results/results_report_20261122`: semantic cohort/condition analyses, training curves, teacher-mask examples, and a supporting cable/fusion/boundary gallery.
- `results/triton_20261121/production/analysis_final`: completed semantic evaluation and depth interventions.
- `DATASET_CREATION.md`: reference assistance versus RGB-only automatic-label generation, human reviews and frozen versions.
- `rubric_check.json`: current PDF integrity and rubric checks.
- `source_assets.json`: approved source hashes and generated figure provenance.

GTE is one external room with five static repeats per recording. IDs include assisted groups and recording props are not exhaustively annotated, so full-frame unmatched predictions are diagnostic. ISOD test results cover 30 reviewed frames, not the entire 300-frame split. The short box experiment is not convergence-matched to ESANet; five validation optima occur at the last epoch. Physical dimensions are not invented from image area or unvalidated object depth. Prop-aware inference crops, expanded reviewed test coverage and verified physical-object evaluation remain future work.
