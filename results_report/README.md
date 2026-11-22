# Results report

`REPORT.pdf` combines the approved bibliography survey, adapted methodology, results and conclusion in the IEEE conference template. Introduction and State of the Art remain verbatim from the approved methodology source; the original methodology report is unchanged. Results and conclusion occupy pages 5–6 (four columns), references page 7, and supporting appendices pages 8–21. Deadline: November 22, 17:59 Europe/Paris. Nothing has been submitted automatically.

Build with `zsh build.sh`. The directory includes its class, bibliography and figure assets.

The report preserves the teammate's experiment/quantitative/qualitative/interpretation structure, cross-dataset table, object-set breakdown and dot/whisker graph style. Numbers are recalculated from the current models and labels, rather than copied from the earlier experiments. Additional comparisons distinguish binary semantic masks, instance segmentation and box-only detection.

All reported learned conditions have three seeds: 18 ESANet runs, three U-Net runs, three YOLOv8n-seg runs and six box-detector runs. ESANet histories contain 500 epochs. U-Net was rerun for 15 epochs with actual training and validation BCE+Dice loss logging. The six box detectors were retrained from the same COCO initialization with max 200 epochs and validation patience 20. Reference seeds stopped at 72/73/65 epochs; automatic seeds at 67/98/84. Checkpoints are selected on ISOD validation data, never GTE test scores. Longer training does not uniformly improve external generalization.

The primary common comparison uses obstacle coverage and the published SegmentMeIfYouCan component metric, with explicit adaptations for supplied object/group IDs and rasterized native boxes. It tolerates useful splits and merges while penalizing excess foreground. Strict native box AP is a separate diagnostic. The appendix includes all training curves, component metrics, per-item scores, teacher examples, all-seed cloth comparisons, failure analysis and measured CPU/GPU inference cost.

Evidence:

- `../results/extended_training_20261122`: longer box runs, logged U-Net reruns, saved predictions, native evaluation, CPU/GPU timings and execution state.
- `../results/instance_detection_20261122/input`: unchanged frozen box-reference labels and manifest.
- `../results/results_report_20261122/revision`: updated domain/category/component analyses and training audit.
- `../results/results_report_20261122`: source-supervision analyses and automatic-label examples.
- `../results/triton_20261121/production/analysis_final`: semantic evaluation and depth interventions.
- `DATASET_CREATION.md`: reference assistance versus RGB-only automatic labels, human reviews and frozen snapshots.
- `rubric_check.json`: current PDF integrity and rubric checks.
- `source_assets.json`: source and generated-figure hashes.

All GTE predictions and histories required for this report, including the third RGB/SAM semantic run, are local. No further GTE VPN connection is required for the current report. Large trained weights remain on their recorded servers; the repository contains labels, code, histories, scores and provenance rather than all weights.

Limits: GTE is one external room with five static repeats per recording; reference IDs include groups and recording props are not exhaustively annotated. Full-frame extra-region scores are therefore diagnostic. ISOD test scores use 30 reviewed images, not the entire 300-frame test split. Semantic and box references are documented, distinct frozen versions. Physical dimensions are not inferred from unvalidated depth. Expanded reviewed testing, verified instances, prop-aware evaluation and robot-hardware timing remain future work.
