# Results report draft

REPORT.tex preserves the approved introduction and related work verbatim and retains most methodology from methodology_report/REPORT.tex. Targeted methods edits update reference creation/review counts, automatic prompts, completed experiment cohorts, training durations and the evaluation domain. Results are inserted from results_section.tex. The original methodology report is unchanged. Local class, bibliography and image assets allow this directory to build independently of that source directory.

Evidence: results/results_report_20261122 and results/triton_20261121/production/analysis_final. Full-frame GTE scores are diagnostic; prop-aware crop inference and expanded reviewed ISOD test coverage are not claimed.

Build with `zsh build.sh` from this directory using a LaTeX installation. The output is REPORT.pdf. The combined document has eight pages: the results/conclusion occupy pages 5–6, references page 7 and the preserved collection appendix page 8. The deadline supplied by Jakub is November 22 at 17:59, superseding the earlier 23:59 note. Nothing has been submitted automatically.

The main presentation uses a model/supervision table as context, a range/size plot, and one matched qualitative figure covering shape, near/far appearance and the poster counterexample. Coverage recall, localization recall and false components are reported separately: a tighter mask is not automatically a newly recovered object. Overlap-threshold sensitivity was checked at 0.25/0.50/0.75. The figures deliberately show a fusion failure alongside aggregate gains. Seed spread is shown instead of ranking the best run. The three-prompt teacher is kept separate from reference assistance and older automatic-label versions.

Training curves, exact teacher examples, four outcome-selected/paired qualitative cases, background statistics and label provenance are supporting artifacts in results/results_report_20261122. The cases distinguish a newly covered target, a boundary-only gain, a fusion loss and the low-contrast counterexample; shown-view adjusted IoU and all-seed/repeat rates are disclosed. They are not extra pages of the submitted results section. Native instance AP, physical dimensions without calibration, incomplete three-prompt U-Net/YOLO repetitions and clean prop-aware/cropped results are excluded because the required evidence is not complete.

Rubric and integrity checks are recorded in rubric_check.json. source_assets.json records the approved source and copied asset hashes; figures/selection.json records the main example choices.
