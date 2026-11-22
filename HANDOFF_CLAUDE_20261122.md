# Handoff: results report (Claude session, 2026-11-22 evening)

Start here. This supersedes the "Immediate priorities" and "Important unresolved scientific issues" sections of `HANDOFF_20261122.md` (Codex). Read `results_report/REVIEW_NOTES.md` (Jakub's writing rules) and `results_report/RUBRIC.md` (Canvas rubric) before editing.

## Deadline and state

- Jakub says the deadline is **2026-11-22 23:59**; the Canvas text says "Niedziela do 17:59". Jakub knows. Nothing has been submitted; never submit or message anyone.
- Branch `results-report-revision` (created from `main`). Commits: `2212f343` (300-frame ISOD test, results rewrite, appendix trim), `9142f4f0` (60-epoch YOLO reruns, methodology rewrite). **Uncommitted:** the Table II caption/hanging-line fix in `results_report/REPORT.tex` + rebuilt `REPORT.pdf`, and this file. Nothing pushed.
- `results_report/REPORT.pdf`: 17 pages. Survey pp. 1–2 (verbatim, do not edit), methodology pp. 3–4, results + conclusion pp. 4–6, references p. 6, appendices A–H pp. 7–17. Build: `zsh results_report/build.sh`.
- The report is hand-written TeX (Jakub chose to stay in TeX, not the REPORT.md workflow). Sources: `REPORT.tex` (survey + methodology), `results_section.tex`, `appendix.tex`, generated `main_results_table.tex`, `full_scores_table.tex`.

## Decisions Jakub made (do not reopen)

1. Results are organized as findings: Depth / Automatic labels / Masks, boxes and cost; caveats only in the conclusion's limitations paragraph.
2. Automatic labels are framed as "good enough, large annotation saving", not a precision/recall tradeoff.
3. **ISOD test uses all 300 frames.** References `data/ISOD_reference_test_v2` built with the same assistance pipeline as training; the 30 reviewed frames override proposals (agreement pooled IoU 0.929). Never call it a subset/provisional.
4. Stay in TeX. GTE servers are done for good: all 12 GTE-only ESANet checkpoints are on Triton `~/itr_weights/esanet_gte` (checksums in `SHA256SUMS`; local copies in `results/gte_export_20261122/weights`, gitignored). They reproduce saved predictions to <=0.07% pixels.
5. YOLO models: fixed **60-epoch** standard Ultralytics schedule, no early stopping (the earlier 200-epoch early-stopped runs were selected before LR decay; superseded). Seg picks checkpoint by validation obstacle IoU (5-epoch saves; chose 45/55/50). Boxes pick best validation fitness 0.9·AP50:95+0.1·AP50 (chose ref 40/50/45, SAM 50/50/60).
6. Prompt-selection wording: "compared specific item names with broad categories on 30 reviewed training frames; three broad prompts worked well and are simple".
7. Appendix: trimmed, not expanded. Kept what Jakub asked for (curves, label protocol, full scores, examples, seed consistency, cost, false-region analysis).
8. Methodology framing: depth test on ESANet (only network with matched RGB/D/RGB-D versions; masks, so segmentation); U-Net / YOLOv8n-seg / YOLOv8n boxes are RGB references with different outputs; all scored as obstacle regions. Training in one table (Table II). Plain-language metric definitions.

## Key numbers (three-seed means; source of truth `results/results_report_20261122/report_numbers.json`)

| Method | ISOD IoU | ISOD obj. recall | ISOD comp F1 | GTE IoU | GTE obj. recall | GTE comp F1 | GTE false reg./fr |
|---|---|---|---|---|---|---|---|
| ESANet RGB ref | .711 | .854 | .591 | .486 | .628 | .356 | 3.65 |
| ESANet D ref | .318 | .388 | .131 | .141 | .383 | .067 | 8.24 |
| ESANet RGB-D ref | .708 | .867 | .583 | .554 | .643 | .428 | 2.84 |
| U-Net ref | .724 | .750 | .693 | .487 | .464 | .388 | 2.40 |
| YOLOv8n-seg ref (60 ep) | .725 | .751 | .735 | .450 | .600 | .522 | 0.96 |
| YOLOv8n boxes ref (60 ep) | .498 | .806 | .501 | – | .675 | .380 | 1.29 |
| ESANet RGB SAM | .689 | .870 | .566 | .469 | .781 | .346 | 6.42 |
| ESANet RGB-D SAM | .680 | .883 | .543 | .498 | .753 | .356 | 6.03 |
| YOLOv8n boxes SAM (60 ep) | .522 | .802 | .517 | – | .720 | .393 | 1.51 |
| SAM 3 direct | .730 | .838 | .829 | .418 | .945 | .673 | 2.54 |

GTE paired bootstrap (64 recordings, seeds pooled): RGB-D−RGB IoU +0.069 [+0.031,+0.105], object recall +0.014 [−0.019,+0.050]; SAM−ref object recall RGB +0.153, RGB-D +0.110, boxes +0.045 [+0.009,+0.084]. Depth intervention (RGB-D ref, GTE object recall): normal 0.643, depth removed 0.078, other recording's depth 0.641. Box AP on GTE (Ultralytics val) ref/SAM: AP50 .599/.630, AP50:95 .407/.445. Cost: SAM 183 ms vs YOLO 4.0 ms (H200), 13.3 s vs 34 ms (CPU).

False regions under SAM labels are mostly unannotated real objects (tape measure 21/30, bin/box 7/30 in a pixel-weighted sample): `results/false_regions_20261122`, Appendix G.

## Pipeline (rerun in this order after any change)

All with `.venv-ml/bin/python` (torch, ultralytics, smp, pymupdf; gitignored) from `code/`, `MPLCONFIGDIR=<scratch>/mpl`:
1. `prepare_report_revision.py` – GTE scoring of all runs (writes `results/results_report_20261122/revision/*`).
2. `score_isod_test300.py` – ISOD 300-frame scoring → `results/test_references_20261122/isod_test300_scores.json`.
3. `bootstrap_gte_contrasts.py` → `revision/bootstrap_contrasts.json`.
4. `collect_report_numbers.py` → `report_numbers.json` (prints the summary table).
5. `make_final_report_assets.py` – main/full tables, cloth figure (seed 2 with 15-observation counts), RGB-baseline and box curves, SAM-label example figure.
6. `zsh ../results_report/build.sh`, then render pages (`pdftoppm`) and check layout. Numbers in prose are hand-written: update them from `report_numbers.json`.

Other scripts: `export_isod_test_references.py` (test refs), `make_detection_report_figures.py` (Fig. 1 pipeline), Triton controllers `triton_boxes_e60.py`, `triton_seg_e60.py`, `e60.slurm`; test prediction job `results/test_references_20261122/deploy/`.

## Where things live

- Triton (`duszaj1@code.triton.aalto.fi`, needs Aalto VPN): runs `/scratch/work/duszaj1/itr_extended_20261122/runs/*_e60*` (weights there), RGB-D runs `/scratch/work/duszaj1/itr_20261121/production/runs`, GTE ESANet weights `~/itr_weights/esanet_gte`. Scratch quota nearly full (~0.5 GB); home has ~12 GB.
- Local predictions (not in git): `results/test_references_20261122/pred/`, `results/extended_training_20261122/runs/*/pred|gte`.
- `data/` is gitignored (test references v2 not in git).

## Open / optional

- Commit the Table II fix + this handoff (Jakub approves commits per round).
- Methodology reviewer comment not checked: crop the GTE photo figure (Fig. 2) to save space.
- Embedded text in `figures/qualitative_cases.pdf` (Appendix G cable figure) still carries defensive notes ("outcome-selected… not evidence of frequency"); could be regenerated plainer.
- Stale unused files in `results_report/`: `domain_table.tex`, `shared_detection_table.tex`, `domain_spread_table.tex`, `full_component_table.tex`, `table_rows.tex`, `cloth_counts_table.tex`.
- Jakub's style: concise, plain, no jargon in main text, no "X, not Y", no development history in the paper, captions short.
