# Current project handoff — 2026-11-22

This is the active entry point. Earlier Codex/Claude handoffs and the full revision history are preserved; their pending-job and old-test-subset states are superseded.

## Completed

- All **36 learned runs** are complete, including the final six automatic-SAM-label U-Net/YOLOv8n-seg runs. Every model/label-source pair has three seeds.
- Final restarted jobs **21782791 / 21782792** all exited **0:0**. U-Net:15 epochs, chosen11/8/4. YOLOv8n-seg:60-epochs, chosen `epoch45.pt / epoch45.pt / epoch40.pt` (filenames are zero-based). Checkpoint selection used reference validation data, not test scores.
- All methods have evaluation on **300 ISOD test images** and **320 GTE-Floor images**. ISOD v2 references are270 assisted masks plus30 reviewed overrides; never revert to the earlier30image evaluation.
- The final report includes both new rows, their numerical discussion, and both-label-source RGB training curves. The earlier36 runs/table mismatch is fixed.
- Main results/conclusion remain within the first six pages; references are on page6. Appendices have been repaginated to keep headings with content and make examples readable. The approved survey is unchanged.
- Scratch cleanup removed44obsolete optimizer/resume/gate/duplicate checkpoints, freeing **656 MiB**. Selected models, datasets, current60-epoch runs and active outputs were preserved. Exact deletion audit: `results/scratch_cleanup_20261122/scratch_cleanup_20261122.json`.
- No further GTE access is needed. All 12formerlyGTE-only ESANet checkpoints are mirrored on Triton. New SAM baseline outputs are on Triton home, avoiding the full scratch quota.

## Final results to remember

Three-seed means, reference → automaticSAM labels:

| Model | ISOD object recall | GTE object recall | GTE pixel IoU | GTE component F1 |
|---|---|---|---|---|
| U-Net | .750 → .733 | .464 → .472 | .487 → .559 | .388 → .357 |
| YOLOv8n-seg | .751 → .747 | .600 → .662 | .450 → .578 | .522 → .514 |

On ISOD, all label-source differences in object recall are within.02; pixel IoU changes by−.04to+.02. These are not claims of identical mask accuracy. DirectSAM has the highest componentF1 (.829ISOD / .673GTE) and GTE recovery .945, but costs183ms ontheH200 partition versus4ms forYOLO boxes.

Source of truth: `results/results_report_20261122/report_numbers.json`. Paired recording-bootstrap contrasts: `revision/bootstrap_contrasts.json`. New model metadata and hashes: `results/extended_training_20261122/sam_baselines_artifact_registry.json`. Paired-source environment/optimizer checks: `results/results_report_20261122/cohort_integrity.json`.

## Canonical workflow

From repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/itr-mpl .venv-ml/bin/python code/refresh_results_evaluation.py
```

This runs GTE scoring, full300 ISOD scoring, bootstrap, numeric collection and assets, with logs in `results/results_report_20261122/final_refresh/`. Update hand-written prose from the numeric JSON, then:

```sh
zsh results_report/build.sh
.venv-ml/bin/python code/audit_results_report.py --visual-reviewed
```

Use `--visual-reviewed` only after actually rendering/inspecting the current PDF. The audit verifies36completed runs,300/320frame coverage, actual schedules, source preservation, references, fonts and layout anchors.

The older `make_report_revision_assets.py` main/table routine and `revision/domain_comparison.json` include legacy30frame diagnostics and are not the canonical report pipeline. Current tables come from `make_final_report_assets.py` and `report_numbers.json`.

## User decisions to preserve

Read `results_report/REVIEW_NOTES.md`: one integrated experiment design; finding-based results (depth / automatic labels / masks-boxes-cost); concise plain language; no edit-history or formatting-fix prose in the paper; no “teacher” or “condition” group jargon; clear label names; short captions. Retain all scientific limits in the conclusion. Do not force an equivalence claim if the metrics differ. Preserve the survey and TeX workflow. Do not submit the assignment or message anyone.

User-stated deadline from the Claude session:23:59 on November 22, Europe/Paris; the Canvas wording differs and user is aware. Nothing has been submitted.

## Storage / Git

- Worktree branch: **results-report-revision**. Do not switch to/reset the older `/private/tmp/itr-sam-v2-publish` worktree.
- Triton: `duszaj1@code.triton.aalto.fi`; selected GTE weights `~/itr_weights/esanet_gte`; other runs under `/scratch/work/duszaj1/itr_extended_20261122/runs`.
- NewSAM U-Net/seg runs are symlinked to `~/itr_runs/unet_logged_sam_s*` / `~/itr_runs/yoloseg_e60_sam_s*`. Large weights stay there; report outputs are local.
- Current/new test-reference masks are `data/ISOD_reference_test_v2` (annotations only). Baseline frozen training/validation/SAM manifests exactly match the already-published `codex/sam-v2-overnight` branch.
- Keep model weights, raw RGB/depth, prediction caches, Python envs, build intermediates and raw conversation archives out of Git. Curate code, annotations, metadata, histories, final numeric evidence and PDF.
- Unrelated legacy code modifications may still be in the main workspace; do not reset or blanket-stage them.

## History and remaining work

`results/session_handoff_20261122/REVISION_LOG.md` records material changes. Full visible Codex and Claude conversation/tool archives are local in that folder, ignored byGit; export manifests provide hashes and record counts. Private reasoning/system instructions/binary images are excluded. See its README for refresh commands.

Remaining project/final-article work is listed as future work in the report: validated physical dimensions/heights, prop-aware ignore regions, on-robot timing, more environments, and better verified individual-instance labels. Those are not unfinished training jobs for this results-report round.
