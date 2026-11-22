# Native box detection, November 22

Six completed YOLOv8n **detection** models: reference/automatic labels x seeds 0/1/2, 20 epochs each. These are separate from the earlier YOLOv8n-seg polygon runs and the ESANet semantic runs.

- `input/reference_manifest.json`: 1,400 train, 300 shared validation and 320 external GTE frames; one saved ID becomes one enclosing box, including disconnected pieces. IDs can be assisted groups, not certified individuals.
- `input/reference/labels` and `input/gte/labels`: versioned reference boxes.
- `automatic/automatic/labels/train`: 1,400 RGB-only SAM-generated instance-box labels; generation receives no reference masks.
- `automatic/manifest.json`: pinned checkpoint, per-label hashes, prompts, native instance and parent/child decisions.
- `runs/*/{settings.json,args.yaml,results.csv,gte_metrics.json}`: actual configurations and training/evaluation histories.
- `analysis/summary.json`: AP plus one-to-one matching at fixed confidence 0.25 and box IoU 0.5.
- `analysis/presentation.json`: matched-configuration checks, cohort means/seed SD and selected checkpoint epochs.
- `remote_analysis/artifact_registry.json`: hashes for all 42 saved run artifacts, including best/last checkpoints and full predictions on Triton.
- `analysis/training_curves.pdf`: the new 20-epoch detector histories.

Full predictions and weights remain at `duszaj1@code.triton.aalto.fi:/scratch/work/duszaj1/itr_instances_20261122/runs`. They are not copied into Git. Neither the laptop nor a live VPN connection is required to preserve or complete these already-finished jobs.

Reference boxes use the latest instance pool (85 reviewed binary training masks, 1,315 assisted). All validation binary masks were reviewed, but the per-object separation is assisted. The automatic branch uses fresh native SAM inference with the fixed three prompts, duplicate resolution and conservative parent/child handling, not the original ISOD mask.

Scores use annotated GTE hazard units, including groups; recording props remain unannotated. Do not call this clean operational object AP, or compare box recall directly with mask coverage/localization. Five of six validation optima occur at the final epoch, so the 20-epoch budget limits conclusions.

Recompute from local full prediction exports with `code/analyse_instance_detection.py --base results/instance_detection_20261122 --out results/instance_detection_20261122/analysis`, then run `code/make_native_detection_figures.py`. Model computation was on Triton; local evaluation/plots use saved predictions only.
