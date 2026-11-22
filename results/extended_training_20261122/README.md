# Longer box runs and logged U-Net validation

All six native YOLOv8n box runs and all three U-Net semantic reruns are complete and locally retrieved. No GTE VPN connection is needed for the current report.

The box experiment preserves the frozen 1,400-frame training and shared 300-frame validation labels from `../instance_detection_20261122`. Each run starts from the identical COCO checkpoint. The policy is max 200 epochs, patience 20 on validation box AP50:95, seeds 0/1/2 for reference and RGB-only SAM instance labels. It is a fresh longer-schedule experiment, not a continuation with a truncated learning-rate schedule. Earlier 20-epoch runs remain intact.

| Labels | Actual epochs, seeds 0/1/2 | Validation-best epochs from CSV |
|---|---|---|
| Reference | 72 / 73 / 65 | 52 / 53 / 45 |
| Automatic | 67 / 98 / 84 | 47 / 78 / 64 |

Reported checkpoint selection uses the trainer's best validation checkpoint. CSV-derived best epochs are an audit of rounded logs. GTE is not used for model selection. Longer training improves automatic-label external box AP but does not uniformly improve reference-label external generalization. No test-based rollback to the older reference models is made.

U-Net reruns preserve the existing 15-epoch reference semantic recipe, including the frozen 70-reviewed/1,330-assisted training version. The change logs validation BCE+Dice loss from existing validation logits. All three histories contain finite training loss, validation loss and validation IoU. Checkpoints maximize pooled validation obstacle IoU; selected epochs are 8/12/15. In the first two runs the selected epoch was recovered from a unique history maximum because the separate selection metadata copy was added after those jobs started. Their completion metadata is derived from those histories; the third run stores `selection.json` directly. Report scores use the new reruns.

`analysis/summary.json` gives strict native box AP and confidence-ordered matching at confidence .25 / IoU .5. The primary shared mask/box evaluation is in `../results_report_20261122/revision/component_comparison.json`: boxes are filled and united, then evaluated with the adapted SegmentMeIfYouCan component protocol. The protocols answer different questions.

`cost/cpu.json` and `cost/cuda.json` contain real same-image, batch-one inference timings. SAM shares its image encoder across all three prompts. Model loading/file reading are excluded; preprocessing, forward, postprocessing and output transfer are included. GPU uses YOLO FP16 / SAM BF16; CPU uses FP32. Memory definitions differ by hardware and are specified in the report. These are server configurations, not robot timings.

Remote base: `/scratch/work/duszaj1/itr_extended_20261122` on Triton. Large weights remain there; local run histories, provenance, completions and predictions support all report recalculation. Git publishes small metadata, histories and analysis, not model binaries or full prediction caches. Gate 21773723, reference array 21773724, automatic array 21773725, U-Net array 21773726, CPU/GPU benchmark jobs 21774376/21774377.
