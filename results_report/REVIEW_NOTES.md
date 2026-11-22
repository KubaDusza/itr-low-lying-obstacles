# Report review rules

These rules apply across the report, including appendices, tables, captions and text inside figures.

- Present a coherent experiment design. Explain the purposes of binary masks, instance masks and boxes together. Remove development-history wording such as “added detector,” “why add detection,” “new runs” and “we do not describe.”
- Explain the scientific question and finding. Remove narration of edits, logging repairs, matching axis limits and other formatting choices. Implement those choices silently.
- Keep necessary interpretation details in concise captions: outputs, colors, sample sizes and checkpoint criterion. Do not repeat the same details in surrounding prose.
- Use clear dataset names. “GTE-Floor pixel IoU” means mask agreement on the separate dataset; it is not a new “external IoU” metric and does not directly count recovered objects.
- Verify training claims against saved histories and settings. Current U-Net baselines have three reference-trained RGB runs of 15 epochs. Current YOLOv8n-seg baselines have three reference-trained RGB runs of 20 epochs. Longer YOLOv8n box runs are a different training task.
- Do not equate a completed epoch budget with convergence. All three 20-epoch YOLOv8n-seg runs select the final saved checkpoint by validation mask IoU. Prepare longer validation-stopped runs; do not call them complete or replace report values before verification. U-Net validation loss mostly plateaus, but its 15-epoch budget alone does not establish convergence.
- State checkpoint selection plainly: validation obstacle IoU for ESANet/U-Net; saved-checkpoint validation mask IoU for YOLOv8n-seg; validation box AP50:95 for box-only YOLO. Test scores do not select checkpoints.
- Distinguish the available ISOD test subset from the full split. Only 30/300 test images have reviewed obstacle references (10/location). Current semantic ISOD numbers are provisional. Box-only detectors currently have GTE evaluation, not ISOD test evaluation. All reported methods have 320-frame GTE evaluation.
- Keep object recovery separate from pixel agreement and spatial accuracy. Use the stated published component protocol for useful split/merge tolerance; strict box AP is a separate diagnostic.
- Preserve the teammate's sound experiment/quantitative/qualitative/interpretation structure, graph style and per-object-set analysis. Recalculate all results from current evidence.
- Preserve the approved survey wording; adapt the methodology where needed to describe the actual experiment accurately.
- Proofread whole sections after edits. Check all repeated instances, not just the highlighted occurrence. Rebuild, render affected pages and check pagination before delivering.

- ISOD test uses all 300 frames (assisted references, 30 reviewed). Do not describe it as a subset or provisional.
- Results are organized as findings: depth, automatic labels, masks/boxes/cost, limitations. Caveats go in the Limitations paragraph only.
