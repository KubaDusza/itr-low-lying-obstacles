# Frozen floor-obstacle labels

These PNG masks are the supervision and evaluation targets used for the ISOD and GTE-Floor experiments. Binary masks encode background as 0 and floor obstacles as 255.

| Directory | Frames | Purpose |
|---|---:|---|
| [ISOD_reference_train_v1](ISOD_reference_train_v1) | 1,400 | ISOD-assisted training, with 70 human-reviewed masks and 1,330 assisted prelabels. |
| [ISOD_reference_val_v1](ISOD_reference_val_v1) | 300 | Human-reviewed ISOD ground truth for validation and checkpoint selection. These masks also serve as the reference for SAM calibration. |
| [ISOD_reference_test_v1](ISOD_reference_test_v1) | 30 | Human-reviewed ISOD test subset with 10 evaluation frames at each held-out location. |
| [ISOD_sam_train_v1](ISOD_sam_train_v1) | 1,400 | Unattended RGB-only SAM 3 masks for the same training frames. |
| [OWN_manual](OWN_manual) | 320 | Manually corrected GTE-Floor masks and object IDs for external evaluation. |
| [ISOD_sam_train_v2](ISOD_sam_train_v2) | 1,400 | Three-prompt unattended SAM training targets; original reference validation remains unchanged. |
| [ISOD_sam_all_v2](ISOD_sam_all_v2) | 2,000 | Same teacher on every ISOD image, including 300 validation and 300 test images. These are automatic predictions, not human validation/test targets. |
| [OWN_sam_v2](OWN_sam_v2) | 320 | Uncorrected three-prompt SAM predictions for external teacher evaluation. |

The ISOD manifests record frame paths, shapes, mask hashes, and supervision provenance. Use each mask's location and frame ID to find matching RGB and depth files in the [original ISOD download](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset). The split by location is in [splits.json](splits.json). The GTE-Floor manifest maps each annotation to its recording under [OWN](OWN). Object IDs start at 1 and follow the order of items within each object set in [items.csv](../data_collection/items.csv). Zero is background in the ID images.

## ISOD attribution and terms

The ISOD mask sets are adaptations of the Indoor Small Object Dataset by Yu Chen, Chirag Rastogi, Zheyu Zhou, and William R. Norris. The original data provides drivable-area annotations. This project prepared obstacle masks through ISOD-assisted annotation and human review, and produced a second training set with RGB-only SAM 3. The ISOD-derived files are shared under [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/), as listed by the [original dataset](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset).

Please cite [Chen et al., *A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications*](https://arxiv.org/abs/2306.08814) when using ISOD.

## Three-prompt SAM version

The version-2 generator takes full RGB and runs the fixed SAM checkpoint separately for `small object on the floor`, `cable on the floor`, and `things on the floor`. Masks with detection confidence at least 0.60 are unioned at mask probability 0.50. There is no original ISOD-mask intersection, depth filtering, negative subtraction, or human editing in these automatic outputs.

The prompt study used 30 reviewed training frames, with vocabulary inspection and bounded mask-union/exclusion comparisons. The final three-prompt simplification used a 300-frame validation ablation and was explicitly accepted by the project owner. The 30-frame manual-budget simulation therefore includes validation-informed final pruning; it should not be presented as a completely blind 30-label-only result. The accepted recipe and its evidence are in [recipe.json](../results/sam_v2_20261121/recipe.json).

The 1,400 training PNGs are byte-identical to their counterparts in the 2,000-image automatic set. Dataset manifests include shape, foreground counts, content hashes and SAM-cache/recipe provenance. Training checkpoint selection continues to use the 300 reviewed reference validation masks. Existing reference-trained models are reused, and 15 SAM-supervised runs are queued across five model/input setups and three seeds. Large checkpoints and full logs remain on GTE; see the [queue plan](../results/sam_v2_20261121/queue_plan.json) and deployment configuration under `experiments/sam_v2_20261121`.
