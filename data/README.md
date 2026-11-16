# Frozen floor-obstacle labels

These PNG masks are the supervision and evaluation targets used for the ISOD and GTE-Floor experiments. Binary masks encode background as 0 and floor obstacles as 255.

| Directory | Frames | Purpose |
|---|---:|---|
| [ISOD_reference_train_v1](ISOD_reference_train_v1) | 1,400 | ISOD-assisted training, with 70 human-reviewed masks and 1,330 assisted prelabels. |
| [ISOD_reference_val_v1](ISOD_reference_val_v1) | 300 | Human-reviewed ISOD ground truth for validation and checkpoint selection. These masks also serve as the reference for SAM calibration. |
| [ISOD_reference_test_v1](ISOD_reference_test_v1) | 30 | Human-reviewed ISOD test subset with 10 evaluation frames at each held-out location. |
| [ISOD_sam_train_v1](ISOD_sam_train_v1) | 1,400 | Unattended RGB-only SAM 3 masks for the same training frames. |
| [OWN_manual](OWN_manual) | 320 | Manually corrected GTE-Floor masks and object IDs for external evaluation. |

The ISOD manifests record frame paths, shapes, mask hashes, and supervision provenance. Use each mask's location and frame ID to find matching RGB and depth files in the [original ISOD download](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset). The split by location is in [splits.json](splits.json). The GTE-Floor manifest maps each annotation to its recording under [OWN](OWN). Object IDs start at 1 and follow the order of items within each object set in [items.csv](../data_collection/items.csv). Zero is background in the ID images.

## ISOD attribution and terms

The four ISOD mask sets are adaptations of the Indoor Small Object Dataset by Yu Chen, Chirag Rastogi, Zheyu Zhou, and William R. Norris. The original data provides drivable-area annotations. This project prepared obstacle masks through ISOD-assisted annotation and human review, and produced a second training set with RGB-only SAM 3. The ISOD-derived files are shared under [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/), as listed by the [original dataset](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset).

Please cite [Chen et al., *A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications*](https://arxiv.org/abs/2306.08814) when using ISOD.
