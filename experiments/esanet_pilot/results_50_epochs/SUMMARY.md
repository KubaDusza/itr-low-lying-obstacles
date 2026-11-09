# ESANet — First 50-Epoch Experiments on ISOD

Date: 2026-11-09
Status: Exploratory experiment — NOT final paper results.

## 1. Research question

Does adding depth information to RGB improve the segmentation
of small, low-lying indoor obstacles?

## 2. Experimental setup

- Architecture: ESANet R34-NBt1D.
- Dataset: ISOD, 2,000 RGB-depth image pairs.
- Split: 1,400 train / 300 validation / 300 test images.
- Split by site: 14 train / 3 validation / 3 test.
- Modalities: RGB and RGB-D.
- Training: 50 epochs, seed 0.
- Initialization: pretrained ImageNet weights.
- Ground truth: automatically generated ISOD_gt v0.
- Mask fingerprint: 322545f4dca9e751.
- Project code: esanet-pilot, initial commit f430ad3.
- Official ESANet commit: 820c5bb.
- Learning-rate scheduler: OneCycleLR, pct_start=0.1.
- Model selection: best validation obstacle IoU.
- Hardware: NVIDIA RTX 3090 GPUs on Shire.

The dataset audit reported no structural problems.
All 2,000 masks match the automatic annotation rule:
they have NOT been manually corrected.

## 3. Validation results

| Model | Best obstacle IoU | Best epoch |
|---|---:|---:|
| RGB | 0.7088 | 41 |
| RGB-D | 0.7137 | 37 |

RGB-D achieved a slightly higher best validation IoU.

## 4. Test results

| Metric | RGB | RGB-D |
|---|---:|---:|
| Obstacle IoU | 0.7416 | 0.7206 |
| Precision | 0.7835 | 0.7548 |
| Recall | 0.9327 | 0.9409 |
| Parameters | 32.05 M | 46.88 M |
| Training time | 26.6 min | 38.5 min |
| Peak GPU memory | 3645 MiB | 5684 MiB |

RGB achieved 2.10 percentage points higher test IoU.

RGB-D achieved slightly higher recall, but lower precision.

## 5. Learning curves

Both models learned successfully.

Training loss decreased substantially.
Validation obstacle IoU increased and then fluctuated
around approximately 0.70.

Validation loss increased during later epochs despite
continued improvements in training loss, suggesting
possible overfitting or increasing prediction confidence
on incorrectly classified pixels.

The best checkpoint was selected using validation IoU,
not the final epoch or the lowest validation loss.

See:
- figures/training_loss.png
- figures/validation_loss.png
- figures/validation_iou.png

## 6. Error analysis

All 300 test predictions were evaluated again using
the project's original isod.score and isod.ignore_region
functions.

The aggregate IoU values matched test_eval.json exactly.

The two modalities perform differently on some images.
However, these differences cannot yet be attributed
specifically to depth information.

Visual inspection revealed that many large predicted
regions lie inside ignored areas and do not affect
the reported obstacle IoU.

Example: MEB_CM/99 has only 55 positive obstacle pixels
in the provisional reference mask.

Per-image and per-site results are available in
analysis/per_image_results.csv and
analysis/per_site_results.csv.

## 7. Scientific limitations

1. Only one random seed was evaluated.
2. ISOD_gt v0 masks are automatically generated and incomplete.
3. RGB-D has more parameters than RGB.
4. The experiment uses a 50-epoch learning-rate schedule.
5. No Depth-only ESANet experiment has been completed.
6. Metrics measure agreement with v0 labels, not necessarily
   detection of every real physical obstacle.
7. The depth audit found values reaching 65535 mm.
   Their meaning and treatment deserve verification.

The test set must not be used for hyperparameter tuning.

## 8. Provisional conclusion

Both ESANet variants can learn to segment obstacles
according to the current ISOD annotations.

In this exploratory experiment, adding depth did not
improve overall test obstacle IoU.

However, this does NOT demonstrate that depth is useless.
A stronger conclusion requires improved annotation
quality, more seeds, and the Depth-only baseline.

## 9. Recommended next steps

1. Review the quality of ISOD obstacle annotations.
2. Complete the limited SAM annotation study.
3. Train and evaluate the Depth-only baseline.
4. Repeat experiments with multiple seeds.
5. Compare results using one agreed annotation version.
6. Examine performance on small and difficult obstacles.

## 10. Reproducibility and files

rgb/ and rgbd/ contain the original configurations,
full training histories and test evaluation metrics.

figures/ contains the three learning curves.

analysis/ contains the ISOD audit and per-image and
per-site evaluation tables.

The trained best.pt checkpoints and all prediction
masks remain on Shire under:

/data/cs_courses/jgomezel/esanet_pilot/

The dataset and model checkpoints are NOT stored in GitHub.
