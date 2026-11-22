# Full ISOD test evaluation

The current benchmark uses all300images at HDCL_HW_A, MEB_CM and MEB_HW_B (100each). `data/ISOD_reference_test_v2` contains the frozen binary obstacle references and manifest:270 assisted proposals from the same preparation pipeline as training, and30human-reviewed overrides. Proposal agreement on those30 reviewed frames is pooledIoU0.929; this does not certify every assisted instance.

`isod_test300_scores.json` is the canonical evaluation for36 learned runs plus directSAM and the floor-plane fit. Rawpredictions in `pred/` and run folders are local caches, not Git artifacts. `code/score_isod_test300.py` defines all source mappings and component/reference-unit rules.

Baseline training/validation and SAM-label snapshots are already published on the `codex/sam-v2-overnight` branch. Their manifest blobs match this workspace exactly: ISOD_reference_train_v1, ISOD_reference_val_v1, ISOD_reference_test_v1, ISOD_sam_train_v2 and ISOD_sam_all_v2. This revision adds the new300-frame reference version without modifying those frozen snapshots. No rawRGB/depth images are added.
