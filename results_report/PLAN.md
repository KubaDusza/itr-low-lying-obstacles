# Results report: revision plan (2026-11-22, deadline 23:59)

> Status 2026-11-22 evening: plan executed; current state and decisions in `../HANDOFF_CLAUDE_20261122.md`.

## 1. What is wrong with the current version

Most of the numbers are correct. The problem is that the report presents metrics instead of findings.

1. **No thesis.** Each results paragraph opens with a vague claim ("transfer only partially", "the better model depends on the metric"). After reading pages 5–6 you still cannot say what was found.
2. **The strongest findings are buried or missing.**
   - *Depth intervention.* RGB-D GTE pixel IoU falls from 0.554 to 0.147 when depth is removed, but stays at 0.547 when the depth image comes from another recording. Component recall moves the same way: 0.507, then 0.062, then 0.492. The fusion model needs depth to be present but barely uses the scene-specific geometry. The report gives this one sentence and no numbers.
   - *The teacher beats every student.* Direct SAM 3 reaches 0.945 object recall and 0.673 component F1, against at most 0.78 and 0.52 for any trained model. This is the central fact for the automatic-label question, yet it only appears in the Interpretation paragraph.
   - *Physical pattern.* Depth finds crumpled or raised items whatever their colour (the white tissue on the poster: D 15/15, RGB 0/15). RGB finds anything that contrasts with the background. A flat white towel on a white poster is invisible to every model. This is the "what can and cannot be detected" analysis you asked for, but it sits in an appendix count table.
3. **The headline qualitative figure shows a failure that the appendix then has to explain away** (the seed-0 fusion miss). A main-text example should be representative, with the across-seed counts shown in the figure itself.
4. **The tables are metric dumps.** Table III has eight rows, including pixel F1 (redundant with P/R/IoU) and raw FP/FN pixel counts, and ISOD object recall is "--". Table V mixes every method. Neither table is built around a single claim.
5. **The statistics got worse than the partner's version.** The partner reported paired 95 % bootstrap intervals over recordings. The current version shows only the seed SD. With three seeds and an IoU SD of 0.075–0.088, the RGB-D-over-RGB gain (+0.068) cannot be judged without paired intervals.
6. **Defensive hedging is spread through every section** ("not a clean operational false-alarm rate", "not proof of", "provisional"). The caveats belong in one Limitations paragraph.
7. **The methodology is full of bookkeeping** (70/1,330 versus 85/1,315, Torch versions, the 20-px polygon cutoff) but never says why three output types are compared.
8. **The appendix runs to 14 pages and repeats itself**: two separate cost tables, the same cloth caveats three times, and narration of the curves.

## 2. The story (what the reader should leave with)

Three questions, three answers, each stated as a sentence in the results text and repeated in the conclusion.

| Question | Answer (current evidence; verify numbers) | Main evidence |
|---|---|---|
| **Q1. Does depth help?** | Depth alone is weak (GTE object recall 0.38). Adding depth to RGB gives no reliable average gain. ISOD IoU is identical, the GTE IoU difference is within seed variation, and object recall rises only 0.628 → 0.643. Fusion depends on depth being present (IoU collapses without it) but hardly on the specific scene's depth (swapping it changes little). Depth helps on raised, low-contrast items and fails on flat ones. | Modality table with paired bootstrap Δ and the intervention rows; range/size figure; object-set table; cloth gallery |
| **Q2. Can RGB-only SAM labels replace reference labels?** | Largely yes. With no human mask input for the 1,400 training images, models trained on SAM labels match or beat assisted-reference training on GTE. Object recall: RGB 0.628 → 0.781, RGB-D 0.643 → 0.753, boxes 0.676 → 0.744. Component F1 is unchanged within seed SD for RGB (0.356/0.346) and boxes (0.366/0.357); the one exception is RGB-D (0.428 → 0.356). Box AP50:95 improves (0.352 → 0.438). The cost is about 2× the isolated false regions per frame, and reviewed validation references are still needed to select prompts and checkpoints. Direct SAM is better still (0.945 / 0.673). | Label-source table or scatter with paired bootstrap Δ; teacher-quality numbers (val IoU 0.604, test IoU 0.757) |
| **Q3. Masks, boxes or the teacher?** | Boxes recover as many objects as the best masks (0.676 vs 0.643) but describe extent more coarsely. YOLOv8n-seg has the best F1 of the trained models. SAM is best overall at 45× (GPU) to 392× (CPU) the latency. | Same scatter, plus one cost sentence (table goes to the appendix) |
| **What stays undetected** | Screws, hex keys (object recall ≤0.23 for every model), distant items, and flat objects with the background's colour. | Object-set table; cloth gallery |

## 3. Main-text layout (results and conclusion, 4 IEEE columns)

**IV-A Experiments** (about 8 lines). The 30-run matrix and the two test sets, one sentence on the ISOD 30/300 scope, and a pointer to the appendix matrix.

**IV-B Depth** (findings first)
- **Table I (new, compact):** rows RGB / D / RGB-D / RGB-D with depth removed / RGB-D with swapped depth. Columns: ISOD-30 pixel IoU, then GTE pixel IoU, object recall and component F1. Last row: Δ(RGB-D − RGB) with a paired 95 % recording bootstrap CI. All values are three-seed means. The caption states the finding.
- **Fig. 1:** the range/size panels in the partner's style. Keep them, but switch the whiskers to recording-bootstrap CIs, as the partner had.
- **Table II:** object sets in the partner's format. Keep it; add a component-F1 or YOLOv8n-seg column only if it fits.

**IV-C Training labels and output type**
- **Fig. 2 (new, the key figure):** object recall (x) against false regions per frame or component F1 (y) for every configuration. Reference→SAM pairs are joined by arrows, direct SAM is a star and the floor-plane fit is a marker. Seed whiskers. This one figure replaces Table V and carries both Q2 and Q3.
- One cost sentence (SAM 183 ms vs YOLO 4 ms on the H200, 13.3 s vs 34 ms on CPU).

**IV-D What the models miss** (qualitative)
- **Fig. 3:** a reworked cloth gallery with floor and poster rows and RGB / D / RGB-D / YOLO columns, using a representative seed. Each panel is annotated with its 15-observation recovery counts, so the figure carries the variation and the appendix does not have to defend it.
- Two or three sentences on depth versus contrast, flat versus crumpled, and white on white.

**IV-E Limitations** (one paragraph, every caveat in one place). ISOD 30/300, assisted references and grouped IDs, one GTE room with repeats that are not independent, recording props left unannotated, D455 intrinsics, and the difference in capacity between RGB and RGB-D.

**V Conclusion and future work.** The three answers as three sentences, then future work: the full reviewed ISOD test, prop-aware scoring, on-robot timing and distillation that keeps the teacher's precision.

## 4. Methodology edits (adapted section; Intro/SoA untouched)

- Add one paragraph on why three output types: binary masks give occupied area for a costmap, instances and boxes count hazards, and boxes are cheapest to label and run.
- Move the reviewed/assisted counts, Torch versions and the polygon cutoff to the appendix. Keep one sentence on the reference construction.
- Statistics paragraph: three seeds, and paired bootstrap over the 60 object recordings (seeds pooled within a recording), so the reader knows what the whiskers mean.
- Model table: change "Epochs" to "Training budget / checkpoint criterion" and fill in the actual YOLOv8n-seg budget once the long runs are integrated.

## 5. Appendix (target about 8 pages, every item referenced from the main text)

A. Object sets (keep). B. Configuration and labels: run matrix, reference construction counts, SAM recipe, teacher accuracy with two example figures (cut from three). C. Evaluation protocol plus the full scores table plus per-item recovery. D. Training curves, three figures with one sentence each and no narration. E. Seed consistency with all-seed cloth figures, with the count table folded into Fig. 3. F. Failure types (isolated vs attached FP) and the cable/pawn examples. G. Cost, merged into one table (pipeline latency on CPU and H200 plus parameters).

Delete: the duplicate RTX 3090 forward-only table, the stale `table_rows.tex`, and the repeated caveats.

## 6. New computation needed

1. Paired recording bootstrap (10k resamples) for RGB-D − RGB and SAM − reference: IoU, object recall and component F1, from `recording_summary.csv` plus the component scores.
2. Object recall and component F1 for the two depth-intervention conditions, which are already scored in `revision/summary.json`.
3. Integrate the longer YOLOv8n-seg runs (Triton 21776992: s0 done, s1/s2 running at 17:55). Retrieve them, run predictions on GTE and ISOD-30, score them, and replace the 20-epoch cohort everywhere: table, curves, matrix and text.
4. Draw Fig. 2 (tradeoff scatter) and rework Fig. 3 (gallery with counts); redo the Fig. 1 whiskers.

## 7. Decisions (Jakub, 17:58)

- Story approved. Q2 is reframed as "automatic labels are good enough and save the annotation effort", not as a tradeoff.
- ISOD test: keep the 30 reviewed frames from the three held-out locations. They were never used for training, prompt selection or checkpoint selection. Drop the "provisional" wording and state the scope once.
- Stay in TeX for now (no REPORT.md migration).

## 8. Timeline

| Time | Step |
|---|---|
| 18:00–18:30 | Decisions; bootstrap and intervention numbers |
| 18:30–19:30 | Fig. 1/2/3 and Table I; YOLO-seg retrieval and scoring |
| 19:30–21:00 | Rewrite of results, conclusion and methodology edits (TeX) |
| 21:00–22:00 | Appendix cut to about 8 pages |
| 22:00–23:15 | Proofread every section against REVIEW_NOTES; build; render each page |
| 23:15–23:59 | Buffer; final PDF |
