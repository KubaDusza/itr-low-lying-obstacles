# Decision and revision log

This log separates verified artifacts from historical intent reconstructed from the preserved visible conversation. Update it with each material decision, run, result substitution or report revision.

## Verified Git checkpoints (Europe/Paris)

- 2026-11-21 20:57 — `1f49d27e`: reviewed GT versions, quality audits and grouping corrections preserved.
- 2026-11-21 21:31 — `520be72a`: verified scoring and figure jobs chained to Triton experiments.
- 2026-11-22 10:15 — `7043612b`: verified IEEE results draft and paired coverage/localization analysis.
- 2026-11-22 14:36 — `9f7e8321`: native box detector comparison and report revision; short20-epoch experiment.
- 2026-11-22 16:57 — `77bb636d`: six longer box runs, three logged U-Net reruns, complete analysis appendix and actual CPU/GPU cost measurements. Report21pages, results/conclusion5–6. All completed learned conditions use3seeds.

## Historical decisions retained from the conversation

- Adapt submitted survey/methodology in place; preserve useful wording, do not wholesale rewrite or include report-type headings.
- Simplify floor-plane explanation; remove unnecessary training/evaluation implementation detail and textbook metric equations.
- Keep GTE collection concise; remove total/reference-only rows where redundant; retain no-object controls within each background condition.
- Ground-truth assistance and the SAM automatic-label experiment are distinct. Human edits mainly improve obstacle references. RGB-only automatic generation must not read original ISOD annotations.
- Full-image SAM first; no default clipped-image pipeline. Ground-truth assistance can intersect original masks. Automatic-label generation cannot.
- Include low-lying items on the floor, including Lego/crayons; reject furniture/walls/on-desk items. User accepted a large vacuum as an ambiguous retained target.
- Human-reviewed masks persist and must be included in regression audits. Fix clicking/grouping bugs; manual cuts often reflect linked editing fragments.
- Separate duplicate pile parents from constituent instances; disconnected visible parts of one physical item can share an ID.
- Use GPUs on GTE/Triton, not the Mac; respect free/reserved resources; avoid computer-use/photo inspection unless needed.
- Three final SAM prompts: small object on the floor / cable on the floor / things on the floor. Four-prompt version with poop was not the final choice. No negative subtraction in the accepted recipe.
- Analysis must explain depth contribution, range, appearance/contrast, form and failures, with paired RGB/depth/model examples and seed variation. Object recovery matters alongside mask IoU.
- Use standard split/merge-tolerant component evaluation; native box AP is secondary. Do not equate covering a poster with localizing its objects.
- Preserve the partner's sound report structure, graph style, cross-dataset and per-object-set tables. Recalculate current numbers; do not copy their old experiment claims.
- All appendix details requested: training curves, label protocols, full scores, examples, seed consistency, compute cost. Keep captions short and useful.

## 2026-11-22 review after commit77bb636d

User's comments identify report-wide failures, not isolated word substitutions:

- Verify15/20epoch RGB baseline claims. Histories confirm3U-Net reference runs15epochs and3YOLOseg reference runs20epochs.
- Fixed20YOLOseg budget is suspect: all3 selectfinal checkpoint by validationIoU. U-Net last validation losses mostly plateau, but do not claim guaranteed convergence.
- Remove “added detector” / “why add detection?” across methods/results/appendix. Present binary masks, instance masks and boxes as one experiment design.
- Report both evaluation datasets accurately: semantic models have30reviewed ISOD test frames; all models have320GTE frames; box-only model has noISOD test result.
- Thirty reviewed test frames are not a completed full300frame ISOD evaluation. Explicitly mark provisional; remaining270references unfinished. User objects strongly; expanding this remains open.
- Replace “external IoU” with “GTE-Floor pixel IoU” and distinguish mask quality from object recovery.
- Remove plot-formatting/edit-history narration everywhere, including text embedded inside figures. Delete axis-range explanations and “full-resolution” wording from report-facing curve labels/captions. Keep actual scaling correct silently.
- State actual checkpoint criterion in methods/captions, not generic “best model.”
- Record all comments as general review rules and proofread the whole report, not only highlights.

Implemented source/figure revisions are in current workspace; latest build after these changes returned21pages. They need final PDF QA and publishing. Survey source remains verbatim.

## New longer segmentation training

- Aalto reconnected by user; no GTE switch needed.
- Runtime gate21776991 and dependent3seed array21776992 submitted on Triton.
- Controller `code/triton_segmentation_extended.py`; deploy `results/extended_training_20261122/seg_deploy/`.
- Frozen semantic labels unchanged; polygon conversion matches existing policy. Max200 / Ultralytics validation-fitness patience20; final checkpoint selected by pooled validation maskIoU over5epoch saves plusfinal. ExplicitAdamW settings prevent auto-optimizer changes with the longer budget.
- **Not yet verified or incorporated.** Record gate result, actual durations, selected checkpoints and evaluation before changing any report number.

## Handoff preservation

- Local append-only session log found;12compactionevents do not erase earlier visible records from that log.
- Export at about2026-11-22 17:27Paris preserved295user/530assistant messages and4640execution records. Private reasoning/system instructions/binary images excluded; obvious credentials redacted.
- Full visible transcript/execution archives stay local; commit only concise handoff, review rules, manifest and this decision log.
- Latest user instruction: continue autonomously until usage runs out, logging as work proceeds.

### Whole-report review, about17:34Paris

- Runtime gate passed; segmentation seeds0/1 started, seed2 queued. No new segmentation result substituted yet.
- Reviewed approved survey, main results and full appendix prose. Removed repeated edit-history language from methods/training appendix and plot annotations; retained scientific metric/protocol explanations.
- Caught and corrected a factual timing-protocol wording error: preprocessing does not change between CPU/GPU for a given pipeline; precision does. Removed an unsupported causal explanation of the latency-gap change.
- Current build again21pages with results5–6; no extra page caused by method overflow. Final render/audit/push still required after current edits.

### Training scheduling adjustment

- Raised array21776992 throttle from2to3 so the scheduler can allocate a third free GPU if available; resource/QoS limits still apply. No test-based tuning or dataset change.
- Latest report source audit passed21pages,26referenced figures/tables, mainresults5–6, no undefined references/overfull boxes/Type3 fonts; PDF SHA2561076fd27e4815c95183ec0679b0944c34bd7b522e3d0082b8a7f401ae259ad57.

### Further integrated-methodology review

- Removed the review chronology and specific sunglasses/loop correction story from paper-facing methods. Actual70/1330semantic versus85/1315box reference counts remain; detailed correction provenance stays in dataset/handoff records.
- Identified and documented a scientifically relevant target difference: YOLOseg polygon conversion omits contours below20pixel area and degenerate contours; native boxes retain all IDs. Longerseg runs deliberately preserve that existing label policy to isolate training-budget changes.

## Codex resumed Claude handoff, 2026-11-22 evening

- Read HANDOFF_CLAUDE_20261122.md, latest Claude visible messages, current branch commits and current analysis scripts. Active branch is results-report-revision at878db627; not the older publishing worktree branch. Nothing pushed/submitted by this resumed pass.
- Preserve Claude/user decisions: all300 ISOD test references (30reviewed overrides +270assisted); fixed60-epoch YOLO schedules; revised finding-based results; new label group names; no “teacher”/“condition” jargon.
- Verified remote SAM U-Net seeds0/1/2 all complete15epochs, selectedepochs11/8/4. SAM YOLOseg runs are still active/queued; do not assume all6finished. New output dirs symlink toTritonhome afterscratch-quota failures.
- User explicitly authorized removing stale scratch artifacts. Inventory before deletion; keep datasets, current selectedmodels andactive output dirs. Deletions will be recorded in results/scratch_cleanup_20261122.

### Resumed review and cleanup

- All three restarted SAM U-Net runs are VERIFIED and retrieved (15epochs; chosen11/8/4). SAM YOLOseg seed0 is complete60epochs, selectedepoch45; seeds1/2 remain active.
- Removed44verified-stale checkpoint/duplicate files, freeing0.640635GiB. Selected/best full-run models, datasets, current60epoch runs andactiveSAM outputs retained. Exact hashes/paths/reasons in results/scratch_cleanup_20261122/scratch_cleanup_20261122.json.
- Reconciled REVIEW_NOTES.md with Claude/user decisions, removing contradictory old30frame andearly-stopping instructions from active guidance. Historical versions remain in handoffs/Git.
- Tightened a few scientific scopes without changing finding structure: ESANet is one model family, not identicalcapacity; unchanged objectrecall is not unchanged predictions; cloth-miss statement applies tothreeESANet versions; inspected false-region examples are not a population estimate.

### Additional resumed-pass verification

- The remaining large scratch directories are other model/Gemma projects; pdf_check contains unrelated project data, not a disposable Python environment. They were not deleted merely because of their names/age.
- RGB-baseline curve generator previously omitted the two newSAMarms; updated it to plot both label sources with per-seed traces and means. Fixed checkpoint-axis indexing (Ultralytics epoch filenames are zero-based) and handled last.pt selections.
- Flow figure used Type3 fonts because pipeline() was invoked outside its former main() setup; made embedding local to the generator and aligned its labels with the approved source names.
- Scoped the automatic-label claim to RGB+fixed prompts instead of claiming nohumanmasks were used at all (prompt development uses reviewed examples). DirectSAM strength is stated in the actual metrics, not blanket superiority. Final wording will be checked against all36run results.

### Final six-arm integration completed

- All six restarted jobs completed with scheduler exit0:0. Retrieved all300ISOD/320GTE outputs; recorded selected/last weight hashes in sam_baselines_artifact_registry.json.
- Ran canonical five-stage scoring/assets pipeline successfully (GTE44.6s, ISOD30080.8s, bootstrap2.2s, collection0.4s, assets11.9s), then regenerated both-source curves and clean supplementary figures. Full per-step logs preserved.
- New GTE means: U-NetSAM objectrecall.472 versus.464reference, pixelIoU.559vs.487; YOLOsegSAM recall.662vs.600, pixelIoU.578vs.450. ISOD recovery changes staywithin.02 acrossmodels; pixelIoU differences span−.04to+.02. Updated prose instead of retaining an unsupported equal-mask-accuracy claim.
- Included both new SAM rows and curves, robust checkpoint handling and corrected one-based plot positions. Reference/SAM initialization, environments and YOLOhyperparameters matched; validation-label manifests match.
- Added clean appendix section breaks, fitted allseed gallery with its introduction, enlarged cable examples and removed embedded revision/defensive footers. Final draft18pages; main results/conclusion stillwithinpages4–6, references6.
- Canonical active handoff now HANDOFF_CURRENT.md. Historical handoffs retained with pointers; REVIEW_NOTES reconciled; Claude visible/execution archives preserved locally alongside Codex history. No assignment submission.

### Completed evidence/figure consistency pass

- GTE filled-box pixelIoU had been omitted from numeric collection although ISOD reported the same metric. Added aggregate pixelIoU to native-box records and the collector, then reran only GTE preparation/collection/assets (ISOD/bootstrap inputs unchanged). Both datasets now report the same four metrics for every trained model.
- The numerical audit requires three finite values for all four metrics in every model/label arm. Final report keeps the main results/references within six pages; appendix repagination removes orphaned headings and keeps figures in their sections.
- Current300reference masks are curated for Git along with manifest; baseline frozen annotation manifests exactly match the already-published dataset branch. No rawRGB/depth, model binaries, prediction caches or private conversations are staged.

### Final review checkpoint

- Final report18pages, all36runs verified on300ISOD/320GTE; main results/conclusion withinpages4–6 and references6. All23figure/table labels are referenced; no LaTeX warnings/overflows orType3fonts. Survey remains verbatim.
- GTEfilled-box pixelIoU is now.221reference/.236SAM, reported consistently withISOD. All four metrics have three finite seed values for every trained arm.
- Caption wording scopes three-seed averages to trained models; directmethods are fixed. Shortened TableII heading while retaining checkpoint-protocol information below the table.
- Latest active state and continuation commands are in HANDOFF_CURRENT.md. Raw Codex/Claude histories remain local and ignored byGit. No assignment submission or messages to others.

## 2026-11-22 21:48 CET — figure, counts and wording review

- Moved the modality legend outside the distance/size panels: the previous unboxed legend overlapped an actual depth datapoint. Data and uncertainty bars unchanged.
- Configuration table now reports three seeds per label source rather than six total runs; no training changes.
- Replaced reference supervision, SAM supervision, assisted annotations and hazard-unit boxes with direct descriptions. Kept the distinction between masks checked by hand and automatically prepared masks.
- Added a two-page inline methodology text/table/caption diff against the saved October 25 methodology, plus the exact TeX diff and source hashes. Pictures are omitted from the text diff; detailed current settings also appear in Appendices B–C. Survey unchanged.
- Rebuilt the 18-page report and visually checked the changed pages and both diff pages; reran report audit. No new experiment or metric changes.
