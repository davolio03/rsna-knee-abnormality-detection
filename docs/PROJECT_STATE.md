## Publication verification (2026-10-30)

The 30 changed/new test modules passed offline in the training environment: 289 tests. The separate SIWC authentication, Responses and CLI suites passed in the existing labeler environment: 155 tests. A parent spot-check of the ResNet50 adapter, runner and package passed 13 tests. Whole-project discovery in the training environment is **not green**: 527 tests, four failures, twelve errors and four skips. Missing labeler dependencies explain the SIWC errors and three failures in that environment; a Windows synthetic child-process timeout cleanup failure remains unresolved. These scoped successes do not replace a whole-project pass. No live provider inference or new training was performed during publication verification.

## Current: first full-cohort RadImageNet epoch completed (2026-10-30)

[E041](../experiments/E041_resnet50_pretraining_comparison.md) records successful paired 8+8-study private timing trials on identical series-burden selections. RadImageNet projects 4.315 h raw / 6.556 h conservative per full epoch; ImageNet 3.997 h / 6.078 h. Both allocate about 414 MiB peak CUDA memory. This is conditional cost evidence, not a medical-pretraining quality comparison. The full ImageNet control remains pending.

The canonical RadImageNet run trained and validated one complete epoch across all 2,850/1,448 fold-0 studies from the fresh pinned generic pretrained weights. Parent verified private Version 1 / execution `363626193` succeeded after 15,642.2 seconds (4 h 20 m 42 s), processing 13,195 observed validation cells. Validation masked BCE was 0.552242986240162 and Brier 0.12105172885048329, versus training-only constant BCE 0.6161211079852825 and Brier 0.14907895693661133; RadImageNet was better on both metrics for 12/12 targets. This is one epoch of fixed soft-report-label fit, not evidence that RadImageNet caused a benefit over ImageNet or establishes independent clinical accuracy/generalization. The atomic `checkpoint-epoch-01.pt` has SHA-256 `83518ebe59f8a3ecf39b2d8da35aa4c618afeeb04a78b315b5cda13ff56014a8` in private output `full-run01`. Only checkpoint-based continuation is available; the five-epoch target is not complete, and no subsequent epoch has been launched. The live Kaggle quota was 27:29/30 h used, leaving 2 h 31 m; the full ImageNet control remains pending. Local GPU setup has not begun; the verified local `.venv-training` has PyTorch `2.14.1+cpu` and CUDA unavailable, despite the RTX 3060 Laptop GPU and driver 610.78. [Tracker](../odd/tasks/resnet50-medical-pretraining-comparison.md).

## Latest completed baseline: five-epoch masked submission succeeded (2026-10-29)

Private Resume Run04 Version 1 / execution `363315909` completed all five target epochs. The pinned final report records validation masked BCE 0.5721198783180238 and Brier 0.12899160491401088; checkpoint SHA `f1fb2d250eb601cfb5f03e92e01d2cd020cb5ba6be9fd345576c160d38771d7b`. Checked image-free comparison on the same 13,195 observed validation cells gives training-only constant BCE 0.6161211079852856 and Brier 0.14907895693661136. The model improves 11/12 target BCE values and 12/12 Brier values; PF OA BCE is worse. This supports fixed soft-report-label fit beyond constants, not independent clinical accuracy/AUC or a proven leaderboard improvement.

A new separate pinned checkpoint adapter and source-only canary/full submission package are prepared under private `data/private/masked-model-submission/run01/`; legacy HSR12 successful adapter/artifact remains unchanged. The scoped 166 tests passed, with source/bootstrap/payload/isolated-import checks; independent verification passed 141 scoped tests plus actual training-fingerprint/publication checks. Parent verified private canary Version 1 / execution `363464695` succeeded in 47.9 seconds with 3/3 visible studies, about 20.22-second GPU inference and zero skipped series/slices. Parent then verified the separate uncapped Version 2 / execution `363465727` completed its visible run in 65.4 seconds (`canary=false`, 3/3 studies, about 23.38-second inference, zero recoveries) and submitted exactly Version 2/root `submission.csv`. The competition list shows the new entry Notebook Running (after deadline); the terminal submission succeeded with public 0.701/private 0.683 (old 0.642/0.620). Required inputs are competition, existing offline codecs and exact trained Resume Run04 output. No retraining, Gold use or new dataset transfer is needed. Old reference public 0.642/private 0.620; any future difference changes both cohort and labels, not isolated label causality. See [E040](../experiments/E040_masked_model_evaluation_submission.md) and [feature tracker](../odd/tasks/masked-model-evaluation-submission.md). Git remains read-only; no automatic watcher is configured.

## Current: HSR11 full audit completed; HSR12 opt-in recovery authorized (2026-10-26)

HSR11 private Run02 execution `362520997` completed successfully after 12,871.8 seconds: 4,349/4,349 non-Gold studies and 24,035/24,035 series; diagnostic CSV 1,302,373 bytes; CUDA peak allocated 122,657,280 bytes. The earlier 33-study log was interim. This is public-training runtime evidence only. The original hidden Run08 failure/HSR05 remains unclassified and unresolved.

HSR12 partial-recovery v1 remains opt-in and strict defaults are preserved. Parent verification found and paused v4 before import/execution; the bounded safety corrections now pass 132 applicable tests. Parent independently verified finalized v5-final and completed the private visible canary (3/3 studies, 15 series, zero recoveries). Exactly one authorized competition submission was made from the candidate; the competition currently shows “Notebook Running (after deadline)” and score is pending. No hidden logs are available; the original failure cause and HSR05 remain unresolved. See [HSR tracker](../odd/tasks/hidden-submission-repair.md), [E032](../experiments/E032_hidden_submission_candidate.md), and D026.
# Project State

## Current: masked soft-label training implementation independently verified (2026-10-27)

An isolated soft-target dataset/objective/runner and planning-default CLI validate pinned E034 manifest/folds and retain fractional targets plus observation masks. Parent independent verification passed the correction batch: frozen ImageNet1K V1 ResNet-18/Adam defaults and weight provenance, stable observed-only masked loss with zero gradient for an extreme unknown cell, pre-update rejection of observed weighted overflow, and safe CLI handling of unexpected exceptions. The nine-module selected regression command ran 85 tests (84 passed, 1 existing Windows symlink skip); parent spot-checked both focused modules with 20 passing tests. The local fold-0 metadata plan hashed the explicit offline weights file without deserialization and reported 2,850 training / 1,448 validation studies without image inspection or model construction. Synthetic CPU tests exercised both balance modes. All eight protected hashes stayed unchanged. No real DICOM/image QC/training, Kaggle packaging/upload/launch, or model-performance conclusion is included; the balance heuristic remains unproven. The broader discovery suite was not run. Legacy binary training and the old C2 baseline remain unchanged. See [E035](../experiments/E035_masked_soft_training_system.md) and the [task tracker](../odd/tasks/masked-soft-training.md).

## Current: full non-Gold soft-label prefilter prepared and independently verified (2026-10-27)

The authorized local prefilter prepared a separate private artifact from the pinned original Steven v1 probability file. It covers all 4,349 non-Gold studies and 24,035 series; 51 all-unknown studies are audited but excluded from supervised folds, leaving 4,298 eligible studies across 4,175 normalized-report groups. Three deterministic folds (seed 42) have nonzero observed positive and negative support for every target in both training and validation. Parent verification independently confirmed all rows, group isolation, fold allocation, weights, privacy, pins, and unchanged C2/folds/protected HSR notebook ZIP; the applicable regression suite passed 31 tests with one existing Windows symlink skip, compile/diff-check passed, and an 8-test spot-check passed. Training-only capped class weights remain a preparation heuristic, not applied or evaluated. No image was inspected, no model trained, and no change was made to the existing integer-casting reader, 500-study C2 cohort, folds, or model. The full discovery suite was not run. See [E034](../experiments/E034_full_cohort_prefilter.md) and the [task tracker](../odd/tasks/full-cohort-prefilter.md).

## Current: report-label source comparison completed; training comparison not run (2026-10-27)

The approved original Steven v1 report-probability file was pinned and compared descriptively against the frozen 500-study C2 cohort. The source matched all 4,407 official study IDs; exact Gold-58 IDs were excluded before probability parsing, leaving 4,349 non-Gold rows. All 500 C2 records joined with the existing 167 / 163 / 170 fold memberships unchanged. `0.5` remains masked/not addressed; paired metrics are descriptive agreement with C2, not ground truth. No training, model selection, AUC, or Gold evaluation occurred. No source, cohort, fold, model, or existing training reader was changed. See [E033](../experiments/E033_report_label_source_comparison.md) and the [task tracker](../odd/tasks/report-label-source-comparison.md). A future soft-target model comparison needs a separately authorized adapter; the current training reader casts labels to integers and must not consume the private overlay.

HSR12 status correction: the single parent-authorized v5-final competition submission (execution `362631647`) has since reached terminal success with public score 0.642 and private score 0.620. This does not identify the cause of the earlier hidden failure or close HSR05; see the HSR tracker and E032 for the point-in-time history.

## Current: HSR10 Run03 private diagnostic completed; hidden competition issue unresolved (2026-10-26)

Run08 Version 3 / execution `362238893` successfully installed the offline codec bundle (13.1s), reported all five target decoder preflights available, processed the visible 3-study/15-series sample through CUDA inference (20.9437s), verified checkpoint/report pins, and wrote root `/kaggle/working/submission.csv` (1.03 KB). The competition submission later ended with `NotebookThrewException` after approximately 45–75 minutes. Kaggle exposed only a generic unhandled error; there is no traceback or verified root cause. Do not claim the codec availability gap or run07 wheel-tag bug explains this hidden failure.

HSR10 v4 was run privately as Run03 Version 1 / execution `362492412` (Save & Run All, Internet OFF, T4x2, pinned environment, four inputs). Parent verified terminal success in 373.7 seconds. Final safe aggregate logs report 100/100 studies, 546 series, Axial 131 / Coronal 197 / Sagittal 218, 30,037-byte diagnostic CSV, inference elapsed 345.50719228 seconds, and adapter elapsed 345.683879726 seconds. The actual dataset, preprocessing, trained model, production runner, CSV write/readback and validator completed. CUDA peak allocated memory was 114,660,352 bytes (~109.35 MiB); this is not whole-GPU memory or process RSS. Interim logs at about 149 seconds showed 37/100 studies; 546 was the planned total, not a processed count then. The diagnostic intentionally did not write `/kaggle/working/submission.csv` and was not submitted to the competition. Checkpoint and protocol were unchanged. This completes HSR10 testing, but HSR05 and the original hidden competition failure remain unresolved/unclassified. See [E032](../experiments/E032_hidden_submission_candidate.md) and the [HSR tracker](../odd/tasks/hidden-submission-repair.md).

## HSR10.6 historical builder checksum defect (2026-10-26)

Parent reports Run02 Version 1 / execution `362263657` failed at source-archive verification after 15.2 seconds, before codec or model work. Reproducing against the uploaded v3 notebook shows its embedded base64 source ZIP hashes to `f075c14cd389168c3d51342eb8ddf7ac86fe4cb3f411806ae9ff6af432dc0d99`, while the notebook comparison constants and actual archive use `a24273eca1f7f6b08923bacdcf83d231fa452b713133fe4102afd25ae341a1c7`. HSR10.6 added a generated-notebook checksum regression, fixed the package builder, and prepared a fresh private v4 diagnostic candidate. This packaging failure does not establish a cause for the prior hidden competition exception. Parent owns any Kaggle rerun. See [E032](../experiments/E032_hidden_submission_candidate.md) and the [HSR tracker](../odd/tasks/hidden-submission-repair.md).

## HSR10.7 interim Run03 progress (historical; parent-reported)

Parent independently confirmed v4 package hashes and byte proofs, reran the 108-test scoped suite, and spot-checked the package regression (1 passed, 0.095s). Parent created [RSNA Knee Production Runner Public100 Audit Run03](https://www.kaggle.com/code/davidsulleiro/rsna-knee-production-runner-public100-audit-run03), then launched Version 1 / execution `362492412` using Save & Run All, Internet OFF, T4x2, pinned environment, and four inputs: official competition data, trusted model output from execution `361175523`, private codec package, and private JSON plan. Interim logs at approximately 149 seconds showed 37/100 studies; 546 was the planned total series count, not yet a verified processed count. Terminal completion is now recorded in the current status above. The original hidden competition failure and HSR05 remain unresolved. See [E032](../experiments/E032_hidden_submission_candidate.md) and the [HSR tracker](../odd/tasks/hidden-submission-repair.md).

## Historical run08 visible-sample success; hidden competition failure (2026-10-25)

Parent reports run08 notebook Version 3 / execution `362238893` completed offline codec installation in 13.1 seconds and preflighted all five target decoders: JPEG Lossless P14/SV1 (`.57/.70`) via pylibjpeg, JPEG-LS Lossless/Near-Lossless (`.80/.81`) via pyjpegls and pylibjpeg, and JPEG2000 Lossless (`.90`) via Pillow. Full visible inference processed 3/3 studies and 15 series on CUDA in 20.9437 seconds, verified the expected checkpoint/report pins, and wrote the root `/kaggle/working/submission.csv` (1.03 KB). Parent spot-check passed 9 offline bootstrap tests in 0.232 seconds. The parent submitted Version 3 to the competition; it initially ran, then ended with `NotebookThrewException` after approximately 45–75 minutes. Kaggle exposed only a generic unhandled error with no traceback. The run07 bootstrap failure was traced locally to validation expecting a combined wheel tag rather than the wheels' separate WHEEL `Tag:` headers; HSR09 corrected the validator. Run08 success verifies bundle install and the visible sample, not hidden-set transfer syntaxes or the original hidden exception cause. No fallback model or clinical/model/protocol change occurred. See [E032](../experiments/E032_hidden_submission_candidate.md) and the [HSR tracker](../odd/tasks/hidden-submission-repair.md).

## Historical run06 candidate preparation (2026-10-25)

Prepared a new disabled-by-default, output-free full-inference candidate at ignored `data/private/fold0-submission/run06-submission-candidate/`. It preserves the trusted fold-zero pins, embeds the same 19-member source archive as run05, adds only a safe notebook-start runtime/decoder-availability event, and publishes the competition CSV at `/kaggle/working/submission.csv`. Archive member hashes and isolated `python -I` import passed. The new real JPEG2000 Lossless synthetic DICOM case passed through actual production preprocessing, `run_inference`, CSV readback and the smoke validator; no local valid JPEG Lossless P14/SV1 or JPEG-LS encoder/wheel is available. Pydicom 3.0.2 reports `.57`, `.70`, `.80`, and `.81` unavailable and `.90` available via Pillow. No cause for hidden `NotebookThrewException` was reproduced; no model/preprocessing behavior change, Kaggle execution, submission, wheel install, or checkpoint deserialization occurred. Candidate is not submission-ready until parent observes the actual Kaggle codec/runtime preflight and decides whether compatible offline wheels are needed. Browser UI is currently unavailable. See [E032](../experiments/E032_hidden_submission_candidate.md) and [HSR task tracker](../odd/tasks/hidden-submission-repair.md).

## SR01 synthetic failure-class reproduction complete; historical cause unproven

Added an isolated synthetic test module that composes the existing fold-zero dataset, preprocessing, inference runner, CSV publication/readback, and smoke validator with synthetic pydicom inputs; checkpoint loading and model forward are stubbed. Worker focused checks passed 7 tests (7.855s); the specified seven-module scope passed 88 (15.568s, no skips). A separate stub-only 1,300-study/series bookkeeping case took 4.921s with Python tracemalloc peak 5,541,528 bytes; it is not a decode/GPU/RSS benchmark. Parent spot-check separately passed seven focused tests in 10.623s and measured the stub scale case in 7.303s / 5,563,491 bytes. Independent SR02 verification passed the exact seven-module scoped suite (88 tests, no skips; verifier-reported 19.373s), confirming only checkpoint-loader/model stubs in the positive composition. The independent scale case was 5.73s / Python tracemalloc peak 5,541,712 bytes; this is not decode/GPU/RSS performance. SR03 is recorded in E031 and the tracker.

Local JPEG Lossless SV1 decoder availability is absent; JPEG2000 Lossless is available via Pillow, but no compressed bytes were tested. This and public comparator implementation review keep codec/input handling and resource/hidden-I/O among hypotheses; none is established as the cause of the hidden generic `NotebookThrewException`. Existing visible 3-study/15-series success and the later CPU/GPU diagnostics cover distinct scopes, not full hidden-scale production inference plus CSV. No production change, remote execution, new submission, or actual checkpoint deserialization occurred. SR03 interpretation is recorded in E031 and the tracker. See [E031](../experiments/E031_submission_failure_reproduction.md) and [SR01–SR03 tracker](../odd/tasks/submission-failure-reproduction.md).

## Full-study fold-zero model/runtime diagnostic complete; no failure reproduced (2026-10-24)

One separately authorized private GPU run tested the 81 sibling series from the same 20 studies used in CA12/CA16. Kaggle Version 6 / execution `361908116` completed after 82.1s on T4x2 / `cuda:0`; no competition Submit action/new submission or training occurred. The ID-free report SHA-256 is `00d35ca312a85dfd6f63030b86856a3ce6bfcc1eca65c2c98156b721c6ee185f`; report and screenshot are under ignored `data/private/full-study-runtime-audit/run01-preparation-v4/`. Expected/verified provenance maps matched across eight fields. The full-study plan matched the same 20 selected studies and all 81 ordered sibling series from pinned metadata (Axial 22 / Coronal 20 / Sagittal 39; 658 exclusions).

Report status is complete: 20/20 studies and 81/81 series attempted/completed/successful, 81 preprocessed, zero failed studies/series, no runtime failure, checkpoint verified and loaded. Series success means preprocessing-stage success. Aggregate output counters: 240 finite logits and 240 finite probabilities; logit range -3.5538 to 2.2309. Audit runtime 54.564s; peak PyTorch allocated GPU memory 97,880,576 bytes (~93.35 MiB), not total VRAM. Environment was Python 3.13.15, PyTorch 2.11.0, torchvision 0.26.0, CUDA 12.8, `cuda:0` on a two-device T4x2 runtime. The report contains no per-series result rows or per-study probability values.

No failure was reproduced on this bounded same-study multi-series sample. It is not proof of every pixel-stack decode, all-cohort/all-plane behavior, full production submission-contract compatibility, model quality/generalization, or resolution of the historical hidden exception. Compared with CA16's 31.056s audit for 20 isolated series, the 54.564s full-study audit processes 81 series; this is not a controlled benchmark and cannot isolate the cost of pooling. The latest diagnostic status supersedes the previous section's proposed next step to test sibling aggregation. Next recommendation: read-only packaging and production-submission-contract review before any further run; any further execution requires separate authorization. Earlier writer `audit_checkpoint_failed` remains unclassified; FS-02's separate scoped rerun passed 189 tests with one skip and a parent five-test spot check, not a global-suite-green claim. No new source/model/preprocessing changes or submission occurred. See [full-study audit](../odd/tasks/full-study-runtime-audit.md) and [E030](../experiments/E030_training_compatibility_audit.md).

## Bounded fold-zero model/runtime diagnostic complete; no failure reproduced (2026-10-24)

One authorized private GPU run (Kaggle Version 5 / execution 361898806, T4x2 / CUDA device 0) completed successfully after 58.0s. The ID-free report SHA-256 is `0e0bb3a5548ebffb28d26e1d40f92b19c2fd3cb0e94b63771b7e7a546a6cc1e2`; screenshot and report are under ignored `data/private/training-model-runtime-audit/run01-preparation-v2/`. All 20 pinned fold-zero inference inputs are distinct selected Sagittal series from CA12, and all expected/verified provenance hashes matched local plan, source inventory, checkpoint, training report and `train_series.csv`; checkpoint status was verified and loaded.

The report is complete: 20/20 series succeeded, zero failures/runtime failures, and aggregate counts were 240 finite logits / 240 finite probabilities. Finite logit range: -2.8658 to 2.3247. Runtime elapsed 31.056s; peak PyTorch allocated GPU memory 88,310,272 bytes (~84.2 MiB), not total VRAM. Environment: Python 3.13.15, PyTorch 2.11.0, torchvision 0.26.0, CUDA 12.8; cuda:0 on two-device T4x2. This aggregate report contains no per-series ordinal result rows or per-study probability values. No competition Submit action/new submission occurred; earlier versions were preserved.

This demonstrates the pinned checkpoint/model can run on these 20 isolated Sagittal inputs after selected-slice preprocessing; it is not a model-quality/AUC result and does not test sibling-series study aggregation, all planes/cohort, or the full competition inference contract. No failure was reproduced; the historical hidden exception remains unconfirmed, not proven fixed. CA13-CA16 are complete as performed; CA14 scoped verification was 184 tests with one skip (17.169s) plus a five-test parent spot check; earlier broad discovery failures remain unclassified, so do not claim global green. The remaining natural model/runtime boundary is full production inference with sibling-series aggregation for these same studies, requiring separate authorization; no such job was prepared/launched. No retraining, source/model/preprocessing/checkpoint changes, full-test/Gold data, or new submission. See [E030](../experiments/E030_training_compatibility_audit.md) and `odd/tasks/training-compatibility-audit.md`.

## Fresh-training CPU compatibility audit complete: zero failures in scoped sample (2026-10-23)

Private CPU execution 361594587 succeeded in 327.1s total (audit 304.042973s). All 100 frozen fresh studies / 546 series were attempted, completed and successful, with zero failures. Remote report SHA-256 2975a6b3065a2e3bf72ded080aed211c03acef71338744677f2bc9581160612b and the frozen plan hash were verified; all 546 expected ordinal pairs are unique and successful. The audit read 18,327 headers and produced 4,368 finite selected-slice outputs. Plane counts: Axial 131 / Coronal 197 / Sagittal 218; row and column ranges: 256-1280. Only uncompressed explicit_vr_little_endian was observed (decoder available, plugins []); JPEG/compressed coverage is absent. Report, analysis proof and screenshot are under ignored data/private/training-compatibility-audit/run01-preparation-v2/. [Execution 361594587](https://www.kaggle.com/code/davidsulleiro/rsna-knee-training-compatibility-audit-run01?scriptVersionId=361594587).

CA01-CA04 are complete as performed; authorization is consumed, with no retry. The exact current CPU preprocessor worked on this outcome-blind training sample, but the historical hidden failure was not reproduced and its actual cause remains unknown. This does not validate rare data/geometry/compression, all pixels, GPU/model/runtime/submission-contract behavior, general hidden compatibility or medical quality. No model, checkpoint, labels, radiology reports or predictions were read. No new submission is authorized. Proposed next diagnostic, not executed: inventory uncovered training header strata before a predeclared targeted audit; a later end-to-end model/runtime test would require separate scope. Preserve negative evidence and prior failures. Details: [E030](../experiments/E030_training_compatibility_audit.md); submissions: [E029](../experiments/E029_fold0_kaggle_submission.md). Source, packages, model, manifest and folds are unchanged; no new tests were needed for passive report analysis. Git metadata remains read-only; RDD is off. Fold-1/2 history below is unchanged.

## Current: fold-1/2 frozen-protocol runs launched; terminal results pending (2026-10-22)

The previously authorized frozen ImageNet V1 fold-0 execution `361175523` completed all 1,665 steps across five epochs on 333/167 studies. Macro AUC was 0.6457976721 versus 0.5365993973 for full-backbone adaptation; paired delta +0.1091982748, study-bootstrap 95% interval [+0.0624354, +0.1584720]. Runtime was 4,554.6 seconds and peak CUDA allocation 126,066,688 bytes. It exceeded the unfrozen single-fold macro AUC in this paired analysis, but this is one development fold, not external-test evidence or proof of a causal mechanism; Fracture AUC was lower (0.5173 vs 0.5550) with only eight validation positives. Evidence is in ignored `data/private/resnet18-frozen-backbone/run01-upload/frozen-terminal.log` and `frozen-paired-analysis.json`.

LD-11/LD-12 implementation and fold-0 preparation were independently verified (130 tests in 11.185s, seven ResNet tests in 2.848s, plus pinned-weight CPU probe: 120 backbone state tensors unchanged after Adam, head gradient 3.8971/nonzero update, 6,156 trainable parameters). Parent uploaded private reader dataset Version 11 and launched the one authorized fold-0 execution once; no retry.

LD-14 fold-1 and fold-2 packages were prepared locally under fresh ignored v2 directories and separate chooser ZIPs. Protocols derive from pinned assignments: fold 1 = 337/163 / 1,685 steps; fold 2 = 330/170 / 1,650. The inaccurate v1 markdown packages remain preserved, superseded, and were never uploaded/launched. V2 markdown identifies each selected fold and the single frozen-backbone control. LD-15 independent verification is complete: 132-test suite in 14.761s, 11 targeted package checks in 10.162s, and parent spot-check of 11 tests in 10.057s; pins and all 28 chooser members per fold passed. Parent uploaded/pinned private dataset Versions 12 and 13 and launched one run per fold (LD-16; no retries). Fold 1 execution `361209064` has startup evidence confirming 337/163, five epochs / 1,685 steps, CUDA, frozen backbone and 6,156 trainable head parameters. Fold 2 execution `361209896` has run-start evidence confirming 330/170 and five epochs / 1,650 steps on CUDA, plus training-start confirmation of the frozen backbone and 6,156 trainable head parameters. Both were last observed Running (about five minutes for fold 1 and one minute for fold 2); metrics and terminal outcomes are pending. Commit remains blocked by managed read-only Git metadata; RDD is OFF. See [E028](../experiments/E028_resnet18_pretrained_control.md).

## LD-08 ResNet-18 prerequisite preparation (historical snapshot; later verified)

User authorized the matched random-vs-ImageNet1K V1 ResNet-18 experiment. Downloaded the official V1 state dict into ignored `data/private/resnet18-comparison/weights/resnet18-f37072fd.pth` (46,830,571 bytes, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; URL-prefix hash `f37072fd` matches). Safe CPU `weights_only=True` load and strict ResNet-18 state-dict compatibility passed after hash verification. Installed official CPU torchvision `0.29.1+cpu` into existing `.venv-training` only after inspecting its Windows CPython 3.11 wheel metadata (`torch>=2.14.0`); existing `torch 2.14.1+cpu` remained unchanged. Synthetic CPU model construction/forward passed; CUDA is unavailable locally, so Kaggle remains the execution target. Weight terms may be dataset-derived and are not assumed to be identical to the BSD-3-Clause torchvision software license; see [E028](../experiments/E028_resnet18_pretrained_control.md). At the time of this historical prerequisite snapshot, LD-09 implementation awaited independent verification and no remote training run had started; subsequent verification and LD-10 launches are recorded above.

## LD-07 real-input CPU diagnostic completed (2026-10-21)

Private dataset Version 9 was uploaded and verified at 26 files / 1.63 MB; its 26-file transport tree and dataset contents were unchanged for the run. The corrected notebook selected only between the two allowlisted dataset mounts by verifying the inventory SHA; the actual selected mount path was not printed. Parent verified the private dataset and official competition inputs, CPU-only accelerator, private visibility, internet off, and draft session off, then launched the single authorized run. Execution Version 1 (`360841828`) completed in 254.5 seconds. The eight-study audit found finite inputs and zero exact duplicates among 28 pair comparisons. Across-study input mean/std ranges were 0.1834457278251648–0.3010384539763133 / 0.20877141058444976–0.3189350763956706. The fresh scratch CPU probe ran exactly two Adam steps on two studies: status `ok`; encoder/head gradient norms 0.11206283593294178 / 0.15020050147137515; maximum parameter deltas 0.0020013563334941864 / 0.0020013321191072464; input-induced logit delta 0.00017346891399938613; post-update output delta 0.02310626022517681. This is a diagnostic probe, not full training, GPU training, validation or generalization evidence. Zero exact duplicates do not establish clinically adequate preprocessing, and causal root cause remains unknown. One-run authorization is consumed; no retry. LD-04/LD-06 comparison is not implemented and required dependency/weights remain absent; a matched random-vs-ImageNet V1 ResNet-18 comparison is a useful next experiment, not evidence the baseline defect is fixed. Aggregate-only summary: `data/private/training-learning-diagnostics/learning-path-run01-upload/run01-aggregate-summary.json` (SHA-256 `2d469aafcd55fda3053c8c3fe52e937cec159200965049b57ebc370934833cc4`). Details are in [E027](../experiments/E027_training_learning_diagnostics.md).

## LD-05 synthetic learning-path check (2026-10-21)

Added a bounded, opt-in CPU probe for the actual scratch CNN. Synthetic tests verify distinct inputs affect logits, encoder/head gradients are finite and nonzero, both parameter groups move after Adam, and outputs change; invalid, non-finite, and constant inputs do not pass. Fixed a global seeding issue so the helper preserves caller CPU RNG without invoking accelerator seeding APIs; non-CPU masks now fail safely. The exact focused 10-module suite passed 116 tests. Fresh disabled-default preparation v8 is under ignored `data/private/training-learning-diagnostics/sanity-run01-preparation-v8/`; its 23-member ZIP preserves the prior 22 package member bytes and adds the helper. It audits aggregate inputs for the first eight fold-0 training studies, then performs exactly two total CPU optimizer steps on the first two. A synthetic ZIP-import smoke passed; no real images were read, upload or GPU run occurred, and no dependency/weight download or checkpoint load occurred. LD-05 local implementation/preparation passed independent verification (116 focused tests plus two targeted regressions); parent repeated seven helper tests. The real-image CPU probe remains unrun; details are in [E027](../experiments/E027_training_learning_diagnostics.md). E028 records a protocol-only random-vs-ImageNet V1 ResNet-18 comparison draft; no model or weights were obtained. Commit remains blocked by managed read-only Git metadata.

## Current training diagnostics and run04 terminal status (2026-10-20)

Run04 execution `359470765` completed successfully: 1,665 steps over 333 training and 167 validation studies, five epochs, and macro AUC `0.5203435261555952`. Prediction JSON contained 167 rows and recomputed to the same AUC; all 12 target AUCs were defined. The report did not include training-loss curves, and no local report copy has been verified. This terminal evidence supersedes the earlier pending snapshot in the historical run04 section below; preserve that earlier history.

LD-01 epoch telemetry is independently verified (104 focused tests plus parent spot checks); its local commit remains blocked by managed read-only Git metadata. LD-02's eight-study fold-0/seed-42 diagnostic completed in one authorized corrected v4 run against private dataset Version 8: 8 studies, 240 steps, 30 complete epochs, initial fixed-subset BCE 0.6906175 and final 0.4713898. The predeclared <=0.10 memorization target was not met. Final BCE was 0.0085939 below the mean per-target constant-prevalence BCE baseline (0.4799836), with low per-target across-study prediction dispersion; this is training-subset evidence only, not validation/generalization evidence or proof of a causal model/input defect. The remote run took 357.8 seconds; the notebook measured 323.731 seconds. One target had no positives, so its AUC was undefined; 11/12 target AUCs were defined, but none are validation metrics. LD-03 is complete as a performed-and-interpreted diagnostic, with the target not met. Do not retry. Before scaling or the LD-04 pretrained comparison, investigate input/gradient/learning behavior on a focused basis; LD-04 remains unstarted.

A simple future-runtime estimate from the historical 83.315-second / 40-step run was about 500 seconds (~8 minutes 20 seconds) for 240 steps; the measured v4 run completed in 357.8 seconds. The dataset decodes on demand with no cross-step cache, so runtime can differ. Full aggregate evidence and interpretation are recorded in [E027](../experiments/E027_training_learning_diagnostics.md) and the private ID-free summary at `data/private/training-learning-diagnostics/sanity-run01-upload-v2/run02-aggregate-summary.json`. The earlier v1 synthetic timeout remains unexplained. The one authorized retry was consumed; no further retry or LD-04 launch is authorized by this result. Local commits remain blocked by read-only Git metadata.

## Run04 preparation for one authorized private full-fold relaunch (2026-10-17)

The user authorized exactly one private Kaggle GPU training run using the existing private reader dataset Version 6 and official competition input. Local preparation and parent verification are complete. Parent launched the one authorized private run once; the initial viewer snapshot showed it running with no output yet, so terminal outcome remains pending. The fixed protocol is fold 0 / seed 42, 333 training and 167 validation studies, scratch model, five epochs / 1,665 planned steps, CUDA, 256-pixel inputs, at most eight slices, width 64, learning rate 0.001. Gold-58 remains excluded. No other data, model, cohort, fold, prompt, preprocessing, or protocol changes; no automatic retry.

The fresh disabled-default notebook/config are under ignored `data/private/kaggle-first-fold-background/run04-preparation/`, with distinct output `/kaggle/working/kaggle-first-fold-baseline/run04`. The trusted run03 notebook/config were used as templates; only V6 inventory/archive pins and the output path changed in the notebook, with matching config values. The already-uploaded V6 ZIP was reused unchanged (SHA-256 `226ec46ab3ec944d67e2537b4ba35598eae90be826de875de9710baa97183d82`; inventory SHA-256 `fcd6c6bfd9349f760a4a97f4a2de368538ca949cca22bbf121803890a153550c`; 22 members). Relative to V5, only `src/rsna_knee/data/dicom_preprocessing.py` differs; manifest/fold pins remain `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.

Notebook SHA-256 `31cda55cf9d9cf912dc8db371cd6c6dab6b187090b62d652f453a92baf6416ca`; config SHA-256 `c506749e6f555af4bf7f7f2ea48baa657e9455658463600abc5168622e6c56f2`. The prepared code cell is 16,297 characters, disabled by default, SHA-256 `0b4b23f2749a47687b11ecdd46b5fc1afdf3eb0afcd5c515cff167f2e778d7e0`; the one-flag enabled clipboard cell is 16,296 characters, SHA-256 `8f4a5558d47fc0b291a729ce68611984ae9367ea7fc73c6558daa39e47952807`. Notebook has empty outputs; exact source comparison confirms only the output path and V6 inventory/archive pins changed from run03. V6 ZIP/inventory member hashes, embedded inventory bytes, manifest/fold pins, protocol config, notebook JSON, and Python syntax were verified. The focused runner/data/package/notebook suite passed 102 tests. Parent independently passed 18 additional checks and 15 notebook/package tests. No new package was built or uploaded; no DICOMs, reports, pixels, GPU, or remote execution were accessed in local preparation. See [E026](../experiments/E026_kaggle_first_fold_baseline.md) and the ignored tracker for details.

Parent launched exactly one private run from notebook Version 1, execution `359470765`: [private run04 status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-first-fold-training-run04?scriptVersionId=359470765). Reader dataset Version 6 and the official competition were explicitly pinned; visibility is private, T4x2 selected, interactive session off. The full enabled cell was read back at 16,296 characters with SHA-256 `8f4a5558d47fc0b291a729ce68611984ae9367ea7fc73c6558daa39e47952807`, matching the prepared cell with only `RUN_TRAINING = False` changed to `True`. One Save Version → Save & Run All was accepted. The initial viewer snapshot showed `Running` for 10.5 seconds, T4x2, and 0 B output. No epoch, step, metric, checkpoint, or terminal result is established. This is the single authorized attempt; do not retry or restart.

**Next step:** Await parent-observed terminal evidence and record it without inferring success or failure from the initial running snapshot.

## CPU orientation-header audit complete; local boundary correction verified (2026-10-17)

The one authorized private CPU orientation-header audit completed in notebook Version 1 / execution `359458891`, using private reader dataset Version 5 and the attached official competition input. Its ID-free summary reports 12,319 headers read in 115.273 seconds; the first threshold-exceeding series (ordinal 355) was fully read across 43 headers, with 23 headers above the existing `1e-5` component threshold and zero invalid orientations or read failures. Maximum component delta/spread was `0.000010000000000010001`; maximum row/column/normal angles were `0`, `0.0005712049259344931`, and `0.0005593304958631258` degrees. This is a bounded training-header observation, not a cohort-wide estimate or clinical conclusion.

A user-authorized local fix now retains the same scientific orientation threshold while allowing only a two-ULP, operand-scale arithmetic-rounding margin in the orientation component checks. Spacing and projected-position comparisons, data selection, models, and folds are unchanged. Synthetic regression tests reproduce the exact-boundary false rejection and verify larger deltas/rotations still fail. Corrected V6 package and disabled-default CPU preprocessing-canary artifacts are in `orientation-preprocessing-canary-run01-v4`; the package differs from V5 only in `src/rsna_knee/data/dicom_preprocessing.py`. Parent verification passed and one private CPU canary completed successfully for the single targeted training series, ordinal 355. Its aggregate reports 43 input files, finite output shape `[8, 1, 256, 256]`, normalized range `[0.0, 1.0]`, and 2.651 seconds of preprocessing. This confirms only the bounded series-level preprocessing path, not full-cohort readiness or model training. The focused suite ran 100 tests: 99 passed and one symlink-capability test was skipped because the sandbox cannot create directory symlinks. No real DICOM data was opened locally. The one canary authorization is consumed; no rerun or training is authorized. See [E026](../experiments/E026_kaggle_first_fold_baseline.md) for provenance and exact hashes.

## Run03 controlled retry failed; no further retry (2026-10-17)

The prior queued run03 execution `359370509` was canceled; the one explicitly authorized identical retry, execution `359386220`, failed after 289.4 seconds with 0 B output. Its safe event reported only `data_reason_code=inconsistent_orientation`, family `training`, source `run_training:492`; actual orientation deltas and failing series remain unknown. That training retry authorization is consumed. See E026 for the preserved cancellation evidence and history.

## First full-fold Kaggle baseline: V1 and diagnostic run02 failed (2026-10-17)

FB-01/02 and launch recording are complete; private Version 1 / execution
`359170889` failed after 258.3 seconds with 0 B output. Logs at 247.3 seconds
reported `training_blocked` / `training_failed`, followed by Papermill
`RuntimeError(training_failed)` from notebook main line 232. No metrics, report,
predictions, or checkpoint were produced. The underlying exception, failure
stage, and completed step count remain unknown; do not infer GPU out-of-memory
or a specific failed step. The original Version 1 authorization was consumed.
After a new explicit user authorization, exactly one diagnostic run was
prepared, uploaded as private package Version 4, and launched as notebook
Version 1 / execution `359359350`. The parent observed failure after 400.7
seconds with 0 B output. At 390.4 seconds, the safe event reported
`stage=training`, `exception_family=value`, `exception_chain=[value]`, and
source `training.baseline.run_training:477`. That expression evaluates
`train_dataset[row_index]` before entering `_sample_loss`, so the location does
not prove a dataset/preprocessor failure; the origin may be dataset access or
the loss path. Do not infer a specific decoder, preprocessing, numeric-loss,
or OOM cause. No metrics, report, predictions, or checkpoint were produced.
The one-run authorization is consumed; do not retry or restart.

Local diagnostics now preserve only allowlisted exception family and bounded
exception-family chain, coarse stage, and validated internal source location; notebook and CLI tests verify raw error
text, paths, IDs, reports, and locals are not exposed. This code was not included
in Kaggle package Version 3 or the failed Version 1, so it cannot diagnose that
historical run. The separately authorized diagnostic run02 used the matching
Version 4 package with the same fixed trial/provenance and a distinct `run02`
output directory. A local strict-TDD correction now preserves only known
allowlisted `DicomPreprocessingError` reason codes through the dataset, runner,
CLI, and notebook diagnostic validation. It does not determine the historical
run02 cause, change data/model policy, or authorize a package rebuild or new
remote run. See [E026](../experiments/E026_kaggle_first_fold_baseline.md).

## Kaggle eight-study overfit completed; no validation or publication (2026-10-17)

KG-01/02 were independently verified and committed locally as `f79c0cb`.
Verification passed 78 focused tests, the full legacy suite (287 tests run, OK,
7 optional skips), all 24 ZIP-member/source/inventory checks, and isolated ZIP
import. KG-01 provides explicit CUDA availability/no-fallback behavior,
deterministic settings, and synchronized peak-allocation telemetry. KG-02
provides the allowlisted private package, exact manifest/fold hashes, inventory,
and disabled-by-default notebook/config. See
[E024](../experiments/E024_kaggle_eight_study_cuda_probe.md) for the earlier
synthetic GPU probe and resolved upload blocker, and
[E025](../experiments/E025_kaggle_eight_study_overfit.md) for the observed
bounded fit.

The separate synthetic CUDA probe passed on Kaggle with PyTorch
`2.11.0+cu128`, CUDA `12.8`, and Tesla T4 hardware: deterministic algorithms
were enabled, gradients were finite, BCE loss was `0.6535296440124512`, and
peak allocation was `18,198,528` bytes. This probe is distinct from the later
real-image study fit.

KG-03 completed on the private Version 3 package. The mounted package passed
all embedded-inventory member hashes before model import. A fresh GPU kernel
completed the predeclared fold-0/seed-42 overfit cap: eight training studies,
five epochs, 40 optimizer steps, 256-pixel inputs, and at most eight slices per
series. Mean BCE on that same bounded training subset fell from
`0.6906174942851067` to `0.48356025479733944`; runtime was `83.31482125599996`
seconds, and peak CUDA allocation was `222,188,032` bytes. Individual losses
were not monotonically decreasing. This is a short optimization smoke test, not
evidence of complete memorization, validation performance, generalization, or
clinical utility. No validation, predictions, AUC, or Gold-58 evaluation ran.

The private report and checkpoint are preserved under ignored
`data/private/kaggle-eight-study-overfit/observed-training-run01/`; their hashes
and runtime are recorded in E025. The GPU session was confirmed off. Notebook
source/results remain a draft: the attempted Save Version action was rejected
before execution because publication was not explicitly authorized; no save-
version operation ran and the prior three dataset versions remain preserved.
The parent will seek separate consent before private notebook publication.

**Historical next step at that point:** No further training/full-fold action
was authorized by the eight-study smoke test alone. The later user authorization
and first-fold preparation are recorded above; actual launch remains pending.

## Provisional local training runner prepared (2026-10-16)

TB-03 now provides a plan-only-by-default CLI and output-free notebook for the
fixed fold-bound scratch baseline. The public runner revalidates dataclass
instances so `dataclasses.replace` cannot bypass configured bounds. Bounded
overfit is train-only and reports start/end loss on the same at-most-16-study
subset. Single-fold mode uses the complete selected train/validation partitions
unless an explicit step
cap is recorded as partial. Private outputs include a state-dict checkpoint,
an ID-free report, and fold-mode predictions keyed only by partition row index.
Synthetic runner and existing training/preprocessing suites passed 73 tests in
the isolated runtime. These tests did not decode competition images or fit on
the cohort. At the time of this earlier TB-03 snapshot, the checked-in local config still
pointed to an update-required private image root and CUDA was blocked pending
KG-01/02. The later package and CUDA-probe status is recorded above. Earlier
independent verification passed 73 focused tests; see
[baseline runner details](BASELINE.md#local-training-runner-tb-03-provisional).

**Historical next step at that time:** Prepare the private Kaggle package and
validate GPU execution. KG-01/02, the synthetic GPU probe, and the bounded KG-03
fit are now complete; the current evidence and remaining authorization boundary
are recorded in the status section at the top of this document.

## Reusable trainable components prepared (2026-10-16)

TB-02 adds a fold-bound lazy study dataset, a provisional scratch study-level
CNN with mask-aware spatial pooling and equal slice/series aggregation, and
tie-correct 12-target ROC-AUC reporting. Focused synthetic CPU forward/backward,
dataset, and metric tests passed. These tests did not read competition DICOMs or
fit on the private cohort; no model-quality or generalization evidence exists.
The dataset requires the exact manifest-bound deterministic fold artifact and
fails closed on malformed inputs rather than dropping studies. See
[D025](DECISIONS.md#d025--establish-reusable-scratch-study-level-training-components)
and [baseline setup](BASELINE.md#reusable-trainable-components-provisional).

**Next step:** Complete TB-03's bounded overfit/single-fold runner and verify
its safeguards; actual fitting on competition data remains outside this work.

## Five-study Kaggle preprocessing pilot completed (2026-10-16)

The authorized CPU run completed PP-02/PP-01 preprocessing for the first five
manifest studies: 30/30 listed series succeeded, scanning 1,067 headers and
selecting 240 slices (eight per series). Every tensor was finite, shaped
`(8, 1, 256, 256)`, and within `[0, 1]`; there were zero failed series and zero
failure placeholders. The parent directly inspected the original 30-series
processed-image/mask montage and observed recognizable knees across views with
no blank panels. This is bounded observed pilot evidence, not independent live
rerun, raw-image comparison, anatomical/clinical accuracy, full-series or codec
coverage, offline compatibility, or training readiness. Series indices 9 and
10 had validity masks with 84.375% valid area and letterbox margins; the other
28 masks were fully valid all-white validity fields. These masks encode valid
pixels, not segmentation labels. See [E023](../experiments/E023_kaggle_five_study_preprocessing_pilot.md).

The UI-confirmed notebook source was saved as Quick Version 3, titled
“Five-study preprocessing pilot - observed success”; the two prior versions
were preserved. Quick Save confirmed source, not retained remote outputs, and no
additional run was made to save results. The parent stopped the session and
the UI confirmed the CPU session was off. The downloaded ID-free aggregate
report and montage are preserved under ignored
`data/private/kaggle-five-study-preprocessing/observed-run01/`. The report's
locally verified SHA-256 is
`94c944300cc3d9569fe1c85269c49d75b059211984a4afbf57f90956675b11ae`.

The Kaggle environment was CPU-only (`accelerator=None`), with internet left
enabled; this is not an offline test. No package install, training, labeling,
Gold-58 use, new-cohort traversal, or public publication occurred. Editor
recovery succeeded without security changes; the original cause of the earlier
editor failure remains unknown.

**Next step:** Review E023 and decide whether to pursue additional authorized
preprocessing validation before any separate training smoke-test decision.

## Five-study preprocessing pilot prepared (2026-10-16)

The bounded harness in `src/rsna_knee/data/preprocessing_pilot.py` and its five
synthetic tests are independently verified. Focused suites passed 5 pilot,
18 adapter and 25 PP-01 tests; full project suite passed outside the restricted
sandbox (283 tests, zero failures, five skips). The original manifest is unchanged.

The private dataset Version 2 and output-free transfer package were prepared
and independently audited before the successful follow-up pilot above. See
E023 for the final observed result and evidence limits.

## Explicit DICOM-to-PP-01 adapter (PP-02; 2026-10-16)

PP-02 now provides an explicit-path DICOM series adapter that reuses the raw
reader's patient-space ordering geometry, validates header and decoded layout,
selects at most eight uniformly spaced slices before decode, and passes native
numeric metadata and decoded arrays to PP-01. Its result reports ID-free counts,
projected slice positions, tensors, masks, and transforms. The synthetic adapter
suite and PP-01 regressions pass under bundled Python with real NumPy/Pillow
processing; pydicom is not installed, so tests inject synthetic datasets. No
competition images were accessed and codec compatibility, image quality,
actual-image policy, and training readiness remain unverified. See
[PP-02 details](BASELINE.md#pp-02-explicit-dicom-series-adapter).

**Verification:** Independent verification passed: 18 adapter tests and 25 PP-01
regressions; full project suite outside sandbox: 282 tests, zero failures, four
skips (optional dependencies/platform conditions).

**Next step:** Run a separately authorized bounded Kaggle preprocessing pilot
and visually inspect representative images before any training smoke test.

## Decoded-array preprocessing prototype (2026-10-16)

PP-01 now has a versioned local prototype for already-decoded monochrome arrays,
plus synthetic numerical coverage for sampling, per-series intensity handling,
padding, MONOCHROME1 inversion, and pixel-space letterboxing. The prototype
rejects unsafe real-numeric inputs and uses mask-weighted bilinear interpolation
at padding edges. Integer raw pixels and padding tags outside ±2^53 are rejected
before float64 conversion; boolean percentile parameters are rejected. All 25
focused synthetic tests pass under bundled Python
3.12.14 with NumPy 2.3.5 and Pillow 12.3.0. The full suite passed outside the
restricted sandbox (281 tests, three skips). This is not evidence
that the choices are appropriate for competition images: percentiles, resolution,
and slice cap remain provisional. Independent verification passed; actual-image
validation remains pending. The prototype does not integrate with the raw DICOM
reader, train a model, access images, or establish clinical validity. See
[PP-01 details](BASELINE.md#pp-01-decoded-array-preprocessing-prototype) and
[D024](DECISIONS.md#d024--keep-baseline-decoded-array-preprocessing-provisional).

**Next step:** Integrate the decoded-array adapter and separately validate the
preprocessing contract on authorized representative images before deciding
whether the baseline can advance to a training smoke test. Do not interpret the
synthetic tests as image-quality, clinical, or model-performance evidence.

## Local baseline and bounded real-image pilot (2026-10-16)

The report-free private cohort manifest is generated and independently checked:
500 ordered non-Gold studies, 6,000 binary labels, 2,784 series, and prompt-era
counts 64 legacy / 408 v1 / zero v2 / 28 v3. Five input SHA-256 values match;
no report or evidence text is exported. Three deterministic report-grouped folds
have also been generated from 488 normalized-report groups using group-OR labels
and seed 42. Writer-side audits preserve both classes for all 12 targets in every
train/validation split with zero group overlap; independent fold verification is
complete. See [baseline preparation](BASELINE.md). BP-03 adds a bounded raw
DICOM reader scaffold and synthetic tests. After the historical local-only
snapshot below was written, an authorized private Kaggle CPU pilot mounted the
competition input and decoded 240 slices from 30 series across the first five
manifest studies, with zero observed decode failures. This is parent-observed
draft-cell evidence, not an independent live rerun, full-series check, or
full-cohort readiness claim; see [E022](../experiments/E022_kaggle_five_study_dicom_reader.md).
The BP-03 focused suite passed thirteen tests; the full suite ran 280 tests with
no failures and two skips, and a synthetic notebook harness decoded one stub
slice without exposing identifiers. This is harness evidence only, not decoding
of competition images. The earlier BP-01 focused run had eight tests and one
Windows symlink skip; its full suite outside the restricted sandbox had 261 tests, no
failures, and two skips. BP-02 writer checks ran six focused tests without skips;
the full suite outside the sandbox ran 267 tests with no failures and two skips.

**Next step:** Define and validate a preprocessing/geometry policy, then
separately authorize an overfit smoke test or training. The five-study decode
does not resolve complete geometric consistency, codec coverage, full-cohort
runtime, or label quality.

## Training-readiness audit of 500 silver labels (2026-10-16)

The local aggregate audit is complete; see [E021](../experiments/E021_500_study_training_readiness_audit.md). All 500 unique records match the ordered non-Gold eligible prefix, and all 6,000 target items and 4,374 nonempty literal-source citations pass validation. This verifies structure/source integrity, not clinical label correctness or evidence relevance. The cohort is not randomly sampled and contains 64 legacy, 408 v1, zero v2 and 28 v3 records; prompt-era differences are confounded by order.

All 12 targets retain both classes across the declared three-/five-fold diagnostic configurations (seeds 17/42/123), including normalized-report-group sensitivity. Fracture is sparse: 24 positives overall, 4-5 per five-fold validation or eight per three-fold validation. Seven normalized repeated-report groups contain 19 studies; study-only allocation separates 5-6 groups across folds, while group allocation separates none. These are possible dependencies, not proven patient or MRI duplicates; patient independence is unknown. Synovitis has empty evidence for 423 of 433 negative labels, a review priority rather than demonstrated label errors.

Metadata/report-length proxies are descriptively similar to the remaining 3,849 eligible studies, but do not establish clinical, patient, scanner/site or language representativeness. No candidate image files were found under local `data`; Kaggle image availability is not negated by this local observation. The historical 50-study geometry characterization does not verify decoding/preprocessing for this cohort.

**Next step:** Resolve image access/preprocessing and label-quality/dependency checks, then define a development split outside Gold-58 and train a first baseline with learning curves. The audit supports conditional pilot feasibility, not final sample sufficiency, automatic uncertainty filtering or a blanket need for more labels. No final split, training, further inference or cohort expansion occurred. Independent recomputation verified aggregate counts and all 48 folds' arithmetic; two of 12 allocation configurations were independently replayed exactly. Ignored local HTML/JSON reports hold the full aggregate detail; they are not published predictions. Earlier sections below remain historical snapshots.

## SIWC target reached: 500/500 (2026-10-16)

The parent-monitored run reached 500 saved non-Gold records using verified v3 source `2a5e9db`. Auto028 advanced 472 to 484 and stopped on an invalid evidence reference; auto029 advanced 484 to 498 and stopped similarly; auto030 advanced 498 to 500 and exited 0. Each invocation preserved its previous checkpoint prefix. The job is completed and no owned provider process remains active. Independent audit passed: 500 unique ordered-prefix records from 4,349 eligible studies, with all 58 Gold studies excluded; all 500 records passed the actual loader and all 6,000 target evidence checks. This is operational completion, not clinical-quality validation.

Saved provenance is 64 legacy, 408 v1, zero v2 and 28 v3 records, with prompt boundaries 64/472/472. V3 resolves selected source-line IDs locally before the unchanged public schema/literal guard; it does not establish evidence relevance or label accuracy. Preserve checkpoints, journal and private capture history. No further cohort expansion or provider run is authorized by completing the 500 target. See D022/E020; sections below retain historical snapshots.

Final audit also confirmed the last 28 v3 records contain 258 nonempty exact catalogue-line excerpts and 78 empty evidence fields, with no saved `evidence_id` field. The v3 phase made 30 attempts: 28 completed and 2 evidence failures. Checkpoint/journal and 27 capture archives were hash-stable; the current capture slot is absent and the final process is inactive. Original canary/baseline hashes were not independently reverified.

## Historical SIWC stop: 472/500 (2026-10-15)

The monitored campaign is incomplete: 472 saved non-Gold records, 28 remaining, and no active provider process (parent-confirmed terminal outcome). Version 1 added 408 accepted records after the original 64. Version 2 in source `6a380bd` was attempted three times at offset 472; auto024, auto025 and auto026 each exited 2 on `evidence_not_grounded`, accepting zero additional records. Its two bounded fresh relabels are exhausted. The v2 prompt clarification did not resolve this case; no further speculative prompt change, evidence relaxation or failed-row skip is justified by these results.

Preserve the existing 64 legacy plus 408 v1 records, v1 boundary at 64, v2 boundary at 472 (zero accepted v2 rows), current private rejected-evidence capture and all prior archives. Prefix preservation is reported by parent monitoring; independent final aggregate audit is pending. Frozen Gold-58 prompts, model, labels/schema and D018/strict evidence validation remain unchanged. This is operational evidence, not clinical-quality validation. See D021/E019. The sections below retain historical implementation and run snapshots.

## Authorized SIWC literal-evidence prompt revision (2026-10-15)

The user authorized a prompt clarification and continuation toward 500 total. D020 adds the SIWC-only `c2-literal-evidence-v1` instructions before REPORT, preserving source characters and separating JSON escaping from decoded evidence. Strict validation and D018 recovery are unchanged; frozen E007/Gold-58 prompts are not modified. A safe idempotent journal boundary will identify the first new-prompt offset (planned: 64); the existing 64 records remain legacy C2. Local implementation is under verification, with independent review and live continuation pending. No provider request or target advancement is claimed for this change. The previous terminal stop below remains historical evidence. See E019.

## Latest SIWC full-cohort status (2026-10-15)

The bounded run01 checkpoint is stopped at 64/500, with 436 successful records still needed and no active provider process. After D018, auto006 advanced 24 to 26, auto007 advanced 26 to 30, and auto008 advanced 30 to 64; auto009 and auto010 made no further progress. Each invocation exited 2 on source-evidence validation, preserving prior checkpoint entries and private capture history. The two bounded fresh relabels at offset 64 are exhausted. A proposed D019 formatting recovery was abandoned and reverted before any commit or live use: independent review found an unsafe space-before-source-period case, and the actual retained fragment did not meet the proposed conservative guards. The source remains D018-only; see E018 for this negative result. Do not weaken evidence or numeric safeguards, skip the failed study, or accept unsupported clinical output to force 500. Preserve the total-500 scope, prior records, canary, private diagnostics, and provider safeguards. Runtime progress establishes neither label quality nor clinical validity.

## Fresh evidence matcher baseline (D015, extended by D017; 2026-10-15)

Fresh SIWC Responses share one boundary-aware matcher for mismatch classification and source-span recovery. It maps matches back to the earliest exact original report substring, preserving punctuation inside the quote while allowing punctuation outside it. D017 extends per-codepoint case matching to single-codepoint Unicode caseless equivalents while retaining conservative lowercase fallback for multi-codepoint expansions; exact source spans and cached strictness remain unchanged. Repaired fields are copied individually; any remaining unmatched field is still rejected by strict all-target validation, with no partial checkpoint. See D015-D017 for history and current boundaries. This offline matcher update diagnoses the captured failure class at a character level, not evidence relevance, clinical validity, or label quality; no retry or target advancement is recorded here.

## Bounded rejection-capture preparation (D016; 2026-10-15)

`label-all` now exposes the existing fixed-slot rejected-evidence capture only as an explicit opt-in with a positive total `--limit`; default behavior remains capture-off. The fixed private destination is preflighted before cohort-scope journal writes or authentication/provider construction, and the same sink is forwarded across sequential windows. Synthetic CLI tests verify the capture slot/offset payload shape, default-off behavior, missing-limit rejection, and occupied-slot refusal without auth, request, or journal activity. Those implementation checks were local only. A subsequent authorized capture-enabled attempt stopped at 15/500 and retained one fragment for local diagnosis; no label-quality result follows. The original implementation proof and the later runtime outcome are distinct.

## Current fresh matcher extension (D017; 2026-10-15)

The shared fresh-only matcher now permits single-codepoint Unicode caseless equivalents through a per-character mapping, but falls back to conservative lowercase when full case folding expands to multiple codepoints. Synthetic tests cover Greek small mu/micro sign and sigma/final sigma while retaining rejection of sharp-S-to-`ss`, accent changes, partial words, and punctuation changes. The original report substring remains the stored value; cached records stay exact-literal strict. Responses 55/55 and CLI 45/45 pass. Subscription ran 42 tests with the known Windows timeout-cleanup failure and one symlink skip; the full suite ran 241 tests with 239 passing, the same failure, and one skip. `py_compile`, `git diff --check`, independent safety review, and the parent focused check passed. A separate read-only local probe confirmed that one retained fragment maps to an exact source substring, not that any full discarded response is valid. At that point the checkpoint was still 15/500; later activity is summarized above.

## Current fresh context recovery policy (D018; 2026-10-15)

After the primary literal/case/whitespace/source-span matcher fails, fresh evidence with at least two exact lexical terms may recover only from one unique minimum source-word window in exactly one matching sentence/line segment (multiple matching segments are rejected regardless of length). The true minimum is selected before requiring at least one intervening source word; a shorter zero-gap window cannot be discarded in favor of a longer gapped window. The entire original contiguous source substring is restored, retaining all intervening context. Ambiguity, ties, word changes, reordering, partial tokens, zero-gap punctuation-only changes, or crossing `. ! ?` / CR/LF boundaries remain rejected. Cached records and final literal validation are unchanged. Responses 60/60 and CLI 45/45 pass. Subscription ran 42 tests with the known Windows timeout-cleanup failure and one symlink skip; the full suite ran 246 tests with 244 passing, the same failure, and one skip. `py_compile` and `git diff --check` pass; independent review and the parent focused check passed. This is narrow source-context extraction, not semantic inference or an evidence-quality/clinical-validity conclusion. No new provider call was made by D018; current saved checkpoint is 24/500.

## Initial SIWC run01 launch snapshot (2026-10-15; E015)

One explicitly authorized full-cohort `label-all` invocation was launched with a new isolated run01 output and checkpoint; it did not import or resume the canary. At the first safe post-launch inspection, private job metadata reported the process as running, with 3 started, 2 completed, 0 failed, and 2 checkpoint records. This is a launch-time snapshot, not a current-status guarantee. The run targets 4,349 eligible non-Gold rows with sequential windows of 14, zero retries, and checkpointing after each completed case. The prior-runtime estimate of about 29 hours 34 minutes is variable and not a completion promise; sleep or runtime termination may interrupt the managed session. Check observed job status before any separately authorized resume; do not automatically restart an in-flight row. Operational progress does not establish label quality, evidence relevance, or clinical validity. See E015.

## Latest SIWC canary status (2026-10-15)

One authorized final-case invocation after D013 exited 0 in 23.01 seconds: started 1, completed 1, failed 0. The checkpoint advanced from 13 to 14; the prior 13 records' byte-line hashes and the existing E012 capture artifact hash were unchanged. Capture opt-in was off to preserve the existing capture slot, and no safe mismatch category was emitted. This successful response does not prove that D013 recovery executed. The canary now has 14 saved records across multiple attempts and policy revisions, not 14 consecutive first-attempt successes; no label-quality, evidence-relevance, or clinical-validity conclusion follows. The full cohort was not launched in this bounded invocation. See E014.

## Previous SIWC canary status (2026-10-15; before E014)

One authorized two-case resume after D012 exited 2 in 39.70 seconds: started 2, completed 1, failed 1. The checkpoint advanced from 12 to 13, and the prior 12 records' byte-line hashes were unchanged. The existing E012 capture artifact remained unchanged; capture opt-in was off, so no new rejected evidence was retained. The failure at eligible offset 13 carried the safe category `case_only`; this static case-fold substring category does not reveal the actual quote or establish its quality. The completed case does not prove D012 recovery executed. The canary remains incomplete at 13 of 14 records, with one failed and none unattempted; no full-cohort run or retry occurred. See E013.

D012 policy and code are unchanged for the historical REP-02 run. This bounded result neither validates the E012 full response nor provides evidence that the D012 whitespace branch ran. No label-quality, evidence-relevance, or clinical-validity conclusion follows.

As of 2026-10-15, D013 extends only fresh-response token comparison to per-token Python `str.lower()` while preserving the earliest exact original source span. Focused synthetic regression tests and independent correctness/privacy verification passed locally. The full suite retains one known Windows child-cleanup failure and one unavailable directory-symlink test. REP-02 was run before D013 and does not validate this update. Cached validation and exact literal acceptance are unchanged; the canary remains 13/14 with one failed case.

## Prior SIWC canary window (2026-10-14)

After E011, one authorized bounded run attempted the nine remaining canary cases with capture opt-in. It exited 2 in 143.52 seconds: 8 started, 7 completed, 1 failed, and 1 was not attempted. The checkpoint advanced from five to twelve, with the prior five records unchanged. The failed response at eligible offset 12 emitted `whitespace_only`; one fixed local capture was verified at 143 UTF-8 bytes. A safe in-memory comparison found two normalized substring matches and two identical contiguous-token spans, so no unique source span existed and the unchanged recovery correctly rejected the ambiguity. The conditional full-cohort condition was unmet; `label-all` was not launched. The canary remains incomplete: one failed and one unattempted. This does not establish the cause of E010, generalize to uncaptured responses, or support label-quality/evidence-relevance/clinical-validity claims. The captured fragment remains private local data; manual removal is required when no longer needed. See E012; E008-E011 remain historical records.

Before the later REP-02 resume, D012 changed only the fresh whitespace-equivalent tie-break: repeated or overlapping exact token spans select the earliest original source substring. Implementation, focused synthetic regression tests, and independent safety verification had passed; the full suite retained the known Windows child-cleanup failure and one unavailable directory-symlink test. At that stage no live rerun had occurred and the E012 full response remained unvalidated; see the latest status above for the subsequent bounded resume.

- **Last updated:** 2026-10-15
- **Current phase:** Phase 1 - Dataset Understanding / EDA
- **Status:** In progress. Phase 0 remains complete against the acceptance criteria in `docs/ROADMAP.md`. One authorized constant-prediction smoke submission completed successfully, providing submission/scoring-path evidence only; no model-performance claim. Gold-58 candidate identification, the light non-Gold CSV metadata inventory, and the broadened deterministic 50-study DICOM geometry characterization are complete; Phase 1 as a whole is not.
- **Current objective:** Continue Phase 1 characterization only within separately authorized scope; the 50-study DICOM geometry characterization is complete. Any Gold-58 prompt-strategy comparison must follow the nested study-level cross-validation limits in D004.
- **Current baseline:** None.
- **Current label strategy:** Not selected.
- **Current validation:** Not defined.
- **Gold-58 policy:** D004 permits comparison of a short, predeclared report-labeling prompt-strategy set via nested study-level cross-validation. The aggregate-only feasibility screen on 2026-10-10 reported 58 studies with all 12 targets present and positive/negative representation for every target in all outer and inner folds of the configured 3x3 nested split; MCL is sparse (9 positives overall; 2 in each inner validation fold). This establishes fold feasibility, not metric precision or label validity. The user approved the exact candidate prompt text documented in E007; the local registry contains that fixed three-candidate set. The approved comparison protocol uses macro balanced accuracy as primary, mean inner-validation score for candidate selection, and a fixed simpler-candidate tie-break; macro Brier score for uncertainty versus error is descriptive only. The planned runner target is `gpt-6-luna` with `model_reasoning_effort=high` through Codex CLI and ChatGPT subscription OAuth, with no API key. The existing CLI auth method reported ChatGPT. The user separately authorized the observed comparison invocations and one later checkpoint resume through the existing OAuth route; those assertions are not independent terms or competition-rule determinations. G58-06 failed after 2 successful coordinates and 1 `response_validation` / `invalid_response`; G58-10 failed with exit status 1 and sanitized `evidence_length_invalid` after 86 completed checkpoint entries (about 13m36s). Neither G58-06 nor G58-10 produced a valid aggregate summary or comparison metrics, and no raw output was inspected. G58-12 subsequently resumed the fixed run from 86 cached successes, completed all 406 coordinates (320 new calls) with zero remaining failures, and produced a summary that passed the strict local schema validator. The selected candidates were Candidate 3, Candidate 2, and Candidate 1 for outer folds 1?3; mean outer macro balanced accuracy was 0.819117 and mean descriptive macro Brier was 0.156024. Per-target values and fold details are recorded in E007. The incompatible pre-schema checkpoint/key/failure record remain preserved in private archives; the current checkpoint/key and aggregate output remain private. Model/effort entitlement, terms, competition rules, and licensing are not independently determined here; any further invocation requires fresh explicit authorization. Outer folds evaluate the inner-selection procedure only and must not guide prompt iteration; results are not a separate independent final test. No Gold use for model training or downstream classifier model selection without separate authorization. A private metadata-only query identified 58 all-targets-present candidate rows and created a private exclusion artifact; IDs are not recorded here. Expert-label provenance/identity is not independently verified from label values.
- **Current model:** None.
- **Current best OOF:** None.
- **Current best Kaggle submission:** None; one constant-prediction smoke submission received public score 0.500. This is not a model result.
- **Active hypothesis:** None.
- **Current blockers / prerequisites:** The completed light Kaggle inventory is limited to CSV keys and documented series metadata; it is not a full corpus/scanner/site audit. Its three-study test split is too small to infer reliable distribution alignment or representativeness. The 50-study header characterization is bounded and descriptive, not a full-corpus estimate or proof of complete 3D reconstruction, acquisition protocol, pathology signal, or model suitability; one orientation-variant series remains a candidate for later targeted inspection. No additional data-access authorization is implied by completed queries. Full local schema/data-quality inspection, practical full-runtime environment, and Kaggle API/CLI access remain unverified prerequisites for later inference/deployment planning. The successful Kaggle score proves only the submission/scoring path; detailed hidden rerun logs, processed-row count, duration, and per-target outputs were not inspected. MIRA and winner-license text leave derived-artifact scope and distribution/commercial-use interpretation unresolved; seek organizer clarification before distribution or commercial deployment, but this is not a blanket blocker for private research.
- **Next phase:** Continue Phase 1 - Dataset Understanding / EDA.
- **Next recommended action:** The G58-12 nested comparison has completed under its separate, now-consumed authorization, as recorded below; do not repeat it or use outer-fold results to iterate prompts without fresh explicit authorization. Preserve the current private checkpoint/key and archived pre-schema state. The observed CLI auth method and official Codex documentation record workflow/auth evidence only, not legal or competition approval. Separately select the next Phase 1 characterization question and confirm its data-access scope before work. The geometry results describe only the deterministic 50-study sample and do not establish full-corpus behavior; targeted inspection of the orientation-variant series is a possible follow-up, not yet a selected activity. No model, label strategy, validation protocol, or performance result has been established. Seek organizer clarification before any distribution or commercial deployment of competition-derived artifacts.

The bounded Codex subscription pilot's corrected eight-request continuation completed once for each fresh offset 6–13 (all CLI and runner statuses 0; 179,750 ms total). Including the earlier offset-5 failure, the authorized nine-attempt campaign ended with one failed attempt whose transmission outcome remains unknown and eight successful requests; the failed attempt was not resent. This establishes bounded operational feasibility only, not sustained scale, label quality, or clinical accuracy. Labels remain uninspected and unvalidated; no label strategy or full-scale campaign is selected. The next labeling step, if separately authorized, is quality validation on a non-Gold development set.

Official overview/evaluation/data descriptions were checked on 2026-10-10 ([overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)). The full rules, host clarification, and submissions page were read under user-authorized browser access on 2026-10-02; the authenticated UI displayed accepted-rules status. A limited metadata-only aggregate inspection also ran locally on 2026-10-02: train and series row/ID aggregates and selected series metadata counts are recorded in `DATA.md`. Report and target values were excluded; no identifiers were emitted, and no DICOM was opened. A separate private Kaggle notebook observation dated 2026-10-10 identified 58 rows with all 12 target fields present and created a private exclusion artifact; it read no target values, reports, or image/DICOM data. This is a candidate-set identification only, not independent verification of expert-label provenance or identity. One authorized constant-prediction smoke submission completed successfully with public score 0.500; this is evidence of the submission/scoring path, not a model-performance result. Detailed private execution metadata is intentionally omitted from this public record. No raw data was downloaded; no training, publication, or final-selection change occurred. Local synthetic validation passed 9 standard-library tests. Full local schema/data quality, representative full-runtime feasibility, and Kaggle API/CLI access remain unverified; these do not negate the documented Phase 0 acceptance. No model, label strategy, validation protocol, or performance result has been established.
Official overview/evaluation/data descriptions were checked on 2026-10-10 ([overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)). The full rules, host clarification, and submissions page were read under user-authorized browser access on 2026-10-02; the authenticated UI displayed accepted-rules status. A limited metadata-only aggregate inspection also ran locally on 2026-10-02: train and series row/ID aggregates and selected series metadata counts are recorded in `DATA.md`. Report and target values were excluded; no identifiers were emitted, and no DICOM was opened. A separate private Kaggle notebook observation dated 2026-10-10 identified 58 rows with all 12 target fields present and created a private exclusion artifact; it read no target values, reports, or image/DICOM data. This is a candidate-set identification only, not independent verification of expert-label provenance or identity. The final accepted light private Kaggle metadata inventory excluded those 58 studies before training-series aggregation and recorded study/series counts and documented series metadata in `DATA.md`; that final run did not examine reports, target values, or image/DICOM payloads. The inventory is not a full corpus/scanner/site audit, and three test studies are insufficient for distribution-alignment or representativeness conclusions. One authorized constant-prediction smoke submission completed successfully with public score 0.500; this is evidence of the submission/scoring path, not a model-performance result. Detailed private execution metadata is intentionally omitted from this public record. No raw data was downloaded; no training, publication, or final-selection change occurred. Local synthetic validation passed 9 standard-library tests. Full local schema/data quality, representative full-runtime feasibility, and Kaggle API/CLI access remain unverified; these do not negate the documented Phase 0 acceptance. No model, label strategy, validation protocol, or performance result has been established.

## Local nested-validation implementation

The reusable study-level multilabel allocator is committed as `dc7cfdf` and independently verified. The nested outer/inner split API is implemented locally with strict test-first evidence; all 44 synthetic unittest tests and the usage doctest pass, and an independent audit found no technical blockers. T2 is committed locally on main as `65ae48f`; the user selected direct-main local delivery without a PR. These checks do not establish real-data fold feasibility, label quality, or model performance; no private Gold data was used. Clone-local receipt-driven development is disabled by explicit user choice; T1's earlier native review remains inconclusive and is not claimed as approved. No push or pull request has been made for this work.

## Gold-58 response-contract hardening (2026-10-11)
The failed comparison's historical `response_validation` / `invalid_response` cause is not diagnostic of the underlying response problem: the earlier validator collapsed parse and schema failures into one code. A local Codex output schema and privacy-safe granular validator reason codes are now added as prevention and future diagnosis hardening only; they do not establish the cause of the prior failure or prove that the Codex CLI accepts the schema in a live run. Schema/transport versions now bind checkpoint reuse, so the existing private partial checkpoint is incompatible with this implementation. Preserve that checkpoint and key unchanged; any future execution requires fresh explicit authorization and a separately chosen checkpoint namespace/reset strategy. No private artifacts were inspected, migrated, reset, or reused, and no provider request was made.

## G58-08 schema acceptance gate (2026-10-11)
The user authorized one schema-acceptance check using a synthetic report through the existing Codex adapter, with one fixed nested comparison only if the check passed. Preflight passed: the full required suite passed (84 tests); input and canonical-manifest aggregate validation confirmed 58 rows/keys, 12 targets, 3 outer and 9 inner folds, and 144 target-fold checks with both classes (minimum positive count 2; minimum negative count 3); all private and planned archive paths were ignored and archive destinations absent; `codex login status` reported ChatGPT OAuth without account identifiers.

The single synthetic adapter call using the first fixed candidate and `gpt-6-luna`/high returned sanitized `process_exit` / `arguments_error` (exit status 2, no errno) in about 0.2 seconds. This identifies the adapter's failure classification only; the underlying CLI argument cause is not inferred. No accepted structured result was produced. Per the explicit gate, the full comparison was not launched, the old checkpoint/key/failure record was not moved or overwritten, and no aggregate comparison result or metrics were produced. A later attempt requires fresh explicit authorization.

## G58-08 CLI error narrowing (2026-10-11)
Read-only diagnostics found the installed Codex CLI is version `0.162.1`; `codex exec --help` exits successfully and recognizes `--output-schema`. Inspection of the generated Windows `cmd.exe` command confirmed that the schema flag is present and the absolute schema path (which contains spaces) is quoted. This does not confirm an unsupported flag or an obvious schema-path quoting defect. The sanitized `arguments_error` classification does not identify its underlying cause; no retry or inference was performed.

## Windows Codex argument transport (2026-10-11)
A synthetic local regression test reproduced a Windows adapter defect: passing the `cmd.exe /d /s /c` invocation as a Python argument list caused the quoted output-schema path to split into multiple arguments in the child process. The adapter now passes a raw command-line string to `cmd.exe`, leaving its quoted fixed schema-path token intact; the report remains on stdin, and POSIX continues to use a normal argument list. A harmless fake `codex.cmd` wrapper and Python child verified argv token boundaries and stdin preservation, including spaces, ampersand, and parentheses in the synthetic path. This confirms the local adapter defect, not the exact cause of the prior live `arguments_error`; raw historical stderr was not inspected. Parent may perform one separately authorized synthetic live check after independent verification. No Gold data or comparison inference is authorized by this repair.

The separately authorized synthetic adapter acceptance call subsequently succeeded: one `codex_cli_infer` call with candidate 1 and fixed Luna/high returned a locally validated 12-target response in 10.1 seconds (exit 0). This establishes only that this synthetic response passed transport and local validation, not general output quality or model reliability. The full Gold-58 comparison remains unrun; no private checkpoint/key/failure/input/output was changed. Any future full run needs fresh authorization and a separate namespace or deliberate preserve-and-reset strategy for the now-incompatible pre-schema checkpoint. The Windows command construction has a known portability limitation: `%NAME%` expansion by `cmd.exe` is not covered; the current repository path has no percent character.

## Gold-58 G58-10 full comparison attempt (2026-10-11)

- The single separately authorized run followed the successful synthetic G58-09 acceptance check; fixed prompts, schema, candidate set, seed-42 nested 3x3 folds, metrics, model, and effort were unchanged.
- Preflight passed: full suite 85 tests; ChatGPT OAuth method confirmed without account identifiers; aggregate-only validation confirmed 58 rows and unique manifest keys, 12 targets, 3 outer/9 inner folds, 144 feasible fold-target checks, minimum per-target positive/negative totals 9/23 and minimum fold counts 2/3. No row-level values were emitted.
- The incompatible pre-schema checkpoint, key, and failure record were archived under ignored same-directory names; current-schema checkpoint/key remain private.
- The one runner invocation exited 1 after approximately 13m36s with sanitized reason `evidence_length_invalid`; safe checkpoint count was 86. No valid summary or metrics exist, and no retry occurred.
- This is a failed nested-CV execution, not an independent final test or clinical result. No outer-fold outcome was used for prompt iteration; any future run requires fresh explicit authorization.

## G58-11 evidence validation update (2026-10-11)
The user approved treating evidence longer than 200 characters as a nonfatal warning and preserving the complete string without truncation. The threshold remains an operational preference with no established empirical basis. Exact schema/target structure, boolean labels, evidence type, and finite uncertainty in [0,1] remain strict. The warning is fixed and includes no evidence, target, or identifier; it is not evidence that the label is correct or reliable. The prompt texts, output schema, checkpoint fingerprint, generation contract, model/effort, folds, and metrics are unchanged.

## G58-12 single resume authorization (2026-10-11)
The user authorized one resume from the existing checkpoint's 86 successful entries, using the same input/manifest, checkpoint/key, and fixed comparison protocol with up to 320 remaining planned calls. That authorization has been consumed; the operation's completed result is recorded below. Preserve the cached successes and private artifacts; no checkpoint/key reset or generic retry was allowed.

G58-11 implementation verification: strict-TDD synthetic boundary tests accept 200 characters silently and accept 201 with one fixed privacy-safe warning while preserving the exact evidence string; other validation remains strict. Synthetic resume integration reused 86 cached successes and made only 320 fake calls. Full suite passed (87 tests); no private checkpoint or live inference was touched. Commit `c4ef06b` records the warning behavior. Independent verification passed all 87 tests and the parent-focused check; RDD was off and risk was high (`process_boundary`), with no RDD lifecycle started. D008 records the user's warning-policy decision.

### G58-12 completed resume (2026-10-12)
The single authorized resume exited 0 after approximately 53 minutes, reused 86 successful entries, completed 320 remaining calls, and ended with 406 completed coordinates and zero failures. The aggregate summary passed the local strict schema validator; mean outer macro balanced accuracy was 0.819117 and mean descriptive macro Brier was 0.156024. Fold and per-target details are in E007. No further invocation is authorized without fresh user approval.
## Local SIWC labeling provider (2026-10-13)

A separate local Sign in with ChatGPT CLI is implemented in `scripts/siwc_label_reports.py` and documented in `docs/SIWC_LABELING_SETUP.md`. It offers explicit `signin`, account-catalog `models`, and bounded `label` commands, alongside the existing Codex CLI path. `label` transfers eligible non-Gold report text to the external Responses service; both plan-use OAuth and a separate per-run report-transfer confirmation are required. The confirmation flags are not competition, privacy, or licensing approval. No SIWC sign-in, account-model lookup, report transfer, or inference was performed; no account model entitlement or current Luna availability is established. Gold-58 report text is excluded from this ordinary labeling path; its exact 58-ID list is used only as an exclusion guard.

**Current limitation and next safe step:** no report transfer or inference is authorized by the CLI implementation or this status update. Before any SIWC labeling invocation, obtain separate explicit authorization for the specific data, external destination/service, and operation, and verify applicable privacy, competition, and licensing constraints. Continue the existing Phase 1 characterization objective; this provider implementation does not change the project phase, select a label strategy, or establish model performance.

### SIWC credential persistence update (2026-10-14)

SIWC now stores only the refresh token and required non-secret account/grant metadata. Initial ID tokens are validated then discarded; access tokens remain process-local and are refreshed on demand, including after a new CLI process starts. Legacy persisted access tokens are not trusted. Synthetic SIWC regression tests and `git diff --check` are the applicable local checks; no live OAuth request, credential access, or provider validation was performed. This storage hardening does not change the Phase 1 objective or authorize report transfer or inference.

### SIWC Candidate 2 parity correction (2026-10-14)

The SIWC Responses adapter now targets the frozen Candidate 2 prompt via the existing comparison registry; no new prompt text or output coercion is introduced. Earlier user-provided validation findings informed safe diagnostics; private pilot output is not reproduced here. A small user-run local Candidate 2 pilot has since completed; its private outputs remain local and ignored. New C2 output and diagnostics defaults are isolated under `data/outputs/siwc-c2-report-labels.jsonl` and `data/outputs/siwc-c2-pilot-diagnostics.jsonl`; explicit reuse of the former generic SIWC defaults is rejected before private input or account-profile access, leaving legacy files untouched. Safe diagnostics remain limited to allowlisted metadata and exclude report content, target names, raw values, and dynamic type names. Non-empty evidence must occur as a literal substring of its corresponding source report; invalid fresh evidence is rejected before save and cached evidence is checked before skip. A mismatch preserves the existing output. This is provenance checking only, not clinical validation. No general prompt/model checkpoint fingerprint guarantee is added. This work does not alter Phase 1 status, D006, Gold-58 exclusions, or authorization boundaries; implementation work did not access report data or credentials, or perform inference or network/API calls.

### SIWC literal-evidence diagnostics (2026-10-14)

Offline implementation classifies rejected non-empty evidence as `whitespace_only`, `case_only`, `unicode_only` (NFC canonical equivalence), or `no_literal_match`. Classification is diagnostic only: exact substring acceptance is unchanged, and `no_literal_match` does not establish hallucination. Only these fixed categories may reach the failure journal or CLI; report/evidence text remains excluded.

One separately authorized retry of the same third eligible case used the existing checkpoint and journal with no model, prompt, or authorization change. It made one attempt (started 1, completed 0, failed 1) and reported `whitespace_only`; the checkpoint remained at two records, prior records were unchanged, and the failed result was not saved. No further requests were made. This identifies the retry's mismatch category only; the original discarded response remains unclassified. No claim is made about label quality, evidence relevance, or clinical validity.

A separate bounded policy now permits fresh-response recovery only when non-whitespace tokens map to one unique contiguous report span; accepted evidence is replaced with that exact source substring before the unchanged literal guard. Ambiguous matches and non-whitespace differences remain rejected, while cached records stay strict. This does not establish evidence relevance or clinical validity and does not authorize further inference.

### SIWC local rejected-evidence capture capability (2026-10-14)

The bounded `label` CLI now has an explicit opt-in, off by default, for one fixed local capture of the first rejected fresh evidence string plus eligible-row offset. It excludes the full source report and identifiers, refuses an existing or unsafe destination, and enforces a 16 KiB UTF-8 limit without truncation. Exact acceptance and checkpoint behavior are unchanged; capture failure preserves the original validation error. Focused tests and independent privacy review passed; the full suite retains the known Windows child-cleanup failure and one directory-symlink test unavailable on this host. No live capture occurred. The fixed file is sensitive, unencrypted local data and remains until manual removal; this feature does not reconstruct the discarded E010 response or authorize further inference. See D011 and `SIWC_LABELING_SETUP.md`.

### SIWC full-cohort scaling authorization (2026-10-14)

The user has explicitly authorized three steps: build a resumable full-cohort runner, perform a fresh canary, then infer all eligible non-Gold reports. The local implementation preserves the frozen Candidate 2 prompt and `gpt-6-luna`, checks the exact Gold-58 exclusion and eligibility/count before authentication, uses fresh isolated output/checkpoint and diagnostics paths, and processes sequential windows of at most 14 records. Provider-denied and usage-limit errors stop execution without automatic retry or bypass; Gold-58 remains excluded. This implementation update does not claim that the fresh canary or full-cohort run has occurred and makes no model-quality or clinical-validity claim. The existing 14-record output remains untouched and is not resumed by this rollout.

### SIWC system certificate trust repair (2026-10-14)

A credential-free TLS diagnostic found that Requests' default certificate bundle rejected the official auth metadata endpoint while Python's system-trust path succeeded. The SIWC CLI now initializes the project dependency `truststore==0.10.5` at command startup, before it constructs authentication or Responses clients, so HTTPS uses native system trust with certificate and hostname verification still enabled. This is process-local integration; no root store, global Python configuration, or verification bypass was changed. Offline startup/order and missing-dependency regressions pass. The subsequent live metadata and model-catalog verification is recorded below; report transfer and inference were not performed as part of this repair.

### SIWC temporal claim validation repair (2026-10-14)

Offline real-Authlib regressions exposed that Authlib's default zero temporal leeway rejected future `iat`/`nbf` claims before the existing five-second application tolerance could apply. Verification now shares that bounded tolerance while explicitly rejecting missing or expired `exp` at the verifier boundary and retaining strict signature, issuer, audience, subject, and nonce validation. Refresh and model-catalog failures preserve only validated, allowlisted diagnostics. Focused auth (43), Responses (36), and CLI (31) suites pass; the full suite run before the final coverage-only auth test had one known synthetic Windows child-process timeout failure. Parent's authorized elevated verification received HTTP 200 from credential-free official auth metadata and successfully listed `gpt-6-luna` from the existing selected profile. No sign-in/profile creation or report transfer/inference occurred. The exact rejected temporal claim was not established, and this verification does not establish label quality or clinical validity.


## LD-14 additional frozen-backbone folds prepared locally (2026-10-22)

The user authorized preparing and later launching exactly one additional frozen ImageNet V1 head-only run for each existing fold 1 and 2. Fold 1 uses 337 training / 163 validation studies / 1,685 planned steps; fold 2 uses 330 / 170 / 1,650. Each uses a fresh ImageNet V1 initialization, seed/head seed 42, five epochs, Adam 0.001, CUDA, 256 px inputs and up to eight slices. No model/runner/data/fold/label behavior changed. The existing fold-0 default remains intact.

The initial fold1/2 packages (`fold1-run01-*`, `fold2-run01-*`) are preserved as superseded v1 snapshots. Before upload, review found inherited markdown falsely describing fold 0 and a two-run comparison; neither v1 package was uploaded or launched. Fresh disabled, output-free v2 packages are under ignored `data/private/resnet18-frozen-backbone/fold1-run01-preparation-v2/` and `fold2-run01-preparation-v2/`; separate upload roots and chooser ZIPs are in `fold1-run01-upload-v2/` and `fold2-run01-upload-v2/`. Each has 23 package members and 28 transport files. Outer chooser ZIPs: fold 1 43,913,417 bytes / SHA-256 `6ee6d4b5344f96977f94b891189f4ba4c894eea55f5975b440b5421478e08cd3`; fold 2 43,913,416 bytes / SHA-256 `ba5aec36b5517d641b71f51a0dca76038eee10059d81cdee33b099e409b33173`. Each chooser contains the recursive package tree, config/notebook, offline weights and base64 package archive, with no raw inner ZIP. Shared archive SHA-256 is `3946c3a545034cf31c7f03dffae85e4f85ce5e91fcd9b4decdbd0ade35328275`; inventory SHA-256 is `15d3b32974d9a79106a1d1d285432ffd391ee3034b76eb77ca953d0b11513e92`; manifest/fold pins are unchanged. The corrected v2 packages passed the full 11-module verification (132 tests in 12.847s); `git diff --check` passed with line-ending warnings only. Parent independent verification and parent-owned uploads/launches remain pending; local package creation is not training evidence.


\r\n


## HSR12 verifier correction batch (local; 2026-10-26)

V4 was paused before import or execution after independent review found non-identical duplicate metadata acceptance, broad OSError recovery, spoofable pydicom error checks, stale per-study recovery state, and a dangling-symlink classification gap. HSR12.6–12.8 added RED-first cases and narrowed each boundary. Exact full-row equality is now required for duplicate metadata; only FileNotFoundError/EOFError and exact pydicom parser exception types are recoverable; PermissionError, generic OSError, forged exception classes, symlinked paths and programming/runtime errors stay fatal. Per-index journal state is cleared before work. Unknown pydicom plugin failures remain fatal.

The named applicable 10-module suite passed 132 tests in 13.779s with no skips. Full discovery (435 tests) remains not green: 7 failures, 22 errors, 4 skips. The complete names/root evidence were not preserved because returned output was truncated; no baseline comparison was made, so the results remain unclassified. Final local candidate is `data/private/fold0-submission/run12-partial-recovery-candidate-v5-final/kaggle_fold0_submission.ipynb` (SHA-256 `361ac91914b0ee5ef85aabb47da272cb238bb4ab730974a3566b5d46b520ab21`, 107,452 bytes), with source ZIP SHA-256 `47aae9f98cd24fa1d5dd2c2e3433e01774f71e147377e34906cbafc8ef8b0462` (71,673 bytes). Generated source pin prefix, archive/payload byte identity, 20-member inventory, notebook compilation, isolated import without pydicom, and protected model source bytes passed. `RUN_INFERENCE=True` and recovery policy v1 is explicit; no Kaggle run or submission occurred. Parent verification/canary is pending. HSR05 and the original hidden failure remain unresolved/unclassified. See [HSR tracker](../odd/tasks/hidden-submission-repair.md) and [E032](../experiments/E032_hidden_submission_candidate.md).


## HSR12 private canary and one submission (parent-reported; outcome pending)

Parent executed only finalized v5-final in private [RSNA Knee Fold0 Partial Recovery Run01](https://www.kaggle.com/code/davidsulleiro/rsna-knee-fold0-partial-recovery-run01), Version 1 / execution `362631647` (Internet OFF, T4x2, pinned environment). The 48.2-second visible canary processed 3/3 studies and 15 series with recovery v1 enabled and zero skipped series/slices; CUDA runner time was 23.280744492 seconds. Root submission CSV and internal report/journal/CSV readbacks were verified. Parent made exactly one competition submission using the frozen v5-final/root CSV. The competition page currently says “Notebook Running (after deadline)” at ~41 seconds; score is pending, not a final success or error. Hidden logs are unavailable. Candidate/checkpoint/report pins are unchanged. HSR05/original hidden failure remain unresolved. Screenshot is `data/private/fold0-submission/run12-partial-recovery-candidate-v5-final/kaggle-submission-running.jpg`.

## Masked soft-training package preparation (2026-10-27)

Parent independent execution verification paused run03 before upload: default PLAN installed codecs/imported pydicom, the codec target was not exposed on the import path, decoder availability was not checked, known Kaggle mount layouts were missed, output paths were ephemeral, and the inventory misreported eligible series (23,765) as 24,035. Run03 remains preserved and must not be uploaded. Bounded RED-first correction produced fresh `data/private/masked-soft-training-preparation/run04/` (source ZIP SHA-256 `aeb8c066195c172bb919c05574203c55e4d7b491f1ad3051dfcee86d41381d49`; notebook SHA-256 `0f79fd5583c5dd7f3a98422d3665a99929b07003a26f867a4d04bc625d762429`; inventory SHA-256 `25833614d3b274956cfa5284f7c0502ccc7d63d9d6c78d87cdbcf169e0a40707`). The selected 12-module suite ran 120 tests (119 passed, one existing Windows symlink skip). Executed generated-notebook tests and an actual default metadata-only PLAN returned 2,850 train / 1,448 validation, with no pydicom/model import or codec install; local images remain absent. At package-preparation completion, no real image QC, Kaggle upload/launch, or training had occurred; the later bounded decode-only preflight is recorded below. Independent package verification of run04 passed. This does not establish real-image QC or training readiness; no broad suite was run and the five-epoch runtime remains unqualified. Run03 remains a failed, preserved candidate and must not be uploaded. The authorized private decode-only preflight then succeeded in the same private notebook Version 2 / execution `362966961`: 8/8 studies, 43 series, 344 selected slices, 28.3068 seconds decode elapsed (58.3 seconds total), CPU/None accelerator, Internet OFF, full training false, no model constructed. This bounded smoke test does not qualify the remaining eligible cohort, clinical image quality, or five-epoch training runtime. See [RDP tracker](../odd/tasks/private-masked-decode-preflight.md) and [E037](../experiments/E037_private_masked_decode_preflight.md). See [UMP tracker](../odd/tasks/unweighted-masked-preparation.md), [E036](../experiments/E036_unweighted_masked_training_package.md), and D030.
