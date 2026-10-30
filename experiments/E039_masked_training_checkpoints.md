# E039 — Resumable masked-soft baseline training

## Hypothesis and baseline

Atomic per-epoch checkpoints can make the existing five-epoch masked-soft baseline safely continuable across bounded Kaggle executions without changing the scientific protocol. E038 measured a sampled five-epoch projection of 12.98 hours raw and 19.56 hours after its 1.5× safety multiplier plus fixed overhead; that pilot was a cost-only estimate, not a quality result or full-cohort decode proof. The user separately authorized full baseline training with checkpoint-based continuation.

The frozen baseline is fold 0, 2,850 training studies and 1,448 validation studies from the pinned 4,298-eligible-study non-Gold manifest; frozen ImageNet1K V1 ResNet-18 trunk, seeded study head, Adam at 0.001, seed 42, unweighted masked BCE, 256-pixel inputs and at most eight selected slices per series. No model selection, labels, fold membership, cohort, or objective changes are in scope.

## Predeclared checkpoint protocol

- Persist model and Adam optimizer state, completed epoch history, and Python, NumPy, Torch CPU and CUDA RNG state only after a full training epoch and validation pass.
- Use a weights-only-safe, versioned schema and same-directory temporary file plus atomic rename; verify checkpoint readback and hash. Retain every completed epoch checkpoint, so a subsequent interruption cannot destroy the last valid epoch.
- Validate an explicitly supplied checkpoint against the full five-epoch protocol, metadata/fold/weights pins, model/preprocessing/source hashes, device and runtime versions before dataset/model construction and before optimizer updates. Reject missing, corrupt, incompatible, or unsupported checkpoints; never silently restart.
- Require the explicit expected checkpoint SHA-256 beside the resume path and verify it before `torch.load`. Fingerprint all packaged Python sources by relative module path and bind `max_steps`; exclude filesystem roots and per-execution controls.
- Validate model and Adam tensor finiteness/hyperparameters before continuing. CPU resumes require empty CUDA RNG state; CUDA resumes require one valid RNG tensor per visible device, validated before any global RNG mutation.
- Resume means continue toward five total epochs. Do not overwrite the mounted source checkpoint; each execution has a fresh output directory.
- Bound Kaggle executions to two completed epochs and an eight-hour elapsed budget. Check the budget at epoch boundaries using 1.5× the previous epoch duration as a reserve. The first epoch is allowed when no prior duration exists. Emit aggregate-only epoch/checkpoint progress.
- Keep the normal generated notebook in plan-only mode; the separately named authorized launch variant runs full training only after successful decoder preflight. It starts with `RESUME_CHECKPOINT=None` and a fresh `OUTPUT_RUN_NAME`.

## Verification protocol and evidence

Synthetic CPU proof compares uninterrupted five-epoch training against a two-epoch segment plus explicit three-epoch continuation, including model parameters, Adam state, full history, NumPy RNG and Torch RNG. It verifies incompatible/source-mutated/max-step checkpoints fail before factories, expected-SHA mismatch prevents load, nonfinite Adam moments fail before optimizer step, CUDA RNG mismatch causes no RNG mutation, and an epoch-two exception retains epoch one's checkpoint. Aggregate-only training-start and every-100-study progress events are checked for counts and identifier redaction. Package checks verify default plan-only mode and the separate bounded launch variant with explicit resume path and digest fields.

Observed focused command:

```text
$env:PYTHONPATH='src'; .venv-training/Scripts/python.exe -m unittest tests.test_masked_soft_training tests.test_kaggle_masked_training_package -v
Ran 39 tests ... OK
```

The repo's `.venv` lacks Torch/NumPy; `.venv-training` supplied Torch 2.14.1+cpu and NumPy 2.4.6, so it was used for numerical equivalence tests without installation. The run is synthetic/local CPU only: no real DICOMs, competition images, ImageNet weights deserialization, CUDA training, or Kaggle execution occurred. The eight-hour policy is a guardrail, not evidence each epoch fits; the parent must inspect each actual segment and explicitly pass the checkpoint into the next fresh execution.

## Generated artifacts

- Current standard package run09: `data/private/masked-soft-training-preparation/run09/`.
- Current authorized full-training variant: `data/private/masked-checkpoint-training/run03/kaggle_masked_soft_training_authorized.ipynb`.
- Both builds have the same 27-member source ZIP SHA-256 `af5aadb3be99ae92e497b0ea85f66ee9ed25bdaa1b510b8861f3303700924303`; plan notebook SHA-256 `464dfc6e4a77bffbe115c362873386c42a93e0de996859ec6c080e7af2580149`; authorized launch notebook SHA-256 `8b539568b1206cfee0c61a7f485d5422b4b60eb5f81a087758f58dd6e2f111ab`. Generated-notebook tests exercise both the first segment and an explicit resume handoff with a fresh run label.
- Local implementation/tests did not train real DICOMs or run CUDA. The parent owns remote Kaggle operations and checkpoint continuation.

## Parent-observed remote launch — first segment still running

The parent uploaded only the three changed code/metadata assets from run09 to the existing private dataset, created Dataset Version 4 with seven total files, pinned that version explicitly, and imported the authorized launch notebook run03 into the existing private notebook. No public sharing, paid resources, or extra data assets were used.

- Private notebook Version 6, execution `363001906`, titled “Full cohort epochs 1-2 of 5 with checkpoints”: [open the private execution log](https://www.kaggle.com/code/davidsulleiro/rsna-knee-masked-decode-preflight-run01/log?scriptVersionId=363001906).
- Environment: T4x2, Internet OFF, original runtime environment; inputs were explicitly pinned Dataset Version 4 and the official competition image input.
- Dataset inventory SHA-256: `0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3`. Uploaded source ZIP SHA-256 matches run09: `af5aadb3be99ae92e497b0ea85f66ee9ed25bdaa1b510b8861f3303700924303`.
- Compared with the already verified run06 package, the run09 27-member source ZIP changes only `rsna_knee/training/masked_soft.py`; cohort manifest/folds, ResNet weights, codec bundle and their pins were preserved.
- The 59.1-second UI observation showed `training_started`: 0 completed, target 5 epochs, two-epoch execution cap, 28,800-second budget, 2,850 training and 1,448 validation studies/steps. At 431.6 seconds, logs showed epoch 1 training progress 100/2,850; at 487.3 seconds, the UI still showed Running.
- No error was observed in those snapshots, but no completed epoch or checkpoint file/hash/readback is confirmed yet. Do not claim a completed epoch, checkpoint, full model, or quality result. The first-segment launch action is done; remote training remains pending.
- Parent-provided evidence is retained under `data/private/masked-checkpoint-training/run03/`: `kaggle-version6-live-logs.txt`, `kaggle-version6-running.json`, `kaggle-version6-running.png`, `kaggle-training-prelaunch.png`, and `kaggle-dataset-v4-created.png`.
- Independent verification passed the 39-test suite, including exact toy CPU equivalence for uninterrupted five epochs vs two epochs plus three-epoch continuation with all model/Adam/history/Torch CPU/NumPy RNG state. This is not GPU resume proof or a model-quality claim.

## First-segment terminal result and authorized continuation

Parent subsequently verified Version 6 / execution `363001906` completed successfully in 16,984.2 seconds (about 4 h 43 min), with 81.71 MB of published output. It completed **one** full train-plus-validation epoch, checkpointed it with readback, and returned `paused_checkpoint_available`. It did not fail and did not complete five epochs. The eight-hour boundary policy correctly refused another epoch: the measured epoch took 16,912.17 seconds, so the remaining budget could not accommodate the 1.5× reserve for another epoch.

- Saved checkpoint: `checkpoint-epoch-01.pt`; SHA-256 `cf3a2d1c85f1ee8a85bdd4b636cbb8a586647d53abef5859ea55f7956158dff6`.
- Epoch-one train masked BCE: 0.6355818985040186 on 25,740 observed cells.
- Validation masked BCE: 0.594582709055161; Brier: 0.13963097877915392 on 13,195 observed cells from 1,448 studies.
- Per-target validation masked BCE (observed cells): ACL 0.541756 (1,347); MCL 0.475370 (1,334); medial meniscus 0.700272 (1,379); lateral meniscus 0.563421 (1,321); medial OA 0.619279 (1,182); lateral OA 0.591897 (1,101); PF OA 0.720789 (1,217); effusion 0.645460 (1,335); synovitis 0.541959 (243); Baker's 0.627688 (753); contusion 0.616899 (1,130); fracture 0.407595 (853). Cross-target losses are not a fair quality ranking because target prevalence and soft-label uncertainty differ.
- Runtime reported Python 3.13.15, Torch 2.11.0+cu128, NumPy 2.1.3 and CUDA 12.8. Resume must retain that runtime and T4x2 device topology as well as the original source/data/model pins; fingerprint validation rejects incompatible replacements.
- One epoch alone establishes neither improvement over the previous baseline nor adequate discrimination. No AUC or independent clinical ground-truth performance was reported by this run.

The user explicitly authorized launching the next segment after inspection. Parent owns the browser and exact prior-output mounting. Local preparation uses a fresh `run02` output and explicit path plus SHA, continues toward five total epochs, and keeps the two-epoch cap/eight-hour boundary policy. Based on the first actual epoch cost, another one-epoch segment is plausible; the cap is not a promise of two completed epochs.

## Prepared continuation candidate

Parent observed the successful notebook's exact input mount and output tree, and created a new **private** fork, RSNA Knee Masked Resume Run02. The fork retains the two original inputs plus the prior Version 6 notebook output, T4x2, Internet OFF and the original 2026-10-22 environment. No training launch is yet claimed here.

The continuation candidate is `data/private/masked-checkpoint-training/run04/kaggle_masked_soft_training_resume_authorized.ipynb` (14,289 bytes; SHA-256 `615904820387f6a82301896fe0e8009c8c4748581fb49bf3e7331ed171487070`). Compared with the authorized first-segment notebook, exact raw-byte replacement changes only three constants: `OUTPUT_RUN_NAME="run02"`, `RESUME_CHECKPOINT` and `RESUME_CHECKPOINT_SHA256`. All other bytes, metadata, source and pins are preserved. The explicit mounted checkpoint is:

```text
/kaggle/input/notebooks/davidsulleiro/rsna-knee-masked-decode-preflight-run01/data/private/masked-soft-training/run01/checkpoint-epoch-01.pt
```

Its required digest remains `cf3a2d1c85f1ee8a85bdd4b636cbb8a586647d53abef5859ea55f7956158dff6`. The notebook compiles. A synthetic directory-layout check confirmed prepared-asset discovery ignores the observed nested `/notebooks/owner/title` mount even if it contains an inventory marker, so it cannot compete with the explicitly pinned prepared dataset under the bounded discovery rule. The focused 39-test command was rerun and passed in 28.004 seconds. No real checkpoint has been deserialized locally and no production code changed.

## Parent-observed live continuation — epoch two pending

Parent imported the prepared run04 notebook into the new **private** RSNA Knee Masked Resume Run02 fork and launched Version 1 / execution `363139918`: [open the continuation](https://www.kaggle.com/code/davidsulleiro/rsna-knee-masked-resume-run02?scriptVersionId=363139918). Dataset Version 4 was explicitly verified by its selected version radio. T4x2, Internet OFF and the original 2026-10-22 environment were preserved; the prior Version 6 notebook output is the third input.

At 54.8 seconds the live log emitted `training_started` with `completed_epochs=1`, target 5, 2,850 training / 1,448 validation studies, two-epoch execution cap and 28,800-second budget. At 101.1 seconds the UI showed Running. This verifies the expected checkpoint digest, compatibility, weights-only load, model/Adam and RNG restoration succeeded in the live GPU path: this is continuation, not a fresh restart. It does not establish epoch-two completion, a newly written checkpoint, or training-quality improvement.

Evidence: `data/private/masked-checkpoint-training/run04/resume-live-logs.txt` and `kaggle-resume-running.png`. The launch action is complete; training remains active. No error was observed in those snapshots, and no automatic monitor or further continuation is configured.

## Second-segment terminal result — three of five epochs complete

Parent subsequently verified private RSNA Knee Masked Resume Run02 Version 1 / execution `363139918` reached terminal success in 23,190.1 seconds (about 6 h 26 min 30 s). It completed epochs two and three, preserving the first epoch's history, and paused cleanly with `completed_epochs=3`, `target_epochs=5` and `paused_checkpoint_available`.

- Epoch two duration: 11,405.38034 seconds; checkpoint SHA-256 `9fc37cf9c6ba1e6d505ec471f78d7cfb77db0ef68f08c4042f3a7674b700160d`.
- Epoch three duration: 11,714.59904 seconds; latest `checkpoint-epoch-03.pt` SHA-256 `1bea3c357b7283e0eb831d247e077f9d48f2444520900c80252c736364c3eafe`, under `data/private/masked-soft-training/run02/`.
- Train masked BCE at epochs one/two/three: 0.6355818985040186 / 0.6055127162280801 / 0.5929830039731967.
- Validation masked BCE: 0.594582709055161 / 0.5847537230043847 / 0.5787277466857149 (2.67% lower at epoch three versus epoch one).
- Validation Brier: 0.13963097877915392 / 0.13514776351427632 / 0.1323149686105554 (5.24% lower), on the same 13,195 observed cells from 1,448 studies.
- Eleven of twelve target BCE values improved versus epoch one; synovitis was slightly worse (0.5419595 to 0.5428570). Preserve this exception rather than reporting only favorable aggregates.

This is evidence of improved fit to the fixed soft-report-label validation targets, not a clinical ground-truth evaluation, AUC result or proof of competitive generalization. The user explicitly authorized the final segment for epochs four and five. It must resume the verified epoch-three checkpoint with a fresh `run03` output and preserve all prior source/data/model/runtime pins, five total target epochs, two-epoch cap and 28,800-second boundary budget. No source or strategy change is authorized.

## Prepared final-segment candidate

Parent observed the exact input root `/kaggle/input/notebooks/davidsulleiro/rsna-knee-masked-resume-run02` and created the private RSNA Knee Masked Resume Run03 fork. The final candidate is `data/private/masked-checkpoint-training/run05/kaggle_masked_soft_training_final_authorized.ipynb`, 14,279 bytes, SHA-256 `61790c8d51cf392c2befdc8a5abcd257750d93ed685b0002956ce4831c166671`.

Relative to run04, exact raw-byte replacement changes only `OUTPUT_RUN_NAME` to `run03`, `RESUME_CHECKPOINT` to the observed mounted epoch-three checkpoint below, and its expected SHA-256 to `1bea3c357b7283e0eb831d247e077f9d48f2444520900c80252c736364c3eafe`. All other notebook bytes, metadata, logic and source/data/model pins are unchanged. Compile passed; the targeted package suite (`tests.test_kaggle_masked_training_package`) passed 18 tests in 7.049 seconds. No new production behavior or source change required TDD; previous strict-TDD implementation and continuation proofs remain applicable.

```text
/kaggle/input/notebooks/davidsulleiro/rsna-knee-masked-resume-run02/data/private/masked-soft-training/run02/checkpoint-epoch-03.pt
```

The final target remains five total epochs, with cap two and the 28,800-second boundary budget. Parent verified inherited T4x2, Internet OFF and original 2026-10-22 environment; the visible weekly quota was 15:02/30h before launch. The bounded prepared-asset discovery does not traverse nested notebook-output mounts, so the inherited previous outputs cannot create prepared-input ambiguity. Parent owns input cleanup, final dataset version check and remote launch. No final-segment execution is yet claimed.

## Parent-observed final-segment launch — epochs four/five pending

Parent launched **private** RSNA Knee Masked Resume Run03 Version 1 / execution `363245997`: [open the final segment](https://www.kaggle.com/code/davidsulleiro/rsna-knee-masked-resume-run03?scriptVersionId=363245997). The new fork has exactly three inputs: official competition images, RSNA Knee Masked Soft Inputs Run04 explicitly pinned to Dataset Version 4 (selected version radio verified), and the successful Resume Run02 output. The obsolete first-run output was removed only from this new fork. Private sharing, T4x2, Internet OFF and the original 2026-10-22 environment were verified.

At 64.5 seconds, `training_started` reported `completed_epochs=3`, target 5, 2,850 training / 1,448 validation studies, two-epoch cap and 28,800-second budget. At 90.0 seconds, the UI showed Running. This confirms checkpoint SHA verification, compatibility, model/Adam/RNG restoration succeeded; it did not restart from zero. No completed epoch four/five, new checkpoint or new validation metric is yet confirmed. Parent independently spot-checked the 18 package tests, which passed in 5.811 seconds.

Evidence is retained in `data/private/masked-checkpoint-training/run05/kaggle-final-segment-running.png` and `final-segment-live-logs.txt`. The final-segment launch action is complete; five-epoch training completion remains pending. No automatic watch or additional run has been configured.

## Run03 terminal result — one remaining epoch

Parent verified private RSNA Knee Masked Resume Run03 Version 1 / execution `363245997` completed successfully in 16,772.3 seconds (4 h 39 min 32 s), but completed **epoch four only**, then returned `paused_checkpoint_available` with four of five total epochs. The runtime boundary guard stopped cleanly; the earlier intended final segment did not complete epoch five.

The saved epoch-four checkpoint is `data/private/masked-soft-training/run03/checkpoint-epoch-04.pt`, SHA-256 `1ec0942869156e1bc991de6e020ccbc1a4b7152c631916d41da8d137cf7ad5e2`. Epoch-four train masked BCE was 0.5847246087022363; validation masked BCE 0.5748012696123566 and Brier 0.13038617525936208, compared with epoch-three validation BCE 0.5787277466857149 and Brier 0.1323149686105554. These are fixed soft-report-label fit metrics, not AUC or independent clinical evaluation.

The user explicitly authorized the remaining epoch-five continuation. It must resume this exact epoch-four checkpoint with fresh output `run04`, five total target epochs, unchanged cap two / 28,800-second boundary budget, and preserved source/data/model/runtime. Parent owns observed mount-path verification and private Kaggle launch; local notebook preparation is pending that path. No new source behavior is required.

## Prepared remaining-epoch candidate

Parent observed the exact prior-output root `/kaggle/input/notebooks/davidsulleiro/rsna-knee-masked-resume-run03` and created the private RSNA Knee Masked Resume Run04 fork. The remaining-epoch candidate is `data/private/masked-checkpoint-training/run06/kaggle_masked_soft_training_epoch5_authorized.ipynb`, 14,279 bytes, SHA-256 `a022036977a643ea6f17e34def9adf73296527d6593d3e303f8dc8437bee6aad`.

Exact raw-byte comparison with run05 proves only three constants changed: fresh output `run04`, the explicit epoch-four checkpoint path below and matching expected digest `1ec0942869156e1bc991de6e020ccbc1a4b7152c631916d41da8d137cf7ad5e2`. All other notebook bytes, metadata, logic and pins remain unchanged. Compile passed; the targeted package command (`tests.test_kaggle_masked_training_package`) passed 18 tests in 7.169 seconds. No production source change or real checkpoint deserialization occurred locally.

```text
/kaggle/input/notebooks/davidsulleiro/rsna-knee-masked-resume-run03/data/private/masked-soft-training/run03/checkpoint-epoch-04.pt
```

The inherited environment remains T4x2, Internet OFF and original 2026-10-22. Parent observed weekly quota 19:42/30h before launch, leaving about 10h18 for the eight-hour bounded run. The total target stays five; with four completed, only one epoch remains regardless of the unchanged two-epoch cap. Parent owns obsolete-input removal only from the new fork, Dataset Version 4 verification and remote launch. No remaining-epoch execution is yet claimed.

## Parent-observed remaining-epoch launch — completion pending

Parent launched **private** RSNA Knee Masked Resume Run04 Version 1 / execution `363315909`: [open epoch-five execution](https://www.kaggle.com/code/davidsulleiro/rsna-knee-masked-resume-run04?scriptVersionId=363315909). Exactly three inputs are mounted: official competition images, private Dataset Version 4 explicitly verified by selected version radio, and successful Resume Run03 output. Obsolete Run02 output was removed only from the new fork. Private sharing, T4x2, the pinned 2026-10-22 environment and Internet OFF were verified.

At 63.7 seconds, `training_started` reported restored `completed_epochs=4`, target 5, 2,850 training / 1,448 validation studies, cap two and 28,800-second budget. The UI was Running at about 80 seconds with no observed error. This confirms the checkpoint digest, compatibility, model/Adam and RNG restoration succeeded; training did not restart from zero. It does not establish completed epoch five, a final model, a new checkpoint or new validation metrics.

Parent spot-checked the targeted package suite: 18 tests passed in 6.181 seconds. Evidence: `data/private/masked-checkpoint-training/run06/kaggle-epoch5-running.png` and `epoch5-live-logs.txt`. Task eight closes the launch action only. No automatic watch, duplicate run or additional epoch is configured.

## Five-epoch terminal completion (parent-verified 2026-10-29)

Private Resume Run04 Version 1 / execution `363315909` has since completed all five target epochs. Its final `training-report.json` reports `status=complete`, `completed_epochs=5`, and `training_completed_no_quality_claim`. Latest `checkpoint-epoch-05.pt` SHA-256 is `f1fb2d250eb601cfb5f03e92e01d2cd020cb5ba6be9fd345576c160d38771d7b`. Parent observed the numbered checkpoint, final `model-state.pt` and `training-report.json` in the actual notebook output. The copied final report has SHA-256 `ff8f77a2f4d5507e6e6c76f0830abf50d87f45c52930c234e8d46ee1154ec2f1`. Final validation masked BCE is 0.5721198783180238 and Brier 0.12899160491401088. Full training is complete; a ranking/clinical quality claim does not follow automatically. The user authorized constant comparison and a separately pinned submission, recorded in E040.

## Next step

Follow E040 for checked constant-baseline comparison, new-model inference validation and the parent-owned authorized submission. Preserve all segment checkpoint/history evidence; no additional training or automatic run is required.
