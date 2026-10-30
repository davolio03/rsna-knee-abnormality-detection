# E027: Training learning diagnostics

## Purpose

Record the LD-01 reporting change that makes optimization progress inspectable without adding data access or evaluation work. This is instrumentation only; it is not a training experiment and contains no study-level or image-derived records.

## Report contract

Both overfit and fold reports now include `training_loss_by_epoch`. Each emitted epoch entry contains its one-based epoch number, planned and completed study counts, completed optimization-step count, a `complete` or `partial` status, and the finite mean of online per-study BCE values measured immediately before each optimizer update.

The mean is an optimization-step aggregate, not a fixed-checkpoint evaluation loss. In overfit mode, the existing `train_loss_start` and `train_loss_end` continue to represent fixed-subset evaluation losses before and after fitting. Fold mode adds no training-set pass; its existing final validation pass and metrics remain unchanged. Fold reports do not expose per-step loss arrays.

## Verification evidence

- Synthetic two-epoch overfit and fold cases assert epoch order, planned/completed study and step counts, finite aggregates, existing overfit fields, and absence of fold per-step loss arrays.
- A three-step cap over a two-study dataset asserts a complete first epoch and an explicitly partial second epoch.
- Focused training and preprocessing suite: 104 tests passed (see task tracker for the exact command and observed TDD RED result).

## Interpretation limits

The online mean mixes losses computed at different model states within each epoch; it is not directly comparable to the fixed-subset checkpoint evaluation loss. These synthetic checks validate reporting behavior only and make no claim about real-data learning or generalization.

## LD-02: Eight-study sanity protocol preparation (2026-10-20)

A separate disabled-default package/config/notebook is prepared under ignored `data/private/training-learning-diagnostics/sanity-run01-preparation/`. It keeps the historical fold 0, seed 42, ordered eight-study training subset, scratch CNN (width 64), 256-pixel inputs, eight-slice cap, Adam at 0.001, and sets the bounded ceiling to 30 epochs / 240 steps. The predeclared same-subset BCE target is `<= 0.10`; it is a diagnostic criterion, not a guaranteed result or evidence of validation/generalization. No run occurred.

The package inventory pins every source member, the actual baseline source, manifest `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`, folds `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`, and package SHA-256 `d148146df4ed703b852f441ad8f09b4944674bd85c546b6ec864f285aab536be`. Inventory SHA-256 is `8a714d69e8300cc40ac9c785c36d0fa48b6721dfb747a3eac369fa5954363f14`; baseline source SHA-256 is `4b94283f6854e6175324cbdc120dcb160da670ea61e7d1442fe8dae299ed7228`. Prepared config and notebook remain disabled; notebook outputs are empty, it validates those pins, and it prints only aggregate summaries.

The final overfit evaluation pass reports per-target positive/negative counts, nullable AUC, and probability mean, standard deviation, minimum, and maximum across the training subset. One-class targets explicitly receive `auc: null`; these values are computed during the existing final subset pass, with no extra image decoding. The original v1 isolated-package synthetic probe exceeded a 60-second subprocess timeout and produced no result; its cause was not established and this historical failure is retained below.

Cost forecast only: the prior 40-step fit took 83.315 seconds, so a linear estimate for 240 steps is about 500 seconds (8 minutes 20 seconds). The dataset decodes studies on demand with no cross-step cache, so runtime is uncertain; no automatic retry or execution is implied. The successful run04 full-fold terminal outcome (execution `359470765`, 1,665 steps, 333/167 train/validation, macro AUC `0.5203435261555952`) is recorded in the current project-state summary; no local report copy was verified.

### Final v2 independent verification

The corrected v2 package passed independent re-verification: 107 focused tests, all 22 exact inventory/source-member hashes, unchanged manifest/fold hashes, disabled/output-free notebook, and an actual ZIP-imported two-step synthetic CPU run in 1.922 seconds. A constant-across-studies prediction probe confirms that all 12 per-target standard deviations are zero even when target means differ. The parent separately passed 24 runner/package tests and `git diff --check`. These checks establish local behavior and package wiring only; the real eight-study / 240-step experiment remains unrun. The historical v1 60-second timeout has no established root cause and is not represented as fixed. Commits remain pending because Git metadata is managed read-only.

## LD-02 correction: per-target dispersion and preparation v2 (2026-10-20)

Independent verification identified that v1's pooled `probability_std` mixed target and study variation, so it could be nonzero when each target's prediction was identical across all studies. The ambiguous pooled field was removed. Each per-target record now reports mean/std/min/max across studies. A regression uses identical per-study predictions but different logits between targets; it requires zero standard deviation for every target while preserving distinct target means. The test first failed with `KeyError: 'probability_std'`, then passed after the per-target implementation. The existing final evaluation pass is unchanged; no extra evaluation pass was added.

The v1 private preparation remains unchanged as historical evidence. The canonical prepared bundle is now the fresh ignored `data/private/training-learning-diagnostics/sanity-run01-preparation-v2/`: package SHA-256 `71937e7fcc51fee411a8243f3e17a25774f873b1b932906cf26faf8a0d309a26`, inventory SHA-256 `baca495c0054d2653ee94fbd4244c2dd77988f36b0ef8e711a6e7397e12ec51c`, and corrected baseline source SHA-256 `a75f3ea7a08ce1d8a9b7df277b011662fc24c6536b35ae046fe3caf853274931`. Manifest/folds remain pinned to `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`; inventory contains 22 members. Focused suite: 107 tests passed. Independent parent verification also passed exact member/inventory/source/config/hash and disabled-notebook checks, plus a 2-step CPU ZIP wiring smoke using eight synthetic 8px rows (1.968 sec with one Torch thread). That smoke validates package wiring only, not 240-step performance or learning. The v1 60-second timeout remains unexplained; no claim is made that its cause was resolved.

## LD-02 pre-launch transport correction (2026-10-20)

The parent uploaded v2 as Kaggle dataset Version 7, but the UI showed 46 files: the intended 22-file `package/` tree, the inner archive's 22 recursively extracted members under a second directory, inventory and config. Because Kaggle removed the uploaded raw ZIP after recursive extraction, the prepared notebook's `ARCHIVE_PATH.read_bytes()` would fail. Training was not launched; the one authorized GPU-run attempt remains unconsumed.

Builder transport is corrected without changing the 22-member package or training source/protocol: the intended upload contains exactly the package tree (22 files), `inventory.json`, config JSON, and an ASCII `.zip.b64` archive file (25 files total). The notebook strict-decodes with `base64.b64decode(..., validate=True)`, verifies the decoded ZIP SHA-256 against the unchanged package provenance, and inspects members through `ZipFile(BytesIO(...))`. The fresh ignored preparation is `data/private/training-learning-diagnostics/sanity-run01-preparation-v3/`; the exact 25-file transport copy for upload is `data/private/training-learning-diagnostics/sanity-run01-upload-v2/`. Package SHA-256 `71937e7fcc51fee411a8243f3e17a25774f873b1b932906cf26faf8a0d309a26`; inventory SHA-256 `baca495c0054d2653ee94fbd4244c2dd77988f36b0ef8e711a6e7397e12ec51c`; source SHA-256 remains `a75f3ea7a08ce1d8a9b7df277b011662fc24c6536b35ae046fe3caf853274931`; manifest/folds remain `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`. Encoded transport file SHA-256 `9bd51ba0763aeec30645b1040986a1d527b07d119b9dc0d51d2b2e8c1fb3cb1f`. The strict-transport regression first failed with missing `transport_dir`, then passed; focused suite now passes 108 tests. Parent uploaded Version 8 and used its one authorized execution; the actual notebook path failure and its local correction are recorded in the following section.

## LD-02 notebook dataset mount correction after authorized launch (2026-10-20)

The parent used the single authorized Save & Run All against private dataset Version 8, which contained the corrected 25-file transport. The execution terminated after 16.8 seconds with no output artifacts; logs showed `FileNotFoundError` reading `inventory.json` at notebook line 28, before baseline import, image access, or model training. The UI-confirmed dataset copy path was `/kaggle/input/rsna-knee-five-study-reader-private`; the generated notebook incorrectly used `/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private`. The competition image path was verified correct. No metrics or checkpoint exist, LD-03's diagnostic objective is not complete, and that one-run authorization is consumed. No retry is authorized.

Strict TDD: the upload regression first failed because notebook text did not contain the UI-confirmed dataset mount; it passed after changing only the notebook `INPUT_ROOT`. Fresh disabled, output-free preparation: `data/private/training-learning-diagnostics/sanity-run01-preparation-v4/`, notebook SHA-256 `c228ce5bf44f83a68d5027cb58f53e79bab6a3f72e1e8fce1d8ecc02d9255180`. Package SHA-256 stays `71937e7fcc51fee411a8243f3e17a25774f873b1b932906cf26faf8a0d309a26`, inventory `baca495c0054d2653ee94fbd4244c2dd77988f36b0ef8e711a6e7397e12ec51c`, baseline source `a75f3ea7a08ce1d8a9b7df277b011662fc24c6536b35ae046fe3caf853274931`; manifest/folds remain unchanged. The full focused suite passed 108 tests. This was local-only: no dataset upload, GPU, or second execution occurred. At that point, a further run required new authorization; the user has since authorized one controlled v4 retry, as recorded below.

## v4 retry authorization was initially blocked by Kaggle editor loading (2026-10-20)

The user explicitly authorized exactly one controlled v4 retry using the existing private notebook/session and Version 8 dataset. The retry is authorized but has not begun. The parent could not prepare the editor: the existing run editor remained at “Loading Editor” with notebook cells, imports, and Run All disabled. Reload, reopening the known edit draft, and a fresh same-browser tab with bounded waits did not recover it. No remote code was edited, no upload was made, and Save & Run All was not clicked. The new authorization remains unconsumed; do not ask for it again. Resume only after the editor is usable, then follow the existing one-run scope. LD-03 remains incomplete with no metrics/checkpoint. Screenshot evidence is saved at `data/private/training-learning-diagnostics/sanity-run01-upload-v2/kaggle-retry-editor-loading.png`.

## Initial observation of the v4 retry (2026-10-20; later completed)

After the editor recovered, the parent verified the v4 notebook code readback (8,034 characters; SHA-256 `ed73902f5d2e662d4d38e8d9c44bfdd7abdae008f8c15dc37b14d23afe38c6ab`) against local notebook SHA-256 `c228ce5bf44f83a68d5027cb58f53e79bab6a3f72e1e8fce1d8ecc02d9255180`. The private Version 8 dataset and notebook input radio controls were checked; hardware was T4 x2, Internet off, and session draft off. The parent saved a new notebook version and clicked Save & Run All once, then saved once. The one retry authorization was consumed. The initial viewer state for execution `360568896` was Running at 27.0 seconds with 0 B output; the parent continued monitoring. That was only the initial observation; the final outcome is recorded in the following section. No retry occurred.

## Corrected v4 diagnostic result (2026-10-21)

The single authorized retry completed successfully (Version 2 execution `360568896`) in 357.8 seconds with 118.18 kB of output. The notebook-measured training interval was 323.731 seconds. It completed 240 optimization steps over 30 complete epochs on the same bounded eight-study training subset. No validation/generalization set was involved in the reported final loss or target summaries.

| Measure | Result | Interpretation |
|---|---:|---|
| Initial fixed-subset BCE | 0.6906174943 | Before fitting |
| Final fixed-subset BCE | 0.4713897519 | After 240 steps; lower than initial, but above the predeclared `<= 0.10` target |
| Mean per-target constant-prevalence BCE baseline | 0.4799836397 | Computed as mean Bernoulli entropy from the eight-study positive/negative counts, with pure-class terms equal to zero |
| Final minus constant-prevalence baseline | -0.0085938878 | Small same-subset improvement over constant prevalence; not generalization evidence |
| Defined / undefined per-target AUCs | 11 / 1 | One target had 0/8 positives; AUC was undefined. These are same-training-subset diagnostics, not validation metrics. |
| Per-target across-study probability standard deviation range | 0.0013698–0.0159520 | Predictions varied little across these eight studies; this alone does not prove model collapse or explain a cause. |

Online epoch loss fell from 0.6857253 in epoch 1 to 0.4793135 in epoch 8, then to 0.4768710 in epoch 30 (0.0024425 further reduction from epochs 8 to 30). These online means are measured before optimizer updates at changing model states; they are not fixed-checkpoint evaluation losses.

**Conclusion:** LD-03 is complete as a bounded diagnostic that was run and interpreted, but the predeclared memorization target was not achieved. The final loss is only modestly below the same-subset constant-prevalence baseline, and per-target predictions show little between-study spread. These findings motivate a focused inspection of inputs, gradients, and the learning path before scaling or starting the pretrained comparison; they do not identify a causal failure. Do not interpret training-subset AUCs as validation performance. The one-run authorization is consumed and no retry is authorized.

Privacy-safe aggregate-only summary: `data/private/training-learning-diagnostics/sanity-run01-upload-v2/run02-aggregate-summary.json` (SHA-256 `32d12f7f7d57d3a3aa9ba3e327c0daf5054df1d1ae03ff38d08abc1d8467418b`). It contains no study identifiers, target names, raw predictions, or raw logs. The faithful visible-log capture remains private at `data/private/training-learning-diagnostics/sanity-run01-upload-v2/run02-visible-logs.txt`; screenshot evidence is `data/private/training-learning-diagnostics/sanity-run01-upload-v2/kaggle-sanity-run02-completed.png`.

## LD-07 prelaunch mount-path correction (2026-10-21)

Before the authorized CPU audit, parent UI inspection showed the private dataset mounted at `/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private`; the earlier notebook only expected the historical short alias `/kaggle/input/rsna-knee-five-study-reader-private`. The fresh local v9 notebook checks exactly these two known paths and accepts a root only when its `inventory.json` bytes match the embedded SHA-256. It prefers the current UI path if both candidates match and raises the bounded `input_mount_unavailable` error if neither does. The resolver runs only inside the enabled branch; the default notebook remains disabled and output-free. Regression tests exercise fallback, deterministic precedence, safe failure, and the disabled guard with temporary inventories, without reading images.

Fresh ignored preparation `data/private/training-learning-diagnostics/sanity-run01-preparation-v9/` preserves the complete v8 upload tree (26/26 paths and bytes). ZIP SHA-256: `15d3b72ccfdedd063a36dd2771dbd495b1d8679fd70ae74c3658964ebd81f6c0`; inventory: `9e48c3201c56ff667ecf8fe3483a4cc1a6557fcf9dd4e055deb0364d6eba48c9`; notebook: `1db4ba2a9db759fccf54d84e64fcec614caf33d62ee7b24755018807276b804c`; helper: `e2aa3efedf4dfc69314809686e35c9c646d05ae7d88259e52fbdb286d79e8c48`. The outer private dataset Version 9 upload remains unchanged; no reupload was needed. The exact focused 10-module suite passed 116 tests in 5.506 seconds; `git diff --check` passed with line-ending warnings only.

### LD-07 terminal CPU diagnostic (2026-10-21)

The one authorized private CPU diagnostic completed successfully in 254.5 seconds (private notebook execution Version 1, `360841828`). The parent had verified the private dataset Version 9, official competition input, private visibility, CPU-only accelerator, internet disabled, and draft session disabled before a single Save Version → Save & Run All → Save sequence. At the initial viewer snapshot it was Running at 47.8 seconds with 0 B output. The notebook's inventory-hash guard accepted one of its two allowlisted roots; the actual selected mount path was not printed and is not inferred.

The diagnostic audited the first eight fold-0 training studies, then ran two total Adam updates on the first two studies using a fresh CPU scratch model. It did not run full training, GPU training, validation or generalization evaluation.

| Aggregate | Result | Scope |
|---|---:|---|
| Input audit | 8 studies; all finite | Training-subset input audit only |
| Exact input pair comparisons | 28; 0 exact duplicates | No exact duplicate pairs among audited inputs; not proof of clinically adequate preprocessing |
| Across-study mean range | 0.1834457278251648–0.3010384539763133 | Per-study aggregate input mean |
| Across-study standard-deviation range | 0.20877141058444976–0.3189350763956706 | Per-study aggregate input standard deviation |
| Probe | 2 studies, 2 total CPU Adam steps, status `ok` | Fresh scratch model, not full training |
| Encoder / head gradient norm | 0.11206283593294178 / 0.15020050147137515 | Both reported finite |
| Encoder / head maximum parameter delta | 0.0020013563334941864 / 0.0020013321191072464 | After the two probe updates |
| Input-induced logit delta | 0.00017346891399938613 | Distinct audited inputs produced a measurable logit difference |
| Post-update output delta | 0.02310626022517681 | Output changed during the bounded CPU probe |

The test confirms that these real inputs reached a working optimization path in the bounded probe. It does not establish why the earlier eight-study 240-step run missed its criterion, validate preprocessing quality, or demonstrate generalization; the causal root cause remains unknown. The selected dataset mount root was not printed. LD-07 is complete as an executed and interpreted diagnostic. Its single-run authorization is consumed; no retry is permitted.

Privacy-safe aggregate-only summary: `data/private/training-learning-diagnostics/learning-path-run01-upload/run01-aggregate-summary.json` (SHA-256 `2d469aafcd55fda3053c8c3fe52e937cec159200965049b57ebc370934833cc4`). Source log and screenshot are kept privately at `data/private/training-learning-diagnostics/learning-path-run01-upload/run01-visible-logs.txt` and `data/private/training-learning-diagnostics/learning-path-run01-upload/kaggle-cpu-run01-completed.png`; neither is copied into public experiment prose.

LD-04/LD-06's pretrained comparison is not implemented; torchvision/weights remain unavailable and no weight download occurred. A matched ResNet-18 random-vs-ImageNet V1 experiment is a meaningful next architecture comparison, but no result here establishes that a baseline bug is fixed or that a pretrained model is superior. No additional run is authorized.

## LD-05 local synthetic learning-path probe (2026-10-21)

A new opt-in helper, `rsna_knee.training.learning_path.diagnose_scratch_learning_path`, runs a fresh scratch CNN on CPU for one to eight synthetic optimizer steps. It checks aggregate input diversity and input-to-logit sensitivity, finite nonzero encoder and classifier-head gradient norms, separate parameter movement after Adam, and post-update output movement. It returns only status and aggregate scalar diagnostics; it does not mutate an existing model, optimizer, or caller RNG state. Constant, non-finite, malformed, and invalid-mask input cannot return `ok`.

Strict-TDD evidence: the initial four regression tests first failed with `ModuleNotFoundError` because the helper did not exist. After implementation, a constant-input test exposed generic invalid classification and was corrected to `input_not_diverse`. A later cross-study-shape test failed with `invalid_input`; paired studies are now allowed to have different series/slice geometries. Five focused tests now cover these cases. A repeatable current synthetic run returned `status=ok`, 2 steps, aggregate mean-input delta `0.0` (the horizontal flip preserves the mean), input-logit delta `0.0001937675`, encoder/head gradient norms `0.1121488449` / `0.1499759597`, maximum encoder/head parameter deltas `0.0020013545` / `0.0020012558`, and post-update logit delta `0.0232772753`. The changed logits show why a mean-only image statistic is not used as the sole diversity gate. These values establish synthetic wiring only, not evidence about the eight real studies.

The exact focused 10-module suite first passed 112 tests in 6.128 seconds; after adding cross-study shape coverage and packaging, it passed 114 tests. A bounded verifier correction then exposed and fixed two edge cases: `torch.manual_seed` invoked CUDA seeding even inside CPU-only `fork_rng` (monkeypatch RED: `AssertionError: CUDA seeding is forbidden`), and a meta-device mask raised instead of returning `invalid_input` (RED: `RuntimeError: Tensor.item() cannot be called on meta tensors`). The helper now seeds only `torch.random.default_generator` and rejects non-CPU masks. Both regressions pass, as does the latest exact 10-module suite: 116 tests in 5.740 seconds. `git diff --check` passed; Git printed existing CRLF-normalization warnings only.

The fresh final disabled-default notebook/package is under ignored `data/private/training-learning-diagnostics/sanity-run01-preparation-v8/`; v6 and v7 are retained as historical preparations. The ZIP has 23 members (prior 22 member bytes unchanged plus helper); its base64 upload tree has 26 files. It is pinned to the existing manifest and folds and configured to audit aggregate input dispersion over the unchanged first eight fold-0 training studies, then run exactly two total optimizer steps on the first two studies with a fresh CPU model. It emits aggregate metrics only and has no saved output. Package SHA-256 `15d3b72ccfdedd063a36dd2771dbd495b1d8679fd70ae74c3658964ebd81f6c0`, inventory `9e48c3201c56ff667ecf8fe3483a4cc1a6557fcf9dd4e055deb0364d6eba48c9`, config `bbce6b03ba7c4f6060a13f1a2bb7e4639316abc57bf13157f9d465f34a7240b5`, notebook `dae7153c6c1126c6ce33041a89d0bf4b1d1efc937f0c8fd3b2865640b43ff1a9`, and helper `e2aa3efedf4dfc69314809686e35c9c646d05ae7d88259e52fbdb286d79e8c48`; manifest/fold hashes are unchanged. A post-correction bounded CPU ZIP-import smoke returned `status=ok` using synthetic tensors and did not touch acceleration APIs. No real-image read, upload, GPU execution, dependency/weight download, or checkpoint load occurred. The actual eight-study result remains pending parent authorization. LD-05 remains unchecked for independent verification. LD-06 protocol draft is in [E028](E028_resnet18_pretrained_control.md); no model implementation or weight use occurred.

## Final LD-05 local verification (2026-10-21)

The corrected v8 preparation passed independent verification: 116 focused tests, both targeted RNG/device regressions, and full ZIP/inventory/member/notebook/transport checks. The parent independently repeated seven helper tests and a whitespace diff check (CRLF warnings only). This closes local implementation/preparation, not real-image diagnosis: the eight-study input audit and first-two-study two-total-step CPU probe remain unrun. No pretrained encoder has been implemented or tested; E028 is protocol only.
