# E036 — Prepare offline unweighted masked-soft training package

## Hypothesis and baseline

**Hypothesis:** a separately generated, offline-asset-pinned Kaggle notebook can verify the E034 cohort plan without running image decoding, constructing a model, or starting training; a later explicitly selected decode-preflight stage can bound pixel decoding to eight eligible studies before any full-training stage.

**Baseline:** E034's private full non-Gold cohort/fold artifacts and E035's separate masked-soft runner. The legacy 500-study binary reader and package builder remain unchanged.

## Changed variable and fixed configuration

This preparation adds only a package builder, an aggregate-only eight-study decode-preflight helper, a disabled-by-default notebook, focused tests, and this record. The prepared training configuration is fold 0, seed 42, frozen ImageNet1K V1 ResNet-18, Adam at 0.001, five epochs, 256-pixel input, at most eight selected slices, unweighted masked soft BCE, and no artificial maximum-step truncation. No optimizer was run.

## Input and package evidence

- E034 manifest SHA-256: `617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b`.
- E034 fold artifact SHA-256: `b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9`.
- Frozen ImageNet1K V1 weight file: 46,830,571 bytes, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; hashed only, never deserialized.
- Offline codec archive: 7,943,426 bytes, SHA-256 `2af0d5084dc70cb3daee2becf7a81aedb13093461aa64ceb532527c7701e3896`; its manifest and wheel inventory were checked against the pinned CPython 3.13 Linux x86_64 target. Nothing was installed locally.
- The previous HSR notebook/source ZIP, the C2 binary cohort/folds, E034 inputs, and masked-runner/dataset source bytes retained their recorded pins. The prior HSR and C2 files were checked as guards, not copied into the new upload assets.

Fresh private output: `data/private/masked-soft-training-preparation/run03/`. The source ZIP contains 24 allowlisted Python members and has SHA-256 `aeb8c066195c172bb919c05574203c55e4d7b491f1ad3051dfcee86d41381d49`. The notebook SHA-256 is `63a1e3ace3d8312df525daa3cbf12738cc34357ec8e9391a9a6a72b88d5946c9`; inventory SHA-256 is `caafb351b44dc05078d7d352f9bd103fd97461471e1b9eacc826f59ac9b0e448`.

The first run01 output is retained but superseded: its upload listing confirms that it included pinned HSR/C2 guard artifacts. A regression assertion was added and the upload allowlist narrowed. Run02 was not a failed build; its source ZIP, notebook and inventory have the same exact hashes as run03. Run03 is designated as the current, fully read-back package; no earlier output was deleted or overwritten.

## Validation protocol and observed results

- The first focused test run was RED because the package-builder module did not exist.
- After implementation, the focused package/preflight module passed all 7 tests.
- The selected 12-module regression command ran 112 tests: 111 passed and one existing Windows symlink-test skip.
- The source archive was imported in an isolated temporary directory and the pinned metadata-only fold-0 plan was run: status `planned`, 2,850 training studies, 1,448 validation studies. The check observed no `pydicom` import, no model construction, and no weight deserialization.
- Generated notebook code compiles, starts with `RUN_STAGE = "plan"` and `RUN_FULL_TRAINING = False`, has empty cell outputs, verifies staged input identities, installs pinned offline codecs before importing pydicom, and requires a successful bounded preflight before an explicitly enabled full-training stage.
- New source/preflight modules compiled. `git diff --check` completed; Git change tracking is read-only and no commit was created.

## Limits and conclusion

This is a local preparation result only. Local `data/train_images` is absent. The generated eight-study preflight was exercised only with synthetic arrays and mocks; no real DICOM was read and no current remote image availability or visual image QC was established. No model was constructed, no weights were deserialized, no training or performance measurement occurred, and no Kaggle package was uploaded or launched. The five-epoch workload count is not a wall-time forecast. Independent package verification for run04 is complete; actual image availability and image quality remain unverified.

## Next step

Run04 has passed the parent-owned independent package verification. Any later image preflight or training requires separate authorization and a fresh eligible execution context; this preparation initiated no remote operation.

## UMP.4 correction after independent failure

Parent-owned independent execution review paused run03 before upload. It found that PLAN installed codecs and imported pydicom, the installer target was discarded rather than added to `sys.path`, decoder checks did not require `is_available`, mount discovery missed the nested Kaggle dataset layout and alternative competition input root, extraction/output paths were ephemeral rather than aligned with the runner's persistent working root, and the inventory labeled the full 24,035 non-Gold series as eligible instead of the 23,765 in the 4,298 eligible manifest records. Run03 remains unchanged and is not uploadable.

Strict-TDD correction observed RED when new tests executed generated notebook code: the previous unconditional top-level `main()` ran before test injection and exited on the unavailable `/kaggle/input` path. The new notebook exposes an injectable run function and guards interactive top-level execution. Fresh run04 is `data/private/masked-soft-training-preparation/run04/` (chooser folder `data/private/masked-soft-training-preparation/run04/upload/`, notebook `kaggle_masked_soft_training.ipynb`): source ZIP SHA-256 remains `aeb8c066195c172bb919c05574203c55e4d7b491f1ad3051dfcee86d41381d49` (24 members); notebook SHA-256 `0f79fd5583c5dd7f3a98422d3665a99929b07003a26f867a4d04bc625d762429`; inventory SHA-256 `25833614d3b274956cfa5284f7c0502ccc7d63d9d6c78d87cdbcf169e0a40707`. Inventory now distinguishes 4,298 eligible studies / 23,765 eligible series from 24,035 full non-Gold historical series. Run01/02/03 remain preserved.

The generated-code tests now execute PLAN, decode-preflight, and FULL gating paths with faked import/codec/decoder/dataset/runner boundaries. They verify default PLAN makes no codec installation, pydicom import, image dataset, or training calls; target import path is available before pydicom; unavailable decoders block before dataset creation; bounded preflight failure blocks training; both nested/flat asset and competition mounts are handled without recursive DICOM discovery; and the derived output root aligns with the runner's `PRIVATE_RUN_ROOT` without constructing a model. The current default notebook was also executed against the pinned metadata and an empty synthetic mount: `planned`, 2,850 train / 1,448 validation, with no codec file, pydicom, or model import.

The exact selected 12-module suite ran 120 tests: 119 passed and one existing Windows symlink skip. The parent-owned independent verification subsequently passed run04; see the closeout below. No codec installation, pydicom decode, real image, model construction, weight deserialization, optimizer, Kaggle upload, or training was performed.

## UMP.4 independent closeout

The independent run04 recheck passed. The exact selected 12-module suite reported 120 tests (119 passed, one existing Windows symlink skip); compile/diff checks exited 0, and the parent spot-check passed 15 focused tests in 2.541s. An isolated `python -I` execution of the generated notebook's actual PLAN used faked mounts and pinned metadata, returning `planned` with 2,850 training / 1,448 validation studies without pydicom/model import, codec install, image reads, training output, or weight deserialization. Mock stage execution verified bounded flat/nested asset and competition mount discovery, codec target visibility before pydicom, all five decoder availability checks, FULL gating after successful bounded 8-study preflight, and persistent output alignment. Eight protected hashes were unchanged. Run04 current notebook SHA-256 is `0f79fd5583c5dd7f3a98422d3665a99929b07003a26f867a4d04bc625d762429`; inventory SHA-256 is `25833614d3b274956cfa5284f7c0502ccc7d63d9d6c78d87cdbcf169e0a40707`; its 24-member source ZIP remains SHA-256 `aeb8c066195c172bb919c05574203c55e4d7b491f1ad3051dfcee86d41381d49`. Inventory distinguishes 23,765 eligible series from 24,035 full non-Gold historical series. Run03's six independent findings remain historical and it must not be uploaded; run01 and run02 remain superseded as recorded. No real image QC, codec install, model construction, weight deserialization, optimizer/training, Kaggle upload/launch, or broad suite run occurred; the five-epoch runtime remains unqualified. This closes local package preparation only.