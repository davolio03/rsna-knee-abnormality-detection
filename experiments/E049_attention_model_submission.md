# E049 — E048 epoch-five attention submission package

**Status:** corrected candidate `run02` was privately uploaded, its canary/full inference completed, and the full notebook was submitted once. Parent later verified a private score of **0.848** for Version 2; observed execution time was 68.3 seconds on T4x2. No public score is recorded. Earlier local package `run01` remains preserved as invalid negative evidence and must not be transferred.

## Hypothesis and boundary

Package the preselected E048 `target_attention` epoch-five model as a separate Kaggle inference candidate, preserving its trained target-specific slice and series attention. This changes pooling relative to the same-fold equal-mean control; it is not a new training run, independent test, or leaderboard claim. Epoch five is the user-selected comparison point, not a post-hoc best-epoch selection.

| Item | Frozen evidence |
|---|---|
| Paired-run report | `data/private/disease-attention-mil/e048-run01/training-report.json`; 30,355 bytes; SHA-256 `f06b3714514fee1adc43462ca2e75c7d1f1dc5581a990877319283537bf0114e` |
| Attention source checkpoint | `data/private/disease-attention-mil/e048-run01/target_attention/checkpoint-epoch-05.pt`; 214,808,723 bytes; SHA-256 `4c9d8930f54dcacbb88bcecb7f1f449c498dce83f13929fbcf5bb67fb5152290` |
| Native run fingerprint | `13658c21bba3f35bb0d9fd392034af852125f5f2639006e3937d6a82f85108cb` |
| Control / attention validation | Equal mean: masked BCE `0.5139812277577035`, Brier `0.10471935699252992`; attention: masked BCE `0.4927743480438822`, Brier `0.09429479721917607`; each uses the same fold-0 13,195 observed soft-label cells. |
| Selection rule | The comparison was fixed to the completed epoch-five checkpoint; no epoch selection from these reported outcomes. |

The validation metrics are local soft-label fit evidence only. They do not define binary AUC, do not establish generalization or leaderboard improvement, and do not make attention weights clinical explanations. Gold-58, fold assignment, labels, and training were untouched.

## Prepared package

`data/private/attention-model-submission/run02/` is the corrected exclusive staging directory. It contains the model-only export, aggregate report, source ZIP, canary/full notebooks, and inventory. The export was derived only after exact source report/checkpoint/proof pins and strict attention-arm/epoch/history/state checks; it retains model tensors and provenance but excludes the source checkpoint's optimizer and RNG state.

The earlier `run01` directory is intentionally untouched. Independent verification found that its 23-member source ZIP omitted `inference/radimagenet.py`, required by the attention adapter, and isolated import failed with `ModuleNotFoundError`. Both `run01` notebooks also pinned the full source-checkpoint SHA/size rather than their model-only export. Do not upload or run `run01`; `run02` adds the accepted E047 dependency byte-identically and writes notebook pins structurally from the actual export inventory.

| Artifact | SHA-256 / size |
|---|---|
| `attention-model-epoch-05.pt` | 94,645,985 bytes; SHA-256 `d03b37d0fbf99bdd6105259a89fd7574ff72ae8215f13c24cb1df15f8db265bf` |
| `attention-inference-source.zip` | SHA-256 `abb3a076ac0035e780d1c036c9b6c21b6000b331df0301f4448f6a38d1a11e41`; 24 code-only Python members |
| `training-report.json` | SHA-256 `f06b3714514fee1adc43462ca2e75c7d1f1dc5581a990877319283537bf0114e` |
| Canary notebook | `kaggle_radimagenet_attention_canary.ipynb`; SHA-256 `987d8666c448f2ced0a02664849c5c98eebf317f87f593592775d89912c98c67`; max 3 studies |
| Full notebook | `kaggle_radimagenet_attention_submission.ipynb`; SHA-256 `7fa642242911a8f37d78d06a2ce353fd5fa76483544c9742a969477eb54eee64`; uncapped |

The inference closure keeps the existing fold-0 submission writer, test-study/DICOM preprocessing and offline codec bootstrap, plus the frozen ResNet-50 and attention model sources. It excludes training modules, source cache, run plan, optimizer/RNG tensors, and raw DICOM/report data. Following the user's explicit `Si` authorization for these two files, the parent created the private dataset `davidsulleiro/rsna-knee-attention-epoch05` Version 1 with only `attention-model-epoch-05.pt` and aggregate `training-report.json` (license remains unknown, unchanged). The dataset mount is `/kaggle/input/datasets/davidsulleiro/rsna-knee-attention-epoch05`; the official competition input and Internet-offline policy are inherited from the accepted codec package.

## Kaggle execution and submission status

| Event | Observed status |
|---|---|
| Canary | Private notebook `rsna-knee-attention-epoch05-submission-run01`, Version 1 / execution `364641216`; completed successfully in 47.7 seconds total, with 20.712336 seconds in inference. It processed 3/3 visible studies, skipped zero series/slices, reported `canary=true`, and matched the expected attention fingerprint, model/report pins, 224px / eight-slice / microbatch-two configuration. It wrote the root CSV and was **not submitted**. |
| Full inference | The same private notebook, Version 2 / execution `364641924`, ran with `RUN_INFERENCE=True`, `MAX_STUDIES=None`, T4x2, Internet OFF, the original environment, and quota 00:01 / 30 h. It completed successfully in 68.3 seconds total, with 24.260236 seconds in inference; it processed 3/3 visible studies, skipped zero series/slices, reported `canary=false`, and wrote the root CSV with the expected attention/pin evidence. |
| Competition submission | The exact full Version 2 was submitted once. Parent later verified the private score **0.848**; observed execution time was 68.3 seconds on T4x2. The public score was not reported. Do not resubmit. |

Parent-observed private browser evidence is retained under `data/private/attention-model-submission/browser-proof/`: `canary-v1.log`, `full-v2.log`, `canary-v1-success.png`, `private-dataset-created.png`, and `submission-v2-running.png`. E047's public/private `0.825 / 0.802` remain context only, not a causal comparison. No training was launched. The private score does not establish an E050 result or isolate an effect of the sixteen-slice setting.

## Verification and limitations

- Strict-TDD RED observed before implementation: the new focus module failed to import because `radimagenet_attention.py` did not yet exist. A second bounded correction reproduced both package defects as RED; GREEN followed after closure and pin fixes.
- The local training adapter verified report/checkpoint pins before loading, extracted only finite exact-key model tensors, then reloaded the model with `weights_only=True` and strict state matching. It never restores optimizer/RNG state or downloads generic torchvision weights.
- Exact-logit proof compares the exported inference model against a separately instantiated attention architecture loaded directly from the pinned trained epoch-five checkpoint on one SHA-verified cached study; logits were bit-exact (`torch.equal`).
- Both corrected notebook ASTs were read back with export SHA `d03b37d0fbf99bdd6105259a89fd7574ff72ae8215f13c24cb1df15f8db265bf` and size `94,645,985`, plus the exact report SHA and mount from the inventory.
- The actual run02 source ZIP imported all 18 modules in an isolated `python -I` process with the ZIP as the only project path. It has 24 members; all 22 members shared with E047 are byte-identical, with only `attention_mil.py` and the new attention adapter added.
- The post-correction combined requested regression command passed 65 tests; the two new focus modules passed 11 tests. This is not a whole-project test claim.
- The report had no DICOM UID-formatted identifiers; package closure contains no training module, run plan, cache, raw report, optimizer/RNG tensors, or DICOM data.
- The attention adapter preserves direct 224-pixel preprocessing, max eight slices, microbatch two, masked spatial pooling, and separate trained target-specific slice/series attention. A canary can publish the root CSV only when every visible test row was processed; unexpected inference/provenance failures are fatal and do not publish a partial CSV.
- The corrected package output path refuses overwrite. The authorized private upload and one full submission were performed by the parent. No new training, push, or resubmission occurred.

## Next step

No further E049 execution is needed. Do not resubmit or infer a public score. Keep E049's eight-slice result separate from E050's local sixteen-slice validation evidence.
