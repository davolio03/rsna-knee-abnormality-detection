# E053 — Private attention-16 disagreement review sample

**Status:** Bundle prepared; parent independent post-build readback passed; qualified review is pending.

E053 packages a deliberately enriched, blinded report-review queue from the immutable E052 predictions. It is intended to prioritize cases for qualified review—not to measure model performance, estimate prevalence, or establish clinical truth.

## Scope and frozen selection protocol

| Item | Protocol |
|---|---|
| Source | Frozen E050 epoch-five attention-16 predictions and observed soft labels in E052; no new model call |
| Targets | Contusion, Lateral Meniscus, MCL |
| Per-target rows | 10 largest absolute discrepancies among observed labels `>= 0.8`; 10 largest among observed labels `<= 0.2`; 5 agreement controls from each label group with absolute error `<= 0.1` |
| Ordering | Error rows sort by descending absolute error, then ascending source ordinal; controls are selected without replacement with `numpy.random.default_rng(42)` from ordinal-sorted candidates; review rows are shuffled with `random.Random(42)` before pseudonymous case IDs are assigned |
| Result | 90 target-specific rows, 30 per target, covering 89 unique studies; the same study may appear in more than one target queue |

Observed-mask eligibility is required. Error and control selections do not overlap within a target/label group. The protocol and exact inputs are recorded in the private receipt. No identifiers, report text, or per-case predictions are reproduced here.

## Private bundle and reviewer workflow

The bundle is under `data/private/local16-review-sample/run01/` and is ignored by Git. Start at `index.html` and use `review-template.csv`; leave `unblinding.json` closed until blinded review is complete. The unblinding key contains predictions, source soft labels and masks, selection strata, and private study mapping. Blinded case pages show only the target, original report, cached previews, and empty review fields. Original report text is preserved in per-case private JSON and escaped in HTML. Review annotations are not automatically applied to labels, reports, training, or inference.

The bundle contains **1,560 cached 224px PNG previews for 520 series across 89 studies**, with at most three ordered valid slices per series. These are limited cached previews—not full DICOMs and not diagnostic-quality images. Series metadata is included only when present. A qualified reviewer is still required to judge report/image relevance and any clinical questions; this sample cannot establish clinical truth.

## Provenance and verification

The builder revalidated all E052 export rows against the pinned cohort, labels, model/report pins, and provenance before selection. It verified the requested validation-report joins and cache study/series ordering; accessed NPZ payload hashes were checked through the existing safe cache reader. The full 61.2 GB cache verification was **not** repeated. Input hashes, code provenance, output inventory hashes, and aggregate verification evidence are retained in the private receipt and verification record.

- Source implementation: commit `11c6271` (builder and tests).
- Strict-TDD/regression suite: 20 tests passed; the parent independently passed 11 pre-build checks.
- Local post-build readback: all target/stratum quotas and control thresholds, exact report preservation, blinded/unblinded separation, blank review template, local file links, preview count/dimensions, and inventory/source hashes passed.
- Parent independent post-build readback passed for all 1,744 inventoried files and receipt hashes, exact 90/89 selection, preserved multiline reports, blank template, 1,560 grayscale 224×224 previews across 520 series/89 studies, local-only links, and blind/unblind separation. It did not repeat the 1.33 GB selected-payload decode or full 61.2 GB cache verification.
- The default CLI dry plan validates inputs and selection without generating output. The authorized build command was:

  ```powershell
  $env:PYTHONPATH='src'
  & data/private/local-cuda-env/Scripts/python.exe -u scripts/prepare_local16_review_sample.py --build
  ```

  The first invocation stopped at import because `PYTHONPATH=src` was not set and created no output. The corrected, parent-authorized invocation exited 0 in 28.702 seconds. Both private logs are preserved outside the bundle at `data/private/local16-review-sample/build-run01.log` and `data/private/local16-review-sample/build-run02.log`; the aggregate proof is `data/private/local16-review-sample/verification-run02.json`.

The builder refuses to overwrite an existing destination. Do not rerun against `run01`; any separately authorized future build must use a new private output directory.

## Interpretation and boundaries

This is an enriched, selected review queue—not a representative sample, independent test set, accuracy/prevalence estimator, or clinical-performance result. No labels were changed; no training, inference, Gold-58 access, remote transfer, or Kaggle operation occurred. The report records preparation and structural checks only; qualified review remains outstanding.
