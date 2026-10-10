# Feature: Local nested multilabel validation splits

## Objective

Implement a reusable, deterministic, study-level nested 3x3 multilabel split utility for future Gold-58 prompt-strategy evaluation. This feature creates split membership locally from caller-supplied study keys and multilabel targets; it does not compare prompts or execute evaluation strategies.

## Problem and why

The project has an accepted nested study-level prompt-strategy evaluation design (D004), but no reusable local split-generation utility. A reproducible implementation is needed before future evaluation work, while preserving study-level separation and keeping the outer folds isolated from inner selection.

## Scope

- Provide a locally reusable API for deterministic three-fold outer splitting and three-fold inner splitting within each outer-training partition.
- Keep all members of a study together; support multilabel target vectors and stable seed-controlled assignment.
- Validate inputs and report infeasible/invalid split conditions clearly.
- Verify behavior with synthetic fixtures only; real split membership is not to be copied into the repository.
- Do not implement prompt comparison, report labeling, model training, or a Gold-58-specific data loader.

## Constraints and authorized scope

- Authorized: source and test changes required for the reusable split utility, plus a concise usage/API note if needed.
- No private data transfer. Do not inspect or persist private Gold IDs, local labels, reports, outputs, or masks. Do not commit any real split membership.
- Synthetic test data only. Preserve study-level grouping and the D004 outer-fold isolation requirement.
- Avoid a new runtime dependency unless implementation evidence establishes it is necessary; the repository has no dependency manifest and uses standard-library `unittest`.
- Existing exploration found `src/rsna_knee/validation/` empty. A prior authorized aggregate-only feasibility run found a feasible standard iterative-stratification plan; do not copy any real membership or IDs.
- TDD: **strict=true**, source: **explicit user choice for this feature**, runner: `python -m unittest discover -s tests -v`.
- Delivery strategy: `ask-on-risk`.

## Acceptance criteria

1. A public, documented API generates deterministic outer 3-fold partitions from the same ordered inputs and seed, with every study appearing in exactly one outer test fold.
2. For each outer fold, deterministic inner 3-fold partitions use only that outer training set; no outer-test study appears in any corresponding inner train or validation partition.
3. Every returned partition is study-disjoint and collectively covers the intended input partition exactly once; malformed, duplicate, or infeasible inputs fail with actionable errors rather than silently producing invalid folds.
4. Multilabel balancing is implemented with a documented deterministic strategy; tests assert expected balance/coverage behavior on synthetic cases where feasibility is known, without assuming impossible per-class coverage.
5. Repeated generation with fixed inputs and seed is identical; changing the seed is supported without promising that all assignments necessarily change.
6. The configured unittest command passes, including strict TDD evidence (a relevant failing test observed before implementation, followed by passing tests).
7. No private IDs, labels, reports, outputs, or real split membership are added to tracked files.

## Tasks

### T1 - Define deterministic study-level multilabel fold allocation

- [x] Implement the core deterministic fold allocator and input validation using only caller-provided study keys/target vectors.
- [x] Keep the split algorithm independent of Gold-58 paths or data-loading behavior; document seed and tie-breaking behavior.
- **Route:** delegated direct, because implementation requires reading and changing multiple non-trivial source/test files; the existing validation package is empty, so API design and test seams must be established together.
- **Checks:** under strict TDD, add focused synthetic tests first and observe failure; then run `python -m unittest discover -s tests -v`.

### T2 - Compose isolated nested 3x3 partitions

- [ ] Add the public nested split API: outer 3-fold partitions, then inner 3-fold partitions restricted to each outer-training partition.
- [ ] Enforce and test no overlap between an outer test partition and its nested inner train/validation members; keep fold membership exhaustive and disjoint.
- **Route:** delegated direct, because this builds on T1 and spans non-trivial implementation and test behavior; the delegated writer should own code and preparation reads.
- **Checks:** strict TDD for nested-isolation and coverage cases; run `python -m unittest discover -s tests -v`.

### T3 - Document usage and close with reproducibility/safety checks

- [ ] Add a concise API usage note/docstring and synthetic example if needed; do not include real identifiers or membership.
- [ ] Review reproducibility, input failure modes, and repository diff for private-data leakage; record task evidence and update this checklist.
- **Route:** inline for the final structural readback and small documentation/checklist edits; trigger evidence: this is one already-understood document-level closure, not broad code exploration or a new multi-file implementation.
- **Checks:** `python -m unittest discover -s tests -v`; inspect changed paths/content for private data; record any unavailable checks honestly.

## Progress and verification

- [x] Exploration handoff reviewed: empty validation package; standard-library unittest; no dependency manifest; no strict-TDD setting found.
- [x] Strict TDD enabled for this feature by explicit user choice; exact runner recorded above.
- [x] T1 implementation and tests complete; commit identity pending.
- [ ] T2 complete; record commit and RED/GREEN evidence.
- [ ] T3 complete; record commit and final checks.
- **Authored changed-line estimate:** approximately 350 lines across implementation, tests, and concise API documentation; generated files excluded. This is a planning estimate, not a hard cap.
- **T1 strict-TDD evidence:** `python -m unittest discover -s tests -v` first failed before implementation with `ModuleNotFoundError` for `rsna_knee.validation.multilabel`; after implementation, the same command passed all 36 tests. An intermediate test run exposed malformed two-study fixtures using the three-fold default; tests were corrected to request two folds for those validation cases.
- **Current verification:** T1 uses synthetic fixtures only; no private Gold data was accessed. Commit identity will be recorded after the work-unit commit.

## Next step

T1 is ready for work-unit commit and native RDD assessment. Pause for the parent before T2; do not access private Gold contents.
