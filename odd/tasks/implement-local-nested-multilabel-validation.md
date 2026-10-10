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
- Delivery strategy: direct local commits to `main`, explicitly requested by the user; no PR or remote operation is authorized. This supersedes the prior ask-on-risk PR-chain proposal for this feature.

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

- [x] Add the public nested split API: outer 3-fold partitions, then inner 3-fold partitions restricted to each outer-training partition.
- [x] Enforce and test no overlap between an outer test partition and its nested inner train/validation members; keep fold membership exhaustive and disjoint.
- **Route:** delegated direct, because this builds on T1 and spans non-trivial implementation and test behavior; the delegated writer should own code and preparation reads.
- **Checks:** strict TDD for nested-isolation and coverage cases; run `python -m unittest discover -s tests -v`.

### T3 - Document usage and close with reproducibility/safety checks

- [x] Add a concise API usage note/docstring and synthetic example if needed; do not include real identifiers or membership.
- [x] Review reproducibility, input failure modes, and repository diff for private-data leakage; record task evidence and update this checklist.
- **Route:** inline for the final structural readback and small documentation/checklist edits; trigger evidence: this is one already-understood document-level closure, not broad code exploration or a new multi-file implementation.
- **Checks:** `python -m unittest discover -s tests -v`; inspect changed paths/content for private data; record any unavailable checks honestly.

## Progress and verification

- [x] Exploration handoff reviewed: empty validation package; standard-library unittest; no dependency manifest; no strict-TDD setting found.
- [x] Strict TDD enabled for this feature by explicit user choice; exact runner recorded above.
- [x] T1 technically complete; commit `dc7cfdf`; independent audit passed.
- [x] T2 complete; commit `65ae48f`; RED/GREEN evidence recorded below.
- [x] T3 complete; API docs, independent checks, safety readback and closure records accompany the local documentation work-unit commit.
- **Authored changed-line estimate:** at least 505 lines including T1 and T2 source/tests; task/project documentation adds further lines. Direct-main delivery was explicitly selected; no PR is being created. No code or tests were reduced to meet a budget.
- **T1 strict-TDD evidence:** `python -m unittest discover -s tests -v` first failed before implementation with `ModuleNotFoundError` for `rsna_knee.validation.multilabel`; after implementation, the same command passed all 36 tests. An intermediate test run exposed malformed two-study fixtures using the three-fold default; tests were corrected to request two folds for those validation cases.
- **Current verification:** T1 uses synthetic fixtures only; no private Gold data was accessed. T1 commit: `dc7cfdf`. Independent read-only audit reran all 36 tests successfully and 2,800 synthetic coverage/nonempty-partition probes.
- **T2 implemented and committed as `65ae48f`:** `nested_multilabel_splits` returns immutable `NestedSplit(train, test, inner_splits)` and `InnerSplit(train, validation)` records. Defaults are 3x3; outer seed is the supplied seed and inner seed is `seed + 1 + outer_index`. Rows are aligned by original keys and all complements preserve input order. Insufficient allocated outer-training populations reject clearly; no exact-size or class-coverage promise is made.
- **T2 strict-TDD evidence:** Before source implementation, `python -m unittest discover -s tests -v` ran 37 tests and failed with one import error because `nested_multilabel_splits` did not exist. After implementation, all 44 tests passed. After a test-only cleanup, all 44 passed again (2.353 seconds). `python -m doctest src/rsna_knee/validation/multilabel.py` passed; `git diff --check` passed (only Git line-ending conversion warnings).
- **T2 runtime/safety:** Synthetic fixtures exercise the public API directly; external runtime harness is N/A because there is no external execution boundary. No private data or remote operations were accessed. Rollback boundary: remove the nested API/dataclasses and `tests/test_validation_nested.py`, retaining the T1 allocator. Source/test T2 changes total 200 authored lines (90 additions/1 deletion in the module and 109 new test lines), excluding task-record updates.

## Next step

Local implementation and technical checks are complete on main. No push or PR has been performed or authorized. Further evaluation requires separately authorized data scope and a frozen protocol; synthetic tests do not establish real-data feasibility. Old native T1 review remains inconclusive; clone-local RDD stays disabled/unmanaged.

## Workflow update

- User explicitly disabled clone-local RDD; global preference unchanged. Native status: disabled/unmanaged for subsequent work; old T1 review remains inconclusive.
- T1 independent audit: 36 unittest tests and 2,800 bounded synthetic probes passed; no blockers. Fold capacity is a tie-break preference, not a rigid size or per-class-coverage guarantee.
- User superseded the prior PR-chain proposal with direct local main commits; no push or PR is authorized.

- **Independent T2 audit:** 44 tests, doctest and diff whitespace check passed; 300 synthetic cases verified (299 valid partitions, one correctly rejected for insufficient inner population). No blockers. Parent spot-check also passed all 44 tests. T3 API usage criterion is satisfied by the executable docstring; source/test changes contain synthetic fixtures only. T2 code is committed as `65ae48f`; closure documentation is recorded in a separate local work-unit commit. Publication remains unauthorized.

## Direct-main delivery decision

The user explicitly selected local commits directly to main rather than PRs. A local main branch was created at the existing HEAD, preserving all history and the original feature branch. No live remote state was queried; no push or PR is authorized. The 400-line PR budget does not require a chain when no PR is being created. Source/tests were independently audited and full checks passed before this work-unit commit.
