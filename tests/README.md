# Tests

Add tests alongside established data, metric, preprocessing, and model behavior as those components are implemented. Prefer small deterministic tests for schema, study grouping, leakage-sensitive boundaries, and metric behavior.

The synthetic submission contract smoke test uses only Python's standard library. Run it with `python -m unittest discover -s tests -p test_smoke_submission.py -v`; no test framework installation is required. This test covers a synthetic CSV contract, not competition-data compatibility or Kaggle execution.
