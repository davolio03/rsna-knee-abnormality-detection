"""Leakage-aware grouped folds for a validated private cohort manifest."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

from rsna_knee.data.cohort import REPORT_NORMALIZATION_VERSION, write_private_manifest
from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.validation.multilabel import iterative_multilabel_folds


COHORT_MANIFEST_VERSION = "baseline-training-cohort-v1"
FOLD_ARTIFACT_VERSION = "baseline-training-folds-v1"
FOLD_COUNT = 3
FOLD_SEED = 42
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROMPT_VERSIONS = {
    "legacy", "c2-literal-evidence-v1", "c2-literal-evidence-v2", "c2-literal-evidence-v3",
}
_SOURCE_HASH_KEYS = {
    "train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal",
}
_SERIES_FIELDS = {"series_id", "fluid_sensitive", "fat_suppression", "anatomical_plane"}


def _validate_manifest(manifest: object) -> list[dict[str, object]]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != COHORT_MANIFEST_VERSION:
        raise ValueError("The cohort manifest schema is unsupported.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("The cohort manifest configuration is invalid.")
    if configuration.get("report_group_normalization") != REPORT_NORMALIZATION_VERSION:
        raise ValueError("The cohort report-group normalization is unsupported.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("The cohort manifest records are invalid.")
    if type(configuration.get("cohort_size")) is not int or configuration["cohort_size"] != len(records):
        raise ValueError("The cohort manifest record count is inconsistent.")

    source_hashes = manifest.get("source_sha256")
    if (not isinstance(source_hashes, Mapping) or set(source_hashes) != _SOURCE_HASH_KEYS
            or any(not isinstance(value, str) or not _SHA256.fullmatch(value)
                   for value in source_hashes.values())):
        raise ValueError("The cohort manifest source hashes are invalid.")

    provenance = manifest.get("prompt_provenance_counts")
    if (not isinstance(provenance, Mapping) or set(provenance) != _PROMPT_VERSIONS
            or any(type(value) is not int or value < 0 for value in provenance.values())
            or sum(provenance.values()) != len(records)):
        raise ValueError("The cohort prompt-provenance summary is invalid.")

    seen_ids: set[str] = set()
    observed_provenance = {version: 0 for version in _PROMPT_VERSIONS}
    validated = []
    for record in records:
        if (not isinstance(record, Mapping)
                or set(record) != {"study_id", "labels", "series", "prompt_version",
                                   "normalized_report_sha256"}):
            raise ValueError("A cohort record has an invalid schema.")
        study_id = record["study_id"]
        if not isinstance(study_id, str) or not study_id.strip() or study_id in seen_ids:
            raise ValueError("Cohort study IDs must be nonempty and unique.")
        seen_ids.add(study_id)
        labels = record["labels"]
        if (not isinstance(labels, Mapping) or set(labels) != set(LABEL_COLUMNS)
                or any(type(labels[name]) is not int or labels[name] not in (0, 1)
                       for name in LABEL_COLUMNS)):
            raise ValueError("Every cohort record must contain twelve binary integer labels.")
        group_hash = record["normalized_report_sha256"]
        if not isinstance(group_hash, str) or not _SHA256.fullmatch(group_hash):
            raise ValueError("A normalized report group hash is invalid.")
        prompt_version = record["prompt_version"]
        if prompt_version not in _PROMPT_VERSIONS:
            raise ValueError("A cohort record has unknown prompt provenance.")
        observed_provenance[prompt_version] += 1
        series = record["series"]
        if not isinstance(series, list) or not series:
            raise ValueError("Every cohort record must retain series membership.")
        for item in series:
            if (not isinstance(item, Mapping) or set(item) != _SERIES_FIELDS
                    or not isinstance(item["series_id"], str) or not item["series_id"].strip()
                    or type(item["fluid_sensitive"]) is not int
                    or item["fluid_sensitive"] not in (0, 1)
                    or type(item["fat_suppression"]) is not int
                    or item["fat_suppression"] not in (0, 1)
                    or not isinstance(item["anatomical_plane"], str)
                    or not item["anatomical_plane"].strip()):
                raise ValueError("A cohort record contains invalid series metadata.")
        validated.append(dict(record))
    if observed_provenance != dict(provenance):
        raise ValueError("The prompt-provenance counts do not match cohort records.")
    return validated


def _class_counts(records: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    positives = {target: 0 for target in LABEL_COLUMNS}
    for record in records:
        labels = record["labels"]
        for target in LABEL_COLUMNS:
            positives[target] += labels[target]
    negatives = {target: len(records) - positives[target] for target in LABEL_COLUMNS}
    return positives, negatives


def build_grouped_training_folds(manifest: object, *, manifest_sha256: str) -> dict[str, object]:
    """Allocate report-hash groups with group-OR targets and validate study coverage."""
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise ValueError("The source manifest SHA-256 is invalid.")
    records = _validate_manifest(manifest)

    groups: OrderedDict[str, dict[str, object]] = OrderedDict()
    for record in records:
        group_hash = str(record["normalized_report_sha256"])
        group = groups.setdefault(group_hash, {
            "study_ids": [], "labels": {target: 0 for target in LABEL_COLUMNS},
        })
        group["study_ids"].append(str(record["study_id"]))
        for target in LABEL_COLUMNS:
            group["labels"][target] |= record["labels"][target]

    group_keys = list(groups)
    group_targets = [[groups[key]["labels"][target] for target in LABEL_COLUMNS]
                     for key in group_keys]
    folds = iterative_multilabel_folds(
        group_keys, group_targets, n_splits=FOLD_COUNT, seed=FOLD_SEED,
    )
    group_to_fold = {group_key: fold_index
                     for fold_index, fold in enumerate(folds) for group_key in fold}
    group_assignments = [
        {"normalized_report_sha256": group_key,
         "fold": group_to_fold[group_key],
         "study_ids": list(groups[group_key]["study_ids"])}
        for group_key in group_keys
    ]
    record_group = {str(record["study_id"]): str(record["normalized_report_sha256"])
                    for record in records}
    study_assignments = [
        {"study_id": str(record["study_id"]),
         "fold": group_to_fold[record_group[str(record["study_id"])] ]}
        for record in records
    ]
    study_fold = {item["study_id"]: item["fold"] for item in study_assignments}

    fold_audits = []
    overlap_total = 0
    for fold_index in range(FOLD_COUNT):
        validation = [record for record in records if study_fold[record["study_id"]] == fold_index]
        training = [record for record in records if study_fold[record["study_id"]] != fold_index]
        train_groups = {record["normalized_report_sha256"] for record in training}
        validation_groups = {record["normalized_report_sha256"] for record in validation}
        group_overlap = len(train_groups & validation_groups)
        overlap_total += group_overlap
        train_positive, train_negative = _class_counts(training)
        validation_positive, validation_negative = _class_counts(validation)
        if any(not train_positive[target] or not train_negative[target]
               or not validation_positive[target] or not validation_negative[target]
               for target in LABEL_COLUMNS):
            raise ValueError("The fixed grouped allocation cannot preserve both classes for every target.")
        fold_audits.append({
            "fold": fold_index,
            "train_study_count": len(training),
            "validation_study_count": len(validation),
            "group_overlap_count": group_overlap,
            "train": {"positive_counts": train_positive, "negative_counts": train_negative},
            "validation": {"positive_counts": validation_positive,
                           "negative_counts": validation_negative},
        })
    if overlap_total != 0:
        raise ValueError("The grouped fold allocation contains report-group overlap.")
    if len(study_assignments) != len(records) or len(study_fold) != len(records):
        raise ValueError("The fold allocation did not partition every cohort study exactly once.")

    return {
        "schema_version": FOLD_ARTIFACT_VERSION,
        "manifest_sha256": manifest_sha256,
        "configuration": {
            "algorithm": "rsna_knee.validation.multilabel.iterative_multilabel_folds",
            "algorithm_version": "repository-implementation",
            "n_splits": FOLD_COUNT,
            "seed": FOLD_SEED,
            "group_key": "normalized_report_sha256",
            "group_normalization": manifest["configuration"]["report_group_normalization"],
            "group_target_aggregation": "logical OR across study labels",
            "target_order": list(LABEL_COLUMNS),
        },
        "group_count": len(groups),
        "group_assignments": group_assignments,
        "study_assignments": study_assignments,
        "audit": {"group_overlap_count": overlap_total, "folds": fold_audits},
    }


def write_private_folds(artifact: Mapping[str, object], *, output_path: Path,
                        project_root: Path) -> None:
    """Persist fold IDs and assignments only below ignored data/private."""
    write_private_manifest(artifact, output_path=output_path, project_root=project_root)


def load_manifest_and_build_folds(manifest_path: Path) -> dict[str, object]:
    """Read one private manifest and bind fold output to its exact source bytes."""
    try:
        raw = Path(manifest_path).read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("The private cohort manifest could not be read.") from None
    digest = hashlib.sha256(raw).hexdigest()
    return build_grouped_training_folds(manifest, manifest_sha256=digest)
