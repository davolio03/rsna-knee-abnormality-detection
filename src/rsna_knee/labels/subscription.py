"""Fail-closed local labeling utilities for a bounded Codex CLI pilot.

The CLI adapter uses the Codex application command, not the OpenAI API. The
operator must verify subscription-backed ChatGPT OAuth before any real run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Mapping


LABEL_COLUMNS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
STUDY_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"
MAX_PILOT_RECORDS = 10
EXPECTED_GOLD_STUDIES = 58
_MAX_EVIDENCE_CHARS = 200


class LabelingError(ValueError):
    """A safe-to-display validation or execution error (never includes payloads)."""


def _known_ids(gold_ids: Iterable[str] | None) -> set[str]:
    if gold_ids is None:
        raise LabelingError("Gold exclusion IDs are required.")
    result = set()
    for study_id in gold_ids:
        if not isinstance(study_id, str) or not study_id.strip():
            raise LabelingError("Gold exclusion IDs are invalid.")
        normalized = study_id.strip()
        if normalized in result:
            raise LabelingError("Gold exclusion IDs contain duplicates.")
        result.add(normalized)
    if len(result) != EXPECTED_GOLD_STUDIES:
        raise LabelingError("Gold exclusion list must contain exactly 58 unique IDs.")
    return result


def eligible_reports(rows: Iterable[Mapping[str, object]], *,
                     gold_ids: Iterable[str] | None) -> list[Mapping[str, object]]:
    """Return only rows with complete schema, blank targets, and non-Gold IDs."""
    excluded = _known_ids(gold_ids)
    selected = []
    seen = set()
    required = {STUDY_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS}
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise LabelingError("Training row is missing required schema columns.")
        study_id = row[STUDY_COLUMN]
        report = row[REPORT_COLUMN]
        if not isinstance(study_id, str) or not study_id.strip():
            raise LabelingError("Training row has an invalid study key.")
        study_id = study_id.strip()
        if study_id in seen:
            raise LabelingError("Training rows contain duplicate study keys.")
        seen.add(study_id)
        if not isinstance(report, str) or not report.strip():
            raise LabelingError("Training row has a blank or invalid report.")
        if any(_present(row[column]) for column in LABEL_COLUMNS):
            continue
        if study_id in excluded:
            continue
        selected.append(row)
    return selected


def _present(value: object) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def validate_label_result(value: object) -> dict[str, dict[str, object]]:
    """Validate exact 12-target structured output without coercing labels."""
    if not isinstance(value, dict) or set(value) != set(LABEL_COLUMNS):
        raise LabelingError("Model result must contain exactly the 12 target fields.")
    clean: dict[str, dict[str, object]] = {}
    for column in LABEL_COLUMNS:
        target = value[column]
        if not isinstance(target, dict) or set(target) != {"label", "evidence", "uncertainty"}:
            raise LabelingError("Model target has an invalid structure.")
        label, evidence, uncertainty = (target["label"], target["evidence"], target["uncertainty"])
        if type(label) is not bool:
            raise LabelingError("Model target label must be a JSON boolean.")
        if not isinstance(evidence, str) or len(evidence) > _MAX_EVIDENCE_CHARS:
            raise LabelingError("Model target evidence is invalid or too long.")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise LabelingError("Model target uncertainty must be numeric.")
        if not 0 <= uncertainty <= 1:
            raise LabelingError("Model target uncertainty must be between 0 and 1.")
        clean[column] = {"label": label, "evidence": evidence,
                         "uncertainty": float(uncertainty)}
    return clean


def codex_cli_infer(report: str, *, executable: str = "codex",
                    timeout_seconds: int = 180,
                    subscription_oauth_confirmed: bool = False) -> dict[str, dict[str, object]]:
    """Invoke Codex CLI via stdin and read only its final agent message.

    No API endpoint or API-key argument is accepted. Do not call this until the
    operator has independently confirmed ChatGPT OAuth subscription auth.
    """
    if executable != "codex":
        raise LabelingError("Only the Codex CLI executable is supported.")
    if subscription_oauth_confirmed is not True:
        raise LabelingError("Confirm ChatGPT subscription OAuth before invoking Codex.")
    if not isinstance(report, str) or not report.strip():
        raise LabelingError("Report input is blank or invalid.")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 600:
        raise LabelingError("CLI timeout must be between 1 and 600 seconds.")
    target_list = "\n".join(f"- {column}" for column in LABEL_COLUMNS)
    prompt = (
        "Extract provisional binary findings for these targets:\n"
        f"TARGETS:\n{target_list}\n\n"
        "Return only one JSON object with each target mapped to {label:boolean, "
        "evidence:string, uncertainty:number from 0 to 1}. Evidence must be a "
        "short exact report excerpt (maximum 200 characters); use an empty "
        "excerpt when unsupported. Do not add keys.\n\nREPORT:\n" + report
    )
    try:
        process = _run_process(_codex_command(), prompt, timeout_seconds)
    except OSError:
        raise LabelingError("Codex CLI invocation failed or timed out.") from None
    if process.returncode != 0:
        raise LabelingError(
            f"Codex CLI exited with status {process.returncode}; "
            "stdout and stderr details are suppressed for privacy."
        )
    try:
        answer = _final_agent_message(process.stdout)
        return validate_label_result(json.loads(answer))
    except (ValueError, TypeError, json.JSONDecodeError):
        raise LabelingError("Codex CLI response was not valid structured label output.") from None


def _codex_command() -> list[str]:
    if os.name == "nt":
        # Keep report text on stdin; only this fixed command string reaches cmd.exe.
        return ["cmd.exe", "/d", "/s", "/c",
                "codex.cmd exec --json --sandbox read-only -"]
    return ["codex", "exec", "--json", "--sandbox", "read-only", "-"]


def _run_process(command: list[str], input_text: str,
                 timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    """Run one CLI process and reap it; on Windows, terminate its process tree."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                cleanup = subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                if cleanup.returncode != 0:
                    raise OSError
                stdout, stderr = process.communicate(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                _kill_and_reap(process)
                raise LabelingError(
                    "Codex CLI timed out and process tree cleanup failed."
                ) from None
        else:
            _kill_and_reap(process)
        raise LabelingError("Codex CLI invocation timed out.") from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _kill_and_reap(process: subprocess.Popen[str]) -> None:
    """Best-effort direct-child cleanup after process-tree cleanup is unavailable."""
    try:
        process.kill()
        process.communicate(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _final_agent_message(stdout: str) -> str:
    messages = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {}) if isinstance(event, dict) else {}
        if (isinstance(event, dict) and event.get("type") == "item.completed"
                and isinstance(item, dict) and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)):
            messages.append(item["text"])
    if not messages:
        raise LabelingError("Codex CLI did not return an agent message.")
    return messages[-1]


def run_pilot(rows: Iterable[Mapping[str, object]], *, gold_ids: Iterable[str] | None,
              max_records: int, output_path: str | Path,
              project_root: str | Path | None = None,
              subscription_oauth_confirmed: bool = False,
              skip_records: int = 0,
              infer: Callable[[str], object] = codex_cli_infer) -> int:
    """Run/resume a small local pilot; outputs must stay under ignored data/outputs."""
    if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= MAX_PILOT_RECORDS:
        raise LabelingError("max_records must be explicitly between 1 and 10.")
    if (isinstance(skip_records, bool) or not isinstance(skip_records, int)
            or skip_records < 0 or skip_records + max_records > MAX_PILOT_RECORDS):
        raise LabelingError("skip_records and max_records must total no more than 10.")
    if subscription_oauth_confirmed is not True:
        raise LabelingError("Confirm ChatGPT subscription OAuth before running a pilot.")
    root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
    destination = Path(output_path).resolve()
    try:
        destination.relative_to(root / "data" / "outputs")
    except ValueError:
        raise LabelingError("Pilot output must be stored under ignored data/outputs.") from None
    eligible = eligible_reports(rows, gold_ids=gold_ids)
    selected_end = skip_records + max_records
    selected = eligible[skip_records:selected_end]
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = _load_records(destination)
    completed = {record[STUDY_COLUMN] for record in records}
    current_ids = {str(row[STUDY_COLUMN]).strip() for row in eligible[:selected_end]}
    if not completed.issubset(current_ids):
        raise LabelingError("Existing output does not match the current eligible cohort.")
    for row in selected:
        study_id = str(row[STUDY_COLUMN]).strip()
        if study_id in completed:
            continue
        if infer is codex_cli_infer:
            raw_result = infer(str(row[REPORT_COLUMN]), subscription_oauth_confirmed=True)
        else:
            raw_result = infer(str(row[REPORT_COLUMN]))
        result = validate_label_result(raw_result)
        records.append({STUDY_COLUMN: study_id, "labels": result})
        _atomic_write_jsonl(destination, records)
        completed.add(study_id)
    if not destination.exists():
        _atomic_write_jsonl(destination, records)
    return len(records)


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {STUDY_COLUMN, "labels"}:
                raise ValueError
            study_id = record[STUDY_COLUMN]
            if not isinstance(study_id, str) or not study_id.strip() or study_id in seen:
                raise ValueError
            record["labels"] = validate_label_result(record["labels"])
            seen.add(study_id)
            records.append(record)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise LabelingError("Existing local pilot output is invalid; refusing to overwrite.") from None
    return records


def _atomic_write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".labels-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
