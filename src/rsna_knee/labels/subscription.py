"""Fail-closed local labeling utilities for a bounded Codex CLI pilot.

The CLI adapter uses the Codex application command, not the OpenAI API. The
operator must verify subscription-backed ChatGPT OAuth before any real run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping


LABEL_COLUMNS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
STUDY_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"
MAX_PILOT_RECORDS = 14
EXPECTED_GOLD_STUDIES = 58
_MAX_EVIDENCE_CHARS = 200
_DEFAULT_DIAGNOSTICS = Path("data/outputs/codex-pilot-diagnostics.jsonl")
_CAUSE_EXPLANATIONS = {
    "configuration_error": "The Codex CLI reported a configuration problem.",
    "arguments_error": "The Codex CLI rejected an invocation argument.",
    "auth_error": "The Codex CLI reported an authentication problem.",
    "quota_error": "The Codex CLI reported a quota or rate-limit problem.",
    "network_error": "The Codex CLI reported a network transport problem.",
    "model_error": "The Codex CLI reported a model or structured-response problem.",
    "mcp_error": "The Codex CLI reported an MCP tool or server problem.",
    "sandbox_error": "The Codex CLI reported a sandbox or permission problem.",
    "timeout": "The Codex CLI invocation exceeded its time limit.",
    "spawn_error": "The Codex CLI process could not be started.",
    "invalid_response": "The Codex CLI response was not valid structured label output.",
    "unknown": "The failure cause could not be safely classified.",
}


_FAILURE_PATTERNS = (
    ("sandbox_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:sandbox(?:ing)?\s+(?:denied|blocked|failed)|permission denied)\b", re.I)),
    ("configuration_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:invalid config(?:uration)?|configuration error|failed to parse config|unknown config(?:uration)? key)\b", re.I)),
    ("arguments_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:unrecognized arguments?|unknown option|unexpected argument|invalid argument)\b", re.I)),
    ("auth_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:401\s+unauthorized|unauthorized|authentication required|not authenticated|login required)\b", re.I)),
    ("quota_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:429\b|rate limit(?:ed)?|quota exceeded)\b", re.I)),
    ("network_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:connection refused|network unreachable|name or service not known|temporary failure in name resolution|tls handshake|connection reset)\b", re.I)),
    ("model_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:model not found|unknown model|unsupported model)\b", re.I)),
    ("mcp_error", re.compile(r"^\s*(?:error|fatal|codex(?: cli)? error)\s*:?\s*(?:mcp (?:server|tool).{0,30}(?:error|failed|unavailable)|mcp error)\b", re.I)),
)


class LabelingError(ValueError):
    """A safe-to-display validation or execution error (never includes payloads)."""

    def __init__(self, message: str, *, stage: str = "validation",
                 cause_code: str = "unknown", exit_status: int | None = None,
                 errno_value: int | None = None):
        super().__init__(message)
        self.stage = stage
        self.cause_code = cause_code if cause_code in _CAUSE_EXPLANATIONS else "unknown"
        self.exit_status = exit_status if type(exit_status) is int else None
        self.errno_value = errno_value if type(errno_value) is int else None


def _classify_cli_failure(stdout: str, stderr: str) -> str:
    """Classify only fixed allowlisted patterns; never return CLI text."""
    for text in (stderr, stdout):
        for line in text.splitlines():
            candidate = line
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    error = event.get("error")
                    if isinstance(error, dict):
                        code = error.get("code")
                        if isinstance(code, str) and code in {
                            "unauthorized", "authentication_required", "rate_limit_exceeded",
                            "quota_exceeded", "model_not_found", "mcp_error",
                            "sandbox_denied", "invalid_arguments", "invalid_config",
                        }:
                            return {
                                "unauthorized": "auth_error", "authentication_required": "auth_error",
                                "rate_limit_exceeded": "quota_error", "quota_exceeded": "quota_error",
                                "model_not_found": "model_error", "mcp_error": "mcp_error",
                                "sandbox_denied": "sandbox_error", "invalid_arguments": "arguments_error",
                                "invalid_config": "configuration_error",
                            }[code]
                    message = event.get("message")
                    if isinstance(message, str):
                        candidate = message
            except (json.JSONDecodeError, TypeError):
                pass
            for cause, pattern in _FAILURE_PATTERNS:
                if pattern.search(candidate):
                    return cause
    return "unknown"


class _DiagnosticsJournal:
    """Append only fixed-schema, payload-free local attempt diagnostics."""

    def __init__(self, path: str | Path, root: Path, output: Path):
        outputs = root / "data" / "outputs"
        if outputs.exists() and outputs.resolve() != outputs:
            raise LabelingError("Diagnostics journal path is unsafe.", stage="journal_validation")
        raw_path = Path(path).absolute()
        try:
            raw_path.relative_to(outputs.absolute())
        except ValueError:
            raise LabelingError("Diagnostics journal must be under data/outputs.", stage="journal_validation") from None
        if raw_path == output.absolute() or raw_path.resolve() == output.resolve():
            raise LabelingError("Diagnostics journal cannot alias the labels output.", stage="journal_validation")
        if raw_path.is_symlink():
            raise LabelingError("Diagnostics journal path is unsafe.", stage="journal_validation")
        self.path = raw_path
        self.root = outputs.resolve()

    def append(self, record: dict[str, object]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = self.path.parent.resolve()
            resolved_parent.relative_to(self.root)
            if self.path.is_symlink():
                raise OSError
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            flags |= getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_BINARY", 0)
            fd = os.open(self.path, flags, 0o600)
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
                written = 0
                while written < len(payload):
                    written += os.write(fd, payload[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, ValueError):
            raise LabelingError("Diagnostics journal could not be safely written.",
                                stage="journal_persist") from None


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
                    subscription_oauth_confirmed: bool = False,
                    candidate_id: str | None = None,
                    transfer_prerequisites_confirmed: bool = False) -> dict[str, dict[str, object]]:
    """Invoke Codex CLI via stdin and read only its final agent message.

    No API endpoint or API-key argument is accepted. Do not call this until the
    operator has independently confirmed ChatGPT OAuth subscription auth.
    """
    if executable != "codex":
        raise LabelingError("Only the Codex CLI executable is supported.")
    if subscription_oauth_confirmed is not True:
        raise LabelingError("Confirm ChatGPT subscription OAuth before invoking Codex.")
    if candidate_id is not None and transfer_prerequisites_confirmed is not True:
        raise LabelingError(
            "Confirm report-transfer prerequisites before invoking a comparison candidate."
        )
    if not isinstance(report, str) or not report.strip():
        raise LabelingError("Report input is blank or invalid.")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 600:
        raise LabelingError("CLI timeout must be between 1 and 600 seconds.")
    if candidate_id is not None:
        from rsna_knee.labels.prompt_comparison import build_candidate_prompt

        try:
            prompt = build_candidate_prompt(candidate_id, report)
        except ValueError as exc:
            raise LabelingError(str(exc)) from None
    else:
        target_list = "\n".join(f"- {column}" for column in LABEL_COLUMNS)
        prompt = (
            "Extract provisional binary findings for these targets:\n"
            f"TARGETS:\n{target_list}\n\n"
            "Return only one JSON object with each target mapped to {label:boolean, "
            "evidence:string, uncertainty:number from 0 to 1}. Evidence must be a "
            "short exact report excerpt (maximum 200 characters); use an empty "
            "excerpt when unsupported. Do not add keys.\n\nREPORT:\n" + report
        )
    process = _run_process(_codex_command(), prompt, timeout_seconds)
    if process.returncode != 0:
        cause = _classify_cli_failure(process.stdout, process.stderr)
        raise LabelingError(
            f"Codex CLI exited with status {process.returncode}; "
            f"{_CAUSE_EXPLANATIONS[cause]} stdout and stderr details are suppressed for privacy.",
            stage="process_exit", cause_code=cause, exit_status=process.returncode,
        )
    try:
        answer = _final_agent_message(process.stdout)
    except LabelingError as exc:
        raise LabelingError("Codex CLI response was not valid structured label output.",
                            stage="response_parse", cause_code="invalid_response") from None
    try:
        return validate_label_result(json.loads(answer))
    except (ValueError, TypeError, json.JSONDecodeError):
        raise LabelingError("Codex CLI response was not valid structured label output.",
                            stage="response_validation", cause_code="invalid_response") from None


def _codex_command() -> list[str]:
    if os.name == "nt":
        # Keep report text on stdin; only this fixed command string reaches cmd.exe.
        return ["cmd.exe", "/d", "/s", "/c",
                "codex.cmd exec --json --sandbox read-only --model gpt-6-luna "
                "--config model_reasoning_effort=high -"]
    return ["codex", "exec", "--json", "--sandbox", "read-only",
            "--model", "gpt-6-luna", "--config",
            "model_reasoning_effort=high", "-"]


def _run_process(command: list[str], input_text: str,
                 timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    """Run one CLI process and reap it; on Windows, terminate its process tree."""
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
        )
    except OSError as exc:
        raise LabelingError("Codex CLI process could not be started.", stage="process_spawn",
                            cause_code="spawn_error", errno_value=exc.errno) from None
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
                    "Codex CLI timed out and process tree cleanup failed.",
                    stage="process_cleanup", cause_code="timeout",
                ) from None
        else:
            _kill_and_reap(process)
        raise LabelingError("Codex CLI invocation timed out.", stage="process_wait",
                            cause_code="timeout") from None
    except OSError as exc:
        _kill_and_reap(process)
        raise LabelingError("Codex CLI process communication failed.", stage="process_wait",
                            errno_value=exc.errno) from None
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
        raise LabelingError("Codex CLI did not return an agent message.",
                            stage="response_parse", cause_code="invalid_response")
    return messages[-1]


def run_pilot(rows: Iterable[Mapping[str, object]], *, gold_ids: Iterable[str] | None,
              max_records: int, output_path: str | Path,
              diagnostics_path: str | Path = _DEFAULT_DIAGNOSTICS,
              project_root: str | Path | None = None,
              subscription_oauth_confirmed: bool = False,
              skip_records: int = 0,
              infer: Callable[[str], object] = codex_cli_infer) -> int:
    """Run/resume a small local pilot; outputs must stay under ignored data/outputs."""
    if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= MAX_PILOT_RECORDS:
        raise LabelingError("max_records must be explicitly between 1 and 14.")
    if (isinstance(skip_records, bool) or not isinstance(skip_records, int)
            or skip_records < 0 or skip_records + max_records > MAX_PILOT_RECORDS):
        raise LabelingError("skip_records and max_records must total no more than 14.")
    if subscription_oauth_confirmed is not True:
        raise LabelingError("Confirm ChatGPT subscription OAuth before running a pilot.")
    root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
    destination = Path(output_path).resolve()
    try:
        destination.relative_to(root / "data" / "outputs")
    except ValueError:
        raise LabelingError("Pilot output must be stored under ignored data/outputs.") from None
    journal_path = Path(diagnostics_path)
    if not journal_path.is_absolute():
        journal_path = root / journal_path
    journal = _DiagnosticsJournal(journal_path, root, destination)
    eligible = eligible_reports(rows, gold_ids=gold_ids)
    selected_end = skip_records + max_records
    selected = eligible[skip_records:selected_end]
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = _load_records(destination)
    completed = {record[STUDY_COLUMN] for record in records}
    current_ids = {str(row[STUDY_COLUMN]).strip() for row in eligible[:selected_end]}
    if not completed.issubset(current_ids):
        raise LabelingError("Existing output does not match the current eligible cohort.")
    progress = {"started": 0, "completed": 0, "failed": 0}
    for selected_index, row in enumerate(selected):
        study_id = str(row[STUDY_COLUMN]).strip()
        if study_id in completed:
            continue
        attempt_offset = skip_records + selected_index
        started_at = time.monotonic()
        progress["started"] += 1
        common = {"attempt_offset": attempt_offset, "progress": dict(progress)}
        journal.append({"event": "attempt_started", **common})
        try:
            if infer is codex_cli_infer:
                raw_result = infer(str(row[REPORT_COLUMN]), subscription_oauth_confirmed=True)
            else:
                raw_result = infer(str(row[REPORT_COLUMN]))
            result = validate_label_result(raw_result)
            records.append({STUDY_COLUMN: study_id, "labels": result})
            try:
                _atomic_write_jsonl(destination, records)
            except OSError:
                raise LabelingError("Local pilot output could not be safely persisted.",
                                    stage="output_persist") from None
            completed.add(study_id)
        except Exception as exc:
            progress["failed"] += 1
            elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
            cause = getattr(exc, "cause_code", "unknown")
            if cause not in _CAUSE_EXPLANATIONS:
                cause = "unknown"
            stage = getattr(exc, "stage", "inference")
            if stage not in {"process_spawn", "process_wait", "process_cleanup", "process_exit",
                             "response_parse", "response_validation", "inference", "output_persist"}:
                stage = "inference"
            record = {"event": "attempt_failed", "attempt_offset": attempt_offset,
                      "stage": stage, "cause_code": cause,
                      "cause_explanation": _CAUSE_EXPLANATIONS[cause],
                      "elapsed_ms": elapsed_ms, "progress": dict(progress)}
            exit_status = getattr(exc, "exit_status", None)
            errno_value = getattr(exc, "errno_value", None)
            if type(exit_status) is int:
                record["exit_status"] = exit_status
            if type(errno_value) is int:
                record["errno"] = errno_value
            try:
                journal.append(record)
            except LabelingError:
                raise LabelingError(
                    f"Pilot stopped at offset {attempt_offset}; diagnostics journal persistence failed. "
                    f"Progress: started={progress['started']}, completed={progress['completed']}, "
                    f"failed={progress['failed']}.", stage="journal_persist",
                ) from None
            raise LabelingError(
                f"Pilot stopped at offset {attempt_offset}: {_CAUSE_EXPLANATIONS[cause]} "
                f"Progress: started={progress['started']}, completed={progress['completed']}, "
                f"failed={progress['failed']}.", stage=stage, cause_code=cause,
                exit_status=exit_status, errno_value=errno_value,
            ) from None
        progress["completed"] += 1
        elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
        try:
            journal.append({"event": "attempt_completed", "attempt_offset": attempt_offset,
                            "stage": "complete", "elapsed_ms": elapsed_ms,
                            "progress": dict(progress)})
        except LabelingError:
            raise LabelingError(
                f"Pilot stopped after offset {attempt_offset}; diagnostics journal persistence failed. "
                f"Progress: started={progress['started']}, completed={progress['completed']}, "
                f"failed={progress['failed']}.", stage="journal_persist",
            ) from None
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
