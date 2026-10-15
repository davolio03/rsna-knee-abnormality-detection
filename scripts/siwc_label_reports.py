"""Sign in with ChatGPT and run an explicitly authorized local label pilot."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import webbrowser
from urllib.parse import urlsplit
from typing import Sequence, TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rsna_knee.labels.siwc_auth import (  # noqa: E402
    AuthError,
    KeyringCredentialStore,
    SIWCAuthenticator,
    load_or_create_host_id,
)
from rsna_knee.labels.siwc_responses import (  # noqa: E402
    DEFAULT_SIWC_MODEL,
    SIWC_PROMPT_VERSION,
    SIWCResponsesInfer,
    safe_provider_diagnostics,
)
from rsna_knee.labels.subscription import (  # noqa: E402
    MAX_PILOT_RECORDS,
    LabelingError,
    _RejectedEvidenceCapture,
    _DiagnosticsJournal,
    _load_records,
    eligible_reports,
    run_pilot,
)
from label_reports import _read_gold_ids, _read_rows  # noqa: E402


DEFAULT_CALLBACK_PORT = 1455
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "outputs" / "siwc-c2-report-labels.jsonl"
DEFAULT_DIAGNOSTICS = REPOSITORY_ROOT / "data" / "outputs" / "siwc-c2-pilot-diagnostics.jsonl"
REJECTED_EVIDENCE_CAPTURE_PATH = REPOSITORY_ROOT / "data" / "private" / "siwc-rejected-evidence.json"


class SystemTrustError(RuntimeError):
    """A safe, actionable failure to initialize system certificate trust."""


def _initialize_system_trust() -> None:
    """Use the OS certificate store for this CLI process's network clients."""
    try:
        import truststore
    except ImportError:
        raise SystemTrustError(
            "Secure system certificate support is unavailable. Install SIWC "
            "dependencies with: python -m pip install --use-feature=truststore "
            "-r requirements-siwc.txt"
        ) from None
    except Exception:
        raise SystemTrustError(
            "Secure system certificate support could not be loaded."
        ) from None

    try:
        truststore.inject_into_ssl()
    except Exception:
        raise SystemTrustError(
            "Secure system certificate support could not be initialized."
        ) from None


def _callback_handler_type(port: int):
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            self.server.callback_url = f"http://127.0.0.1:{port}{self.path}"
            body = b"Sign-in callback received. You may close this browser tab."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            # The request line contains the one-time authorization code and state.
            return

    return CallbackHandler


def _create_authenticator(redirect_uri: str =
                          f"http://127.0.0.1:{DEFAULT_CALLBACK_PORT}/auth/callback"):
    host_id_path = REPOSITORY_ROOT / "data" / "private" / ".siwc-host-id"
    return SIWCAuthenticator(
        store=KeyringCredentialStore(),
        host_id=load_or_create_host_id(host_id_path),
        redirect_uri=redirect_uri,
    )


def sign_in(*, profile_id: str | None = None,
            callback_port: int = DEFAULT_CALLBACK_PORT,
            timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
            output: TextIO | None = None) -> int:
    if isinstance(callback_port, bool) or not isinstance(callback_port, int) or not 1024 <= callback_port <= 65535:
        raise LabelingError("Callback port must be between 1024 and 65535.", cause_code="auth_error")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 600:
        raise LabelingError("Sign-in timeout must be between 1 and 600 seconds.", cause_code="auth_error")
    output = output or sys.stdout
    redirect_uri = f"http://127.0.0.1:{callback_port}/auth/callback"
    server = HTTPServer(("127.0.0.1", callback_port), _callback_handler_type(callback_port))
    server.timeout = timeout_seconds
    try:
        authenticator = _create_authenticator(redirect_uri)
        request = authenticator.begin_sign_in(profile_id)
        if not webbrowser.open(request.authorization_url, new=1):
            raise AuthError("The sign-in browser could not be opened.")
        server.handle_request()
        callback_url = getattr(server, "callback_url", None)
        if not isinstance(callback_url, str):
            raise AuthError("The sign-in callback was not received before timeout.")
        profile = authenticator.complete_sign_in(callback_url)
        output.write(
            f"Sign-in complete. Profile ID: {profile.profile_id}. "
            f"ChatGPT plan use: {'enabled' if profile.plan_usage_enabled else 'disabled'}.\n"
        )
        return 0
    finally:
        # Do not retain the callback URL/code in server state after validation.
        server.callback_url = None
        server.server_close()


def list_models(*, profile_id: str, output: TextIO | None = None) -> int:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise LabelingError("A saved profile ID is required.", cause_code="auth_error")
    output = output or sys.stdout
    infer = SIWCResponsesInfer(
        authenticator=_create_authenticator(),
        profile_id=profile_id,
        model=DEFAULT_SIWC_MODEL,
    )
    models = infer.list_available_models()
    if not models:
        output.write("No models are currently listed for this account.\n")
        return 0
    output.write("Account-available models (use a listed slug with the label command):\n")
    for model in models:
        output.write(f"- {model['display_name']} ({model['slug']})\n")
    if not any(model["slug"] == DEFAULT_SIWC_MODEL for model in models):
        output.write(
            f"Default model {DEFAULT_SIWC_MODEL} is not listed for this account; "
            "select another listed slug.\n"
        )
    return 0


def run_label_pilot(*, train_csv: Path, gold_exclusions: Path,
                    profile_id: str, model: str,
                    max_records: int, skip_records: int = 0,
                    confirm_siwc_oauth: bool = False,
                    confirm_report_transfer: bool = False,
                    output_path: Path = DEFAULT_OUTPUT,
                    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
                    capture_rejected_evidence: bool = False,
                    output: TextIO | None = None) -> int:
    if confirm_siwc_oauth is not True or confirm_report_transfer is not True:
        raise LabelingError(
            "Both explicit SIWC OAuth and report-transfer confirmations are required."
        )
    if not profile_id.strip():
        raise LabelingError("A saved profile ID is required.", cause_code="auth_error")
    if not model.strip():
        raise LabelingError("An account-available model slug is required.", cause_code="model_error")
    if type(capture_rejected_evidence) is not bool:
        raise LabelingError("Rejected-evidence capture must be an explicit boolean.")
    if (isinstance(max_records, bool) or not isinstance(max_records, int)
            or not 1 <= max_records <= MAX_PILOT_RECORDS):
        raise LabelingError("max_records must be explicitly between 1 and 14.")
    if (isinstance(skip_records, bool) or not isinstance(skip_records, int)
            or skip_records < 0 or skip_records + max_records > MAX_PILOT_RECORDS):
        raise LabelingError("skip_records and max_records must total no more than 14.")
    legacy_paths = {
        (REPOSITORY_ROOT / "data" / "outputs" / "siwc-report-labels.jsonl").resolve(),
        (REPOSITORY_ROOT / "data" / "outputs" / "siwc-pilot-diagnostics.jsonl").resolve(),
    }
    if Path(output_path).resolve() in legacy_paths or Path(diagnostics_path).resolve() in legacy_paths:
        raise LabelingError(
            "The legacy SIWC pilot path is protected to prevent mixing prompt strategies. "
            "Use --output data/outputs/siwc-c2-report-labels.jsonl and "
            "--diagnostics data/outputs/siwc-c2-pilot-diagnostics.jsonl instead."
        )
    rejected_evidence_capture = None
    if capture_rejected_evidence:
        rejected_evidence_capture = _RejectedEvidenceCapture(REPOSITORY_ROOT)
        if rejected_evidence_capture.path != REJECTED_EVIDENCE_CAPTURE_PATH:
            raise LabelingError("Rejected-evidence capture destination is invalid.")
        if not rejected_evidence_capture.prepare():
            raise LabelingError(
                "Rejected-evidence capture slot is unsafe or already exists."
            )
    rows = _read_rows(train_csv)
    gold_ids = _read_gold_ids(gold_exclusions)
    infer = SIWCResponsesInfer(
        authenticator=_create_authenticator(),
        profile_id=profile_id,
        model=model,
    )
    total = run_pilot(
        rows,
        gold_ids=gold_ids,
        max_records=max_records,
        skip_records=skip_records,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        project_root=REPOSITORY_ROOT,
        # The CLI requires explicit provider and transfer confirmations first.
        subscription_oauth_confirmed=True,
        infer=infer,
        rejected_evidence_capture=rejected_evidence_capture,
    )
    output = output or sys.stdout
    output.write(f"Local SIWC pilot output records: {total}\n")
    return 0


def _cohort_paths(output_path: Path, diagnostics_path: Path, *, resume: bool) -> tuple[Path, Path]:
    root = REPOSITORY_ROOT.resolve()
    outputs = (root / "data" / "outputs").resolve()
    raw_output = Path(output_path)
    raw_diagnostics = Path(diagnostics_path)
    output = (raw_output if raw_output.is_absolute() else root / raw_output).resolve()
    diagnostics = (raw_diagnostics if raw_diagnostics.is_absolute() else root / raw_diagnostics).resolve()
    if output == diagnostics:
        raise LabelingError("Full-cohort output and diagnostics paths must be distinct.")
    for path in (output, diagnostics):
        try:
            path.relative_to(outputs)
        except ValueError:
            raise LabelingError("Full-cohort files must stay under ignored data/outputs.") from None
    protected = {
        DEFAULT_OUTPUT.resolve(),
        DEFAULT_DIAGNOSTICS.resolve(),
        (outputs / "siwc-report-labels.jsonl").resolve(),
        (outputs / "siwc-pilot-diagnostics.jsonl").resolve(),
    }
    if output in protected or diagnostics in protected:
        raise LabelingError("Full-cohort paths must be fresh and separate from prior SIWC outputs.")
    if resume:
        if not diagnostics.is_file() or (output.exists() and not output.is_file()):
            raise LabelingError("Resume requires the prior diagnostics journal and regular checkpoint path.")
    elif output.exists() or diagnostics.exists():
        raise LabelingError(
            "A new full-cohort run requires fresh output and diagnostics paths; use --resume only for its checkpoint."
        )
    return output, diagnostics


def _reject_usage_limited_resume(diagnostics_path: Path) -> None:
    """Do not turn a provider usage denial into an automatic retry on resume."""
    entries = _read_cohort_journal(diagnostics_path)
    usage_denial_codes = {
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_usage_unavailable",
    }
    if any(
        entry.get("event") == "attempt_failed"
        and (
            entry.get("cause_code") == "quota_error"
            or entry.get("http_status") == 429
            or entry.get("provider_error_code") in usage_denial_codes
        )
        for entry in entries
    ):
        raise LabelingError(
            "A prior provider usage denial blocks automatic resume; wait for provider guidance before a new run."
        )


def _read_cohort_journal(diagnostics_path: Path) -> list[dict[str, object]]:
    if not diagnostics_path.exists():
        return []
    try:
        entries = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, json.JSONDecodeError):
        raise LabelingError("Prior full-cohort diagnostics are invalid; refusing to resume.") from None
    if not all(isinstance(entry, dict) for entry in entries):
        raise LabelingError("Prior full-cohort diagnostics are invalid; refusing to resume.")
    return entries


def _validate_cohort_limit(limit: int | None) -> None:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise LabelingError("Cohort limit must be a positive integer.")


def _bind_cohort_scope(diagnostics_path: Path, output_path: Path, *, limit: int,
                       eligible_count: int, resume: bool, explicit_limit: bool) -> None:
    """Bind the journal to one total-checkpoint target; only legacy journals migrate."""
    entries = _read_cohort_journal(diagnostics_path)
    markers = [entry for entry in entries if entry.get("event") == "cohort_scope"]
    if markers:
        if len(markers) != 1:
            raise LabelingError("Stored cohort scope is invalid; refusing to resume.")
        marker = markers[0]
        if (set(marker) != {"event", "limit", "eligible_count"}
                or type(marker.get("limit")) is not int or marker["limit"] <= 0
                or type(marker.get("eligible_count")) is not int or marker["eligible_count"] <= 0):
            raise LabelingError("Stored cohort scope is invalid; refusing to resume.")
        if marker["limit"] != limit or marker["eligible_count"] != eligible_count:
            raise LabelingError("Requested limit does not match the stored cohort limit.")
        return
    if resume and not explicit_limit:
        raise LabelingError("A legacy cohort resume requires an explicit limit before migration.")
    # Old run01 journals have no scope marker. A caller may bind them once after
    # the checkpoint has been proven to be a prefix of the selected eligible rows.
    journal = _DiagnosticsJournal(diagnostics_path, REPOSITORY_ROOT.resolve(), output_path)
    journal.append({"event": "cohort_scope", "limit": limit, "eligible_count": eligible_count})


def _validate_prompt_boundary(diagnostics_path: Path, *, checkpoint_count: int) -> bool:
    """Return whether the current revision needs its one-time cohort marker."""
    markers = [entry for entry in _read_cohort_journal(diagnostics_path)
               if entry.get("event") == "prompt_boundary"]
    if not markers:
        return True
    # This is the complete supported history, not a dynamic prompt registry.
    revisions = ("c2-literal-evidence-v1", SIWC_PROMPT_VERSION)
    prior_revision = -1
    prior_offset = 0
    for marker in markers:
        version = marker.get("prompt_version")
        offset = marker.get("start_offset")
        if (set(marker) != {"event", "prompt_version", "start_offset"}
                or version not in revisions or type(offset) is not int
                or not prior_offset <= offset <= checkpoint_count):
            raise LabelingError("Stored prompt boundary is incompatible; refusing to resume.")
        revision = revisions.index(version)
        if revision <= prior_revision:
            raise LabelingError("Stored prompt boundary is invalid; refusing to resume.")
        prior_revision, prior_offset = revision, offset
    return markers[-1]["prompt_version"] != SIWC_PROMPT_VERSION


def _record_prompt_boundary(diagnostics_path: Path, output_path: Path, *,
                            checkpoint_count: int) -> None:
    journal = _DiagnosticsJournal(diagnostics_path, REPOSITORY_ROOT.resolve(), output_path)
    journal.append({"event": "prompt_boundary", "prompt_version": SIWC_PROMPT_VERSION,
                    "start_offset": checkpoint_count})


def run_label_cohort(*, train_csv: Path, gold_exclusions: Path,
                     profile_id: str, model: str,
                     output_path: Path, diagnostics_path: Path,
                     confirm_siwc_oauth: bool = False,
                     confirm_report_transfer: bool = False,
                     resume: bool = False,
                     limit: int | None = None,
                     capture_rejected_evidence: bool = False,
                     output: TextIO | None = None) -> int:
    """Run an eligible prefix in resumable windows of at most 14."""
    _validate_cohort_limit(limit)
    if type(capture_rejected_evidence) is not bool:
        raise LabelingError("Rejected-evidence capture must be an explicit boolean.")
    if capture_rejected_evidence and limit is None:
        raise LabelingError(
            "Rejected-evidence capture requires an explicit positive --limit."
        )
    if confirm_siwc_oauth is not True or confirm_report_transfer is not True:
        raise LabelingError(
            "Both explicit SIWC OAuth and report-transfer confirmations are required."
        )
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise LabelingError("A saved profile ID is required.", cause_code="auth_error")
    if model != DEFAULT_SIWC_MODEL:
        raise LabelingError("Full-cohort labeling is fixed to the frozen Candidate 2 model.",
                            cause_code="model_error")
    if type(resume) is not bool:
        raise LabelingError("Resume must be an explicit boolean.")
    output_path, diagnostics_path = _cohort_paths(
        output_path, diagnostics_path, resume=resume,
    )
    if resume:
        _reject_usage_limited_resume(diagnostics_path)

    rows = list(_read_rows(train_csv))
    gold_ids = _read_gold_ids(gold_exclusions)
    eligible = eligible_reports(rows, gold_ids=gold_ids)
    output = output or sys.stdout
    target_total = len(eligible) if limit is None else limit
    if target_total > len(eligible):
        raise LabelingError("Cohort limit exceeds the eligible non-Gold record count.")
    if not eligible:
        output.write("Eligible non-Gold records: 0.\nNo inference was started.\n")
        output.flush()
        return 0

    checkpoint_records = _load_records(output_path) if resume else []
    if len(checkpoint_records) > target_total:
        raise LabelingError("Existing checkpoint exceeds the requested cohort limit.")
    expected_prefix = [str(row["StudyInstanceUID"]).strip()
                       for row in eligible[:len(checkpoint_records)]]
    cached_ids = [record["StudyInstanceUID"] for record in checkpoint_records]
    if cached_ids != expected_prefix:
        raise LabelingError("Existing checkpoint is not a prefix of the requested eligible cohort.")

    needs_prompt_boundary = _validate_prompt_boundary(
        diagnostics_path, checkpoint_count=len(checkpoint_records),
    )
    rejected_evidence_capture = None
    if capture_rejected_evidence:
        rejected_evidence_capture = _RejectedEvidenceCapture(REPOSITORY_ROOT)
        if rejected_evidence_capture.path != REJECTED_EVIDENCE_CAPTURE_PATH:
            raise LabelingError("Rejected-evidence capture destination is invalid.")
        if not rejected_evidence_capture.prepare():
            raise LabelingError(
                "Rejected-evidence capture slot is unsafe or already exists."
            )

    _bind_cohort_scope(
        diagnostics_path, output_path, limit=target_total,
        eligible_count=len(eligible), resume=resume, explicit_limit=limit is not None,
    )
    if needs_prompt_boundary:
        _record_prompt_boundary(diagnostics_path, output_path,
                                checkpoint_count=len(checkpoint_records))
    output.write(f"Eligible non-Gold records: {len(eligible)}.\n")
    output.write(f"Checkpoint records: {len(checkpoint_records)}/{target_total}.\n")
    output.flush()

    # Authentication is constructed only after Gold exclusion and eligibility preflight.
    infer = SIWCResponsesInfer(
        authenticator=_create_authenticator(), profile_id=profile_id, model=model,
    )
    total = len(checkpoint_records)
    for offset in range(0, target_total, MAX_PILOT_RECORDS):
        batch_size = min(MAX_PILOT_RECORDS, target_total - offset)
        try:
            total = run_pilot(
                rows,
                gold_ids=gold_ids,
                max_records=batch_size,
                skip_records=offset,
                output_path=output_path,
                diagnostics_path=diagnostics_path,
                project_root=REPOSITORY_ROOT,
                subscription_oauth_confirmed=True,
                allow_full_cohort_checkpoint=True,
                infer=infer,
                rejected_evidence_capture=rejected_evidence_capture,
            )
        except (LabelingError, KeyboardInterrupt):
            try:
                total = len(_load_records(output_path))
            except LabelingError:
                pass
            output.write(f"Checkpoint records: {total}/{target_total}.\n")
            output.flush()
            raise
        output.write(f"Checkpoint records: {total}/{target_total}.\n")
        output.flush()
    if total != target_total:
        raise LabelingError("Checkpoint did not reach the requested cohort limit.",
                            stage="checkpoint_validation")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    signin = subparsers.add_parser("signin", help="Add or reauthorize a local ChatGPT account.")
    signin.add_argument("--profile-id", help="Existing profile ID to reauthorize; omit to add an account.")
    signin.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT)
    signin.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)

    models = subparsers.add_parser("models", help="List models visible to a saved account.")
    models.add_argument("--profile-id", required=True)

    label = subparsers.add_parser("label", help="Run a bounded, explicitly authorized local label pilot.")
    label.add_argument("--train-csv", type=Path, required=True)
    label.add_argument("--gold-exclusions", type=Path, required=True)
    label.add_argument("--profile-id", required=True)
    label.add_argument("--model", required=True,
                       help="Exact account catalog slug, obtained from the models command.")
    label.add_argument("--max-records", type=int, required=True,
                       help="Explicit batch size (1-14); combined skip and batch limit is 14.")
    label.add_argument("--skip-records", type=int, default=0)
    label.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    label.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    label.add_argument(
        "--capture-rejected-evidence", action="store_true",
        help="Opt in to one bounded local capture of rejected fresh evidence.",
    )
    label.add_argument(
        "--confirm-siwc-oauth", action="store_true", required=True,
        help="Explicitly confirm this selected ChatGPT account has granted plan-use permission.",
    )
    label.add_argument(
        "--confirm-report-transfer", action="store_true", required=True,
        help="Explicitly authorize sending eligible non-Gold report text to OpenAI for this run.",
    )

    cohort = subparsers.add_parser(
        "label-all", help="Label all eligible non-Gold rows in safe batches of at most 14."
    )
    cohort.add_argument("--train-csv", type=Path, required=True)
    cohort.add_argument("--gold-exclusions", type=Path, required=True)
    cohort.add_argument("--profile-id", required=True)
    cohort.add_argument("--model", required=True)
    cohort.add_argument("--output", type=Path, required=True,
                        help="Fresh ignored checkpoint path; preserve it for explicit resume.")
    cohort.add_argument("--diagnostics", type=Path, required=True,
                        help="Distinct fresh payload-free diagnostics path.")
    cohort.add_argument("--resume", action="store_true",
                        help="Resume only the matching run using its existing two paths.")
    cohort.add_argument("--limit", type=int,
                        help="Total checkpoint target from the eligible prefix, including cached records.")
    cohort.add_argument(
        "--capture-rejected-evidence", action="store_true",
        help="Opt in to one bounded local capture of rejected fresh evidence (requires --limit).",
    )
    cohort.add_argument(
        "--confirm-siwc-oauth", action="store_true", required=True,
        help="Confirm the selected account has granted SIWC plan use.",
    )
    cohort.add_argument(
        "--confirm-report-transfer", action="store_true", required=True,
        help="Confirm this authorized run may send eligible non-Gold report text to OpenAI.",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, output: TextIO | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--access-token" or arg.startswith("--access-token=")
           for arg in raw_args):
        parser.error("Access tokens must not be passed on the command line.")
    args = parser.parse_args(raw_args)
    try:
        _initialize_system_trust()
        if args.command == "signin":
            return sign_in(
                profile_id=args.profile_id,
                callback_port=args.callback_port,
                timeout_seconds=args.timeout_seconds,
                output=output,
            )
        if args.command == "models":
            return list_models(profile_id=args.profile_id, output=output)
        if args.command == "label-all":
            return run_label_cohort(
                train_csv=args.train_csv,
                gold_exclusions=args.gold_exclusions,
                profile_id=args.profile_id,
                model=args.model,
                output_path=args.output,
                diagnostics_path=args.diagnostics,
                confirm_siwc_oauth=args.confirm_siwc_oauth,
                confirm_report_transfer=args.confirm_report_transfer,
                resume=args.resume,
                limit=args.limit,
                capture_rejected_evidence=args.capture_rejected_evidence,
                output=output,
            )
        return run_label_pilot(
            train_csv=args.train_csv,
            gold_exclusions=args.gold_exclusions,
            profile_id=args.profile_id,
            model=args.model,
            max_records=args.max_records,
            skip_records=args.skip_records,
            confirm_siwc_oauth=args.confirm_siwc_oauth,
            confirm_report_transfer=args.confirm_report_transfer,
            output_path=args.output,
            diagnostics_path=args.diagnostics,
            capture_rejected_evidence=args.capture_rejected_evidence,
            output=output,
        )
    except KeyboardInterrupt:
        (output or sys.stderr).write(
            "Full-cohort labeling interrupted. The validated checkpoint is preserved; "
            "resume explicitly with the same output and diagnostics paths.\n"
        )
        return 130
    except (AuthError, LabelingError, OSError, SystemTrustError) as exc:
        # Auth and runner exceptions contain only fixed/sanitized diagnostics.
        message = str(exc)
        if isinstance(exc, LabelingError):
            diagnostics = safe_provider_diagnostics(exc)
            # The shared runner may already have included diagnostics in its
            # safe error message. Rebuild that suffix from validated fields so
            # it is rendered once, and never trust stale mutable attributes.
            message = message.split(" Provider diagnostics:", 1)[0]
            if diagnostics:
                message = f"{message} Provider diagnostics: {diagnostics}."
        parser.error(message)


if __name__ == "__main__":
    raise SystemExit(main())
