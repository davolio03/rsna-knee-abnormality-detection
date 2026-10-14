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
    SIWCResponsesInfer,
    safe_provider_diagnostics,
)
from rsna_knee.labels.subscription import (  # noqa: E402
    MAX_PILOT_RECORDS,
    LabelingError,
    eligible_reports,
    run_pilot,
)
from label_reports import _read_gold_ids, _read_rows  # noqa: E402


DEFAULT_CALLBACK_PORT = 1455
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "outputs" / "siwc-c2-report-labels.jsonl"
DEFAULT_DIAGNOSTICS = REPOSITORY_ROOT / "data" / "outputs" / "siwc-c2-pilot-diagnostics.jsonl"


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
                    output: TextIO | None = None) -> int:
    if confirm_siwc_oauth is not True or confirm_report_transfer is not True:
        raise LabelingError(
            "Both explicit SIWC OAuth and report-transfer confirmations are required."
        )
    if not profile_id.strip():
        raise LabelingError("A saved profile ID is required.", cause_code="auth_error")
    if not model.strip():
        raise LabelingError("An account-available model slug is required.", cause_code="model_error")
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
    if not diagnostics_path.exists():
        return
    try:
        entries = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, json.JSONDecodeError):
        raise LabelingError("Prior full-cohort diagnostics are invalid; refusing to resume.") from None
    if not all(isinstance(entry, dict) for entry in entries):
        raise LabelingError("Prior full-cohort diagnostics are invalid; refusing to resume.")
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


def run_label_cohort(*, train_csv: Path, gold_exclusions: Path,
                     profile_id: str, model: str,
                     output_path: Path, diagnostics_path: Path,
                     confirm_siwc_oauth: bool = False,
                     confirm_report_transfer: bool = False,
                     resume: bool = False,
                     output: TextIO | None = None) -> int:
    """Run all eligible non-Gold rows in safe, resumable windows of at most 14."""
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
    output.write(f"Eligible non-Gold records: {len(eligible)}.\n")
    if not eligible:
        output.write("No inference was started.\n")
        return 0

    # Authentication is constructed only after Gold exclusion and eligibility preflight.
    infer = SIWCResponsesInfer(
        authenticator=_create_authenticator(), profile_id=profile_id, model=model,
    )
    total = 0
    for offset in range(0, len(eligible), MAX_PILOT_RECORDS):
        batch_size = min(MAX_PILOT_RECORDS, len(eligible) - offset)
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
        )
    output.write(f"Checkpoint records: {total}.\n")
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
            output=output,
        )
    except KeyboardInterrupt:
        (output or sys.stderr).write(
            "Full-cohort labeling interrupted. The validated checkpoint is preserved; "
            "resume explicitly with the same output and diagnostics paths.\n"
        )
        return 130
    except (AuthError, LabelingError, OSError) as exc:
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
