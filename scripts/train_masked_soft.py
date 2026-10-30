"""Plan or explicitly execute the opt-in masked soft-label training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="UTF-8 JSON config; planning is the default")
    parser.add_argument("--execute", action="store_true", help="explicitly opt in to model training")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="explicit checkpoint file from a prior completed execution")
    parser.add_argument("--resume-checkpoint-sha256",
                        help="expected SHA-256 of the explicitly selected checkpoint")
    parser.add_argument("--max-epochs-per-run", type=int)
    parser.add_argument("--max-runtime-seconds", type=int)
    args = parser.parse_args(argv)
    try:
        config_data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if args.resume_checkpoint is not None:
            config_data["resume_checkpoint"] = str(args.resume_checkpoint)
        if args.resume_checkpoint_sha256 is not None:
            config_data["resume_checkpoint_sha256"] = args.resume_checkpoint_sha256
        if args.max_epochs_per_run is not None:
            config_data["max_epochs_per_run"] = args.max_epochs_per_run
        if args.max_runtime_seconds is not None:
            config_data["max_runtime_seconds"] = args.max_runtime_seconds
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        config = MaskedSoftConfig.from_mapping(config_data)
        result = run_masked_soft_training(config, execute=args.execute)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print(json.dumps({"status": "failed", "reason_code": "config_unavailable"}, sort_keys=True))
        return 2
    except (TypeError, ValueError) as error:
        reason = getattr(error, "reason_code", None)
        if type(reason) is not str or not reason.isascii() or not reason.replace("_", "").isalnum():
            reason = "training_unavailable"
        print(json.dumps({"status": "failed", "reason_code": reason}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "failed", "reason_code": "training_unavailable"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
