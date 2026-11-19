"""Put the weights of a new attention16 checkpoint into the pinned E051 model export.

The E051 inference code checks the export header and the sanitized report against E050 pins.
For a model with the same architecture, this script keeps the E051 header and report and replaces
only ``model_state``. The submission notebook then needs only new ``SOURCE_ROOT`` and
``CHECKPOINT_SHA256`` values (the export size does not change). The header still names the E050
source checkpoint, so the experiment record must name the real source.

Example (E056, layer3 fold-0 model):
    PYTHONPATH="src;." python scripts/build_attention16_weight_swap.py \
        --checkpoint data/private/cv-attention16-layer3/kaggle-run01/fold0/target_attention/checkpoint-epoch-05.pt \
        --base data/private/attention-sixteen-submission/run02 \
        --output data/private/attention-sixteen-submission/layer3-fold0-run01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from rsna_knee.inference import attention_sixteen as a16

EXPORT = "attention-model-epoch-05.pt"
REPORT = "training-report.json"


def swap(checkpoint: Path, base: Path, output: Path) -> dict:
    proof = json.loads(checkpoint.with_suffix(".sha256").read_text(encoding="utf-8"))
    digest = a16._sha256(checkpoint)
    if proof.get("checkpoint_sha256") != digest or proof.get("completed_epochs") != 5:
        raise ValueError("checkpoint does not match its .sha256 proof or is not epoch 5")
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if source.get("arm") != "target_attention" or source.get("completed_epochs") != 5:
        raise ValueError("expected a completed target_attention checkpoint")
    exported = torch.load(base / EXPORT, map_location="cpu", weights_only=True)
    state = a16._validated_state(source["model_state"], a16._model())
    if set(state) != set(exported["model_state"]):
        raise ValueError("state keys differ from the base export")
    changed = sum(not torch.equal(state[k].float(), exported["model_state"][k].float()) for k in state)
    output.mkdir(parents=True, exist_ok=False)
    payload = {**{k: v for k, v in exported.items() if k != "model_state"},
               "model_state": {k: t.detach().cpu().contiguous() for k, t in state.items()}}
    torch.save(payload, output / EXPORT)
    shutil.copy2(base / REPORT, output / REPORT)
    sha, size = a16._sha256(output / EXPORT), (output / EXPORT).stat().st_size
    report_raw = (output / REPORT).read_bytes()
    report_sha = hashlib.sha256(report_raw).hexdigest()
    model = a16.load_attention16_model(output / EXPORT, json.loads(report_raw), sha,
                                       expected_report_sha256=report_sha, checkpoint_size=size)
    loaded = model.state_dict()
    if not all(torch.equal(loaded[k], state[k]) for k in state):
        raise ValueError("readback differs from the source weights")
    return {"export_sha256": sha, "export_size": size, "report_sha256": report_sha,
            "source_checkpoint_sha256": digest, "source_fingerprint": source["fingerprint"],
            "tensors": len(state), "tensors_changed": changed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="epoch-5 training checkpoint")
    parser.add_argument("--base", type=Path, required=True, help="E051 package directory (run02)")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the export")
    args = parser.parse_args(argv)
    print(json.dumps(swap(args.checkpoint, args.base, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
