"""Score fold checkpoints and their ensemble on the 58 expert-labelled Gold studies (D037).

Reads the Gold samples of the extra cache (``scripts/build_extra_cache_kernel.py``) and the expert
labels from the official ``train.csv``. Each run is a set of epoch-5 fold checkpoints; the ensemble
averages their logits, as in the E057 submission. Gold identifiers and labels stay private:
outputs go under ``data/private``.

Example:
    python scripts/score_gold58.py --run E055=data/private/cv-attention16-layer3/fold2/target_attention/checkpoint-epoch-05.pt,... \
        --output data/private/gold58-evaluation/E055.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from rsna_knee.training.cross_validation import TARGETS  # noqa: E402

EXTRA_CACHE = ROOT / "data/private/attention-sixteen-slices/cache-extra-run01/local16-cache-extra-000.zip"


def gold_labels(train_csv: Path) -> dict[str, list[float]]:
    with train_csv.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("ACL", "").strip() != ""]
    labels = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGETS] for row in rows}
    if len(labels) != 58 or any(value not in (0.0, 1.0) for values in labels.values() for value in values):
        raise ValueError("expected 58 Gold studies with 0/1 labels")
    return labels


def gold_samples(archive_path: Path, labels: dict[str, list[float]]):
    """Yield ``(study_sha256, images, masks, label_row)`` for every cached Gold study."""
    import numpy as np
    import torch
    by_hash = {hashlib.sha256(study.encode()).hexdigest(): row for study, row in labels.items()}
    with zipfile.ZipFile(archive_path) as archive:
        records = [r for r in json.loads(archive.read("cache-manifest.json"))["records"] if r["role"] == "gold"]
        if {r["study_sha256"] for r in records} != set(by_hash):
            raise ValueError("the extra cache does not hold exactly the 58 Gold studies")
        for record in records:
            raw = archive.read(record["file"])
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("cache sample hash mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                count = record["series_count"]
                images = [torch.from_numpy(data[f"images_{i:03d}"].copy()) for i in range(count)]
                masks = [torch.from_numpy(data[f"masks_{i:03d}"].copy()) for i in range(count)]
            yield record["study_sha256"], images, masks, by_hash[record["study_sha256"]]


def score(run: dict[str, list[Path]], *, archive: Path, train_csv: Path, device: str) -> dict:
    import numpy as np
    import torch
    from rsna_knee.training.attention_mil import AttentionMILConfig, _model_for_arm
    from rsna_knee.validation.masked_auc import masked_multilabel_auc

    labels = gold_labels(train_csv)
    samples = list(gold_samples(archive, labels))
    y = np.asarray([row for *_, row in samples])
    config = AttentionMILConfig(repository_root=ROOT,
                                weights_path=ROOT / "data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt",
                                output_dir=ROOT / "data/private/gold58-evaluation", device=device)
    result = {"studies": len(samples), "label_definition": "Gold-58 expert labels", "runs": {}}
    for name, checkpoints in run.items():
        logits = []
        for checkpoint in checkpoints:
            model = _model_for_arm(config, "target_attention")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.get("completed_epochs") != 5:
                raise ValueError(f"{checkpoint} is not an epoch-5 checkpoint")
            model.load_state_dict(state["model_state"], strict=True)
            model.to(device).eval()
            with torch.no_grad():
                logits.append(np.stack([
                    model([i.to(device) for i in images], [m.to(device) for m in masks]).float().reshape(-1).cpu().numpy()
                    for _, images, masks, _ in samples]))
        members = [masked_multilabel_auc(y, np.ones_like(y, bool), 1 / (1 + np.exp(-l)), TARGETS) for l in logits]
        ensemble = masked_multilabel_auc(y, np.ones_like(y, bool), 1 / (1 + np.exp(-np.mean(logits, axis=0))), TARGETS)
        result["runs"][name] = {"checkpoints": [str(c) for c in checkpoints],
                                "member_macro_auc": [m["macro_auc"] for m in members],
                                "ensemble": ensemble}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=checkpoint[,checkpoint...] (fold checkpoints of one run)")
    parser.add_argument("--archive", type=Path, default=EXTRA_CACHE)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {}
    for item in args.run:
        name, _, paths = item.partition("=")
        runs[name] = [Path(path) for path in paths.split(",")]
    result = score(runs, archive=args.archive, train_csv=args.train_csv, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, values in result["runs"].items():
        print(name, "members", [round(v, 4) for v in values["member_macro_auc"]],
              "ensemble", round(values["ensemble"]["macro_auc"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
