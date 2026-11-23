"""Build a Kaggle notebook that averages the logits of several attention16 fold exports.

The pinned E051 inference code (via the E056 notebook) loads one model through
``rsna_knee.inference.attention_sixteen.load_attention16_model``. The notebook wraps that function
so it loads every member export (each verified by its own SHA-256 pin) and returns a module that
averages their logits. Each study is preprocessed once, so runtime barely changes. The first
member replaces the E056 model pins; the others are loaded by the wrapper.

Members are private model-only datasets built with ``scripts/build_attention16_weight_swap.py``.
Experiments: E057 (``--experiment E057``, the E055 fold models), E058 (the E058 fold models), E059
(the four final all-data models plus the E058 fold models) and E059-final4 (the final models only).

Example:
    python scripts/build_attention16_ensemble_notebook.py --experiment E058 \
        data/private/attention-sixteen-submission/labels7-ensemble-kernel
    kaggle kernels push -p data/private/attention-sixteen-submission/labels7-ensemble-kernel
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/attention-sixteen-submission"
OWNER = "davidsulleiro"
E056_SOURCE = "/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-layer3-fold0"
E056_SHA256 = "f4fa4e242d8f4aa60e60edf48223051203d3a92c247898824ab963e4151391d7"

EXPERIMENTS = {
    "E057": {
        "description": "the three E055 fold models (layer3 fine-tuning, epoch 5)",
        "comment": "the three E055 fold models",
        "heading": "E057 attention16 layer3 3-fold ensemble submission",
        "kernel": "rsna-knee-attention16-layer3-ensemble-submission",
        "title": "RSNA Knee Attention16 Layer3 Ensemble Submission",
        "notebook": "kaggle_attention16_layer3_ensemble.ipynb",
        "members": [
            ("rsna-knee-attention16-layer3-fold0", E056_SHA256),
            ("rsna-knee-attention16-layer3-fold1", "7bd7bc45d9c8c0d574cb1291a5097a0d5e40c4f68bce3b48c0c38fbcc59b5a2c"),
            ("rsna-knee-attention16-layer3-fold2", "a06dc46de5fbf83fda1b446da080bcf18d13874a1b02782755f0b090585a9341"),
        ],
    },
    "E058": {
        "description": "the three E058 fold models (7th-place labels, layer3 fine-tuning, epoch 5)",
        "comment": "the three E058 fold models",
        "heading": "E058 attention16 labels7 3-fold ensemble submission",
        "kernel": "rsna-knee-attention16-labels7-ensemble-submission",
        "title": "RSNA Knee Attention16 Labels7 Ensemble Submission",
        "notebook": "kaggle_attention16_labels7_ensemble.ipynb",
        "members": [
            ("rsna-knee-attention16-labels7-fold0", "a85e7600a88462f18ec214e6f82a620374dfde0c4ca38a17eb51ef388362a471"),
            ("rsna-knee-attention16-labels7-fold1", "1a6f7823158647388ca662a93f7b437e0364ca2ad6e3633d8ca1a944583d9a7b"),
            ("rsna-knee-attention16-labels7-fold2", "4d24d68abfbec1479e35530a1e609ef1ce394c579e17d343c2790e8687f8aaa2"),
        ],
    },
}
E059_FINAL = [
    ("rsna-knee-attention16-final-seed42", "92bb5da3ab38cb846eb3e02b0b2cac02683e3e032e983e9b16408b5fe60a4721"),
    ("rsna-knee-attention16-final-seed43", "c75eb7b40ae9ef6572f3028c98b5a4880e0fe9fae8e9851527e04b6b5b630df5"),
    ("rsna-knee-attention16-final-seed44", "5f3effe31a301517b90f881467fd1e417e5c14cffdcd274ff2c3ad5918e776f5"),
    ("rsna-knee-attention16-final-seed45", "80f06cf5dab4abda6f3e5ad1d90cd8d0ca35bac8d0eb76e4412faea94935cae7"),
]
EXPERIMENTS["E059"] = {
    "description": "the four E059 final all-data models and the three E058 fold models (epoch 5)",
    "comment": "the four E059 final models and the three E058 fold models",
    "heading": "E059 attention16 final 7-member ensemble submission (primary)",
    "kernel": "rsna-knee-attention16-final-ensemble7-submission",
    "title": "RSNA Knee Attention16 Final Ensemble7 Submission",
    "notebook": "kaggle_attention16_final_ensemble7.ipynb",
    "members": E059_FINAL + EXPERIMENTS["E058"]["members"],
}
EXPERIMENTS["E059-final4"] = {
    "description": "the four E059 final all-data models (epoch 5)",
    "comment": "the four E059 final models",
    "heading": "E059 attention16 final 4-member ensemble submission (secondary)",
    "kernel": "rsna-knee-attention16-final4-submission",
    "title": "RSNA Knee Attention16 Final4 Submission",
    "notebook": "kaggle_attention16_final4.ipynb",
    "members": E059_FINAL,
}


def mount(slug: str) -> str:
    return f"/kaggle/input/datasets/{OWNER}/{slug}"


def patch(experiment: str) -> str:
    """Code inserted before the inference call: loads the members after the first one."""
    rows = "".join(f"        ({mount(slug)!r}, {sha!r}),\n" for slug, sha in EXPERIMENTS[experiment]["members"][1:])
    return f'''    # {experiment}: average the logits of {EXPERIMENTS[experiment]["comment"]}. The pinned inference code loads one
    # model through attention_sixteen.load_attention16_model; this wraps it to load all members.
    import torch
    import rsna_knee.inference.attention_sixteen as _a16
    ENSEMBLE_MEMBERS = [
{rows}    ]
    _load_single = _a16.load_attention16_model

    class MeanLogitEnsemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, images, masks):
            return torch.stack([member(images, masks) for member in self.members]).mean(0)

    def load_ensemble(checkpoint, report, checkpoint_sha256, **kwargs):
        members = [_load_single(checkpoint, report, checkpoint_sha256, **kwargs)]
        for root, sha in ENSEMBLE_MEMBERS:
            members.append(_load_single(Path(root) / 'attention-model-epoch-05.pt', report, sha, **kwargs))
        print(json.dumps({{'event': 'ensemble_loaded', 'members': len(members)}}), flush=True)
        return MeanLogitEnsemble(members)

    _a16.load_attention16_model = load_ensemble
'''


PATCH = patch("E057")


def build(out: Path, experiment: str, source: Path | None = None) -> Path:
    spec = EXPERIMENTS[experiment]
    source = source or BASE / "layer3-fold0-kernel/kaggle_attention16_layer3_submission.ipynb"
    nb = json.loads(source.read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    anchor = "    summary = run_attention16_inference(\n"
    assert code.count(anchor) == 1
    code = code.replace(anchor, patch(experiment) + anchor)
    first_slug, first_sha = spec["members"][0]
    for old, new in ((E056_SOURCE, mount(first_slug)), (E056_SHA256, first_sha)):
        assert code.count(f"'{old}'") == 1
        code = code.replace(f"'{old}'", f"'{new}'")
    code = code.replace("'/kaggle/working/attention16-inference-run01'", "'/kaggle/working/attention16-ensemble-run01'")
    code = code.replace("Path('/kaggle/working/attention16-inference-run01/submission.csv')",
                        "Path('/kaggle/working/attention16-ensemble-run01/submission.csv')")
    assert code.count("attention16-ensemble-run01") == 2
    compile(code, "nb", "exec")
    nb["cells"][1]["source"] = code.splitlines(keepends=True)
    nb["cells"][0]["source"] = [f"# {spec['heading']}\n\nE051 run02 inference "
                                f"code, unchanged; the notebook averages the logits of {spec['description']}.\n"]
    out.mkdir(parents=True, exist_ok=False)
    (out / spec["notebook"]).write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    meta = {"id": f"{OWNER}/{spec['kernel']}", "title": spec["title"], "code_file": spec["notebook"],
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{OWNER}/{slug}" for slug, _ in spec["members"]]
                               + [f"{OWNER}/rsna-knee-offline-dicom-codecs"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="E057")
    args = parser.parse_args(argv)
    build(args.out, args.experiment)
    print("built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
