"""Build the E057 Kaggle notebook: the E056 notebook plus a 3-fold logit-averaging ensemble.

The pinned E051 inference code loads one model through
``rsna_knee.inference.attention_sixteen.load_attention16_model``. The notebook wraps that function
so it loads the three E055 fold exports (each verified by its own SHA-256 pin) and returns a module
that averages their logits. Each study is preprocessed once, so runtime barely changes.

Example:
    python scripts/build_attention16_ensemble_notebook.py data/private/attention-sixteen-submission/layer3-ensemble-kernel
    kaggle kernels push -p data/private/attention-sixteen-submission/layer3-ensemble-kernel
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "data/private/attention-sixteen-submission"
PATCH = '''    # E057: average the logits of the three E055 fold models. The pinned inference code loads one
    # model through attention_sixteen.load_attention16_model; this wraps it to load all members.
    import torch
    import rsna_knee.inference.attention_sixteen as _a16
    ENSEMBLE_MEMBERS = [
        ('/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-layer3-fold1', '7bd7bc45d9c8c0d574cb1291a5097a0d5e40c4f68bce3b48c0c38fbcc59b5a2c'),
        ('/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-layer3-fold2', 'a06dc46de5fbf83fda1b446da080bcf18d13874a1b02782755f0b090585a9341'),
    ]
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
        print(json.dumps({'event': 'ensemble_loaded', 'members': len(members)}), flush=True)
        return MeanLogitEnsemble(members)

    _a16.load_attention16_model = load_ensemble
'''
if __name__ == "__main__":
    out = Path(sys.argv[1])
    nb = json.loads((base / "layer3-fold0-kernel/kaggle_attention16_layer3_submission.ipynb").read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    anchor = "    summary = run_attention16_inference(\n"
    assert code.count(anchor) == 1
    code = code.replace(anchor, PATCH + anchor)
    code = code.replace("'/kaggle/working/attention16-inference-run01'", "'/kaggle/working/attention16-ensemble-run01'")
    code = code.replace("Path('/kaggle/working/attention16-inference-run01/submission.csv')", "Path('/kaggle/working/attention16-ensemble-run01/submission.csv')")
    assert code.count("attention16-ensemble-run01") == 2
    compile(code, "nb", "exec")
    nb["cells"][1]["source"] = code.splitlines(keepends=True)
    nb["cells"][0]["source"] = ["# E057 attention16 layer3 3-fold ensemble submission\n\nE051 run02 inference code, unchanged; the notebook averages the logits of the three E055 fold models (layer3 fine-tuning, epoch 5).\n"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "kaggle_attention16_layer3_ensemble.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    meta = {"id": "davidsulleiro/rsna-knee-attention16-layer3-ensemble-submission",
            "title": "RSNA Knee Attention16 Layer3 Ensemble Submission",
            "code_file": "kaggle_attention16_layer3_ensemble.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"davidsulleiro/rsna-knee-attention16-layer3-fold{k}" for k in range(3)]
                               + ["davidsulleiro/rsna-knee-offline-dicom-codecs"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built")
