# %% [markdown]
# # Bounded raw DICOM reader benchmark
#
# This notebook measures a small, deterministic sample of raw DICOM header and
# pixel decoding. It does not train, normalize, reorient, preprocess, or establish
# a Kaggle runtime contract. Configure all three local paths explicitly; the
# expected image layout is `<image-root>/<study-id>/<series-id>/*.dcm`.

# %%
import json
import os
import sys
from pathlib import Path

SOURCE_ROOT_VALUE = os.environ.get("SOURCE_ROOT", "").strip()
MANIFEST_VALUE = os.environ.get("PRIVATE_MANIFEST", "").strip()
IMAGE_ROOT_VALUE = os.environ.get("COMPETITION_IMAGE_ROOT", "").strip()
SOURCE_ROOT = Path(SOURCE_ROOT_VALUE).expanduser()
PRIVATE_MANIFEST = Path(MANIFEST_VALUE).expanduser()
COMPETITION_IMAGE_ROOT = Path(IMAGE_ROOT_VALUE).expanduser()

try:
    source_available = bool(SOURCE_ROOT_VALUE) and (SOURCE_ROOT / "src").is_dir()
    manifest_available = bool(MANIFEST_VALUE) and PRIVATE_MANIFEST.is_file()
    image_root_available = bool(IMAGE_ROOT_VALUE) and COMPETITION_IMAGE_ROOT.is_dir()
except OSError:
    raise RuntimeError("Configured local paths could not be checked safely.") from None

if not source_available:
    raise RuntimeError("Set SOURCE_ROOT to the local repository root containing src/.")
if not manifest_available:
    raise RuntimeError("Set PRIVATE_MANIFEST to the ignored local cohort manifest.")
if not image_root_available:
    raise RuntimeError("Set COMPETITION_IMAGE_ROOT to the explicitly configured image directory.")

sys.path.insert(0, str(SOURCE_ROOT / "src"))

# %% [markdown]
# The first five studies in manifest order are selected by default; all their
# series are considered in sorted order. Per-series slice selection is bounded
# and sorted by patient-space geometry. No corpus-wide recursive scan is used.
# The runtime must already provide `pydicom` and `numpy`; this notebook never
# installs packages or accesses the network.

# %%
try:
    manifest = json.loads(PRIVATE_MANIFEST.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise RuntimeError("The configured private cohort manifest could not be read.") from None

from rsna_knee.data.dicom_benchmark import run_dicom_benchmark

aggregate = run_dicom_benchmark(
    manifest=manifest,
    image_root=COMPETITION_IMAGE_ROOT,
    max_studies=5,
)
print(json.dumps(aggregate, indent=2, sort_keys=True))

# %% [markdown]
# Interpret this output only as bounded raw-reader evidence. Actual image-root
# availability, decoding performance on competition images, geometric coverage,
# preprocessing policy, and the Kaggle runtime remain unverified until run in
# their explicitly authorized environment. No image arrays or identifiers are
# displayed or written by this scaffold.
