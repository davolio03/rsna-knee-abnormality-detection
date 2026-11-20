from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load():
    spec = importlib.util.spec_from_file_location("build_extra_cache_kernel",
                                                  ROOT / "scripts" / "build_extra_cache_kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_extra_cache_kernel"] = module
    spec.loader.exec_module(module)
    return module


class StudyListTests(unittest.TestCase):
    def test_roles_cover_gold_extra_and_controls_without_labels(self):
        builder = load()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            train = ["StudyInstanceUID,Report," + ",".join(TARGETS)]
            train += ["g1,r," + ",".join(["1"] * 12), "g2,r," + ",".join(["0"] * 12)]
            train += [f"{s},r," + ",".join([""] * 12) for s in ("c1", "c2", "c3", "x1")]
            (base / "train.csv").write_text("\n".join(train) + "\n", encoding="utf-8")
            series = ["StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive,Fat_Suppression"]
            series += [f"{s},{s}-b,Sagittal,1,1" for s in ("g1", "g2", "c1", "c2", "c3", "x1")] + ["g1,g1-a,Coronal,0,0"]
            (base / "series.csv").write_text("\n".join(series) + "\n", encoding="utf-8")
            (base / "manifest.json").write_text(json.dumps({"records": [{"study_id": s} for s in ("c1", "c2", "c3")]}))
            labels = ["StudyInstanceUID," + ",".join(TARGETS)]
            labels += [f"{s}," + ",".join(["0.1"] * 12) for s in ("c1", "c2", "c3", "x1")]
            (base / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")
            studies = builder.study_list(train_csv=base / "train.csv", series_csv=base / "series.csv",
                                         manifest=base / "manifest.json", labels=base / "labels.csv")
            roles = {study["study_id"]: study["role"] for study in studies}
            self.assertEqual(roles, {"g1": "gold", "g2": "gold", "x1": "extra", "c1": "control", "c2": "control"})
            g1 = next(study for study in studies if study["study_id"] == "g1")
            self.assertEqual([s["series_id"] for s in g1["series"]], ["g1-a", "g1-b"])
            self.assertNotIn("labels", json.dumps(studies))
            out = builder.build(out=base / "kernel", studies=studies, slug="u/extra")
            compile((out / "extra_cache_runner.py").read_text(encoding="utf-8"), "runner", "exec")
            metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["enable_internet"])
            self.assertIn("u/extra-studies", metadata["dataset_sources"])


if __name__ == "__main__":
    unittest.main()
