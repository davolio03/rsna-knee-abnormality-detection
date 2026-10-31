import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_kaggle_local_masked_cache_export import (
    build_cache_export_notebook,
    build_full_chunk_export_notebooks,
    read_successful_prepared_inventory_pin,
)


ROOT = Path(__file__).resolve().parents[1]
V4_INVENTORY_PIN = "0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3"
AUTHORITY_NOTEBOOK = ROOT / "data/private/resnet50-pretraining-preparation/run02/kaggle_resnet50_radimagenet_full_epoch01.ipynb"


class KaggleLocalMaskedCacheExportTests(unittest.TestCase):
    def test_emits_private_cpu_only_bounded_notebook_with_frozen_source_pin(self):
        source = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
        helper = ROOT / "scripts/local_masked_tensor_cache.py"
        with tempfile.TemporaryDirectory() as temporary:
            pin = read_successful_prepared_inventory_pin(AUTHORITY_NOTEBOOK)
            result = build_cache_export_notebook(source, helper, Path(temporary) / "export.ipynb",
                                                 prepared_inventory_sha256=pin)
            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            compile(code, "generated_local_cache_export.py", "exec")
            self.assertIn('EXPECTED_SOURCE_SHA256 = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"', code)
            self.assertEqual(pin, V4_INVENTORY_PIN)
            self.assertIn(f'EXPECTED_PREPARED_INVENTORY_SHA256 = "{V4_INVENTORY_PIN}"', code)
            self.assertIn('STAGE = "plan"', code)
            self.assertIn("RUN_BOUNDED_EXPORT = False", code)
            self.assertIn("MAX_STUDIES_PER_PARTITION = 8", code)
            self.assertIn('for partition in ("train", "validation")', code)
            self.assertIn("find_prepared_asset_root(input_root)", code)
            self.assertIn("find_competition_root(input_root)", code)
            self.assertIn("torch.cuda.is_initialized()", code)
            self.assertIn("allow_pickle=False", code)
            self.assertEqual(notebook["cells"][0]["outputs"], [])
            self.assertLess(code.index('if stage == "plan":'), code.index('codec_bundle_path = assets'))
            self.assertIn('stage == "smoke_export" and run_bounded_export is not True', code)
            self.assertIn('write_cache_archive"](output_root, archive_path)', code)
            raw = Path(result["notebook_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), result["notebook_sha256"])

            second = build_cache_export_notebook(source, helper, Path(temporary) / "export-again.ipynb",
                                                 prepared_inventory_sha256=pin)
            self.assertEqual(raw, Path(second["notebook_path"]).read_bytes())

    def test_refuses_unpinned_source_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.zip"
            source.write_bytes(b"not the frozen source")
            helper = ROOT / "scripts/local_masked_tensor_cache.py"
            with self.assertRaisesRegex(ValueError, "source_pin_mismatch"):
                build_cache_export_notebook(source, helper, Path(temporary) / "out.ipynb",
                                             prepared_inventory_sha256=V4_INVENTORY_PIN)

    def test_authorized_smoke_variant_changes_only_the_two_explicit_opt_in_flags(self):
        source = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
        helper = ROOT / "scripts/local_masked_tensor_cache.py"
        with tempfile.TemporaryDirectory() as temporary:
            pin = read_successful_prepared_inventory_pin(AUTHORITY_NOTEBOOK)
            default = build_cache_export_notebook(source, helper, Path(temporary) / "default.ipynb",
                                                  prepared_inventory_sha256=pin)
            authorized = build_cache_export_notebook(
                source, helper, Path(temporary) / "authorized.ipynb", prepared_inventory_sha256=pin,
                authorized_smoke_export=True)
            default_code = "\n".join("".join(cell["source"]) for cell in
                                    json.loads(Path(default["notebook_path"]).read_text())["cells"])
            authorized_code = "\n".join("".join(cell["source"]) for cell in
                                        json.loads(Path(authorized["notebook_path"]).read_text())["cells"])
            expected = default_code.replace('STAGE = "plan"  # plan | smoke_export',
                                            'STAGE = "smoke_export"  # plan | smoke_export', 1)
            expected = expected.replace("RUN_BOUNDED_EXPORT = False", "RUN_BOUNDED_EXPORT = True", 1)
            self.assertEqual(authorized_code, expected)
            self.assertEqual(authorized_code.count('STAGE = "smoke_export"'), 1)
            self.assertEqual(authorized_code.count("RUN_BOUNDED_EXPORT = True"), 1)
            self.assertIn("MAX_STUDIES_PER_PARTITION = 8", authorized_code)

    def test_rejects_run04_inventory_pin_instead_of_successful_dataset_v4_pin(self):
        source = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
        helper = ROOT / "scripts/local_masked_tensor_cache.py"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "prepared_inventory_pin_mismatch"):
                build_cache_export_notebook(
                    source, helper, Path(temporary) / "wrong-pin.ipynb",
                    prepared_inventory_sha256="25833614d3b274956cfa5284f7c0502ccc7d63d9d6c78d87cdbcf169e0a40707")

    def test_full_chunk_notebooks_are_plan_default_and_only_enable_their_exact_chunk(self):
        source = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
        helper = ROOT / "scripts/local_masked_tensor_cache.py"
        planner = ROOT / "scripts/local_masked_cache_chunk_plan.py"
        plan = ROOT / "data/private/local-radimagenet-epoch02/cache-chunk-plan.json"
        with tempfile.TemporaryDirectory() as temporary:
            defaults = build_full_chunk_export_notebooks(
                source, helper, planner, plan, Path(temporary) / "plan", authorized=False)
            exports = build_full_chunk_export_notebooks(
                source, helper, planner, plan, Path(temporary) / "export", authorized=True)
            self.assertEqual(len(defaults), 7)
            self.assertEqual([row["chunk_id"] for row in defaults],
                             [row["chunk_id"] for row in exports])
            for default, export in zip(defaults, exports):
                default_nb = json.loads(Path(default["notebook_path"]).read_text(encoding="utf-8"))
                export_nb = json.loads(Path(export["notebook_path"]).read_text(encoding="utf-8"))
                default_code = "".join(default_nb["cells"][0]["source"])
                export_code = "".join(export_nb["cells"][0]["source"])
                expected = default_code.replace('STAGE = "plan"  # plan | chunk_export',
                                                'STAGE = "chunk_export"  # plan | chunk_export', 1)
                expected = expected.replace("RUN_CHUNK_EXPORT = False", "RUN_CHUNK_EXPORT = True", 1)
                self.assertEqual(export_code, expected)
                self.assertIn(f"CHUNK_ID = '{export['chunk_id']}'", export_code)
                self.assertIn('"max_series_per_chunk":4000', export_code)
                compile(export_code, export["notebook_path"], "exec")

    def test_frozen_full_chunk_notebooks_match_current_builder_payload(self):
        source = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
        helper = ROOT / "scripts/local_masked_tensor_cache.py"
        planner = ROOT / "scripts/local_masked_cache_chunk_plan.py"
        plan = ROOT / "data/private/local-radimagenet-epoch02/cache-chunk-plan.json"
        frozen = ROOT / "data/private/local-radimagenet-epoch02/cache-chunks-v2"
        with tempfile.TemporaryDirectory() as temporary:
            generated = build_full_chunk_export_notebooks(
                source, helper, planner, plan, Path(temporary), authorized=True)
            for row in generated:
                frozen_path = frozen / Path(row["notebook_path"]).name
                self.assertEqual(Path(row["notebook_path"]).read_bytes(), frozen_path.read_bytes())
                self.assertEqual(hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                                 row["notebook_sha256"])


if __name__ == "__main__":
    unittest.main()
