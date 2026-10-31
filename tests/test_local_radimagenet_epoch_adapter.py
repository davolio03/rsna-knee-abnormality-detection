import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts import local_masked_full_cache as cache
from scripts import train_local_radimagenet_epoch as runner


class FullCacheAdapterTests(unittest.TestCase):
    def test_native_training_call_enforces_precision_inside_call(self):
        import torch
        native = Mock()
        def capture(*args,**kwargs):
            native.observed = (torch.backends.cuda.matmul.allow_tf32,
                               torch.backends.cudnn.allow_tf32,
                               torch.backends.cudnn.benchmark,
                               torch.get_float32_matmul_precision(),
                               torch.is_autocast_enabled("cuda"))
            return {"status": "captured"}
        native.run_recipe_training.side_effect = capture
        previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                  torch.backends.cudnn.benchmark,torch.get_float32_matmul_precision())
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            torch.backends.cudnn.benchmark=True
            torch.set_float32_matmul_precision("medium")
            with torch.autocast(device_type="cuda",enabled=True):
                result=runner._execute_native_training(native,{},lambda *_:None,lambda *_:None)
            self.assertEqual(result,{"status":"captured"})
            self.assertEqual(native.observed,(False,False,False,"highest",False))
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=previous[:3]
            torch.set_float32_matmul_precision(previous[3])

    def test_train_preflight_requires_one_cuda_device_and_three_gib_free(self):
        runner._validate_training_device(cuda_available=True,device_count=1,free_memory_bytes=3*1024**3)
        for values,code in (((False,1,4*1024**3),"target_runtime_mismatch"),
                            ((True,2,4*1024**3),"target_runtime_mismatch"),
                            ((True,1,3*1024**3-1),"gpu_free_memory_below_safety_margin")):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError,code):
                runner._validate_training_device(cuda_available=values[0],device_count=values[1],free_memory_bytes=values[2])

    def test_real_plan_and_all_archive_pins_validate(self):
        root = Path(__file__).resolve().parents[1]
        result = cache.validate_full_cache(root)
        self.assertEqual(result["studies"], 4298)
        self.assertEqual(result["series"], 23765)
        self.assertEqual(result["slices"], 190120)
        self.assertEqual(result["training_studies"], 2850)
        self.assertEqual(result["validation_studies"], 1448)

    def test_invalid_coverage_receipt_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps({"status": "pass", "studies": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                cache.validate_receipt(path)

    def test_receipt_rejects_wrong_identity_counts_and_gold_claim(self):
        base={"status":"pass","schema_version":"local-masked-full-cache-verification-v1",
              "studies":4298,"series":23765,"slices":190120,
              "partition_study_counts":{"train":2850,"validation":1448},
              "no_overlap":True,"no_missing_or_extra_studies":True,"gold_intersection_count":0,
              "training_started":False,"source_sha256":cache.SOURCE,"manifest_sha256":cache.MANIFEST,
              "folds_sha256":cache.FOLDS,"plan_sha256":cache.PLAN,"all_archive_hashes_rechecked":True}
        for key,value in (("studies",4297),("no_overlap",False),("gold_intersection_count",1),
                          ("source_sha256","0"*64),("all_archive_hashes_rechecked",False)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                receipt=dict(base); receipt[key]=value
                path=Path(tmp)/"receipt.json"; path.write_text(json.dumps(receipt),encoding="utf-8")
                with self.assertRaises(ValueError): cache.validate_receipt(path)

    def test_missing_pin_fails_before_training_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError): runner.make_plan(root,train=True)
            self.assertFalse((root/"data/private/local-radimagenet-epoch02/epoch02-run01").exists())

    def test_plan_only_does_not_migrate_or_train(self):
        plan = runner.make_plan(Path("."), train=False)
        self.assertEqual(plan["mode"], "plan")
        self.assertEqual(plan["completed_epochs"], 1)
        self.assertEqual(plan["target_epochs"], 5)
        self.assertEqual(plan["epochs_to_run"], 1)


if __name__ == "__main__":
    unittest.main()
