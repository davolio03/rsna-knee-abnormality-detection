import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts import local_masked_full_cache as cache
from scripts import train_local_radimagenet_epoch as runner


class FullCacheAdapterTests(unittest.TestCase):
    def test_final_segment_resume_selects_verified_epoch_three_and_two_epoch_cap(self):
        root=Path(__file__).resolve().parents[1]
        resume=runner._resolve_resume(root,3)
        self.assertEqual(resume["completed_epochs"],3)
        self.assertEqual(resume["checkpoint_sha256"],"05f0442abf4e3d53713464476b230a6ef45895eb24c58f49fb989a340fc0fcb1")
        plan=runner.make_plan(root,train=False,resume_epoch=3,epochs_this_run=2)
        self.assertEqual(plan["next_epoch"],4)
        self.assertEqual(plan["epochs_to_run"],2)
        self.assertEqual(plan["final_epoch"],5)
        self.assertEqual(plan["resume_checkpoint_sha256"],resume["checkpoint_sha256"])

    def test_epoch_four_and_five_cap_rejects_invalid_and_out_of_range_counts(self):
        runner._validate_run_cap(3,2)
        runner._validate_run_cap(3,1)
        for resume_epoch,cap,code in ((3,0,"run_epoch_cap_unsupported"),(3,-1,"run_epoch_cap_unsupported"),
                                      (3,3,"run_epoch_cap_unsupported"),(2,2,"two_epoch_run_requires_epoch_three"),
                                      (4,2,"two_epoch_run_requires_epoch_three"),(3,4,"run_epoch_cap_unsupported")):
            with self.subTest(resume_epoch=resume_epoch,cap=cap), self.assertRaisesRegex(ValueError,code):
                runner._validate_run_cap(resume_epoch,cap)

    def test_epoch_three_resume_refuses_wrong_pin_or_lineage_evidence(self):
        root=Path(__file__).resolve().parents[1]
        with unittest.mock.patch.object(runner.migration,"_sha256_file",return_value="0"*64):
            with self.assertRaisesRegex(ValueError,"epoch_three_checkpoint_pin_mismatch"):
                runner._resolve_resume(root,3)
        report={"status":"paused_checkpoint_available","completed_epochs":3,
                "fingerprint":runner.NATIVE_FINGERPRINT,"checkpoint_sha256":"a"*64,
                "checkpoint_path":"C:/private/checkpoint-epoch-03.pt",
                "epochs":[{"epoch":1},{"epoch":2},{"epoch":3}]}
        provenance={"schema_version":"local-radimagenet-epoch03-provenance-v1",
                    "native_fingerprint":runner.NATIVE_FINGERPRINT,"completed_epochs_before":2,
                    "epochs_to_run":1,"target_epochs":5,"resume_checkpoint_sha256":"b"*64,
                    "resume_checkpoint_path":"C:/private/checkpoint-epoch-02.pt"}
        runner._validate_epoch_three_report(report,provenance,Path("C:/private/checkpoint-epoch-03.pt"),"a"*64,
                                            epoch_two_checkpoint=Path("C:/private/checkpoint-epoch-02.pt"),epoch_two_checkpoint_sha256="b"*64)
        for change in ({"completed_epochs":4},{"epochs":[{"epoch":1},{"epoch":3}]},
                       {"fingerprint":"c"*64},{"checkpoint_sha256":"d"*64}):
            with self.subTest(change=change):
                altered=dict(report);altered.update(change)
                with self.assertRaisesRegex(ValueError,"epoch_three_training_report_mismatch"):
                    runner._validate_epoch_three_report(altered,provenance,Path("C:/private/checkpoint-epoch-03.pt"),"a"*64,
                                                        epoch_two_checkpoint=Path("C:/private/checkpoint-epoch-02.pt"),epoch_two_checkpoint_sha256="b"*64)
        altered_provenance=dict(provenance);altered_provenance["resume_checkpoint_sha256"]="e"*64
        with self.assertRaisesRegex(ValueError,"epoch_three_provenance_mismatch"):
            runner._validate_epoch_three_report(report,altered_provenance,Path("C:/private/checkpoint-epoch-03.pt"),"a"*64,
                                                epoch_two_checkpoint=Path("C:/private/checkpoint-epoch-02.pt"),epoch_two_checkpoint_sha256="b"*64)

    def test_epoch_three_resume_is_pinned_to_verified_epoch_two(self):
        root=Path(__file__).resolve().parents[1]
        resume=runner._resolve_resume(root,2)
        self.assertEqual(resume["completed_epochs"],2)
        self.assertEqual(resume["checkpoint_sha256"],"bd1abee90e19256292a57d04d31eced4341ad7207c403125d03b150cdeacaf72")
        self.assertEqual(resume["native_fingerprint"],"5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920")
        self.assertEqual(resume["next_epoch"],3)
        self.assertEqual(resume["checkpoint"].name,"checkpoint-epoch-02.pt")
        with self.assertRaisesRegex(ValueError,"resume_epoch_unsupported"):
            runner._resolve_resume(root,5)

    def test_epoch_three_rejects_wrong_checkpoint_sha_and_bad_report_history(self):
        root=Path(__file__).resolve().parents[1]
        with unittest.mock.patch.object(runner.migration,"_sha256_file",return_value="0"*64):
            with self.assertRaisesRegex(ValueError,"epoch_two_checkpoint_pin_mismatch"):
                runner._resolve_resume(root,2)
        report={"status":"paused_checkpoint_available","completed_epochs":2,
                "fingerprint":runner.NATIVE_FINGERPRINT,"checkpoint_sha256":"a"*64,
                "checkpoint_path":"C:/private/checkpoint.pt","epochs":[{"epoch":1},{"epoch":2}]}
        provenance={"schema_version":"local-radimagenet-epoch02-provenance-v1",
                    "native_fingerprint":runner.NATIVE_FINGERPRINT,"completed_epochs_before":1,
                    "epochs_to_run":1,"migrated_checkpoint_sha256":"b"*64,"target_epochs":5}
        runner._validate_epoch_two_report(report,provenance,Path("C:/private/checkpoint.pt"),"a"*64,migrated_checkpoint_sha256="b"*64)
        for change in ({"completed_epochs":1},{"epochs":[{"epoch":1}]},{"status":"complete"}):
            with self.subTest(change=change):
                altered=dict(report);altered.update(change)
                with self.assertRaisesRegex(ValueError,"epoch_two_training_report_mismatch"):
                    runner._validate_epoch_two_report(altered,provenance,Path("C:/private/checkpoint.pt"),"a"*64,migrated_checkpoint_sha256="b"*64)

    def test_epoch_three_has_fresh_output_and_default_keeps_epoch_two_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            epoch_two=runner._output_for_resume(root,1)
            target=runner._output_for_resume(root,2)
            final=runner._output_for_resume(root,3)
            self.assertEqual(epoch_two,root/"data/private/local-radimagenet-epoch02/epoch02-run01")
            self.assertEqual(target,root/"data/private/local-radimagenet-epoch03/epoch03-run01")
            self.assertFalse(target.exists())
            self.assertEqual(final,root/"data/private/local-radimagenet-final-epochs/final-run01")
            self.assertFalse(final.exists())

    def test_native_training_call_enforces_precision_inside_call(self):
        import torch
        native = Mock()
        config={"resume_checkpoint":"pinned-epoch-3","resume_sha256":"05f0442abf4e3d53713464476b230a6ef45895eb24c58f49fb989a340fc0fcb1","max_epochs_per_run":2}
        def capture(*args,**kwargs):
            native.observed = (torch.backends.cuda.matmul.allow_tf32,
                               torch.backends.cudnn.allow_tf32,
                               torch.backends.cudnn.benchmark,
                               torch.get_float32_matmul_precision(),
                               torch.is_autocast_enabled("cuda"))
            native.resume=(args[0]["resume_checkpoint"],args[0]["resume_sha256"],args[0]["max_epochs_per_run"])
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
                result=runner._execute_native_training(native,config,lambda *_:None,lambda *_:None)
            self.assertEqual(result,{"status":"captured"})
            self.assertEqual(native.observed,(False,False,False,"highest",False))
            self.assertEqual(native.resume,("pinned-epoch-3","05f0442abf4e3d53713464476b230a6ef45895eb24c58f49fb989a340fc0fcb1",2))
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
        self.assertEqual(Path(plan["output_dir"]),Path("data/private/local-radimagenet-epoch02/epoch02-run01").resolve())
        self.assertEqual(Path(plan["resume_checkpoint"]).name,"checkpoint-epoch-01-migrated.pt")


if __name__ == "__main__":
    unittest.main()
