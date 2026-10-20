"""Packaging and cloud handoff checks; no account, GPU or image dataset required."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import subprocess
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = module("project_kaggle_launcher", ROOT / "scripts/kaggle.py")
runner = module("project_kaggle_runner", ROOT / "kaggle/runner.py")


class KaggleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.job = json.loads((ROOT / "kaggle/job.json").read_text())
        self.cfg = {"owner": "tester", "data_dataset": "tester/data", "resources_dataset": "tester/resources"}

    def tearDown(self):
        self.temp.cleanup()

    def test_default_uses_existing_rfm_without_changing_effective_batch(self):
        launcher.validate_job(self.job)
        config = json.loads((ROOT / self.job["config"]).read_text())
        self.assertEqual(config["model"], "rfm")
        self.assertEqual(config["batch_size"] * config["accumulation"], 32)
        self.assertFalse(self.job["auto_pixel_cache"])

    def test_bad_jobs_fail_before_remote_work(self):
        for key, value in (("fold", "full"), ("max_hours", float("nan")), ("workers", -1),
                           ("config", "../outside.json"), ("output", "artifacts/../weights/overwritten"),
                           ("resume", True), ("auto_pixel_cache", "false")):
            job = copy.deepcopy(self.job)
            job[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.validate_job(job)

    def test_previous_kernel_supports_resume_and_separate_warm_start(self):
        job = {**self.job, "resume": True, "resume_kernel": "tester/old/1"}
        launcher.validate_job(job)
        job.update(resume=False, initialize_from="artifacts/kaggle_train/p1/model.pt")
        launcher.validate_job(job)

    def test_payload_is_allowlisted_and_has_no_secrets_or_data(self):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(launcher.source_payload()))) as z:
            names = z.namelist()
            self.assertIn("glaucoma/engine.py", names)
            self.assertIn("configs/rfm_p2_multiview.json", names)
            for name in names:
                self.assertTrue(name in ("run.py", "requirements-kaggle.txt") or
                                name.startswith(("glaucoma/", "configs/")))
                self.assertNotIn("local.json", name)

    def test_generated_script_and_private_metadata(self):
        with patch.object(launcher, "STATE", self.root):
            folder = launcher.prepare_kernel(self.cfg, {"mode": "smoke"}, "rfm-smoke")
        meta = launcher.read(folder / "kernel-metadata.json")
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["dataset_sources"], [])
        compile((folder / "run_kaggle.py").read_text(), "run_kaggle.py", "exec")

    def test_archive_escape_and_symlink_are_rejected(self):
        for name in ("../outside", "/outside", "safe/../../outside", "safe\\..\\..\\outside"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                item = zipfile.ZipInfo("placeholder")
                item.filename = name  # Preserve a raw backslash on Windows too.
                z.writestr(item, b"bad")
            buffer.seek(0)
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.extract_checked(buffer, self.root)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            item = zipfile.ZipInfo("link")
            item.external_attr = 0o120777 << 16
            z.writestr(item, "../outside")
        buffer.seek(0)
        with self.assertRaises(ValueError):
            runner.extract_checked(buffer, self.root)

    def test_data_mount_and_archive_layouts(self):
        inputs, work = self.root / "inputs", self.root / "work"
        inputs.mkdir()
        work.mkdir()
        archive = inputs / "glaucoma_first.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for split in ("train", "valid", "test"):
                z.writestr(f"glaucoma_first/{split}/", "")
                z.writestr(f"glaucoma_first/{split}.csv", "Image_name,Age\n")
        data = runner.find_data(inputs, work)
        self.assertEqual(data, work / "data/glaucoma_first")
        self.assertEqual(runner.find_data(work, work), data)

    def test_results_keep_resume_evidence_exclude_raw_data_and_caches(self):
        project, work = self.root / "project", self.root / "work"
        work.mkdir()
        output = project / self.job["output"]
        output.mkdir(parents=True)
        for name in ("resume.pt", "model.pt", "holdout.csv", "holdout.json", "status.json", "history.json"):
            (output / name).write_text("evidence")
        (project / "artifacts/pixel_cache").mkdir()
        (project / "artifacts/pixel_cache/big.npy").write_text("excluded")
        (project / "artifacts/rfm_budget.json").write_text("retired budget")
        runner.package_results(project, work, self.job, {"status": "complete"})
        with zipfile.ZipFile(work / "results.zip") as z:
            names = z.namelist()
            self.assertIn(self.job["output"] + "/resume.pt", names)
            self.assertIn(self.job["output"] + "/holdout.json", names)
            self.assertFalse(any("pixel_cache" in n or "rfm_budget" in n for n in names))

    def test_resource_snapshot_keeps_fold_identity_and_rfm_hashes(self):
        audit = self.root / "artifacts/audit"
        audit.mkdir(parents=True)
        weights = self.root / "weights"
        weights.mkdir()
        for name in launcher.AUDIT_FILES:
            (audit / name).write_text("fixture")
        launcher.write(audit / "report.json", {"complete": True, "folds_ready": True, "failures": 0,
                       "fold_fingerprint": launcher.sha(audit / "folds.csv"), "data_fingerprint": "fixed"})
        (weights / "rfm-1.3.pt").write_bytes(b"weight fixture")
        (weights / "rfm_LICENSE.txt").write_text("license fixture")
        launcher.write(weights / "rfm_source.json", {"sha256": launcher.sha(weights / "rfm-1.3.pt")})
        with patch.object(launcher, "ROOT", self.root), patch.object(launcher, "STATE", self.root / "stage"):
            folder = launcher.prepare_resources(self.cfg)
        with zipfile.ZipFile(folder / "resources.zip") as z:
            manifest = json.loads(z.read("resources-manifest.json"))
            self.assertEqual(manifest["data_fingerprint"], "fixed")
            for name, expected in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), expected)
            self.assertFalse(any("green" in n or "rfm_budget" in n for n in z.namelist()))
        for expanded in (False, True):
            inputs, project = self.root / f"inputs-{expanded}", self.root / f"project-{expanded}"
            inputs.mkdir()
            project.mkdir()
            if expanded:
                runner.extract_checked(folder / "resources.zip", inputs)
            else:
                import shutil
                shutil.copyfile(folder / "resources.zip", inputs / "resources.zip")
            runner.stage_resources(inputs, project)
            self.assertEqual((project / "weights/rfm-1.3.pt").read_bytes(), b"weight fixture")
            if expanded:
                (inputs / "weights/rfm-1.3.pt").write_bytes(b"damaged")
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    runner.stage_resources(inputs, project)

    def completed_upload_fixture(self):
        folder, cache = self.root / "stage", self.root / "cache"
        folder.mkdir()
        cache.mkdir()
        archive = folder / "glaucoma_first.zip"
        archive.write_bytes(b"completed upload fixture")
        item = {"path": str(archive.resolve()), "upload_complete": True,
                "timestamp": int(time.time()),
                "start_blob_upload_request": {"contentLength": archive.stat().st_size,
                    "lastModifiedEpochSeconds": int(archive.stat().st_mtime)},
                "start_blob_upload_response": {"token": "test-only-blob-reference"}}
        launcher.write(cache / "upload.json", item)
        return folder, cache, item

    def test_completed_blob_recovery_accepts_only_unchanged_completed_upload(self):
        folder, cache, item = self.completed_upload_fixture()
        self.assertEqual(launcher.completed_upload(folder, cache), item)
        for key, value in (("upload_complete", False), ("timestamp", 0)):
            launcher.write(cache / "upload.json", {**item, key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launcher.completed_upload(folder, cache)
        launcher.write(cache / "upload.json", item)
        (folder / "glaucoma_first.zip").write_bytes(b"changed data")
        with self.assertRaisesRegex(RuntimeError, "changed after upload"):
            launcher.completed_upload(folder, cache)

    def test_creation_failure_recovers_without_repeating_file_transfer(self):
        error = subprocess.CalledProcessError(1, ["kaggle", "datasets", "create"])
        with patch.object(launcher, "cli", side_effect=error) as cli, \
                patch.object(launcher, "completed_upload", return_value={}), \
                patch.object(launcher, "finalize_upload") as finalize:
            launcher.upload_dataset(self.root, "data")
        self.assertEqual(cli.call_count, 1)
        finalize.assert_called_once_with(self.root, "data")

    def test_incomplete_upload_never_triggers_metadata_only_recovery(self):
        error = subprocess.CalledProcessError(1, ["kaggle", "datasets", "create"])
        with patch.object(launcher, "cli", side_effect=error), \
                patch.object(launcher, "completed_upload", side_effect=RuntimeError("incomplete")), \
                patch.object(launcher, "finalize_upload") as finalize:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                launcher.upload_dataset(self.root, "data")
        finalize.assert_not_called()

    def test_version_failure_does_not_create_a_different_dataset(self):
        error = subprocess.CalledProcessError(1, ["kaggle", "datasets", "version"])
        with patch.object(launcher, "cli", side_effect=error), \
                patch.object(launcher, "finalize_upload") as finalize:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.upload_dataset(self.root, "data", version=True)
        finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
