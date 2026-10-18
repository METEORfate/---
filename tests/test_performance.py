import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from glaucoma import engine, performance, pixel_cache
from glaucoma.common import OutputBusy, exclusive_output, read_json, seed_all, write_json
from glaucoma.data import AgeTransform, ImageDataset, load_rows, read_image
import test_training_flow as fixtures

spec = importlib.util.spec_from_file_location("performance_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark-training-io.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PerformanceTests(unittest.TestCase):
    def test_defaults_explicit_overrides_and_rejected_stage(self):
        config = {"batch_size": 4}
        profile = {"stages": {"train": {"accepted": True, "settings": {"workers": 4, "reduced_sync": True}},
                              "validation": {"accepted": True, "settings": {"batch_size": 32, "workers": 0}},
                              "features": {"accepted": False, "settings": {"batch_size": 32}}}}
        self.assertEqual(performance.execution_settings(config, "train")["workers"], 2)
        self.assertFalse(performance.execution_settings(config, "train")["reduced_sync"])
        self.assertFalse(performance.execution_settings(config, "features")["use_pixel_cache"])
        self.assertEqual(performance.execution_settings(config, "validation", profile)["batch_size"], 32)
        self.assertEqual(performance.execution_settings(config, "validation", profile, 2, 8)["batch_size"], 8)
        self.assertEqual(performance.execution_settings(config, "features", profile)["batch_size"], 16)
        profile["stages"]["train"] = {"accepted": False, "settings": {}, "baseline_settings": {"workers": 4}}
        self.assertEqual(performance.execution_settings(config, "train", profile)["workers"], 4)

    def test_profile_binding_and_training_batch_cannot_change(self):
        config = {"model": "green", "size": 32, "batch_size": 4}
        profile = {"format_version": 1, "identity": performance.profile_identity(config, "cpu"),
                   "stages": {"train": {"accepted": True, "settings": {"batch_size": 4}}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            write_json(path, profile)
            self.assertEqual(performance.load_profile(path, config, "cpu"), profile)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                performance.load_profile(path, {**config, "size": 64}, "cpu")
            changed = copy.deepcopy(profile)
            changed["identity"]["environment"]["versions"]["torch"] = "other"
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                performance.load_profile(path, config, "cpu")
            profile["stages"]["train"]["settings"]["batch_size"] = 8
            write_json(path, profile)
            with self.assertRaisesRegex(ValueError, "training batch"):
                performance.load_profile(path, config, "cpu")

    def test_net_cost_threshold_incomplete_repeats_and_numerical_failure(self):
        slow = [{"images_per_second": 100, "total_seconds": 6, "total_images": 512}]*3
        fast = [{"images_per_second": 120, "total_seconds": 5, "total_images": 512}]*3
        checks = [{"passed": True}]*3
        self.assertTrue(performance.recommend(slow, fast, checks, expected_images=5120)["accepted"])
        self.assertFalse(performance.recommend(slow, fast, checks, 20, 5120)["accepted"])
        self.assertFalse(performance.recommend(slow, fast[:2], checks, expected_images=5120)["accepted"])
        self.assertFalse(performance.recommend(slow, fast, [{"passed": False}]*3, expected_images=5120)["accepted"])
        for r in fast:
            r["images_per_second"] = 109
        self.assertFalse(performance.recommend(slow, fast, checks, expected_images=5120)["accepted"])

    def test_cache_payback_and_disk_reserve(self):
        self.assertTrue(performance.cache_payback(.1, .01, 1000, 20, 1024, 40*1024**3)["worth_building"])
        self.assertFalse(performance.cache_payback(.1, .01, 10, 20, 1024, 40*1024**3)["worth_building"])
        self.assertFalse(performance.cache_payback(.1, .01, 1000, 20, 1024, 30*1024**3)["worth_building"])

    def test_predictions_parameters_and_update_count_gate(self):
        result = {"indices": [2, 0, 1, 3], "labels": [0, 1, 0, 1], "probabilities": [.1, .8, .2, .9],
                  "optimizer_steps": 3, "parameters": {"head": torch.tensor([.5])}}
        self.assertTrue(performance.numerical_check(result, copy.deepcopy(result))["passed"])
        for key, value in (("indices", [0, 1, 2, 3]), ("optimizer_steps", 2),
                           ("parameters", {"head": torch.tensor([.51])}), ("probabilities", [.1, .8, .2, .91])):
            changed = copy.deepcopy(result)
            changed[key] = value
            self.assertFalse(performance.numerical_check(result, changed)["passed"])

    def test_real_updates_and_partial_accumulation_are_equivalent(self):
        torch.set_num_threads(2)
        seed_all(42)
        config = {"accumulation": 3, "batch_size": 2}
        data = TensorDataset(torch.rand(14, 3, 8, 8), torch.rand(14, 3), torch.tensor([0., 1.]*7), torch.arange(14))
        initial = fixtures.TinyFundus(config).state_dict()
        outputs = []
        for reduced in (False, True):
            model = fixtures.TinyFundus(config)
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            result = engine.train_batches(model, engine.loader(data, 2, 0), optimizer, scaler, "cpu", config,
                                          reduced_sync=reduced)
            self.assertEqual(result["optimizer_steps"], math.ceil(7/3))
            outputs.append((result, copy.deepcopy(model.state_dict())))
        self.assertAlmostEqual(outputs[0][0]["loss"], outputs[1][0]["loss"], places=7)
        for name in initial:
            self.assertTrue(torch.equal(outputs[0][1][name], outputs[1][1][name]))
        self.assertFalse(torch.equal(initial["head.weight"], outputs[0][1]["head.weight"]))


class PerformanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.TrainingFlowTests.setUp(self)

    def tearDown(self):
        fixtures.TrainingFlowTests.tearDown(self)

    def profile(self):
        path = self.root / "profile.json"
        write_json(path, {"format_version": 1, "identity": performance.profile_identity(self.config, "cpu"),
                          "stages": {"train": {"accepted": True, "settings": {"reduced_sync": True, "workers": 0}},
                                     "validation": {"accepted": True, "settings": {"batch_size": 8, "workers": 0}},
                                     "features": {"accepted": True, "settings": {"batch_size": 4, "workers": 0, "use_pixel_cache": True}}}})
        return path

    def test_existing_signature_can_resume_with_execution_profile(self):
        self.config["epochs"] = 2
        output = self.artifacts / "model"
        class StopInSecondEpoch:
            def check(self):
                if (output / "resume.pt").exists():
                    raise TimeoutError("test interrupted after a saved epoch")
        with self.assertRaises(TimeoutError):
            engine.fine_tune(self.root, self.config, output, workers=0, device="cpu", budget=StopInSecondEpoch())
        saved = torch.load(output / "resume.pt", weights_only=True)
        signature = copy.deepcopy(saved["signature"])
        self.assertEqual(saved["epoch"], 0)
        engine.fine_tune(self.root, self.config, output, workers=0, device="cpu", resume=True, performance_profile=self.profile())
        current = torch.load(output / "resume.pt", weights_only=True)
        self.assertEqual(current["signature"], signature)
        self.assertTrue(read_json(output / "status.json")["complete"])
        control = self.artifacts / "uninterrupted"
        engine.fine_tune(self.root, self.config, control, workers=0, device="cpu")
        expected = torch.load(control / "resume.pt", weights_only=True)
        for key in current["model"]:
            self.assertTrue(torch.equal(current["model"][key], expected["model"][key]))
        self.assertFalse((self.artifacts / "valid_milestones.json").exists())

    def test_cache_discovery_rejects_changed_order_size_and_partial_files(self):
        rows = load_rows(self.root, "train")
        path = self.artifacts / "pixel_cache/train_32_synthetic.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.zeros((len(rows), 32, 32, 3), dtype=np.uint8))
        self.assertIsNone(pixel_cache.find_pixel_cache(rows, 32, self.artifacts, "synthetic"))
        write_json(path.with_suffix(".json"), pixel_cache.cache_signature(rows, 32))
        self.assertEqual(pixel_cache.find_pixel_cache(rows, 32, self.artifacts, "synthetic"), path)
        self.assertIsNone(pixel_cache.find_pixel_cache(rows[::-1], 32, self.artifacts, "synthetic"))
        self.assertIsNone(pixel_cache.find_pixel_cache(rows, 64, self.artifacts, "synthetic"))
        np.save(path, np.zeros((1, 32, 32, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "shape"):
            pixel_cache.find_pixel_cache(rows, 32, self.artifacts, "synthetic")

    def test_active_queue_prevents_benchmark(self):
        with patch.object(benchmark, "ARTIFACTS", self.artifacts):
            with exclusive_output(self.artifacts / "green_cv/run.lock"):
                with self.assertRaises(OutputBusy):
                    with benchmark.idle_slot():
                        self.fail("must not start")

    def test_idle_but_incomplete_cv_does_not_start_benchmark(self):
        write_json(self.artifacts / "green_cv/state.json", {"status": "needs_resume"})
        with patch.object(benchmark, "ARTIFACTS", self.artifacts):
            with self.assertRaisesRegex(RuntimeError, "finish successfully"):
                with benchmark.idle_slot():
                    self.fail("unfinished CV is not completion")

    def test_features_share_cache_and_resume_only_unfinished_rows(self):
        rows = load_rows(self.root, "train")
        cache = self.artifacts / "pixel_cache/train_32_synthetic.npy"
        cache.parent.mkdir(parents=True)
        np.save(cache, np.stack([read_image(r["path"], 32) for r in rows]))
        write_json(cache.with_suffix(".json"), pixel_cache.cache_signature(rows, 32))
        weight = self.root / "weights.bin"
        weight.write_bytes(b"local test weights")
        class StopAfterTwo:
            calls = 0
            def check(self):
                self.calls += 1
                if self.calls > 2:
                    raise TimeoutError("test feature interruption")
        with patch.object(engine, "download_weights", return_value=weight), \
             patch.dict(engine.SPECS, {"green": {"size": 32, "dim": 4}}), \
             patch("glaucoma.data.read_image", side_effect=AssertionError("must use pixels cache")):
            seed_all(42)
            with self.assertRaises(TimeoutError):
                engine.extract_features(self.root, self.config, device="cpu", budget=StopAfterTwo(), performance_profile=self.profile())
            folder = next((self.artifacts / "features").iterdir())
            self.assertEqual(int(np.load(folder / "train_done.npy").sum()), 8)
            seed_all(42)
            path = engine.extract_features(self.root, self.config, device="cpu", performance_profile=self.profile())
            self.assertTrue(read_json(path.with_suffix(".json"))["complete"])
            self.assertEqual(int(np.load(folder / "train_done.npy").sum()), 18)
            self.assertTrue(np.isfinite(np.load(path)).all())
            self.assertEqual(read_json(path.with_suffix(".json"))["pixel_cache"], str(cache))

    def test_benchmark_runs_real_optimizer_and_feature_save_paths(self):
        rows = load_rows(self.root, "train")
        age = AgeTransform.fit([r["age"] for r in rows])
        config = {**self.config, "accumulation": 3}
        seed_all(42)
        payload = {"state_dict": fixtures.TinyFundus(config).state_dict()}
        settings = {"workers": 0, "batch_size": 2, "use_pixel_cache": False, "reduced_sync": False}
        budget = SimpleNamespace(check=lambda: None)
        with patch.object(benchmark, "FundusModel", fixtures.TinyFundus), patch.dict(benchmark.SPECS, {"green": {"dim": 4}}):
            for stage in ("train", "validation", "features"):
                dataset = ImageDataset(rows, 32, age, stage == "train")
                timing, comparison = benchmark.run_workload(stage, settings, config, payload, dataset,
                                                           list(range(18)), budget, "cpu", self.root)
                self.assertEqual(timing["total_images"], 18)
                self.assertEqual(timing["measured_images"], 2)
                self.assertGreater(timing["images_per_second"], 0)
                self.assertTrue(performance.numerical_check(comparison, copy.deepcopy(comparison))["passed"])
                if stage == "train":
                    self.assertEqual(timing["optimizer_steps"], 3)
                    self.assertTrue((self.root / "training_state.pt").exists())
                if stage == "features":
                    self.assertEqual(int(np.load(self.root / "done.npy").sum()), 18)

    def test_budget_interruption_releases_windows_feature_maps(self):
        rows = load_rows(self.root, "train")
        dataset = ImageDataset(rows, 32, AgeTransform.fit([r["age"] for r in rows]))
        settings = {"workers": 0, "batch_size": 2, "use_pixel_cache": False, "reduced_sync": False}
        class StopAfterFirstBatch:
            calls = 0
            def check(self):
                self.calls += 1
                if self.calls > 2:
                    raise TimeoutError("test bounded extraction")
        with patch.object(benchmark, "FundusModel", fixtures.TinyFundus), patch.dict(benchmark.SPECS, {"green": {"dim": 4}}):
            with self.assertRaises(TimeoutError):
                benchmark.run_workload("features", settings, self.config,
                                       {"state_dict": fixtures.TinyFundus(self.config).state_dict()}, dataset,
                                       list(range(18)), StopAfterFirstBatch(), "cpu", self.root)
            # Windows refuses this if the aborted frame still owns its memmaps.
            (self.root / "features.npy").unlink()
            (self.root / "done.npy").unlink()

    def test_oom_and_incomplete_pair_never_publish_recommendation(self):
        config_path = self.root / "config.json"
        checkpoint = self.artifacts / "checkpoint.pt"
        write_json(config_path, self.config)
        torch.save({"config": self.config, "metadata": {"fold": 0}}, checkpoint)
        for name, failure, expected in (("oom", torch.cuda.OutOfMemoryError("synthetic"), "complete"),
                                        ("budget", TimeoutError("synthetic"), "budget_reached")):
            output = self.artifacts / f"acceleration/{name}"
            args = SimpleNamespace(config=config_path, checkpoint=checkpoint, output=output, device="cpu", fold=0,
                                   count=512, stages=["train"], max_minutes=30, expected_passes=None,
                                   build_cache=False, wait_for_idle=False)
            with patch.object(benchmark, "ARTIFACTS", self.artifacts), patch.object(benchmark, "ROOT", self.root), \
                 patch.object(benchmark, "require_audit", return_value={"data_fingerprint": "synthetic"}), \
                 patch.object(benchmark, "load_rows", return_value=load_rows(self.root, "train")), \
                 patch.object(benchmark, "folds_for", return_value=np.arange(18) % 3), \
                 patch.object(benchmark, "run_workload", side_effect=failure):
                benchmark.benchmark(args)
            self.assertEqual(read_json(output / "report.json")["status"], expected)
            self.assertFalse(read_json(output / "profile.json")["stages"]["train"]["accepted"])
            self.assertTrue(read_json(output / "report.json")["protected_files_unchanged"])


if __name__ == "__main__":
    unittest.main()
