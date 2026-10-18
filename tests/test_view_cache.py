import importlib.util
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from glaucoma import common, engine, view_cache
from glaucoma.common import ROOT, read_json, sha256, write_json
from glaucoma.data import AgeTransform, load_rows
from glaucoma.multiview import VIEW_VERSION, image_dataset
from glaucoma.performance import profile_identity
import test_multiview as multi


class NativeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for i in range(8):
            rng = np.random.default_rng(i)
            path = self.root / f"source{i}.jpg"
            array = rng.integers(30, 256, (150, 180, 3), dtype=np.uint8) if i else np.zeros((150, 180, 3), np.uint8)
            Image.fromarray(array).save(path)
            stat = path.stat()
            self.rows.append(dict(name=path.name, path=str(path), age=30+i, label=i % 2,
                                  file_size=stat.st_size, file_mtime_ns=stat.st_mtime_ns))
        self.config = dict(size=32, native_resolution=True, multiview=True, view_version=VIEW_VERSION)
        self.disk = patch("glaucoma.view_cache.shutil.disk_usage", return_value=SimpleNamespace(free=100*1024**3))
        self.disk.start()

    def tearDown(self):
        self.disk.stop()
        self.temp.cleanup()

    def build(self, budget=None):
        return view_cache.build_view_cache(self.rows, self.config, self.root, "data", budget, workers=2)

    def test_exact_pixels_shared_augmentations_status_and_spawn_reopen(self):
        cache = self.build()
        raw = image_dataset(self.rows, self.config, augment=True)
        fast = image_dataset(self.rows, self.config, augment=True, cache=cache)
        for i in range(8):
            random.seed(55+i)
            a = raw[i]
            random.seed(55+i)
            b = fast[i]
            for key in a[0]:
                torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)
            self.assertEqual(a[3], b[3])
        self.assertFalse(fast[0][0]["mask"].any())
        restored = pickle.loads(pickle.dumps(fast))
        self.assertEqual(restored.view_arrays, {})
        random.seed(10)
        a = raw[1][0]
        random.seed(10)
        b = restored[1][0]
        torch.testing.assert_close(a["local"], b["local"], atol=0, rtol=0)
        del fast, restored, raw

    def test_global_is_reused_without_duplicate_decode_or_storage(self):
        global_config = {**self.config, "multiview": False}
        original = view_cache.build_view_cache(self.rows, global_config, self.root, "data", workers=2)
        before = Path(original["global"]).stat().st_mtime_ns
        with patch.object(view_cache, "decode_global", side_effect=AssertionError("must reuse global")):
            combined = self.build()
        self.assertEqual(original["global"], combined["global"])
        self.assertEqual(before, Path(original["global"]).stat().st_mtime_ns)
        self.assertEqual(len(list((self.root / "pixel_cache").glob("*_global.npy"))), 1)
        with patch.object(view_cache, "native_pixels", wraps=view_cache.native_pixels) as decode:
            view_cache.decode_local(self.rows[1]["path"], 32)
            self.assertFalse(decode.call_args.kwargs["include_global"])

    def test_partial_resume_identity_and_incomplete_discovery(self):
        class Stop:
            calls = 0
            def check(self):
                self.calls += 1
                if self.calls > 3:
                    raise TimeoutError("bounded")
        with self.assertRaises(TimeoutError):
            self.build(Stop())
        self.assertIsNone(view_cache.find_view_cache(self.rows, self.config, self.root, "data"))
        with patch.object(view_cache, "decode_global", wraps=view_cache.decode_global) as decode:
            cache = self.build()
            self.assertEqual(decode.call_count, 5)
        self.assertIsNotNone(cache)
        for rows, config, fingerprint in ((self.rows[::-1], self.config, "data"),
                                            (self.rows, {**self.config, "size": 48}, "data"),
                                            (self.rows, self.config, "different")):
            self.assertIsNone(view_cache.find_view_cache(rows, config, self.root, fingerprint))
        path = Path(cache["local"]).with_suffix(".done.npy")
        done = np.load(path, mmap_mode="r+")
        done[0] = False
        done.flush()
        del done
        self.assertIsNone(view_cache.find_view_cache(self.rows, self.config, self.root, "data"))

    def test_combined_space_reserve_and_version_invalidation(self):
        with patch("glaucoma.view_cache.shutil.disk_usage", return_value=SimpleNamespace(free=30*1024**3)):
            with self.assertRaisesRegex(RuntimeError, "30GiB"):
                self.build()
        cache = self.build()
        meta = Path(cache["global"]).with_suffix(".json")
        payload = read_json(meta)
        payload["view_version"] = "stale"
        write_json(meta, payload)
        self.assertIsNone(view_cache.find_view_cache(self.rows, self.config, self.root, "data"))


class CachedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = multi.MultiviewTrainingTests(methodName="test_old_and_new_checkpoint_roundtrip_masks_and_gradients")
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_actual_training_validation_and_checkpoint_equal_to_raw(self):
        f = self.fixture
        rows = load_rows(f.root, "train")
        config = {**f.config, "native_resolution": True, "multiview": True, "view_version": VIEW_VERSION}
        with patch("glaucoma.view_cache.shutil.disk_usage", return_value=SimpleNamespace(free=100*1024**3)):
            cache = view_cache.build_view_cache(rows, config, f.artifacts, "synthetic", workers=2)
        profile = f.root / "native_profile.json"
        write_json(profile, {"format_version": 1, "identity": profile_identity(config, "cpu"),
                            "native_view_cache_verified": True,
                            "stages": {p: {"accepted": True, "settings": {"use_pixel_cache": True}}
                                       for p in ("train", "validation")}})
        raw, fast = f.artifacts / "raw", f.artifacts / "cached"
        engine.fine_tune(f.root, config, raw, workers=0, device="cpu")
        engine.fine_tune(f.root, config, fast, workers=0, device="cpu", performance_profile=profile)
        self.assertEqual(sha256(raw / "holdout.csv"), sha256(fast / "holdout.csv"))
        a, b = [torch.load(p / "model.pt", weights_only=True) for p in (raw, fast)]
        for key in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=0, rtol=0)
        self.assertEqual(b["metadata"]["pixel_cache"], cache)
        self.assertIsNone(a["metadata"]["pixel_cache"])


class BudgetAndBenchmarkTests(unittest.TestCase):
    def test_budget_heartbeat_never_double_counts_or_resets(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_json(folder / "rfm_budget.json", {"used_seconds": 100})
            clock = [1000.]
            with patch.object(common, "ARTIFACTS", folder), patch.object(common.time, "monotonic", side_effect=lambda: clock[0]):
                with common.Budget(True) as budget:
                    clock[0] += 31
                    budget.check()
                    self.assertEqual(read_json(folder / "rfm_budget.json")["used_seconds"], 131)
                    clock[0] += 14
                self.assertEqual(read_json(folder / "rfm_budget.json")["used_seconds"], 145)

    def test_sampler_switch_keeps_indices_and_labels(self):
        spec = importlib.util.spec_from_file_location("native_optimizer", ROOT / "scripts/optimize-p1p2.py")
        optimizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimizer)
        sampler = optimizer.PhaseSampler()
        self.assertEqual(list(sampler), list(range(64)))
        sampler.offset = 64
        self.assertEqual(list(sampler), list(range(64, 128)))

    def test_output_contract_and_parameter_diagnostic_are_separate(self):
        spec = importlib.util.spec_from_file_location("native_optimizer", ROOT / "scripts/optimize-p1p2.py")
        optimizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimizer)
        check = {"max_probability_difference": .0003, "metric_differences": {"auroc": 0., "auprc": 0.},
                 "optimizer_steps_equal": True, "training_order_equal": True, "parameters_close": False}
        self.assertTrue(optimizer.required_check(check))
        self.assertFalse(optimizer.required_check({**check, "max_probability_difference": .0011}))
        self.assertFalse(optimizer.required_check({**check, "training_order_equal": False}))


if __name__ == "__main__":
    unittest.main()
