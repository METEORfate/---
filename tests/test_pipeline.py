import csv
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glaucoma.audit import inspect, make_groups
from glaucoma.common import read_json, write_csv, write_json
from glaucoma.data import AgeTransform, ImageDataset, crop_fov, load_rows, read_image
from glaucoma.deploy import check_submission, export_bundle, write_submission
from glaucoma.metrics import metrics, paired_bootstrap


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, split="train", label=1, age="55"):
        folder = self.root / split
        folder.mkdir(exist_ok=True)
        image = Image.new("RGB", (160, 120), (0, 0, 0))
        ImageDraw.Draw(image).ellipse((25, 5, 135, 115), fill=(130, 70, 35))
        image.save(folder / "example.JPG")
        fields = ["Image_name", "Age"] + (["True_Label"] if split != "test" else [])
        record = {"Image_name": "example.jpg", "Age": age, "True_Label": label}
        write_csv(self.root / f"{split}.csv", [record], fields)
        return load_rows(self.root, split)

    def test_linux_case_mapping_and_original_name(self):
        rows = self.fixture()
        self.assertEqual(Path(rows[0]["path"]).name, "example.JPG")
        self.assertEqual(rows[0]["name"], "example.jpg")

    def test_unlisted_images_fail(self):
        self.fixture()
        Image.new("RGB", (10, 10)).save(self.root / "train" / "extra.JPG")
        with self.assertRaises(ValueError):
            load_rows(self.root, "train")

    def test_age_fit_never_observes_validation(self):
        transform = AgeTransform.fit([40, 60, float("nan")])
        observed = transform.transform([137, float("nan"), 0])
        self.assertEqual(transform.upper, 60)
        self.assertTrue(np.isfinite(observed).all())
        np.testing.assert_array_equal(observed[:, 1:], [[0, 1], [1, 0], [0, 1]])
        np.testing.assert_array_equal(observed, AgeTransform(**transform.to_dict()).transform([137, float("nan"), 0]))

    def test_dark_crop_falls_back(self):
        image = Image.new("RGB", (160, 120), (4, 2, 1))
        result, changed = crop_fov(image)
        self.assertFalse(changed)
        self.assertEqual(result.size, image.size)

    def test_full_field_letterbox_and_large_input(self):
        rows = self.fixture()
        arr = read_image(rows[0]["path"], 392)
        self.assertEqual(arr.shape, (392, 392, 3))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertGreater(arr[:, :, 0].max(), 100)

    def test_truncated_native_pixels_are_rejected(self):
        rows = self.fixture()
        path = Path(rows[0]["path"])
        path.write_bytes(path.read_bytes()[:300])
        rows = load_rows(self.root, "train")
        self.assertFalse(inspect(rows[0])["ok"])

    def test_exact_duplicates_share_group_and_conflicts_are_visible(self):
        first = self.fixture("train", 1)
        second = self.fixture("valid", 0)
        shutil.copyfile(first[0]["path"], second[0]["path"])
        records = [inspect(first[0]), inspect(second[0])]
        groups, exact, nearby, conflicts = make_groups(records, first+second)
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(groups[first[0]["key"]], groups[second[0]["key"]])

    def test_hash_candidates_match_exhaustive_hamming_radius(self):
        import base64
        rng = np.random.default_rng(2026)
        hashes = []
        for i in range(20):
            h = int(rng.integers(0, 2**63, dtype=np.uint64))
            hashes.extend([h, h ^ sum(1 << int(b) for b in rng.choice(63, i % 4 + 1, replace=False))])
        records = [{"key": str(i), "ok": True, "phash": f"{h:016x}", "sha256": str(i),
                    "thumb": base64.b64encode(bytes(1024)).decode()} for i, h in enumerate(hashes)]
        rows = [{"key": str(i), "label": 0, "split": "train"} for i in range(len(records))]
        _, _, nearby, _ = make_groups(records, rows)
        actual = {(int(p["a"]), int(p["b"])) for p in nearby}
        expected = {(i, j) for j in range(len(hashes)) for i in range(j) if (hashes[i] ^ hashes[j]).bit_count() <= 4}
        self.assertEqual(actual, expected)

    def test_submission_preserves_order_threshold_and_precision(self):
        rows = [{"name": "b.jpg"}, {"name": "a.jpg"}, {"name": "c.jpg"}]
        path = self.root / "submission.csv"
        write_submission(path, rows, [0.499999999999999, .5, 1.0])
        self.assertEqual(check_submission(path, rows)["rows"], 3)
        with self.assertRaises(ValueError):
            check_submission(path, list(reversed(rows)))
        self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_invalid_submission_fails(self):
        with self.assertRaises(ValueError):
            write_submission(self.root / "bad.csv", [{"name": "a.jpg"}], [float("nan")])
        with self.assertRaises(ValueError):
            write_submission(self.root / "bad.csv", [{"name": "a.jpg"}]*2, [.2, .3])

    def test_windows_spawn_cache_does_not_pickle_pixel_contents(self):
        import pickle
        rows = self.fixture()
        path = self.root / "pixels.npy"
        np.save(path, np.zeros((1, 392, 392, 3), dtype=np.uint8))
        dataset = ImageDataset(rows, 392, cache=path)
        serialized = pickle.dumps(dataset)
        self.assertLess(len(serialized), 10000)
        restored = pickle.loads(serialized)
        self.assertIsNone(restored.cache)
        self.assertEqual(tuple(restored[0][0].shape), (3, 392, 392))

    def test_paired_bootstrap_identical_predictions(self):
        y, p = [0, 0, 1, 1], [.1, .2, .8, .9]
        interval = paired_bootstrap(y, p, p, repeats=50)
        self.assertEqual(interval["lower_95"], 0)
        self.assertEqual(interval["upper_95"], 0)
        self.assertEqual(metrics(y, p)["auroc"], 1)

    def test_export_does_not_claim_unmeasured_t4_pass(self):
        artifact = self.root / "fake.pt"
        artifact.write_bytes(b"test")
        with self.assertRaises(ValueError):
            export_bundle(artifact, self.root / "export")
        self.assertFalse((self.root / "export").exists())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(4)

    def test_green_official_pooling_and_checkpoint_roundtrip(self):
        import torch
        from glaucoma.models import FundusModel, load_checkpoint, save_checkpoint
        config = {"model": "green", "size": 392, "use_age": True, "head": "linear"}
        model = FundusModel(config).eval()
        torch.manual_seed(42)
        x, age = torch.rand(1, 3, 392, 392), torch.zeros(1, 3)
        with torch.no_grad():
            tokens = model.backbone.forward_features((x-model.mean)/model.std)
            expected = tokens[:, model.backbone.num_prefix_tokens:].mean(1)
            np.testing.assert_allclose(model.features(x).numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
            original = model(x, age)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, config, AgeTransform.fit([40, 60]), {"fold": "full"})
            reloaded, _, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(original, reloaded(x, age), rtol=0, atol=0)
        model.configure_trainable(2)
        self.assertFalse(next(model.backbone.blocks[0].parameters()).requires_grad)
        self.assertTrue(next(model.backbone.blocks[-1].parameters()).requires_grad)

    def test_rfm_block9_feature_shape_without_remote_code(self):
        import torch
        from glaucoma.models import FundusModel
        model = FundusModel({"model": "rfm", "size": 336, "use_age": True}).eval()
        self.assertEqual(len(model.backbone.blocks), 9)
        with torch.no_grad():
            result = model.features(torch.zeros(1, 3, 336, 336))
        self.assertEqual(tuple(result.shape), (1, 1536))
        self.assertTrue(torch.isfinite(result).all())

    def test_cnn_normalization_matches_official_model_config(self):
        import timm
        from glaucoma.models import FundusModel
        model = FundusModel({"model": "effnet", "size": 384, "use_age": False})
        config = timm.data.resolve_model_data_config(model.backbone)
        np.testing.assert_allclose(model.mean.flatten().numpy(), config["mean"])
        np.testing.assert_allclose(model.std.flatten().numpy(), config["std"])


if __name__ == "__main__":
    unittest.main()
