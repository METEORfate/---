import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from glaucoma import feature_mlp as mlp
from glaucoma.common import read_json, write_csv, write_json
from glaucoma.data import AgeTransform
from glaucoma.models import FundusModel, load_checkpoint, save_checkpoint


class TinyMLP(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.head = mlp.make_head(387)
        self.register_buffer("head_mean", torch.zeros(387))
        self.register_buffer("head_scale", torch.ones(387))


class FeatureMLPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"model": "green", "size": 392, "use_age": True, "head": "mlp", "head_standardize": True,
                       "tail_blocks": 0, "epochs": 2, "batch_size": 6, "learning_rate": .001,
                       "weight_decay": .01, "patience": 1}

    def tearDown(self):
        self.temp.cleanup()

    def test_three_fold_fit_does_not_fit_heldout_age_or_use_valid(self):
        rows = [{"name": f"image{i}.jpg", "age": 137 if i % 3 == 0 else 30+i,
                 "label": i % 2} for i in range(18)]
        features = self.root / "train.npy"
        np.save(features, np.random.default_rng(42).normal(size=(18, 384)).astype(np.float32))
        write_json(features.with_suffix(".json"), {"complete": True, "names": [r["name"] for r in rows],
                   "signature": {"model": "green", "size": 392, "data": "synthetic"}})
        write_csv(self.root / "audit" / "folds.csv", [{"Image_name": r["name"], "fold": i % 3}
                   for i, r in enumerate(rows)], ["Image_name", "fold"])
        with patch.object(mlp, "ARTIFACTS", self.root), patch.object(mlp, "require_audit", return_value={"data_fingerprint": "synthetic"}), \
             patch.object(mlp, "load_rows", return_value=rows) as read_rows, \
             patch.object(mlp, "folds_for", return_value=np.arange(18) % 3), \
             patch.object(mlp, "download_weights", return_value=None), patch.object(mlp, "FundusModel", TinyMLP):
            report = mlp.run_trial(self.root, features, self.config, self.root / "result", minutes=1)
            read_rows.assert_called_once_with(self.root, "train")
            self.assertLess(report["folds"][0]["age_transform"]["upper"], 137)
            self.assertTrue(read_json(self.root / "result" / "status.json")["complete"])
            with (self.root / "result" / "oof.csv").open(encoding="utf-8", newline="") as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual([r["Image_name"] for r in predictions], [r["name"] for r in rows])
            self.assertEqual({int(r["fold"]) for r in predictions}, {0, 1, 2})
            self.assertEqual(mlp.run_trial(self.root, features, self.config, self.root / "result", minutes=1), report)

    def test_standardized_head_survives_checkpoint_roundtrip(self):
        model = FundusModel(self.config).eval()
        model.head_mean.copy_(torch.linspace(-1, 1, 387))
        model.head_scale.copy_(torch.linspace(.2, 2, 387))
        images = torch.randn(3, 384)
        ages = torch.randn(3, 3)
        with torch.inference_mode():
            expected = model.head((torch.cat([images, ages], 1)-model.head_mean)/model.head_scale).squeeze(1)
            torch.testing.assert_close(model.classify(images, ages), expected)
        path = self.root / "model.pt"
        save_checkpoint(path, model, self.config, AgeTransform.fit([30, 40]), {"fold": "full"})
        restored, _, _ = load_checkpoint(path)
        with torch.inference_mode():
            torch.testing.assert_close(restored.classify(images, ages), expected)

    def test_expired_trial_stops_before_training(self):
        x = np.ones((8, 4), np.float32)
        with self.assertRaises(TimeoutError):
            mlp.fit_head(x, np.arange(8) % 2, x, np.arange(8) % 2, self.config, 42, time.time()-1)


if __name__ == "__main__":
    unittest.main()
