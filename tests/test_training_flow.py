"""Small synthetic integration tests. These are implementation checks, not scores."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from glaucoma import engine
from glaucoma.common import read_json, write_csv, write_json
from glaucoma.data import load_rows


class TinyFundus(nn.Module):
    """Fast local test double for the backbone interface, no external weights."""
    kind = "effnet"
    size = 32

    def __init__(self, config, pretrained_path=None):
        super().__init__()
        self.backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))
        self.head = nn.Linear(7, 1)

    def configure_trainable(self, tail_blocks):
        return self

    def features(self, images):
        return self.backbone(images)

    def classify(self, features, ages):
        return self.head(torch.cat([features, ages], 1)).squeeze(1)

    def forward(self, images, ages):
        return self.classify(self.features(images), ages)


class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        image_dir = self.root / "train"
        image_dir.mkdir()
        records = []
        for i in range(18):
            name = f"image{i:02}.jpg"
            Image.new("RGB", (40, 32), (30+i*5, 60, 90)).save(image_dir / name.upper())
            records.append({"Image_name": name, "Age": 137 if i % 3 == 0 else 30+i, "True_Label": i % 2})
        write_csv(self.root / "train.csv", records, ["Image_name", "Age", "True_Label"])
        write_csv(self.artifacts / "audit" / "folds.csv",
                  [{"Image_name": r["Image_name"], "fold": i % 3, "group": str(i)} for i, r in enumerate(records)],
                  ["Image_name", "fold", "group"])
        self.config = {"model": "green", "size": 32, "use_age": True, "head": "linear", "tail_blocks": -1,
                       "epochs": 1, "batch_size": 2, "accumulation": 2, "head_warmup_epochs": 0,
                       "auto_pixel_cache": False}
        self.patches = [patch.object(engine, "ARTIFACTS", self.artifacts),
                        patch.object(engine, "require_audit", return_value={"data_fingerprint": "synthetic"}),
                        patch.object(engine, "FundusModel", TinyFundus),
                        patch.object(engine, "download_weights", return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_training_holdout_age_scope_and_resume(self):
        output = self.artifacts / "training"
        engine.fine_tune(self.root, self.config, output, fold=0, epochs=1, workers=0, device="cpu")
        payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(payload["metadata"]["fold"], 0)
        self.assertLess(payload["age_transform"]["upper"], 137)
        with (output / "holdout.csv").open(encoding="utf-8", newline="") as f:
            predictions = list(csv.DictReader(f))
        self.assertEqual(len(predictions), 6)
        self.assertEqual({int(r["fold"]) for r in predictions}, {0})
        engine.fine_tune(self.root, self.config, output, fold=0, epochs=1, workers=0, device="cpu", resume=True)
        self.assertTrue(read_json(output / "status.json")["complete"])
        with self.assertRaises(ValueError):
            engine.fine_tune(self.root, self.config, output, fold=1, epochs=1, workers=0, device="cpu", resume=True)

    def test_probe_full_oof_and_no_official_valid_fit(self):
        rows = load_rows(self.root, "train")
        rng = np.random.default_rng(42)
        x = rng.normal(size=(18, 4)).astype(np.float32)
        path = self.artifacts / "features.npy"
        np.save(path, x, allow_pickle=False)
        write_json(path.with_suffix(".json"), {"complete": True, "names": [r["name"] for r in rows],
                   "signature": {"model": "green", "size": 32, "data": "synthetic"}})
        output = self.artifacts / "probe"
        engine.train_probe(self.root, self.config, path, output)
        with (output / "oof.csv").open(encoding="utf-8", newline="") as f:
            predictions = list(csv.DictReader(f))
        self.assertEqual([p["Image_name"] for p in predictions], [r["name"] for r in rows])
        self.assertEqual({int(p["fold"]) for p in predictions}, {0, 1, 2})
        self.assertTrue(all(np.isfinite(float(p["Probability"])) for p in predictions))
        self.assertEqual(read_json(output / "oof.json")["data_fingerprint"], "synthetic")
        self.assertFalse((self.artifacts / "valid_milestones.json").exists())

    def test_unfinished_fold_cannot_be_collected_or_finalized(self):
        from glaucoma import cli, deploy
        output = self.artifacts / "unfinished"
        write_json(output / "status.json", {"complete": False})
        with self.assertRaisesRegex(ValueError, "Finish or resume"):
            deploy.collect_oof(self.root, [output / "holdout.csv"] * 3, output / "oof.csv")
        config_path = self.root / "config.json"
        write_json(config_path, self.config)
        with self.assertRaisesRegex(ValueError, "Finish or resume"):
            cli.main(["finalize", "--config", str(config_path), "--cv-dirs", *([str(output)] * 3),
                      "--output", str(output / "full")])


if __name__ == "__main__":
    unittest.main()
