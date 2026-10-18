"""Geometry, mask, real model interface and same-fold initialization checks."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from glaucoma import engine, models
from glaucoma.common import read_json
from glaucoma.data import AgeTransform, load_rows
from glaucoma.models import FundusModel, load_checkpoint, save_checkpoint
from glaucoma.multiview import (VIEW_VERSION, flip_images, image_dataset, move_images,
                                native_views, tensor_image, view_boxes)
import test_training_flow as flow


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(3, 4))
        self.norm, self.fc_norm = nn.LayerNorm(4), nn.Identity()

    def forward(self, x):
        return self.norm(self.blocks(x.mean((-2, -1))))


class ViewGeometryTests(unittest.TestCase):
    def test_original_crops_before_resize_and_positions(self):
        rng = np.random.default_rng(3)
        image = Image.fromarray(rng.integers(30, 255, (800, 960, 3), dtype=np.uint8))
        views = native_views(image, 32)
        boxes = view_boxes(*image.size)
        self.assertEqual(views["local"].shape, (5, 3, 32, 32))
        self.assertTrue(views["mask"].all())
        for i, box in enumerate(boxes):
            torch.testing.assert_close(views["local"][i], tensor_image(image.crop(box), 32), rtol=0, atol=0)
            np.testing.assert_allclose(views["coordinates"][i], np.array(box) / [960, 800, 960, 800])
        # An early global resize destroys this high-frequency original information.
        small = image.resize((32, 32)).resize(image.size)
        self.assertFalse(torch.equal(views["local"], native_views(small, 32)["local"]))
        torch.testing.assert_close(flip_images(views)["coordinates"], views["coordinates"])
        torch.testing.assert_close(flip_images(flip_images(views))["coordinates"], views["coordinates"])

    def test_dark_empty_bag_and_local_failure(self):
        views = native_views(Image.new("RGB", (100, 100)), 32)
        self.assertFalse(views["mask"].any())
        with patch("glaucoma.multiview.view_boxes", return_value=[(20, 20, 10, 10)]):
            views = native_views(Image.new("RGB", (100, 100), "white"), 32)
        self.assertFalse(views["mask"].any())
        self.assertEqual(views["local"].sum().item(), 0)

    def test_common_native_augmentation_reproducible_and_full_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.jpg"
            Image.new("RGB", (2200, 2000), (90, 100, 140)).save(path)
            rows = [{"path": path, "age": 55, "label": 1}]
            config = {"size": 32, "multiview": True, "view_version": VIEW_VERSION}
            age = AgeTransform.fit([55])
            with patch("glaucoma.multiview.decode_image", wraps=__import__("glaucoma.data", fromlist=["decode_image"]).decode_image) as decode:
                data = image_dataset(rows, config, age, True)
                random.seed(42)
                first, ages, label, ix = data[0]
                random.seed(42)
                second = data[0][0]
                self.assertTrue(decode.call_args.kwargs["full_decode"])
            torch.testing.assert_close(first["local"], second["local"], atol=0, rtol=0)
            self.assertEqual(label.ndim, 0)  # One image label, never five patch labels.
            self.assertEqual(ix, 0)
            self.assertEqual(ages.shape, (3,))
            with self.assertRaisesRegex(ValueError, "Resized pixel caches"):
                image_dataset(rows, config, age, cache=path)


class MultiviewTrainingTests(flow.TrainingFlowTests):
    def setUp(self):
        super().setUp()
        self.patches.extend([
            patch.object(models, "SPECS", {"green": {**models.SPECS["green"], "dim": 4}}),
            patch.object(models.timm, "create_model", side_effect=lambda *a, **kw: TinyBackbone()),
            patch.object(engine, "FundusModel", FundusModel),
        ])
        for item in self.patches[-3:]:
            item.start()

    def test_global_to_mil_training_resume_and_source_unchanged(self):
        from glaucoma.common import sha256
        config = {**self.config, "native_resolution": True, "view_version": VIEW_VERSION}
        p1, p2 = self.artifacts / "p1", self.artifacts / "p2"
        engine.fine_tune(self.root, config, p1, workers=0, device="cpu")
        source_hash = sha256(p1 / "model.pt")
        local_config = {**config, "multiview": True, "local_chunk_size": 2}
        engine.fine_tune(self.root, local_config, p2, workers=0, device="cpu", initialize_from=p1 / "model.pt")
        payload = torch.load(p2 / "model.pt", weights_only=True)
        self.assertEqual(payload["metadata"]["initialize_from_sha256"], source_hash)
        self.assertGreater(payload["state_dict"]["local_mil.fusion.2.weight"].abs().sum().item(), 0)
        self.assertLess(payload["age_transform"]["upper"], 137)
        engine.fine_tune(self.root, local_config, p2, workers=0, device="cpu", resume=True, initialize_from=p1 / "model.pt")
        self.assertTrue(read_json(p2 / "status.json")["complete"])
        self.assertEqual(sha256(p1 / "model.pt"), source_hash)
        with self.assertRaisesRegex(ValueError, "same holdout"):
            engine.fine_tune(self.root, local_config, self.artifacts / "bad", fold=1,
                             workers=0, device="cpu", initialize_from=p1 / "model.pt")
        self.assertFalse((self.artifacts / "valid_milestones.json").exists())

    def test_old_and_new_checkpoint_roundtrip_masks_and_gradients(self):
        age = AgeTransform.fit([40, 55])
        images = next(iter(engine.loader(image_dataset(load_rows(self.root, "train")[:2],
            {**self.config, "multiview": True, "view_version": VIEW_VERSION}, age), 2, 0)))[0]
        for local in (False, True):
            config = {**self.config, **({"multiview": True, "view_version": VIEW_VERSION} if local else {})}
            model = FundusModel(config).configure_trainable(-1)
            ages = torch.from_numpy(age.transform([40, 55]))
            x = images if local else images["global"]
            expected = model(x, ages)
            path = self.artifacts / f"roundtrip{local}.pt"
            save_checkpoint(path, model, config, age, {"fold": 0})
            restored, _, _ = load_checkpoint(path)
            torch.testing.assert_close(expected, restored(x, ages))
            if not local:
                continue
            torch.testing.assert_close(expected, model(images["global"], ages), atol=0, rtol=0)
            with torch.no_grad():
                model.local_mil.fusion[-1].weight.fill_(.01)
            masked = move_images(images, "cpu")
            masked["mask"] = torch.tensor([[True, False, True, False, True], [False]*5])
            result = model(masked, ages)
            torch.testing.assert_close(result[1], model(masked["global"], ages)[1], atol=0, rtol=0)
            changed = {**masked, "local": masked["local"].clone()}
            changed["local"][~masked["mask"]] = float("nan")
            torch.testing.assert_close(model(changed, ages), result)
            empty = {**masked, "mask": torch.zeros_like(masked["mask"])}
            torch.testing.assert_close(model(empty, ages), model(empty["global"], ages), atol=0, rtol=0)
            result.sum().backward()
            self.assertGreater(model.backbone.blocks[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.local_mil.attention[0].weight.grad.abs().sum().item(), 0)
            features = model.features(masked)
            _, weights = model.local_mil(features["global"], features["local"], features["coordinates"], features["mask"], ages)
            self.assertEqual(weights[~masked["mask"]].sum().item(), 0)
            self.assertEqual(weights[1].sum().item(), 0)

    def test_validation_checks_budget(self):
        config = {**self.config, "multiview": True, "view_version": VIEW_VERSION}
        data = engine.loader(image_dataset(load_rows(self.root, "train"), config), 2, 0)
        class Expired:
            def check(self):
                raise TimeoutError("expired in validation")
        with self.assertRaisesRegex(TimeoutError, "validation"):
            engine.evaluate_loader(FundusModel(config), data, "cpu", budget=Expired())

    def test_packed_encoder_preserves_outputs_gradients_and_checkpoint_api(self):
        config = {**self.config, "multiview": True, "view_version": VIEW_VERSION, "local_chunk_size": 2}
        rows = load_rows(self.root, "train")[:2]
        age = AgeTransform.fit([r["age"] for r in rows])
        images, ages, _, _ = next(iter(engine.loader(image_dataset(rows, config, age), 2, 0)))
        first = FundusModel(config).configure_trainable(-1)
        with torch.no_grad():
            first.local_mil.fusion[-1].weight.fill_(.01)
        second = FundusModel(config).configure_trainable(-1)
        second.load_state_dict(first.state_dict())
        second.performance_view_batch_size = 24
        engine.training_mode(first, False)
        engine.training_mode(second, False)
        a, b = first(images, ages), second(images, ages)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.sum().backward()
        b.sum().backward()
        for (_, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        engine.training_mode(second, True)
        self.assertEqual(second.active_view_batch_size, 0)
        engine.training_mode(second, False)
        data = engine.loader(image_dataset(rows, config, age), 2, 0)
        engine.evaluate_loader(second, data, "cpu")
        self.assertEqual(second.active_view_batch_size, 0)


if __name__ == "__main__":
    unittest.main()
