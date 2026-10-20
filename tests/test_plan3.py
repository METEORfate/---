"""Synthetic integrity/integration checks; these do not claim scientific performance."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from glaucoma.common import digest, read_json, seed_all, sha256, write_csv, write_json
from glaucoma.data import AgeTransform, crop_fov, load_rows
from plan3 import cloud, data, jobs, ledger, models, provenance, results, train, verification

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plan3_launcher", ROOT / "scripts/plan3_kaggle.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TinyEncoder(nn.Module):
    dim = 4

    def __init__(self, config=None, weight=None, source=None):
        super().__init__()
        self.config = config or {}
        self.backbone = nn.Conv2d(3, 4, 1)

    def forward(self, image):
        features = self.backbone(image)
        return features.mean((2, 3)), nn.functional.adaptive_avg_pool2d(features, (2, 2)).flatten(2).transpose(1, 2)


class Fixture(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "train").mkdir()
        records = []
        for i in range(36):
            name = f"image{i:02}.jpg"
            Image.new("RGB", (40, 32), (30+i*4, 65, 90)).save(self.root / "train" / name.upper())
            records.append({"Image_name": name, "Age": 137 if i % 3 == 0 else 25+i, "True_Label": i % 2})
        write_csv(self.root / "train.csv", records, ["Image_name", "Age", "True_Label"])
        self.rows = load_rows(self.root, "train")
        self.artifacts = self.root / "artifacts"
        write_csv(self.artifacts / "audit/folds.csv", [{"Image_name": r["name"], "fold": i % 3, "group": str(i)} for i, r in enumerate(self.rows)],
                  ["Image_name", "fold", "group"])
        self.audit = {"data_fingerprint": "synthetic", "fold_fingerprint": sha256(self.artifacts / "audit/folds.csv")}
        self.patches = [patch.object(provenance, "ARTIFACTS", self.artifacts),
                        patch.object(provenance, "require_audit", return_value=self.audit),
                        patch.object(train, "Encoder", TinyEncoder), patch.object(models, "Encoder", TinyEncoder)]
        for p in self.patches:
            p.start()
        self.context = provenance.Context(self.root)
        self.cfg = dict(model="effnet", size=16, view="global", epochs=2, patience=4, batch_size=4, accumulation=8,
                        head_warmup_epochs=0, backbone_lr=.001, head_lr=.001, pretrained="weights/effnet_source.json")
        self.job = dict(mode="plan3", task="train", fold=0, seed=42, workers=0, max_hours=1,
                        config="configs/plan3/effnet512.json", annotations="artifacts/annotations", output="artifacts/training", resume=False, dependencies=[])

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def boxes(self, count=36):
        records = [dict(zip(data.FIELDS, [r["name"], 40, 32, 12, 8, 24, 20, "valid"])) for r in self.rows[:count]]
        source = self.root / "boxes.csv"
        write_csv(source, records, data.FIELDS)
        spec = self.root / "spec.json"
        write_json(spec, {"format": "csv", "version": "synthetic-v1", "coordinate_space": data.SPACE})
        return source, spec

    def run_tiny(self, output, fold=0, resume=False, kind="classifier", inner_fold=None):
        fit, hold = self.context.split(fold, inner_fold)
        age = AgeTransform.fit([self.rows[i]["age"] for i in fit])
        training = data.Views([self.rows[i] for i in fit], self.cfg, age, augment=True)
        validation = data.Views([self.rows[i] for i in hold], self.cfg, age)
        seed_all(42)
        model = models.Classifier(self.cfg, TinyEncoder(self.cfg))
        meta = self.context.metadata(fit, hold, fold, inner_fold)
        signature = {"config": self.cfg, **meta, "seed": 42}
        return train._train(model, kind, self.cfg, {**self.job, "fold": fold, "inner_fold": inner_fold, "resume": resume}, signature,
                            meta, training, validation, age, None, output, "cpu", None)


class AnnotationTests(Fixture):
    def test_push_detects_zero_exit_rejection_and_waits_only_for_gpu_slot(self):
        folder = self.root / "kernel"
        folder.mkdir()
        write_json(folder / "kernel-metadata.json", {"enable_gpu": True})
        write_json(folder / "package.json", {"source_sha256": "frozen"})
        job = {**self.job, "run_ref": "tester/pilot"}
        rejected = SimpleNamespace(stdout="Kernel push error: Maximum batch GPU session count of 2 reached.", stderr="")
        accepted = SimpleNamespace(stdout="Kernel version 1 successfully pushed.", stderr="")
        with patch.object(launcher.subprocess, "run", return_value=rejected), self.assertRaisesRegex(RuntimeError, "not submitted"):
            launcher.push_kernel({"accelerator": "NvidiaTeslaT4"}, job, folder)
        self.assertEqual(read_json(folder / "submission.json")["status"], "gpu_slot_unavailable")
        with patch.object(launcher.subprocess, "run", side_effect=[rejected, accepted]), \
                patch.object(launcher.time, "sleep") as sleep, patch.object(launcher, "preflight") as gate:
            self.assertEqual(launcher.push_kernel({"accelerator": "NvidiaTeslaT4"}, job, folder, 1), 1)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(gate.call_count, 1)
        self.assertEqual(read_json(folder / "submission.json")["kernel_ref"], "tester/pilot/1")
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout="Kernel push error: Invalid dataset", stderr="")), \
                patch.object(launcher.time, "sleep") as sleep, self.assertRaisesRegex(RuntimeError, "acknowledge"):
            launcher.push_kernel({"accelerator": "NvidiaTeslaT4"}, job, folder, 1)
        sleep.assert_not_called()

    def test_audit_parallelism_is_separate_and_bounded(self):
        jobs.validate({**self.job, "workers": 2, "audit_workers": 16})
        for count in (0, 33, True):
            with self.assertRaisesRegex(ValueError, "audit_workers"):
                jobs.validate({**self.job, "audit_workers": count})

    def test_labelme_conversion_redacts_metadata_and_respects_usability(self):
        source = self.root / "labelme"
        source.mkdir()
        for i, row in enumerate(self.rows[:3]):
            item = {"version": "test", "imagePath": "C:/private/"+Path(row["path"]).name,
                    "imageData": "sensitive-payload-not-to-export", "imageWidth": 40, "imageHeight": 32,
                    "Age": 999, "diagnosis": "not-to-export", "patient_name": "not-to-export",
                    "flags": {"boundary_unclear": i == 1, "unlocalizable": i == 2},
                    "shapes": [{"label": "optic_disc", "shape_type": "rectangle", "points": [[24, 20], [12, 8]],
                                "description": "not-to-export"}]}
            write_json(source / (Path(row["name"]).stem+".json"), item)
        output = self.root / "sanitized"
        csv_path, spec_path, fingerprint = data.convert_labelme(source, self.rows, output)
        records = results.read_predictions(csv_path)
        self.assertEqual([r["status"] for r in records], ["valid", "unclear", "unlocatable"])
        self.assertEqual(float(records[0]["x_min"]), 12)
        self.assertEqual(records[1]["x_min"], "")
        self.assertEqual(list(records[0]), data.FIELDS)
        combined = "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
        self.assertNotIn("not-to-export", combined)
        self.assertNotIn("C:/private", combined)
        self.assertEqual(len(fingerprint), 64)

    def test_partial_approval_does_not_lower_full_gate(self):
        manifest = {"acceptance_scope": "pilot400", "annotated": 401, "valid": 353, "minimum_valid": 300}
        jobs.check_annotation_threshold(manifest, "pilot400")
        with self.assertRaisesRegex(ValueError, "1000"):
            jobs.check_annotation_threshold(manifest)
        with self.assertRaises(ValueError):
            jobs.check_annotation_threshold({**manifest, "acceptance_scope": "full1000"}, "pilot400")

    def test_import_gate_and_tamper(self):
        source, spec = self.boxes()
        output = self.root / "annotations"
        report = data.import_annotations(source, spec, self.rows, self.context.folds, self.audit, output, minimum=30)
        self.assertTrue(report["accepted"])
        self.assertEqual(len(data.require_annotations(output, self.audit)[1]), 36)
        (output / "boxes.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "provenance"):
            data.require_annotations(output, self.audit)

    def test_insufficient_boxes_do_not_auto_accept(self):
        source, spec = self.boxes(6)
        output = self.root / "annotations"
        result = data.import_annotations(source, spec, self.rows, self.context.folds, self.audit, output)
        self.assertFalse(result["accepted"])
        with self.assertRaises(ValueError):
            data.require_annotations(output, self.audit)

    def test_unknown_orientation_and_invalid_box_rejected(self):
        source, spec = self.boxes()
        write_json(spec, {"format": "csv", "version": "v1"})
        with self.assertRaisesRegex(ValueError, "coordinate_space"):
            data.import_annotations(source, spec, self.rows, self.context.folds, self.audit, self.root / "bad", 1)
        source, spec = self.boxes()
        records = results.read_predictions(source)
        records[0]["x_max"] = 90
        write_csv(source, records, data.FIELDS)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            data.import_annotations(source, spec, self.rows, self.context.folds, self.audit, self.root / "bad", 1)

    def test_original_decode_and_crop_coordinates(self):
        image = Image.new("RGB", (200, 100))
        image.paste((120, 120, 120), (50, 5, 150, 95))
        cropped, box = data.fov_geometry(image)
        old, _ = crop_fov(image)
        np.testing.assert_array_equal(np.asarray(cropped), np.asarray(old))
        mapped = data.map_box([50, 10, 70, 30], (40, 0, 160, 100), 240)
        self.assertEqual(mapped, [20, 40, 60, 80])
        roi, coordinates = data.roi_crop(image, [0, 10, 20, 30], 3.)
        self.assertEqual(roi.size, (60, 60))
        self.assertLess(coordinates[-1], 1)

    def test_nested_partitions_exclude_outer_holdout_and_groups(self):
        seen = set()
        for inner in range(3):
            fit, hold = self.context.split(0, inner)
            self.assertTrue(all(self.context.folds[i] != 0 for i in [*fit, *hold]))
            self.assertFalse(set(fit) & set(hold))
            seen.update(hold)
        self.assertEqual(seen, set(np.flatnonzero(self.context.folds != 0)))


class ModelTests(Fixture):
    def views(self, valid=True):
        return {"global": torch.rand(2, 3, 16, 16), "local": torch.rand(2, 2, 3, 16, 16),
                "mask": torch.full((2, 2), valid), "coordinates": torch.rand(2, 2, 5)}

    def test_all_invalid_views_exact_global_fallback(self):
        for view in ("local", "probability", "concat", "cross"):
            model = models.Classifier({**self.cfg, "view": view}, TinyEncoder()).eval()
            images = self.views(False)
            ages = torch.rand(2, 3)
            output = model(images, ages)
            torch.testing.assert_close(output["logits"], output["global"], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_cross_zero_residual_and_gradients(self):
        model = models.Classifier({**self.cfg, "view": "cross"}, TinyEncoder())
        output = model(self.views(), torch.zeros(2, 3))
        torch.testing.assert_close(output["logits"], output["global"], rtol=0, atol=0)
        loss = models.classification_loss(output, torch.tensor([0., 1.]), .3)
        loss.backward()
        self.assertGreater(float(model.residual.weight.grad.abs().sum()), 0)

    def test_cross_supports_different_local_pixel_sizes(self):
        model = models.Classifier({**self.cfg, "view": "cross"}, TinyEncoder())
        views = self.views()
        views["local"] = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 24, 24)]
        output = model(train.move(views, "cpu"), torch.zeros(2, 3))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        models.classification_loss(output, torch.tensor([0., 1.])).backward()

    def test_ranking_direction_single_class_and_sampling(self):
        labels = torch.tensor([1., 0.])
        good = {"logits": torch.tensor([2., -2.], requires_grad=True)}
        bad = {"logits": torch.tensor([-2., 2.], requires_grad=True)}
        self.assertLess(models.classification_loss(good, labels, .3), models.classification_loss(bad, labels, .3))
        self.assertTrue(torch.isfinite(models.classification_loss(good, torch.ones(2), .3)))
        batches = list(train.PairBatches([1, 0, 0, 0, 0, 0], 4, 42, [2]))
        self.assertTrue(all(0 in b and 2 in b for b in batches))

    def test_ema_successful_updates_only(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0)
        ema = models.EMA(model, .5)
        model.weight.data.fill_(2)
        ema.update(model, successful=False)
        self.assertEqual(ema.updates, 0)
        ema.update(model)
        self.assertEqual(float(ema.model.weight), 1.)

    def test_self_distillation_center_and_gradient(self):
        s = [torch.randn(4, 32, requires_grad=True) for _ in range(2)]
        t = [torch.randn(4, 32) for _ in range(2)]
        loss = models.dino_loss(s, t, torch.zeros(1, 32))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(s[0].grad)

    def test_locator_review_hash_and_missed_gate(self):
        path = self.root / "locator.pt"
        path.write_bytes(b"fixture")
        review = self.root / "review.json"
        payload = {"metadata": {"review_names": ["a", "b"]}}
        write_json(review, {"locator_sha256": sha256(path), "checks": [{"Image_name": "a", "missed_disc": False}, {"Image_name": "b", "missed_disc": True}]})
        with self.assertRaisesRegex(ValueError, "5%"):
            train.verify_locator_review(review, path, payload)

    def test_locator_selection_support_also_excludes_holdout(self):
        fit, hold = self.context.split(0)
        meta = self.context.metadata(fit, [], 0, None)
        meta["review_names"] = [self.rows[hold[0]]["name"]]
        with self.assertRaisesRegex(ValueError, "forbidden"):
            provenance.check_support(meta, self.context, fit)


class TrainingTests(Fixture):
    def test_classifier_is_not_started_when_inline_capacity_fails(self):
        from plan3 import __main__ as entry
        path=self.root / "config.json"
        write_json(path,self.cfg)
        with patch.object(entry,"Context",return_value=self.context),patch.object(self.context,"annotations",return_value=({},{})), \
                patch.object(entry,"safe_path",side_effect=lambda p: path if p==self.job['config'] else self.root/p), \
                patch.object(verification,"capacity",side_effect=RuntimeError('synthetic OOM')),patch.object(train,"run_training") as classifier:
            with self.assertRaisesRegex(RuntimeError,'synthetic OOM'):
                entry.execute(self.root,{**self.job,'preflight_capacity':True},'cpu')
            classifier.assert_not_called()

    def test_capacity_covers_optimizer_and_ema_for_classifier_and_ssl(self):
        with patch.object(verification, "pretrained", return_value=(None, {})):
            report = verification.capacity(self.context, self.cfg, self.job, self.root / "capacity", "cpu")
            self.assertTrue(report["includes_adam_state_and_ema"])
            self.assertGreater(report["optimizer_updates"], 0)
            cfg = {**self.cfg, "ssl": True, "local_sizes": [16, 24], "roi_scales": [3, 5]}
            fit, _ = self.context.split(0)
            locator = {"metadata": self.context.metadata(fit, [], 0, None)}
            roi = {r["name"]: {"valid": True, "box": [12, 8, 24, 20]} for r in self.rows}
            with patch.object(train, "load_payload", return_value=locator), patch.object(train, "verify_locator_review"), \
                    patch.object(train, "locate", return_value=roi):
                report = verification.capacity(self.context, cfg, {**self.job, "locator": "artifacts/locator.pt",
                                            "locator_review": "artifacts/review.json"}, self.root / "ssl_capacity", "cpu")
            self.assertGreater(report["optimizer_updates"], 0)

    def test_public_training_path_and_joint_stage_initialization(self):
        weight = self.root / "weight.pt"
        weight.write_bytes(b"synthetic pretrained identity")
        annotation = {"boxes_sha256": "synthetic-annotations", "version": "fixture"}
        resolve = lambda value: self.root / value
        with patch.object(train, "Context", return_value=self.context), \
                patch.object(self.context, "annotations", return_value=(annotation, {})), \
                patch.object(train, "pretrained", return_value=(weight, {})), \
                patch.object(train, "safe_path", side_effect=resolve):
            job = {**self.job, "output": "artifacts/public"}
            train.run_training(self.root, self.cfg, job, "cpu")
            source = train.load_payload(self.artifacts / "public/model.pt")
            cfg = {**self.cfg, "view": "cross", "requires_initialization": True, "local_size": 16}
            joint = models.Classifier(cfg, TinyEncoder(cfg))
            joint.residual.weight.data.fill_(.25)
            source.update(config=cfg, state_dict=joint.state_dict())
            train.atomic_torch(self.artifacts / "joint.pt", source)
            fit, _ = self.context.split(0)
            locator = {"format": "plan3-checkpoint-v1", "kind": "locator", "config": self.cfg,
                       "state_dict": {}, "metadata": self.context.metadata(fit, [], 0, None)}
            locator["metadata"]["review_names"] = [self.rows[fit[0]]["name"]]
            train.atomic_torch(self.artifacts / "locator.pt", locator)
            write_json(self.artifacts / "review.json", {"locator_sha256": sha256(self.artifacts / "locator.pt"),
                        "checks": [{"Image_name": self.rows[fit[0]]["name"], "missed_disc": False}]})
            with patch.object(train, "locate", return_value={}), patch.object(train, "_train", return_value=None) as update:
                train.run_training(self.root, {**cfg, "size": 24}, {**job, "initialize_from": "artifacts/joint.pt",
                                   "locator": "artifacts/locator.pt", "locator_review": "artifacts/review.json"}, "cpu")
                initialized = update.call_args.args[0]
                torch.testing.assert_close(initialized.residual.weight, joint.residual.weight)
                torch.testing.assert_close(initialized.project.weight, joint.project.weight)

    def test_ssl_continuation_saves_teacher_center(self):
        cfg = {**self.cfg, "ssl": True, "epochs": 1, "local_size": 16, "roi_scales": [3, 5]}
        rows = self.rows[:8]
        age = AgeTransform.fit([r["age"] for r in rows])
        roi = {r["name"]: {"valid": True, "box": [12, 8, 24, 20]} for r in rows}
        training = data.Views(rows, cfg, age, roi, True)
        model = models.Distillation(TinyEncoder(cfg))
        output = self.root / "ssl"
        train._train(model, "ssl", cfg, self.job, {"test": "ssl"}, {"fold": 0}, training, training,
                     age, None, output, "cpu", None)
        state = torch.load(output / "resume.pt", weights_only=True)
        self.assertTrue(torch.isfinite(state["ema"]["center"]).all())
        self.assertGreater(float(state["ema"]["center"].abs().sum()), 0)

    def test_last_checkpoint_recovers_without_status_or_history_files(self):
        output = self.root / "recovered"
        self.run_tiny(output)
        (output / "status.json").unlink()
        self.run_tiny(output, resume=True)
        self.assertTrue(read_json(output / "status.json")["complete"])

    def test_resume_full_epoch_equivalence_and_ema_binding(self):
        original_loader = train.data_loader
        calls = 0
        def interrupt(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("synthetic interruption after complete epoch")
            return original_loader(*a, **kw)
        continued = self.root / "continued"
        with patch.object(train, "data_loader", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.run_tiny(continued)
        self.assertTrue((continued / "resume.pt").exists())
        self.run_tiny(continued, resume=True)
        uninterrupted = self.root / "uninterrupted"
        self.run_tiny(uninterrupted)
        first, second = train.load_payload(continued / "model.pt"), train.load_payload(uninterrupted / "model.pt")
        for key in first["state_dict"]:
            torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
        self.assertEqual(first["metadata"]["holdout"], second["metadata"]["holdout"])
        self.assertLess(first["age_transform"]["upper"], 137)
        state = torch.load(continued / "resume.pt", weights_only=True)
        self.assertIn("rng", state)
        self.assertGreater(state["ema_updates"], 0)

    def test_incomplete_and_changed_config_refused(self):
        output = self.root / "train"
        self.run_tiny(output)
        self.cfg["head_lr"] = .01
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_tiny(output, resume=True)

    def test_oof_three_folds_hash_labels_and_support(self):
        paths = [self.root / f"fold{f}" for f in range(3)]
        for fold, path in enumerate(paths):
            self.run_tiny(path, fold)
        output = self.root / "oof.csv"
        result = results.collect(self.context, paths, output)
        self.assertEqual(result["metrics"]["n"], len(self.rows))
        self.assertEqual([r["Image_name"] for r in results.read_predictions(output)], [r["name"] for r in self.rows])
        with self.assertRaisesRegex(ValueError, "exactly three"):
            results.collect(self.context, paths[:2], output)
        records = results.read_predictions(paths[0] / "holdout.csv")
        records[0]["True_Label"] = 1-int(records[0]["True_Label"])
        write_csv(paths[0] / "holdout.csv", records, ["Image_name", "Probability", "True_Label", "fold"])
        with self.assertRaisesRegex(ValueError, "changed"):
            results.collect(self.context, paths, output)

    def test_cross_fold_support_cannot_initialize(self):
        fit, hold = self.context.split(0)
        wrong = self.context.metadata(*self.context.split(1), 1, None)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            provenance.check_support(wrong, self.context, fit)

    def test_three_expert_capsule_and_nested_meta_evaluation(self):
        paths = [self.root / f"fold{fold}" for fold in range(3)]
        for fold, path in enumerate(paths):
            self.run_tiny(path, fold)
        oof = self.root / "outer.csv"
        results.collect(self.context, paths, oof)
        expert = {"oof": "outer.csv", "models": [f"fold{f}/model.pt" for f in range(3)]}
        original_safe = provenance.safe_path
        resolve = lambda value, base=None: original_safe(value, base or self.root)
        artifact = self.root / "capsule/ensemble.json"
        with patch.object(results, "safe_path", side_effect=resolve):
            results.fit_ensemble(self.context, {"fusion": "probability", "experts": [expert]*3}, artifact)
            identity = results.artifact_identity(artifact)
            self.assertEqual(len(read_json(artifact)["experts"]), 3)
            expected = np.mean([results.predict_rows(p / "model.pt", self.rows[:2], "cpu") for p in paths], axis=0)
            actual = results.predict_rows(artifact, self.rows[:2], "cpu")
            np.testing.assert_allclose(actual, expected)
            self.assertEqual(results.artifact_identity(artifact), identity)
            results.verify_artifact(self.context, artifact)
            with self.assertRaisesRegex(ValueError, "full-train"):
                results.verify_artifact(self.context, artifact, require_full=True)
            with self.assertRaisesRegex(ValueError, "TTA"):
                results.fit_ensemble(self.context, {"fusion": "probability", "experts": [expert], "horizontal_tta": True}, self.root / "bad/ensemble.json")

    def test_nested_fusion_and_full_hard_negative_support(self):
        paths = [self.root / f"fold{f}" for f in range(3)]
        for fold, path in enumerate(paths):
            self.run_tiny(path, fold)
        results.collect(self.context, paths, self.root / "outer.csv")
        inner = {}
        for outer in range(3):
            parts = [self.root / f"inner{outer}_{i}" for i in range(3)]
            for index, path in enumerate(parts):
                self.run_tiny(path, outer, inner_fold=index)
            results.collect(self.context, parts, self.root / f"inner{outer}.csv", outer)
            inner[str(outer)] = f"inner{outer}.csv"
        expert = {"oof": "outer.csv", "inner_oof": inner, "models": [f"fold{f}/model.pt" for f in range(3)]}
        original_safe = provenance.safe_path
        with patch.object(results, "safe_path", side_effect=lambda value, base=None: original_safe(value, base or self.root)):
            artifact = results.fit_ensemble(self.context, {"fusion": "logistic", "experts": [expert]}, self.root / "nested/ensemble.json")
        self.assertEqual(artifact["report"]["nested_logistic"]["n"], len(self.rows))
        path = self.root / "full_hard.json"
        results.make_hard_negatives(self.context, self.root / "outer.csv", "full", path)
        selected = results.validate_hard_negatives(path, self.context, np.arange(len(self.rows)), {"fold": "full"})
        self.assertTrue(selected)
        self.assertTrue(all(self.rows[i]["label"] == 0 for i in selected))

    def test_full_epochs_keep_selection_recipe_and_use_cv_weight_policy(self):
        paths = [self.root / f"fold{f}" for f in range(3)]
        for fold, path in enumerate(paths):
            self.run_tiny(path, fold)
        job = {**self.job, "task": "finalize", "fold": "full", "cv_models": [f"fold{f}/model.pt" for f in range(3)]}
        with patch.object(train, "safe_path", side_effect=lambda value: self.root / value):
            count = train.final_epochs(self.context, job, self.cfg)
        self.assertIn(count, (1, 2))
        self.assertIn(job["_cv_final_weights"], ("ema", "raw"))

    def test_oof_rejects_mixed_annotation_provenance(self):
        paths = [self.root / f"fold{f}" for f in range(3)]
        for fold, path in enumerate(paths):
            self.run_tiny(path, fold)
        path = paths[2] / "model.pt"
        payload = train.load_payload(path)
        payload["metadata"]["annotation_sha256"] = "changed"
        train.atomic_torch(path, payload)
        meta = read_json(paths[2] / "holdout.json")
        meta["model_sha256"] = sha256(path)
        write_json(paths[2] / "holdout.json", meta)
        with self.assertRaisesRegex(ValueError, "provenance"):
            results.collect(self.context, paths, self.root / "mixed.csv")


class CloudAndLedgerTests(Fixture):
    def test_preflight_accepts_sdk_string_status_and_rejects_stale_annotations(self):
        folder = self.root / self.job["annotations"]
        folder.mkdir(parents=True)
        (folder / "boxes.csv").write_text("fixture")
        annotation = {"accepted": True, "valid": 1000, "minimum_valid": 1000, "version": "v1",
                      "input_version": "exif-original-xyxy-roi-v1", "coordinate_space": data.SPACE,
                      "boxes_sha256": sha256(folder / "boxes.csv"), **self.audit}
        write_json(folder / "acceptance.json", annotation)
        write_json(self.artifacts / "audit/report.json", self.audit)
        write_json(self.root / self.job["config"], self.cfg)
        weight = self.root / "weights/fixture.pt"
        weight.parent.mkdir()
        weight.write_bytes(b"pretrained fixture")
        write_json(self.root / self.cfg["pretrained"], {"filename": "fixture.pt", "sha256": sha256(weight), "source": "official", "license": "official"})
        state = self.artifacts / "staging"
        (state / "resources").mkdir(parents=True)
        with zipfile.ZipFile(state / "resources/resources.zip", "w") as z:
            z.writestr("resources-manifest.json", json.dumps({"sha256": {self.cfg["pretrained"]: sha256(self.root / self.cfg["pretrained"])}}))
            z.writestr(self.cfg["pretrained"], (self.root / self.cfg["pretrained"]).read_bytes())
            z.writestr(self.job["annotations"]+"/acceptance.json", json.dumps(annotation))
        api = MagicMock()
        api.dataset_list.return_value = [SimpleNamespace(ref="tester/data", is_private=True), SimpleNamespace(ref="tester/resources", is_private=True)]
        api.dataset_status.side_effect = ["ready", SimpleNamespace(status="ready")]
        fake_module = SimpleNamespace(KaggleApi=lambda: api)
        with patch.object(launcher, "ROOT", self.root), patch.object(launcher, "STATE", state), \
                patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": fake_module}), \
                patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout="resource,used,remaining,total,refreshAt\nGPU,0h,30h,30h,unknown\n")) as remote:
            launcher.preflight({"owner": "tester", "data_dataset": "tester/data", "plan3_resources_dataset": "tester/resources"}, self.job)
            self.assertEqual(remote.call_args.args[0], ["kaggle", "quota", "--csv"])
            annotation["version"] = "v2"
            write_json(folder / "acceptance.json", annotation)
            with self.assertRaisesRegex(ValueError, "different annotation"):
                launcher.preflight({"owner": "tester", "data_dataset": "tester/data"}, self.job)
            self.assertEqual(remote.call_count, 1)

    @patch.object(ledger, "verify_artifact")
    def test_evaluation_return_merges_only_matching_predictions_and_ticket(self, verify):
        from glaucoma.engine import save_predictions
        from glaucoma.metrics import metrics
        central = self.root / "central.json"
        write_json(central, {"baseline": {"artifact_sha256": "existing", "status": "complete"}})
        artifact = self.root / "model.pt"
        artifact.write_bytes(b"frozen fixture")
        output = self.root / "evaluation"
        ticket = ledger.reserve(self.context, artifact, "candidate", output / "evaluation_reservation.json", central)
        rows = self.rows[:4]
        probabilities = [.1, .9, .2, .8]
        save_predictions(output / "valid.csv", rows, probabilities)
        report = {"format": "plan3-valid-result-v1", "status": "complete", "milestone": "candidate",
                  "ticket_sha256": digest(ticket), "artifact_sha256": sha256(artifact),
                  "data_fingerprint": self.audit["data_fingerprint"], "fold_fingerprint": self.audit["fold_fingerprint"],
                  "prediction_sha256": sha256(output / "valid.csv"),
                  "report": {"full": metrics([r["label"] for r in rows], probabilities)}}
        write_json(output / "valid_report.json", report)
        ledger.accept_result(output / "valid_report.json", central)
        self.assertEqual(read_json(central)["candidate"]["status"], "complete")
        (output / "valid.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            ledger.accept_result(output / "valid_report.json", central)

    def test_schema_and_fixed_version_inputs(self):
        jobs.validate(self.job)
        for change in ({"fold": "full"}, {"max_hours": float("nan")}, {"output": "../outside"}, {"fold": True}, {"annotations": "C:/secret"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                jobs.validate({**self.job, **change})
        dep = {"alias": "p1", "kernel_ref": "owner/train/1", "results_sha256": "a"*64, "files": {"artifacts/x/model.pt": "b"*64}}
        jobs.validate({**self.job, "dependencies": [dep]})
        with self.assertRaises(ValueError):
            jobs.validate({**self.job, "dependencies": [{**dep, "kernel_ref": "owner/train"}]})

    def test_payload_allowlist_and_legacy_fingerprint_unchanged(self):
        from glaucoma.common import code_fingerprint
        before = code_fingerprint()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(launcher.payload()))) as z:
            self.assertIn("plan3/train.py", z.namelist())
            self.assertIn("configs/plan3/rfm448.json", z.namelist())
            self.assertFalse(any("local.json" in p or p.startswith(("weights/", "artifacts/")) for p in z.namelist()))
        self.assertEqual(before, code_fingerprint())
        cfg = {"owner": "tester", "data_dataset": "tester/data", "accelerator": "NvidiaTeslaT4"}
        with patch.object(launcher, "STATE", self.root / "stage"):
            folder = launcher.prepare(cfg, {**self.job, "config": "configs/plan3/rfm448.json"}, "test-run")
        compile((folder / "run_kaggle.py").read_text(encoding="utf-8"), "generated", "exec")
        self.assertTrue(read_json(folder / "kernel-metadata.json")["is_private"])

    def test_annotation_gate_prevents_remote_launch(self):
        with patch.object(launcher, "ROOT", self.root):
            with self.assertRaises(FileNotFoundError):
                launcher.gate(self.job)

    def test_dataset_selection_ignores_prior_kernel_resources(self):
        inputs = self.root / "inputs"
        dataset = inputs / "datasets/owner/resources"
        dataset.mkdir(parents=True)
        prior = inputs / "previous/project"
        prior.mkdir(parents=True)
        (dataset / "resources-manifest.json").write_text("new")
        (prior / "resources-manifest.json").write_text("old")
        self.assertEqual(cloud.dataset_mount(inputs, "owner/resources"), dataset)

    def test_small_annotation_preview_cannot_open_formal_gate(self):
        source, spec = self.boxes()
        folder = self.artifacts / "annotations"
        data.import_annotations(source, spec, self.rows, self.context.folds, self.audit, folder, minimum=1)
        with patch.object(launcher, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "blocked"):
                launcher.gate(self.job)

    def test_dependency_isolation_missing_and_bad_hash(self):
        inputs, project = self.root / "inputs", self.root / "project"
        inputs.mkdir()
        project.mkdir()
        archive = inputs / "results.zip"
        content = "artifacts/task/model.pt"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(content, b"weight")
            z.writestr("plan3-result-manifest.json", json.dumps({"run_ref": "tester/run"}))
        entry = {"alias": "first", "kernel_ref": "tester/run/1", "results_sha256": sha256(archive), "files": {content: hashlib.sha256(b"weight").hexdigest()}}
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("safe_extractor", ROOT / "kaggle/runner.py")
        legacy = module_from_spec(spec)
        spec.loader.exec_module(legacy)
        cloud.stage_dependencies(inputs, project, {"dependencies": [entry, {**entry, "alias": "second"}]}, legacy.extract_checked)
        self.assertEqual((project / "deps/first" / content).read_bytes(), b"weight")
        self.assertEqual((project / "deps/second" / content).read_bytes(), b"weight")
        with self.assertRaisesRegex(ValueError, "Missing"):
            cloud.stage_dependencies(inputs, project, {"dependencies": [{**entry, "results_sha256": "f"*64}]}, legacy.extract_checked)

    def test_multi_member_fusion_arithmetic(self):
        matrix = np.array([[.1, .4, .7, .9], [.2, .3, .4, .5]])
        np.testing.assert_allclose(results.fuse_values(matrix, "probability"), matrix.mean(1))
        self.assertTrue(np.isfinite(results.fuse_values(matrix, "logit")).all())
        with self.assertRaises(ValueError):
            results.fuse_values(np.array([[np.nan]]), "probability")

    @patch.object(ledger, "verify_artifact")
    def test_ledger_reservation_persists_and_rejects_new_artifact(self, verify):
        ledger_path = self.root / "valid_milestones.json"
        write_json(ledger_path, {"baseline": {"artifact_sha256": "existing", "status": "complete"}})
        artifact = self.root / "model.pt"
        artifact.write_bytes(b"model one")
        ticket_path = self.root / "ticket.json"
        ticket = ledger.reserve(self.context, artifact, "candidate", ticket_path, ledger_path)
        self.assertEqual(read_json(ledger_path)["baseline"]["artifact_sha256"], "existing")
        ledger.check_ticket(ticket, artifact, self.audit)
        artifact.write_bytes(b"model two")
        with self.assertRaisesRegex(ValueError, "reserved"):
            ledger.reserve(self.context, artifact, "candidate", ticket_path, ledger_path)
        with self.assertRaisesRegex(ValueError, "ticket"):
            ledger.check_ticket(ticket, artifact, self.audit)

    @patch.object(ledger, "verify_artifact")
    def test_empty_ledger_cannot_reopen_valid(self, verify):
        ledger_path = self.root / "empty.json"
        write_json(ledger_path, {})
        artifact = self.root / "model.pt"
        artifact.write_bytes(b"model")
        with self.assertRaisesRegex(ValueError, "baseline"):
            ledger.reserve(self.context, artifact, "candidate", self.root / "ticket", ledger_path)


if __name__ == "__main__":
    unittest.main()
