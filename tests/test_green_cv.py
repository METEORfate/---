"""Check CV job boundaries, interruption handling and prediction tampering."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from glaucoma.common import OutputBusy, exclusive_output, read_json, write_csv, write_json


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = script("run-green-cv")
review = script("review-results")


class GreenCVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.calls = []
        self.complete = True
        write_json(self.artifacts / "green_ft2" / "fold0" / "status.json", {"complete": True})

        def run(args, **kwargs):
            self.calls.append(args)
            if "train" in args:
                output = Path(args[args.index("--output") + 1])
                write_json(output / "status.json", {"complete": self.complete})
                (output / "resume.pt").touch()
            if "collect-oof" in args:
                write_json(self.artifacts / "green_ft2" / "oof.json", {"metrics": {"auroc": .95}})

        self.patches = [patch.object(study, "ROOT", self.root), patch.object(study, "ARTIFACTS", self.artifacts),
                        patch.object(study.subprocess, "run", side_effect=run)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_only_missing_folds_then_oof_no_valid_or_test(self):
        study.session(.1, 4)
        trains = [c for c in self.calls if "train" in c]
        self.assertEqual([c[c.index("--fold") + 1] for c in trains], ["1", "2"])
        self.assertTrue(any("collect-oof" in c for c in self.calls))
        self.assertEqual(sum("compare" in c for c in self.calls), 2)
        self.assertFalse(any(v in c for c in self.calls for v in ("evaluate", "predict", "finalize")))
        self.assertEqual(read_json(self.artifacts / "green_cv" / "state.json")["status"], "complete")

    def test_incomplete_fold_prevents_oof_and_resumes(self):
        self.complete = False
        study.session(.1, 4)
        self.assertFalse(any("collect-oof" in c for c in self.calls))
        self.assertEqual(read_json(self.artifacts / "green_cv" / "state.json")["status"], "needs_resume")
        self.complete = True
        study.session(.1, 4)
        self.assertTrue(any("train" in c and "--resume" in c for c in self.calls))

    def test_deadline_blocks_all_commands(self):
        with self.assertRaises(TimeoutError):
            study.session(-1, 4)
        self.assertEqual(self.calls, [])
        self.assertEqual(read_json(self.artifacts / "green_cv" / "state.json")["status"], "stopped")

    def test_existing_study_lock_blocks_second_gpu_queue(self):
        with exclusive_output(self.artifacts / "main_study" / "run.lock"):
            with self.assertRaises(OutputBusy):
                study.session(.1, 4)
        self.assertEqual(self.calls, [])


class PredictionReviewTests(unittest.TestCase):
    def test_rank_auc_ties_and_metadata_corruption(self):
        rows = [{"name": "a.jpg", "label": 0}, {"name": "b.jpg", "label": 1},
                {"name": "c.jpg", "label": 0}, {"name": "d.jpg", "label": 1}]
        records = [{"Image_name": r["name"], "True_Label": r["label"], "fold": 0, "Probability": .5} for r in rows]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oof.csv"
            write_csv(path, records, records[0].keys())
            _, result = review.predictions(path, rows, np.zeros(4))
            self.assertEqual(result["auroc"], .5)
            records[0]["True_Label"] = 1
            write_csv(path, records, records[0].keys())
            with self.assertRaises(ValueError):
                review.predictions(path, rows, np.zeros(4))

    def test_nonfinite_prediction_and_wrong_saved_metric_are_rejected(self):
        with self.assertRaises(ValueError):
            review.same_metrics({"auroc": .9}, {"auroc": .91})
        with self.assertRaises(ValueError):
            review.same_metrics({"auroc": .9}, {"auroc": float("nan")})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oof.csv"
            records = [{"Image_name": "a.jpg", "True_Label": 0, "fold": 0, "Probability": "nan"},
                       {"Image_name": "b.jpg", "True_Label": 1, "fold": 0, "Probability": .9}]
            write_csv(path, records, records[0].keys())
            with self.assertRaises(ValueError):
                review.predictions(path, [{"name": "a.jpg", "label": 0}, {"name": "b.jpg", "label": 1}], [0, 0])


if __name__ == "__main__":
    unittest.main()
