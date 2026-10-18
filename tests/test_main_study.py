"""Verify the bounded experiment queue without starting GPU jobs."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glaucoma.common import read_json, write_csv, write_json

spec = importlib.util.spec_from_file_location("main_study", Path(__file__).resolve().parents[1] / "scripts" / "run-main-study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class MainStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.calls = []
        self.training_complete = True
        write_json(self.artifacts / "nightly" / "state.json", {"status": "complete"})
        self.rows = [{"Image_name": "a.jpg", "Probability": "0.3", "True_Label": "0", "fold": "0"},
                     {"Image_name": "b.jpg", "Probability": "0.7", "True_Label": "1", "fold": "0"},
                     {"Image_name": "c.jpg", "Probability": "0.6", "True_Label": "1", "fold": "1"}]
        write_csv(self.artifacts / "green_probe_age" / "oof.csv", self.rows,
                  ["Image_name", "Probability", "True_Label", "fold"])
        def fake_run(args, **kwargs):
            self.calls.append(args)
            if "train" in args:
                output = Path(args[args.index("--output") + 1])
                write_json(output / "status.json", {"complete": self.training_complete})
                (output / "resume.pt").touch()
                if self.training_complete:
                    write_csv(output / "holdout.csv", self.rows[:2], self.rows[0].keys())
        self.patches = [patch.object(study, "ROOT", self.root), patch.object(study, "ARTIFACTS", self.artifacts),
                        patch.object(study.subprocess, "run", side_effect=fake_run)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_finished_queue_does_not_repeat_training(self):
        study.session(.1, 0)
        self.assertEqual(read_json(self.artifacts / "main_study" / "state.json")["status"], "session_complete")
        self.assertEqual(sum("train" in c for c in self.calls), 1)
        self.assertEqual(sum("--rfm-worker" in c for c in self.calls), 1)
        count = len(self.calls)
        study.session(.1, 0)
        self.assertEqual(len(self.calls), count)
        baseline = self.artifacts / "comparisons" / "green_frozen_fold0.csv"
        self.assertNotIn("c.jpg", baseline.read_text(encoding="utf-8"))

    def test_partial_green_run_is_resumed_next_session(self):
        self.training_complete = False
        study.session(.1, 0)
        self.assertTrue(read_json(self.artifacts / "main_study" / "state.json")["green_needs_resume"])
        self.training_complete = True
        study.session(.1, 0)
        trains = [c for c in self.calls if "train" in c]
        self.assertEqual(len(trains), 2)
        self.assertNotIn("--resume", trains[0])
        self.assertIn("--resume", trains[1])
        self.assertFalse(read_json(self.artifacts / "main_study" / "state.json")["green_needs_resume"])


if __name__ == "__main__":
    unittest.main()
