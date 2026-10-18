import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from glaucoma.common import ROOT, read_json, write_json

spec = importlib.util.spec_from_file_location("p1p2_queue", ROOT / "scripts/run-p1p2.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class FakeBudget:
    used, limit = 2000, 28800

    def __init__(self, *args):
        pass

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, *args):
        pass

    def check(self):
        pass


class P1P2QueueTests(unittest.TestCase):
    def exercise(self, interrupt=False):
        with tempfile.TemporaryDirectory() as folder:
            artifacts = Path(folder)
            calls = []
            def train(root, config, output, **kwargs):
                calls.append((config, output, kwargs))
                if interrupt and len(calls) == 2:
                    raise TimeoutError("existing RFM cap")
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "status.json", {"complete": True})
                torch.save({"metadata": {"epoch": 2, "holdout": {"auroc": .8}}}, output / "model.pt")
            patches = [patch.object(queue, "ARTIFACTS", artifacts), patch.object(queue, "Budget", FakeBudget),
                       patch.object(queue.torch.cuda, "is_available", return_value=True),
                       patch.object(queue.torch.cuda, "empty_cache"),
                       patch.object(queue, "protected_files", return_value={"unchanged": "hash"}),
                       patch.object(queue, "require_audit", return_value={"data_fingerprint": "synthetic"}),
                       patch.object(queue, "sha256", return_value="hash"),
                       patch.object(queue, "load_rows", return_value=[]), patch.object(queue, "patch_preview"),
                       patch.object(queue, "comparison", return_value={"purpose": "synthetic test"}),
                       patch.object(queue.engine, "fine_tune", side_effect=train)]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                if interrupt:
                    with self.assertRaises(TimeoutError):
                        queue.session(hours=1, workers=0)
                else:
                    queue.session(hours=1, workers=0)
            self.assertEqual(len(calls), 2)
            self.assertFalse(calls[0][0].get("multiview", False))
            self.assertTrue(calls[1][0]["multiview"])
            self.assertEqual(calls[1][2]["initialize_from"], calls[0][1] / "model.pt")
            self.assertEqual(calls[0][2]["fold"], 0)
            self.assertLessEqual(calls[0][2]["max_hours"], .35)
            state = read_json(artifacts / "p1p2/state.json")
            self.assertEqual(state["status"], "budget_limited" if interrupt else "complete")
            self.assertTrue(state["protected_sources_unchanged"])

    def test_sequential_same_fold_global_initialization(self):
        self.exercise()

    def test_existing_budget_failure_is_recorded_without_claiming_completion(self):
        self.exercise(interrupt=True)


if __name__ == "__main__":
    unittest.main()
