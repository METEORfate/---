import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glaucoma.common import read_json, write_json

spec = importlib.util.spec_from_file_location("acceleration", Path(__file__).resolve().parents[1] / "scripts" / "complete-acceleration.py")
acceleration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceleration)


class AccelerationFlowTests(unittest.TestCase):
    def test_backup_worker_selection_and_deadline_preserving_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = root / "artifacts"
            checkpoint = artifacts / "green_ft2" / "fold0"
            checkpoint.mkdir(parents=True)
            for name in ("model.pt", "resume.pt", "history.json", "holdout.csv"):
                (checkpoint / name).write_bytes(f"fixture-{name}".encode())
            deadline = time.time()+3600
            write_json(artifacts / "main_study" / "state.json", {"status": "stopped", "deadline_epoch": deadline})
            def fake_run(args, **kwargs):
                if any(str(a).endswith("benchmark-training-io.py") for a in args):
                    write_json(artifacts / "acceleration" / "training_io_benchmark.json", {
                        "results": [{"input": "raw", "workers": 2, "images_per_second": 20},
                                    {"input": "cache", "workers": 2, "images_per_second": 50},
                                    {"input": "cache", "workers": 4, "images_per_second": 45}],
                        "cache_speedup_vs_raw": 2.5, "pixel_and_seeded_augmentation_equality": "PASS"})
            with patch.object(acceleration, "ROOT", root), patch.object(acceleration, "ARTIFACTS", artifacts), \
                 patch.object(acceleration.subprocess, "run", side_effect=fake_run), \
                 patch.object(acceleration.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as launch:
                acceleration.main()
                args = launch.call_args.args[0]
                self.assertEqual(args[args.index("--workers")+1], "2")
                remaining = float(args[args.index("--hours")+1])
                self.assertLessEqual(remaining, 1)
                self.assertGreater(remaining, .99)
            self.assertEqual(read_json(artifacts / "acceleration" / "state.json")["status"], "complete")
            for name in ("model.pt", "resume.pt"):
                self.assertEqual((checkpoint / name).read_bytes(),
                                 (artifacts / "acceleration" / "checkpoint_before" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
