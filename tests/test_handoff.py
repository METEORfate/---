import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from glaucoma.audit import require_audit, run_audit
from glaucoma.common import read_json, write_csv
from glaucoma.data import local_fingerprint
from glaucoma.handoff import import_audit


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.central, self.audit = self.root / "central", self.root / "audit"
        rng = np.random.default_rng(123)
        for split, count in (("train", 12), ("valid", 2), ("test", 2)):
            (self.central / split).mkdir(parents=True)
            rows = []
            for i in range(count):
                name = f"image{i}.jpg"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(self.central / split / name.upper())
                rows.append({"Image_name": name, "Age": 30+i, "True_Label": i % 2})
            fields = ["Image_name", "Age"] + ([] if split == "test" else ["True_Label"])
            write_csv(self.central / f"{split}.csv", rows, fields)
        run_audit(self.central, self.audit, workers=2)
        self.copy = self.root / "copy"
        shutil.copytree(self.central, self.copy)
        for p in self.copy.rglob("*.JPG"):
            stat = p.stat()
            os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns + 100_000_000_000))

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_copy_different_mtime_imports_same_folds(self):
        self.assertNotEqual(local_fingerprint(self.central), local_fingerprint(self.copy))
        output = self.root / "imported"
        imported = import_audit(self.copy, self.audit, output, workers=2)
        self.assertEqual(imported["data_fingerprint"], read_json(self.audit / "report.json")["data_fingerprint"])
        self.assertEqual((self.audit / "folds.csv").read_bytes(), (output / "folds.csv").read_bytes())
        self.assertEqual(require_audit(self.copy, output)["import_status"], "all_image_sha256_verified")
        (output / "folds.csv").write_bytes((output / "folds.csv").read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            require_audit(self.copy, output)

    def test_same_size_modified_image_rejected_before_training(self):
        path = next((self.copy / "train").glob("*.JPG"))
        data = bytearray(path.read_bytes())
        data[150] ^= 1
        path.write_bytes(data)
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "Copied image differs"):
            import_audit(self.copy, self.audit, output, workers=2)
        self.assertFalse(read_json(output / "report.json")["complete"])


if __name__ == "__main__":
    unittest.main()
