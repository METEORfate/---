import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from glaucoma import pixel_cache
from glaucoma.common import read_json
from glaucoma.data import ImageDataset, read_image


class PixelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        rng = np.random.default_rng(42)
        for i in range(7):
            path = self.root / f"image{i}.jpg"
            Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)).save(path)
            stat = path.stat()
            self.rows.append({"name": path.name, "path": str(path), "age": 30+i, "label": i % 2,
                              "file_size": stat.st_size, "file_mtime_ns": stat.st_mtime_ns})
        self.cache = self.root / "cache.npy"
        self.disk = patch.object(pixel_cache.shutil, "disk_usage", return_value=SimpleNamespace(free=100*1024**3))
        self.disk.start()

    def tearDown(self):
        self.disk.stop()
        self.temp.cleanup()

    def test_disk_reserve_is_enforced(self):
        with patch.object(pixel_cache.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, "30GiB"):
                pixel_cache.build_pixel_cache(self.rows, 64, self.cache)

    def test_pixels_and_seeded_augmentations_are_identical(self):
        pixel_cache.build_pixel_cache(self.rows, 64, self.cache, workers=3, flush_every=2)
        cached = np.load(self.cache, mmap_mode="r")
        for i, row in enumerate(self.rows):
            np.testing.assert_array_equal(cached[i], read_image(row["path"], 64))
        del cached
        raw = ImageDataset(self.rows, 64, augment=True)
        fast = ImageDataset(self.rows, 64, augment=True, cache=self.cache)
        for i in range(len(self.rows)):
            random.seed(100+i)
            expected = raw[i][0]
            random.seed(100+i)
            self.assertTrue(torch.equal(expected, fast[i][0]))
        del raw, fast
        self.assertEqual(read_json(self.cache.with_suffix(".progress.json"))["status"], "complete")

    def test_interrupted_build_resumes_only_uncommitted_images(self):
        class StopAfterTwo:
            calls = 0
            def check(self):
                self.calls += 1
                if self.calls > 2:
                    raise TimeoutError("bounded test")
        with self.assertRaises(TimeoutError):
            pixel_cache.build_pixel_cache(self.rows, 64, self.cache, StopAfterTwo(), workers=2)
        self.assertFalse(self.cache.exists())
        done = np.load(self.cache.with_suffix(".done.npy"))
        self.assertEqual(int(done.sum()), 2)
        with patch.object(pixel_cache, "read_image", wraps=read_image) as decode:
            pixel_cache.build_pixel_cache(self.rows, 64, self.cache, workers=2)
            self.assertEqual(decode.call_count, 5)
        with patch.object(pixel_cache, "read_image", side_effect=AssertionError("should reuse")):
            pixel_cache.build_pixel_cache(self.rows, 64, self.cache)

    def test_partial_cache_changed_provenance_is_rejected(self):
        class Stop:
            def check(self):
                raise TimeoutError()
        with self.assertRaises(TimeoutError):
            pixel_cache.build_pixel_cache(self.rows, 64, self.cache, Stop())
        changed = [{**row} for row in self.rows]
        changed[0]["file_mtime_ns"] += 1
        with self.assertRaisesRegex(ValueError, "provenance"):
            pixel_cache.build_pixel_cache(changed, 64, self.cache)


if __name__ == "__main__":
    unittest.main()
