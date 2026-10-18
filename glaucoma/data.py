from __future__ import annotations

import csv
import io
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import ROOT, digest, sha256

PREPROCESS_VERSION = "fov-letterbox-v1"


def load_rows(root, split):
    root = Path(root).resolve()
    if split not in ("train", "valid", "test"):
        raise ValueError("Unknown split")
    folder = root / split
    mapping = {}
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"Unexpected directory/link in {split}: {entry.name}")
            key = entry.name.casefold()
            if key in mapping:
                raise ValueError(f"Case-insensitive filename collision in {split}")
            stat = entry.stat(follow_symlinks=False)
            mapping[key] = (str(folder / entry.name), stat.st_size, stat.st_mtime_ns)
    rows, seen = [], set()
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"Image_name", "Age"} | ({"True_Label"} if split != "test" else set())
        if set(reader.fieldnames or []) != expected or len(reader.fieldnames or []) != len(expected):
            raise ValueError(f"Unexpected CSV schema in {split}: {reader.fieldnames}")
        for record in reader:
            if None in record or any(v is None for v in record.values()):
                raise ValueError("Non-rectangular CSV row")
            name = record["Image_name"]
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Image_name must be a basename")
            if name.casefold() in seen or name.casefold() not in mapping:
                raise ValueError(f"Duplicate or missing image in {split}")
            seen.add(name.casefold())
            age = float(record["Age"]) if record["Age"].strip() else float("nan")
            if not math.isnan(age) and not math.isfinite(age):
                raise ValueError("Non-finite age")
            label = int(record["True_Label"]) if split != "test" else -1
            if split != "test" and label not in (0, 1):
                raise ValueError("Invalid binary label")
            path, size, mtime = mapping[name.casefold()]
            rows.append({"name": name, "path": path, "age": age, "file_size": size,
                         "file_mtime_ns": mtime, "label": label, "split": split, "key": f"{split}:{name}"})
    if seen != set(mapping):
        raise ValueError(f"Unlisted extra files in {split}")
    return rows


def local_fingerprint(root, rows=None):
    root = Path(root)
    if rows is None:
        rows = sum((load_rows(root, s) for s in ("train", "valid", "test")), [])
    return digest({"csv": {s: sha256(root / f"{s}.csv") for s in ("train", "valid", "test")},
                   "files": [(r["key"], r["file_size"], r["file_mtime_ns"]) for r in rows]})


def csv_fingerprints(root):
    return {s: sha256(Path(root) / f"{s}.csv") for s in ("train", "valid", "test")}


def content_fingerprint(root, records):
    """Portable identity; copying identical bytes to another filesystem is harmless."""
    return digest({"csv": csv_fingerprints(root),
                   "files": sorted((r["key"], r["sha256"]) for r in records if r["ok"])})


@dataclass
class AgeTransform:
    median: float
    lower: float
    upper: float
    mean: float
    std: float

    @classmethod
    def fit(cls, ages):
        a = np.asarray(ages, dtype=np.float64)
        good = a[np.isfinite(a)]
        if not len(good):
            raise ValueError("No observed training ages")
        return cls(float(np.median(good)), float(good.min()), float(good.max()),
                   float(good.mean()), max(float(good.std()), 1.0))

    def transform(self, ages):
        a = np.asarray(ages, dtype=np.float64)
        missing = ~np.isfinite(a)
        outside = (~missing) & ((a < self.lower) | (a > self.upper))
        a = np.where(missing, self.median, a)
        a = (np.clip(a, self.lower, self.upper) - self.mean) / self.std
        return np.column_stack([a, missing, outside]).astype(np.float32)

    def to_dict(self):
        return asdict(self)


def crop_fov(image):
    """Conservative border crop, computed on a small preview with dark-image fallback."""
    preview = image.copy()
    preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(preview)
    brightness = arr.max(axis=2)
    mask = brightness > 10
    if mask.mean() < 0.25 or np.percentile(brightness, 90) < 30:
        return image, False
    ys, xs = np.nonzero(mask)
    pw, ph = preview.size
    # Padding protects the edge of the circular field from threshold errors.
    pad = max(pw, ph) * 0.025
    x0, x1 = max(0, xs.min() - pad), min(pw, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(ph, ys.max() + 1 + pad)
    if (x1 - x0) * (y1 - y0) < pw * ph * 0.35:
        return image, False
    w, h = image.size
    box = (int(x0 / pw * w), int(y0 / ph * h), math.ceil(x1 / pw * w), math.ceil(y1 / ph * h))
    return image.crop(box), box != (0, 0, w, h)


def decode_image(path, size=1024, full_decode=False):
    with Image.open(path) as source:
        if not full_decode and source.format == "JPEG":
            source.draft("RGB", (max(1024, size), max(1024, size)))
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    image, _ = crop_fov(image)
    return image


def resize_image(image, size, augment=False):
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    if augment:
        import random
        from PIL import ImageEnhance
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.rotate(random.uniform(-15, 15), Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
    return np.array(image, dtype=np.uint8, copy=True)


def read_image(path, size, augment=False, full_decode=False):
    return resize_image(decode_image(path, size, full_decode), size, augment)


class ImageDataset:
    def __init__(self, rows, size, age_transform=None, augment=False, cache=None):
        self.rows, self.size, self.augment = rows, size, augment
        self.ages = age_transform.transform([r["age"] for r in rows]) if age_transform else np.zeros((len(rows), 3), np.float32)
        self.cache_path = str(cache) if cache else None
        self.cache = np.load(cache, mmap_mode="r", allow_pickle=False) if cache else None
        if self.cache is not None and self.cache.shape != (len(rows), size, size, 3):
            raise ValueError("Image cache shape does not match selected rows")

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        # Windows spawn must reopen the file, never serialize a 40GB memmap.
        state = self.__dict__.copy()
        state["cache"] = None
        return state

    def __getitem__(self, i):
        import torch
        row = self.rows[i]
        if self.cache is None and self.cache_path:
            self.cache = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        if self.cache is None:
            arr = read_image(row["path"], self.size, self.augment)
        else:
            arr = np.array(self.cache[i], copy=True)
            if self.augment:
                import random
                from PIL import ImageEnhance
                image = Image.fromarray(arr)
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = image.rotate(random.uniform(-15, 15), Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
                arr = np.array(image, copy=True)
        return (torch.from_numpy(arr).permute(2, 0, 1).float().div_(255),
                torch.from_numpy(self.ages[i]), torch.tensor(float(row["label"])), i)


def worker_seed(worker_id):
    import random
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

