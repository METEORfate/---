from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class OutputBusy(RuntimeError):
    pass


@contextmanager
def exclusive_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutputBusy(f"Another process owns output {path}") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_fingerprint():
    return digest({p.name: sha256(p) for p in sorted((ROOT / "glaucoma").glob("*.py"))})


def seed_all(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class Budget:
    """Persistent wall-time cap; conservative relative to GPU-hours, including failures."""
    def __init__(self, enabled=False, hours=8):
        self.enabled = enabled
        self.limit = hours * 3600
        self.path = ARTIFACTS / "rfm_budget.json"
        self.used = read_json(self.path)["used_seconds"] if enabled and self.path.exists() else 0
        self.started = None

    def __enter__(self):
        self.started = time.monotonic()
        self.check()
        return self

    def check(self):
        if self.enabled and self.used + time.monotonic() - self.started >= self.limit:
            raise TimeoutError("RFM cumulative 8-hour budget exhausted; retain the existing baseline.")

    def __exit__(self, *exc):
        if self.enabled:
            elapsed = time.monotonic() - self.started
            write_json(self.path, {"used_seconds": self.used + elapsed, "limit_seconds": self.limit,
                                   "accounting": "wall time including downloads, extraction, training and failures"})

