"""Native pre-augmentation views, using the existing durable pixel-cache builder."""
from __future__ import annotations

import shutil
from pathlib import Path
import time

import numpy as np

from .common import digest, read_json
from .data import decode_image
from .multiview import VIEW_VERSION, native_pixels
from .pixel_cache import build_pixel_cache, cache_signature


def local_dtype(size):
    return np.dtype([("pixels", np.uint8, (5, size, size, 3)),
                     ("coordinates", np.float32, (5, 4)), ("status", np.int8, (5,))])


def signature(rows, size, fingerprint, local):
    return {**cache_signature(rows, size), "format": "native-fixed-views-v1",
            "view_version": VIEW_VERSION, "full_decode": True,
            "data_fingerprint": fingerprint, "component": "local5" if local else "global"}


def paths_for(rows, size, artifacts, fingerprint):
    common = signature(rows, size, fingerprint, False)
    prefix = Path(artifacts) / "pixel_cache" / f"native_{size}_{digest(common)[:14]}"
    return {"global": prefix.with_name(prefix.name + "_global.npy"),
            "local": prefix.with_name(prefix.name + "_local.npy")}


def find_view_cache(rows, config, artifacts, fingerprint):
    if config.get("view_version") != VIEW_VERSION:
        return None
    size = config["size"]
    paths = paths_for(rows, size, artifacts, fingerprint)
    required = ("global", "local") if config.get("multiview") else ("global",)
    result = {}
    for key in required:
        path = paths[key]
        expected = signature(rows, size, fingerprint, key == "local")
        if not path.exists() or not path.with_suffix(".json").exists():
            return None
        if read_json(path.with_suffix(".json")) != expected:
            return None
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (len(rows),) if key == "local" else (len(rows), size, size, 3)
        dtype = local_dtype(size) if key == "local" else np.dtype(np.uint8)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("Completed native cache shape or dtype differs")
        del array
        bitmap = path.with_suffix(".done.npy")
        if not bitmap.exists():
            return None
        done = np.load(bitmap, mmap_mode="r", allow_pickle=False)
        if done.dtype != np.bool_ or done.shape != (len(rows),) or not done.all():
            return None
        del done
        result[key] = str(path)
    return result


def decode_global(path, size):
    return native_pixels(decode_image(path, size, full_decode=True), size, local=False)


def decode_local(path, size):
    pixels = native_pixels(decode_image(path, size, full_decode=True), size, include_global=False)
    record = np.zeros((), dtype=local_dtype(size))
    record["pixels"], record["coordinates"], record["status"] = pixels["local"], pixels["coordinates"], pixels["status"]
    return record


def estimated_bytes(count, size, local=True):
    return count * (size*size*3 + (local_dtype(size).itemsize if local else 0)) + 1024*1024


def build_view_cache(rows, config, artifacts, fingerprint, budget=None, workers=6, timings=None):
    if config.get("view_version") != VIEW_VERSION:
        raise ValueError("Unknown native-view preprocessing version")
    size = config["size"]
    paths = paths_for(rows, size, artifacts, fingerprint)
    paths["global"].parent.mkdir(parents=True, exist_ok=True)
    required = ("global", "local") if config.get("multiview") else ("global",)
    extra = 0
    for key in required:
        path = paths[key]
        complete = find_view_cache(rows, {**config, "multiview": key == "local"}, artifacts, fingerprint)
        if complete and key in complete:
            continue
        allocation = path.with_suffix(".partial.npy")
        target_bytes = len(rows) * (size*size*3 if key == "global" else local_dtype(size).itemsize)
        extra += max(0, target_bytes - (allocation.stat().st_size if allocation.exists() else 0))
    if shutil.disk_usage(paths["global"].parent).free - extra - 1024*1024 < 30*1024**3:
        raise RuntimeError("Native view cache would leave less than 30GiB free")
    for key in required:
        started = time.monotonic()
        build_pixel_cache(rows, size, paths[key], budget, workers,
                          decoder=decode_local if key == "local" else decode_global,
                          signature=signature(rows, size, fingerprint, key == "local"),
                          item_shape=() if key == "local" else (size, size, 3),
                          dtype=local_dtype(size) if key == "local" else np.uint8)
        if timings is not None:
            timings[key] = time.monotonic()-started
    return find_view_cache(rows, config, artifacts, fingerprint)
