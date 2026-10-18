"""Lossless, bounded-memory, resumable NumPy pixel caching."""
from __future__ import annotations

import os
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import digest, exclusive_output, read_json, write_json
from .data import PREPROCESS_VERSION, read_image


def cache_signature(rows, size):
    return {"names": [r["name"] for r in rows], "size": size, "preprocess": PREPROCESS_VERSION,
            "files": digest([(r["file_size"], r["file_mtime_ns"]) for r in rows])}


def build_pixel_cache(rows, size, output, budget=None, workers=6, flush_every=256):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(rows, size)
    meta_path = output.with_suffix(".json")
    progress_path = output.with_suffix(".progress.json")
    shape = (len(rows), size, size, 3)
    with exclusive_output(output.with_suffix(".lock")):
        if output.exists() and meta_path.exists() and read_json(meta_path) == signature:
            arr = np.load(output, mmap_mode="r", allow_pickle=False)
            if arr.shape != shape or arr.dtype != np.uint8:
                raise ValueError("Completed pixel cache has the wrong shape or dtype")
            del arr
            return output
        temp = output.with_suffix(".partial.npy")
        completion = output.with_suffix(".done.npy")
        build_meta = output.with_suffix(".build.json")
        resume = temp.exists() and completion.exists() and build_meta.exists()
        if resume and read_json(build_meta) != signature:
            raise ValueError("Partial pixel cache provenance changed; use a new cache path")
        required = int(np.prod(shape, dtype=np.int64))
        # A resumed array already occupies its full disk allocation.
        extra = max(0, required - temp.stat().st_size) if resume else max(0, required - (temp.stat().st_size if temp.exists() else 0))
        if shutil.disk_usage(output.parent).free - extra < 30 * 1024**3:
            raise RuntimeError("Pixel cache would leave less than 30GiB free")
        # Reuse a legacy allocation, but decode every image if it lacks a trusted bitmap.
        array = np.lib.format.open_memmap(temp, mode="r+" if temp.exists() else "w+", dtype=np.uint8, shape=shape)
        done = np.lib.format.open_memmap(completion, mode="r+" if resume else "w+", dtype=np.bool_, shape=(len(rows),))
        if array.shape != shape or array.dtype != np.uint8 or done.shape != (len(rows),) or done.dtype != np.bool_:
            raise ValueError("Partial cache shape differs from the manifest")
        if not resume:
            done[:] = False
            done.flush()
            write_json(build_meta, signature)
        workers = max(1, min(int(workers), 12))
        todo = iter(np.flatnonzero(~done).tolist())
        before = int(done.sum())
        completed, pending = before, []
        started = time.monotonic()
        print(f"pixel cache: {before}/{len(rows)} resumable; workers={workers}", flush=True)

        def progress(status):
            seconds = time.monotonic()-started
            write_json(progress_path, {"status": status, "total": len(rows), "completed": completed,
                       "resumed_images": before, "workers": workers, "seconds": seconds,
                       "images_per_second": (completed-before)/max(seconds, 1e-6), "bytes": required})

        def commit():
            if pending:
                array.flush()  # Publish completion only after the pixels are durable.
                done[pending] = True
                done.flush()
                pending.clear()

        progress("building")
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                queue = deque()
                for _ in range(workers*2):
                    i = next(todo, None)
                    if i is not None:
                        queue.append((i, pool.submit(read_image, rows[i]["path"], size)))
                while queue:
                    i, future = queue.popleft()
                    if budget:
                        budget.check()
                    image = future.result()
                    if image.shape != (size, size, 3) or image.dtype != np.uint8:
                        raise ValueError("Pixel decoder changed output shape or dtype")
                    array[i] = image
                    pending.append(i)
                    completed += 1
                    following = next(todo, None)
                    if following is not None:
                        queue.append((following, pool.submit(read_image, rows[following]["path"], size)))
                    if len(pending) >= flush_every:
                        commit()
                        progress("building")
                        print(f"pixel cache {completed}/{len(rows)}; {(completed-before)/max(time.monotonic()-started,1e-6):.1f} images/s", flush=True)
            commit()
        except BaseException:
            commit()
            progress("stopped_resumable")
            raise
        finally:
            del array, done
        os.replace(temp, output)
        write_json(meta_path, signature)
        progress("complete")
        return output
