"""Compare raw and cached inputs with the same development-fold training workload."""
import argparse
import gc
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import Subset
from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS, sha256, write_json
from glaucoma.data import ImageDataset, load_rows
from glaucoma.engine import autocast, folds_for, loader, synchronize
from glaucoma.models import load_checkpoint
from glaucoma.pixel_cache import cache_signature


def main(count):
    torch.set_num_threads(4)
    checkpoint = ARTIFACTS / "green_ft2" / "fold0" / "model.pt"
    original_hash = sha256(checkpoint)
    model, age, payload = load_checkpoint(checkpoint, "cuda")
    model.configure_trainable(2).train()
    model.backbone.eval()
    for block in model.backbone.blocks:
        if any(p.requires_grad for p in block.parameters()):
            block.train()
    rows = load_rows(ROOT / "glaucoma_first", "train")
    audit = require_audit(ROOT / "glaucoma_first")
    cache = ARTIFACTS / "pixel_cache" / f"train_392_{audit['data_fingerprint'][:10]}.npy"
    from glaucoma.common import read_json
    if read_json(cache.with_suffix(".json")) != cache_signature(rows, 392):
        raise ValueError("Full cache provenance mismatch")
    fit = np.flatnonzero(folds_for(rows) != 0)
    selected = np.random.default_rng(42).choice(fit, min(count, len(fit)), replace=False).tolist()
    raw, cached = ImageDataset(rows, 392, age, True), ImageDataset(rows, 392, age, True, cache)
    for i in selected[:32]:
        random.seed(2026+i)
        a = raw[i][0]
        random.seed(2026+i)
        if not torch.equal(a, cached[i][0]):
            raise AssertionError("Caching changed pixels or augmentation")
    results = []
    for name, dataset, workers in (("raw", raw, 2), ("cache", cached, 2), ("cache", cached, 4)):
        data = loader(Subset(dataset, selected), 4, workers, shuffle=False, seed=42)
        scaler = torch.amp.GradScaler("cuda")
        started, last, waiting, measured = None, time.monotonic(), 0., 0
        torch.cuda.reset_peak_memory_stats()
        for step, (images, ages, labels, _) in enumerate(data):
            arrived = time.monotonic()
            if step >= 8:
                if started is None:
                    started = last
                waiting += arrived-last
            model.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(images.to("cuda", non_blocking=True), ages.to("cuda", non_blocking=True))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.to("cuda", non_blocking=True))
            scaler.scale(loss).backward()
            synchronize("cuda")
            if step >= 8:
                measured += len(images)
            last = time.monotonic()
        seconds = last-started
        result = {"input": name, "workers": workers, "batch_size": 4, "measured_images": measured,
                  "seconds": seconds, "images_per_second": measured/seconds,
                  "loader_wait_fraction": waiting/seconds, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        print(result, flush=True)
        results.append(result)
        del data
        gc.collect()
    assert sha256(checkpoint) == original_hash
    report = {"scope": "Same fit-fold images; tail-2 forward/backward; no optimizer update; exclude 8 startup batches",
              "pixel_and_seeded_augmentation_equality": "PASS on 32 actual train images",
              "checkpoint_unchanged": True, "results": results,
              "cache_speedup_vs_raw": max(r["images_per_second"] for r in results[1:])/results[0]["images_per_second"]}
    write_json(ARTIFACTS / "acceleration" / "training_io_benchmark.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=512)
    args = p.parse_args()
    if args.count < 64:
        p.error("Use at least 64 images")
    main(args.count)
