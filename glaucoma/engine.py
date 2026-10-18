from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .audit import require_audit
from .common import ARTIFACTS, Budget, ROOT, digest, exclusive_output, read_json, seed_all, sha256, versions, write_csv, write_json
from .data import AgeTransform, ImageDataset, PREPROCESS_VERSION, load_rows, read_image, worker_seed
from .metrics import metric_key, metrics
from .models import FundusModel, SPECS, WEIGHTS, download_weights, load_checkpoint, save_checkpoint


def device_for(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the installed PyTorch build")
    return name


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def loader(dataset, batch_size, workers, shuffle=False, seed=42):
    options = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                   generator=torch.Generator().manual_seed(seed), worker_init_fn=worker_seed)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def folds_for(rows):
    with (ARTIFACTS / "audit" / "folds.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    mapping = {r["Image_name"]: int(r["fold"]) for r in records}
    if set(mapping) != {r["name"] for r in rows} or len(mapping) != len(records):
        raise RuntimeError("Fold manifest does not match training rows")
    return np.array([mapping[r["name"]] for r in rows])


def save_predictions(path, rows, probabilities, folds=None):
    if len(rows) != len(probabilities):
        raise ValueError("Prediction length mismatch")
    records = []
    for i, (row, probability) in enumerate(zip(rows, probabilities)):
        p = float(probability)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid predicted probability")
        item = {"Image_name": row["name"], "Probability": format(p, ".17g"),
                "True_Label": row["label"], "fold": int(folds[i]) if folds is not None else ""}
        records.append(item)
    write_csv(path, records, ["Image_name", "Probability", "True_Label", "fold"])


def extract_features(root, config, batch_size=16, workers=2, device="auto", split="train", budget=None):
    audit = require_audit(root)
    device = device_for(device)
    name = config["model"]
    weight = download_weights(name)
    size = config.get("size", SPECS[name]["size"])
    signature = {"model": name, "size": size, "preprocess": PREPROCESS_VERSION,
                 "weight_sha256": sha256(weight), "data": audit["data_fingerprint"],
                 "runtime": versions(), "feature_precision": "cuda_fp16" if device.startswith("cuda") else "cpu_fp32"}
    folder = ARTIFACTS / "features" / f"{name}_{size}_{digest(signature)[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root, split)
    target = folder / f"{split}.npy"
    completion = folder / f"{split}_done.npy"
    meta_path = folder / f"{split}.json"
    with exclusive_output(folder / f"{split}.lock"):
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta["signature"] != signature or meta["names"] != [r["name"] for r in rows]:
                raise RuntimeError("Feature cache provenance mismatch")
            if meta.get("complete"):
                return target
        else:
            meta = {"signature": signature, "names": [r["name"] for r in rows], "complete": False}
            write_json(meta_path, meta)
        shape = (len(rows), SPECS[name]["dim"])
        values = np.lib.format.open_memmap(target, mode="r+" if target.exists() else "w+", dtype=np.float32, shape=shape)
        done = np.lib.format.open_memmap(completion, mode="r+" if completion.exists() else "w+", dtype=np.bool_, shape=(len(rows),))
        todo = np.flatnonzero(~done)
        model = FundusModel(config, weight).to(device).eval()
        data = loader(Subset(ImageDataset(rows, size), todo.tolist()), batch_size, workers)
        processed, started, waiting, previous = 0, time.monotonic(), 0.0, time.monotonic()
        with torch.inference_mode():
            for images, _, _, indices in data:
                waiting += time.monotonic() - previous
                if budget:
                    budget.check()
                images = images.to(device, non_blocking=True)
                with autocast(device):
                    feature = model.features(images)
                feature = feature.float().cpu().numpy()
                if not np.isfinite(feature).all():
                    raise RuntimeError("Non-finite pretrained features")
                values[indices.numpy()] = feature
                values.flush()
                done[indices.numpy()] = True
                done.flush()
                processed += len(images)
                if processed % (batch_size * 25) == 0:
                    print(f"features {name}/{split}: {processed}/{len(todo)}, {processed/(time.monotonic()-started):.1f} images/s", flush=True)
                previous = time.monotonic()
        meta.update(complete=True, versions=versions(), extraction_seconds=time.monotonic()-started,
                    loader_wait_fraction=waiting / max(time.monotonic()-started, 1e-6))
        write_json(meta_path, meta)
        print(f"features complete: {target}", flush=True)
    return target


def fit_logistic(x, y, c=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x)
    linear = LogisticRegression(C=c, max_iter=500, random_state=42).fit(scaler.transform(x), y)
    coefficient = linear.coef_[0] / scaler.scale_
    intercept = float(linear.intercept_[0] - np.dot(coefficient, scaler.mean_))
    return coefficient, intercept


def probability(x, coefficient, intercept):
    from scipy.special import expit
    return expit(np.asarray(x) @ coefficient + intercept)


def train_probe(root, config, features_path, output, budget=None):
    audit = require_audit(root)
    rows = load_rows(root, "train")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features_path = Path(features_path)
    feature_meta = read_json(features_path.with_suffix(".json"))
    if (not feature_meta["complete"] or feature_meta["names"] != [r["name"] for r in rows]
        or feature_meta["signature"]["model"] != config["model"]
        or feature_meta["signature"]["size"] != config["size"]
        or feature_meta["signature"]["data"] != audit["data_fingerprint"]):
        raise ValueError("Feature cache is incomplete, misaligned, or from another backbone")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    y, ages = np.array([r["label"] for r in rows]), np.array([r["age"] for r in rows])
    folds = folds_for(rows)
    results, best = [], None
    for c in (0.01, 0.1, 1.0):
        oof = np.full(len(rows), np.nan)
        for fold in range(3):
            if budget:
                budget.check()
            fit, hold = folds != fold, folds == fold
            age = AgeTransform.fit(ages[fit])
            x = np.concatenate([features, age.transform(ages)], axis=1) if config.get("use_age", True) else features
            coefficient, intercept = fit_logistic(x[fit], y[fit], c)
            oof[hold] = probability(x[hold], coefficient, intercept)
        result = {"C": c, **metrics(y, oof)}
        results.append(result)
        print(f"probe {config['model']} age={config.get('use_age', True)} C={c}: {result}", flush=True)
        if best is None or metric_key(result) > metric_key(best[0]):
            best = result, oof.copy()
    age = AgeTransform.fit(ages)
    x = np.concatenate([features, age.transform(ages)], axis=1) if config.get("use_age", True) else features
    coefficient, intercept = fit_logistic(x, y, best[0]["C"])
    config = {**config, "head": "linear"}
    model = FundusModel(config, download_weights(config["model"]))
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(coefficient[None], dtype=torch.float32))
        model.head.bias.fill_(intercept)
    metadata = {"mode": "frozen_probe_full_refit", "fold": "full", "C": best[0]["C"],
                "data_fingerprint": audit["data_fingerprint"], "preprocess": PREPROCESS_VERSION,
                "fold_fingerprint": sha256(ARTIFACTS / "audit" / "folds.csv"),
                "selection": "C selected on train-only pooled 3-fold OOF; metrics are selection estimates",
                "oof": best[0], "versions": versions(), "seed": 42}
    save_checkpoint(output / "model.pt", model, config, age, metadata)
    save_predictions(output / "oof.csv", rows, best[1], folds)
    write_json(output / "oof.json", {"config": config, "data_fingerprint": audit["data_fingerprint"],
                                    "fold_fingerprint": metadata["fold_fingerprint"], "prediction_sha256": sha256(output / "oof.csv")})
    write_json(output / "report.json", {"candidates": results, "selected": best[0], "metadata": metadata})
    return output / "model.pt"


def age_baseline(root, output):
    rows, valid, test = (load_rows(root, s) for s in ("train", "valid", "test"))
    age = AgeTransform.fit([r["age"] for r in rows])
    coefficient, intercept = fit_logistic(age.transform([r["age"] for r in rows]), [r["label"] for r in rows])
    p = probability(age.transform([r["age"] for r in valid]), coefficient, intercept)
    output = Path(output)
    write_json(output / "model.json", {"age": age.to_dict(), "coefficient": coefficient.tolist(), "intercept": intercept})
    write_json(output / "report.json", {"valid": metrics([r["label"] for r in valid], p), "purpose": "age-only diagnostic; not the primary competition model"})
    save_predictions(output / "valid.csv", valid, p)
    from .deploy import write_submission
    p = probability(age.transform([r["age"] for r in test]), coefficient, intercept)
    write_submission(output / "submission.csv", test, p)


def smoke(root, config, device="auto"):
    """Actual pretrained forward/backward/reload on a bounded TRAIN-only sample."""
    import tempfile
    device = device_for(device)
    rows = load_rows(root, "train")
    chosen = [next(r for r in rows if r["label"] == label) for label in (0, 1)]
    age = AgeTransform.fit([r["age"] for r in rows])
    model = FundusModel(config, download_weights(config["model"])).configure_trainable(2 if config["model"] != "effnet" else -1).to(device)
    data = loader(ImageDataset(chosen, config["size"], age), 2, 0)
    images, ages, y, _ = next(iter(data))
    images, ages, y = images.to(device), ages.to(device), y.to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    start = time.monotonic()
    with autocast(device):
        loss = nn.functional.binary_cross_entropy_with_logits(model(images, ages).float(), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.eval()
    with torch.inference_mode(), autocast(device):
        expected = model(images, ages).float().cpu()
    output = ARTIFACTS / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        path = Path(folder) / "smoke.pt"
        save_checkpoint(path, model, config, age, {"mode": "smoke_only", "fold": "full"})
        restored, _, _ = load_checkpoint(path, device)
        with torch.inference_mode(), autocast(device):
            actual = restored(images, ages).float().cpu()
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)
    synchronize(device)
    result = {"purpose": "implementation smoke test, not model performance", "train_images": 2,
              "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
              "loss": float(loss.detach()), "reload_max_abs_difference": float((actual-expected).abs().max()),
              "seconds": time.monotonic()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "versions": versions()}
    write_json(output / f"{config['model']}_{device}_report.json", result)
    print(result, flush=True)


def build_pixel_cache(rows, size, output, budget=None):
    from .pixel_cache import build_pixel_cache as build
    return build(rows, size, output, budget, workers=int(os.environ.get("GLAUCOMA_CACHE_WORKERS", "6")))


def evaluate_loader(model, data, device, progress_path=None, epoch=None):
    predictions, indices, labels = [], [], []
    model.eval()
    with torch.inference_mode():
        started = time.monotonic()
        for step, (images, age, y, ix) in enumerate(data, 1):
            with autocast(device):
                logits = model(images.to(device, non_blocking=True), age.to(device, non_blocking=True))
            predictions.extend(torch.sigmoid(logits.double()).cpu().tolist())
            indices.extend(ix.tolist())
            labels.extend(y.tolist())
            if progress_path and (step == 1 or step % 100 == 0 or step == len(data)):
                seconds = time.monotonic()-started
                write_json(progress_path, {"phase": "validation", "epoch": epoch, "step": step,
                           "steps_per_epoch": len(data), "samples": len(indices), "seconds": seconds,
                           "images_per_second": len(indices)/max(seconds, 1e-6)})
                print(f"validation epoch={epoch} step={step}/{len(data)}", flush=True)
    return np.array(indices), np.array(predictions), metrics(labels, predictions)


def fine_tune(root, config, output, fold=0, seed=42, epochs=None, workers=2, device="auto", resume=False,
              teacher=None, budget=None, max_hours=10):
    audit = require_audit(root)
    seed_all(seed)
    device = device_for(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root, "train")
    folds = folds_for(rows)
    all_ix = np.arange(len(rows))
    fit_ix = all_ix if fold == "full" else np.flatnonzero(folds != int(fold))
    val_ix = np.array([], dtype=int) if fold == "full" else np.flatnonzero(folds == int(fold))
    age = AgeTransform.fit([rows[i]["age"] for i in fit_ix])
    model = FundusModel(config, download_weights(config["model"])).configure_trainable(config.get("tail_blocks", 2)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.get("backbone_lr", 1e-5)},
        {"params": model.head.parameters(), "lr": config.get("head_lr", 1e-3)}], weight_decay=0.01)
    max_epochs = int(epochs or config.get("epochs", 8))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    run_signature = {"config": config, "fold": fold, "seed": seed, "epochs": max_epochs,
                     "fold_fingerprint": sha256(ARTIFACTS / "audit" / "folds.csv"),
                     "data": audit["data_fingerprint"], "teacher": sha256(teacher) if teacher else None}
    last_path = output / "resume.pt"
    start, best, stale = 0, (-1.0, -1.0), 0
    if last_path.exists():
        if not resume:
            raise RuntimeError("Output already contains a run; pass --resume or choose another output")
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["signature"] != run_signature:
            raise ValueError("Resume config/fold/seed/provenance changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best, stale = saved["epoch"] + 1, tuple(saved["best"]), saved["stale"]
        if (output / "status.json").exists() and read_json(output / "status.json").get("complete"):
            return output / "model.pt"
    teacher_predictor = None
    if teacher:
        from .deploy import Predictor
        teacher_predictor = Predictor(teacher, device)
        for item in teacher_predictor.entries:
            if (item[2]["metadata"]["fold"] != fold
                or item[2]["metadata"].get("data_fingerprint") != audit["data_fingerprint"]):
                raise ValueError("Teacher must exclude the same held-out fold; full-data teachers only distill full-data students")
    cache = None
    cache_path = ARTIFACTS / "pixel_cache" / f"train_{config['size']}_{audit['data_fingerprint'][:10]}.npy"
    if cache_path.exists() and cache_path.with_suffix(".json").exists():
        expected = {"names": [r["name"] for r in rows], "size": config["size"], "preprocess": PREPROCESS_VERSION,
                    "files": digest([(r["file_size"], r["file_mtime_ns"]) for r in rows])}
        if read_json(cache_path.with_suffix(".json")) == expected:
            cache = cache_path
    history_path = output / "history.json"
    history = read_json(history_path) if history_path.exists() else []
    started = time.monotonic()
    interrupted = False
    # A stop during cache construction must not leave resumed training permanently on raw JPEGs.
    if (cache is None and history and config.get("auto_pixel_cache", True)
            and history[-1].get("loader_wait_fraction", 0) > .3):
        cache = build_pixel_cache(rows, config["size"], cache_path, budget)
    for epoch in range(start, max_epochs):
        seed_all(seed + epoch)
        train_data = loader(Subset(ImageDataset(rows, config["size"], age, True, cache), fit_ix.tolist()),
                            config.get("batch_size", 4), workers, True, seed+epoch)
        model.train()
        model.backbone.eval()  # Frozen dropout/BN statistics remain deterministic.
        if epoch >= config.get("head_warmup_epochs", 1):
            if model.kind != "effnet":
                for block in model.backbone.blocks:
                    if any(p.requires_grad for p in block.parameters()):
                        block.train()
        warmup = epoch < config.get("head_warmup_epochs", 1)
        for p in model.backbone.parameters():
            if p.requires_grad:
                p.grad = None
        optimizer.zero_grad(set_to_none=True)
        accumulation = config.get("accumulation", 8)
        loss_sum, count, waiting = 0.0, 0, 0.0
        epoch_started = previous = time.monotonic()
        for step, (images, ages, labels, indices) in enumerate(train_data):
            waiting += time.monotonic() - previous
            if budget:
                budget.check()
            if time.monotonic() - started >= max_hours * 3600:
                interrupted = True
                break  # Partial epoch discarded on resume; only full-epoch states are saved.
            images, ages, labels = images.to(device, non_blocking=True), ages.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device):
                if warmup:
                    with torch.no_grad():
                        features = model.features(images)
                    logits = model.classify(features, ages)
                else:
                    logits = model(images, ages)
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
                if teacher_predictor:
                    with torch.no_grad():
                        soft = teacher_predictor.predict_tensor(images, [rows[i]["age"] for i in indices.tolist()])
                    soft_logits = torch.logit(soft.clamp(1e-6, 1-1e-6))
                    distill = nn.functional.binary_cross_entropy_with_logits(logits.float()/2, torch.sigmoid(soft_logits/2)) * 4
                    loss = .5 * loss + .5 * distill
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            group_start = (step // accumulation) * accumulation
            divisor = min(accumulation, len(train_data) - group_start)
            scaler.scale(loss / divisor).backward()
            if (step+1) % accumulation == 0 or step+1 == len(train_data):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(images)
            count += len(images)
            if step == 0 or (step+1) % 100 == 0:
                print(f"epoch {epoch+1}/{max_epochs} step {step+1}/{len(train_data)} loss={loss_sum/count:.5f}", flush=True)
                elapsed = time.monotonic() - epoch_started
                write_json(output / "progress.json", {"epoch": epoch+1, "step": step+1,
                           "steps_per_epoch": len(train_data), "phase": "head_warmup" if warmup else "fine_tune",
                           "samples": count, "seconds": elapsed, "images_per_second": count/max(elapsed, 1e-6),
                           "loss": loss_sum/count, "loader_wait_fraction": waiting/max(elapsed, 1e-6),
                           "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None})
            synchronize(device)
            previous = time.monotonic()
        del train_data
        if interrupted:
            write_json(output / "status.json", {"complete": False, "reason": "session_time_budget", "last_complete_epoch": epoch,
                                                "resume_from_full_epoch": True})
            break
        scheduler.step()
        result = None
        if len(val_ix):
            data = loader(Subset(ImageDataset(rows, config["size"], age, False, cache), val_ix.tolist()),
                          config.get("batch_size", 4), workers)
            ix, pred, result = evaluate_loader(model, data, device, output / "progress.json", epoch+1)
            del data
            improved = metric_key(result) > best
        else:
            improved = True
        metadata = {"mode": "distillation" if teacher else "fine_tune", "fold": fold, "seed": seed, "epoch": epoch+1,
                    "fold_fingerprint": run_signature["fold_fingerprint"],
                    "data_fingerprint": audit["data_fingerprint"], "preprocess": PREPROCESS_VERSION,
                    "versions": versions(), "holdout": result}
        metadata["loader_workers"] = workers
        metadata["pixel_cache"] = str(cache) if cache else None
        if improved:
            if result:
                best = metric_key(result)
                save_predictions(output / "holdout.csv", [rows[i] for i in ix], pred, folds[ix])
            save_checkpoint(output / "model.pt", model, config, age, metadata)
            stale = 0
        else:
            stale += 1
        state = {"signature": run_signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale}
        temp = last_path.with_suffix(".tmp")
        torch.save(state, temp)
        os.replace(temp, last_path)
        seconds = time.monotonic() - epoch_started
        history.append({"epoch": epoch+1, "loss": loss_sum/count, "holdout": result,
                        "seconds": seconds, "loader_wait_fraction": waiting/max(seconds, 1e-6)})
        write_json(history_path, history)
        print(f"epoch {epoch+1}: {history[-1]}", flush=True)
        if stale >= config.get("patience", 3) and len(val_ix):
            break
        if waiting/max(seconds, 1e-6) > .3 and config.get("auto_pixel_cache", True) and cache is None:
            cache = build_pixel_cache(rows, config["size"], cache_path, budget)
    if not interrupted:
        write_json(output / "status.json", {"complete": True, "last_complete_epoch": len(history)})
    return output / "model.pt"
