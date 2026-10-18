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
from .common import ARTIFACTS, Budget, ROOT, code_fingerprint, digest, exclusive_output, read_json, seed_all, sha256, versions, write_csv, write_json
from .data import AgeTransform, ImageDataset, PREPROCESS_VERSION, load_rows, read_image, worker_seed
from .metrics import metric_key, metrics
from .models import FundusModel, SPECS, WEIGHTS, download_weights, load_checkpoint, save_checkpoint
from .performance import execution_settings, load_profile
from .pixel_cache import find_pixel_cache
from .multiview import empty_diagnostics, image_count, image_dataset, move_images, view_diagnostics


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


def loader(dataset, batch_size, workers, shuffle=False, seed=42, sampler=None):
    options = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                   generator=torch.Generator().manual_seed(seed), worker_init_fn=worker_seed)
    if sampler is not None:
        options.update(sampler=sampler, shuffle=False)
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


def extract_features(root, config, batch_size=None, workers=None, device="auto", split="train", budget=None,
                     performance_profile=None):
    if config.get("multiview") or config.get("native_resolution"):
        raise ValueError("Native-view configs use supervised training, not legacy resized feature caches")
    audit = require_audit(root)
    device = device_for(device)
    profile = load_profile(performance_profile, config, device) if performance_profile else None
    settings = execution_settings(config, "features", profile, workers, batch_size)
    batch_size, workers = settings["batch_size"], settings["workers"]
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
        cache = find_pixel_cache(rows, size, ARTIFACTS, audit["data_fingerprint"]) if settings["use_pixel_cache"] else None
        data = loader(Subset(ImageDataset(rows, size, cache=cache), todo.tolist()), batch_size, workers)
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
                    loader_wait_fraction=waiting / max(time.monotonic()-started, 1e-6),
                    execution=settings, pixel_cache=str(cache) if cache else None)
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
    count = int(config.get("smoke_batch_size", 2))
    chosen = [row for label in (0, 1) for row in [r for r in rows if r["label"] == label][:max(1, count//2)]]
    age = AgeTransform.fit([r["age"] for r in rows])
    model = FundusModel(config, download_weights(config["model"])).configure_trainable(config.get("tail_blocks", 2 if config["model"] != "effnet" else -1)).to(device)
    data = loader(image_dataset(chosen, config, age), count, 0)
    images, ages, y, _ = next(iter(data))
    images, ages, y = move_images(images, device), ages.to(device), y.to(device)
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
    model.active_view_batch_size = 0  # Validation/inference retain the original numerical path.
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
    result = {"purpose": "implementation smoke test, not model performance", "train_images": image_count(images), "config": config,
              "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
              "loss": float(loss.detach()), "reload_max_abs_difference": float((actual-expected).abs().max()),
              "seconds": time.monotonic()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "versions": versions()}
    suffix = "_multiview" if config.get("multiview") else "_native" if config.get("native_resolution") else ""
    write_json(output / f"{config['model']}{suffix}_{device}_report.json", result)
    print(result, flush=True)


def build_pixel_cache(rows, size, output, budget=None):
    from .pixel_cache import build_pixel_cache as build
    return build(rows, size, output, budget, workers=int(os.environ.get("GLAUCOMA_CACHE_WORKERS", "6")))


def evaluate_loader(model, data, device, progress_path=None, epoch=None, budget=None):
    predictions, indices, labels = [], [], []
    views = empty_diagnostics()
    model.eval()
    model.active_view_batch_size = 0
    with torch.inference_mode():
        started = time.monotonic()
        for step, (images, age, y, ix) in enumerate(data, 1):
            view_diagnostics(images, views)
            if budget:
                budget.check()
            with autocast(device):
                logits = model(move_images(images, device), age.to(device, non_blocking=True))
            predictions.extend(torch.sigmoid(logits.double()).cpu().tolist())
            indices.extend(ix.tolist())
            labels.extend(y.tolist())
            if progress_path and (step == 1 or step % 100 == 0 or step == len(data)):
                seconds = time.monotonic()-started
                write_json(progress_path, {"phase": "validation", "epoch": epoch, "step": step,
                           "steps_per_epoch": len(data), "samples": len(indices), "seconds": seconds,
                           "images_per_second": len(indices)/max(seconds, 1e-6)})
                print(f"validation epoch={epoch} step={step}/{len(data)}", flush=True)
    if progress_path and views["images"]:
        write_json(Path(progress_path).with_name("validation_views.json"), {"epoch": epoch, **views})
    return np.array(indices), np.array(predictions), metrics(labels, predictions)


def training_mode(model, warmup):
    model.train()
    model.active_view_batch_size = 0 if warmup else int(getattr(model, "performance_view_batch_size", 0))
    model.backbone.eval()
    if not warmup and model.kind != "effnet":
        for block in model.backbone.blocks:
            if any(p.requires_grad for p in block.parameters()):
                block.train()


def train_batches(model, data, optimizer, scaler, device, config, warmup=False, reduced_sync=False,
                  budget=None, deadline=None, teacher_predictor=None, rows=None, progress_path=None,
                  epoch=1, max_epochs=1, warmup_batches=0, record_order=False):
    """The production update loop, also used by isolated performance benchmarks."""
    training_mode(model, warmup)
    for parameter in model.backbone.parameters():
        if parameter.requires_grad:
            parameter.grad = None
    optimizer.zero_grad(set_to_none=True)
    accumulation = config.get("accumulation", 8)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64) if reduced_sync else 0.0
    count, waiting, updates, startup = 0, 0., 0, 0.
    views = empty_diagnostics()
    sample_order = []
    synchronize(device)
    started = previous = time.monotonic()
    measured_start, measured_count, interrupted = None, 0, False
    for step, (images, ages, labels, indices) in enumerate(data):
        arrived = time.monotonic()
        waiting += arrived-previous
        if step == 0:
            startup = arrived-started
        if budget:
            budget.check()
        if deadline is not None and time.monotonic() >= deadline:
            interrupted = True
            break
        if step == warmup_batches:
            synchronize(device)
            measured_start = time.monotonic()
        view_diagnostics(images, views)
        if record_order:
            sample_order.extend(indices.tolist())
        images, ages, labels = move_images(images, device), ages.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        divisor = min(accumulation, len(data)-group_start)
        scaler.scale(loss/divisor).backward()
        if (step+1) % accumulation == 0 or step+1 == len(data):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += int(scaler.get_scale() >= scale_before)
        if reduced_sync:
            loss_sum.add_(loss.detach().double()*image_count(images))
        else:
            loss_sum += float(loss.detach())*image_count(images)
        count += image_count(images)
        if step >= warmup_batches:
            measured_count += image_count(images)
        if progress_path and (step == 0 or (step+1) % 100 == 0):
            current_loss = float(loss_sum)/count
            elapsed = time.monotonic()-started
            print(f"epoch {epoch}/{max_epochs} step {step+1}/{len(data)} loss={current_loss:.5f}", flush=True)
            write_json(progress_path, {"epoch": epoch, "step": step+1, "steps_per_epoch": len(data),
                       "phase": "head_warmup" if warmup else "fine_tune", "samples": count, "seconds": elapsed,
                       "images_per_second": count/max(elapsed, 1e-6), "loss": current_loss,
                       "loader_wait_fraction": waiting/max(elapsed, 1e-6),
                       "loader_wait_measurement": "host wait; asynchronous work may overlap" if reduced_sync else "synchronized batches",
                       "peak_gpu_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None})
        if not reduced_sync:
            synchronize(device)
        previous = time.monotonic()
    synchronize(device)
    elapsed = time.monotonic()-started
    return {"loss": float(loss_sum)/max(count, 1), "count": count, "waiting": waiting,
            "seconds": elapsed, "startup_seconds": startup, "interrupted": interrupted,
            "optimizer_steps": updates, "sample_order": sample_order if record_order else None,
            "view_diagnostics": views, "measured_images": measured_count,
            "measured_seconds": time.monotonic()-measured_start if measured_start is not None else 0.}


def fine_tune(root, config, output, fold=0, seed=42, epochs=None, workers=None, device="auto", resume=False,
              teacher=None, budget=None, max_hours=10, performance_profile=None,
              validation_batch_size=None, validation_workers=None, initialize_from=None):
    audit = require_audit(root)
    seed_all(seed)
    device = device_for(device)
    profile = load_profile(performance_profile, config, device) if performance_profile else None
    settings = execution_settings(config, "train", profile, workers)
    validation = execution_settings(config, "validation", profile,
                                    validation_workers if validation_workers is not None else workers, validation_batch_size)
    workers = settings["workers"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root, "train")
    folds = folds_for(rows)
    all_ix = np.arange(len(rows))
    fit_ix = all_ix if fold == "full" else np.flatnonzero(folds != int(fold))
    val_ix = np.array([], dtype=int) if fold == "full" else np.flatnonzero(folds == int(fold))
    age = AgeTransform.fit([rows[i]["age"] for i in fit_ix])
    model = FundusModel(config, download_weights(config["model"])).configure_trainable(config.get("tail_blocks", 2)).to(device)
    model.performance_view_batch_size = int((profile or {}).get("view_encode_batch_size", 0))
    if teacher and config.get("multiview"):
        raise ValueError("Tensor distillation cannot reconstruct original-resolution local views")
    initialization = None
    if initialize_from:
        initialization = sha256(initialize_from)
        source = torch.load(initialize_from, map_location="cpu", weights_only=True)
        meta = source["metadata"]
        if (meta["fold"] != fold or meta.get("data_fingerprint") != audit["data_fingerprint"]
                or meta.get("fold_fingerprint") != sha256(ARTIFACTS / "audit" / "folds.csv")
                or source["age_transform"] != age.to_dict()
                or meta.get("preprocess") != PREPROCESS_VERSION):
            raise ValueError("Initialization must exclude the same holdout, with identical data, folds and age transform")
        for key in ("model", "size", "use_age", "head", "head_standardize", "native_resolution", "view_version"):
            if source["config"].get(key) != config.get(key):
                raise ValueError(f"Initialization configuration differs: {key}")
        restored = model.load_state_dict(source["state_dict"], strict=False)
        if restored.unexpected_keys or any(not k.startswith("local_mil.") for k in restored.missing_keys):
            raise ValueError("Initialization is not a compatible global-to-multiview checkpoint")
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.get("backbone_lr", 1e-5)},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad],
         "lr": config.get("head_lr", 1e-3)}], weight_decay=0.01)
    max_epochs = int(epochs or config.get("epochs", 8))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    run_signature = {"config": config, "fold": fold, "seed": seed, "epochs": max_epochs,
                     "fold_fingerprint": sha256(ARTIFACTS / "audit" / "folds.csv"),
                     "data": audit["data_fingerprint"], "teacher": sha256(teacher) if teacher else None}
    if initialization:
        run_signature["initialize_from_sha256"] = initialization
    if config.get("native_resolution") or config.get("multiview"):
        run_signature["view_code_fingerprint"] = code_fingerprint()
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
    cache_path = ARTIFACTS / "pixel_cache" / f"train_{config['size']}_{audit['data_fingerprint'][:10]}.npy"
    native = bool(config.get("multiview") or config.get("native_resolution"))
    if native:
        verified = bool(profile and profile.get("native_view_cache_verified"))
        settings["use_pixel_cache"] = bool(verified and profile.get("stages", {}).get("train", {}).get("accepted") and settings["use_pixel_cache"])
        validation["use_pixel_cache"] = bool(verified and profile.get("stages", {}).get("validation", {}).get("accepted") and validation["use_pixel_cache"])
        from .view_cache import find_view_cache
        discover = lambda: find_view_cache(rows, config, ARTIFACTS, audit["data_fingerprint"])
    else:
        discover = lambda: find_pixel_cache(rows, config["size"], ARTIFACTS, audit["data_fingerprint"])
    cache = discover() if settings["use_pixel_cache"] else None
    history_path = output / "history.json"
    history = read_json(history_path) if history_path.exists() else []
    started = time.monotonic()
    interrupted = False
    # A stop during cache construction must not leave resumed training permanently on raw JPEGs.
    if (cache is None and not performance_profile and settings["use_pixel_cache"] and history and config.get("auto_pixel_cache", True)
            and history[-1].get("loader_wait_fraction", 0) > .3):
        cache = build_pixel_cache(rows, config["size"], cache_path, budget)
    for epoch in range(start, max_epochs):
        seed_all(seed + epoch)
        train_data = loader(Subset(image_dataset(rows, config, age, True, cache), fit_ix.tolist()),
                            config.get("batch_size", 4), workers, True, seed+epoch)
        warmup = epoch < config.get("head_warmup_epochs", 1)
        epoch_started = time.monotonic()
        training = train_batches(model, train_data, optimizer, scaler, device, config, warmup,
                                 settings["reduced_sync"], budget, started+max_hours*3600,
                                 teacher_predictor, rows, output / "progress.json", epoch+1, max_epochs)
        interrupted = training["interrupted"]
        waiting = training["waiting"]
        del train_data
        if interrupted:
            write_json(output / "status.json", {"complete": False, "reason": "session_time_budget", "last_complete_epoch": epoch,
                                                "resume_from_full_epoch": True})
            break
        scheduler.step()
        result = None
        if len(val_ix):
            validation_cache = discover() if validation["use_pixel_cache"] else None
            data = loader(Subset(image_dataset(rows, config, age, False, validation_cache), val_ix.tolist()),
                          validation["batch_size"], validation["workers"])
            ix, pred, result = evaluate_loader(model, data, device, output / "progress.json", epoch+1, budget)
            del data
            improved = metric_key(result) > best
        else:
            improved = True
        metadata = {"mode": "distillation" if teacher else "fine_tune", "fold": fold, "seed": seed, "epoch": epoch+1,
                    "fold_fingerprint": run_signature["fold_fingerprint"],
                    "data_fingerprint": audit["data_fingerprint"], "preprocess": PREPROCESS_VERSION,
                    "versions": versions(), "holdout": result}
        metadata["execution"] = {"train": settings, "validation": validation}
        metadata["performance_profile"] = str(performance_profile) if performance_profile else None
        metadata["view_encode_batch_size"] = model.performance_view_batch_size
        metadata["loader_workers"] = workers
        metadata["pixel_cache"] = cache if isinstance(cache, dict) else str(cache) if cache else None
        if initialization:
            metadata["initialize_from_sha256"] = initialization
        if native:
            metadata["view_version"] = config.get("view_version")
            metadata["native_resolution"] = True
        if config.get("multiview"):
            metadata["view_diagnostics"] = {"train": training["view_diagnostics"],
                "validation": read_json(output / "validation_views.json") if len(val_ix) else None}
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
        history.append({"epoch": epoch+1, "loss": training["loss"], "holdout": result,
                        "seconds": seconds, "loader_wait_fraction": waiting/max(seconds, 1e-6),
                        "optimizer_steps": training["optimizer_steps"], "view_diagnostics": training["view_diagnostics"]})
        write_json(history_path, history)
        print(f"epoch {epoch+1}: {history[-1]}", flush=True)
        if stale >= config.get("patience", 3) and len(val_ix):
            break
        if not performance_profile and settings["use_pixel_cache"] and waiting/max(seconds, 1e-6) > .3 and config.get("auto_pixel_cache", True) and cache is None:
            cache = build_pixel_cache(rows, config["size"], cache_path, budget)
    if not interrupted:
        write_json(output / "status.json", {"complete": True, "last_complete_epoch": len(history)})
    return output / "model.pt"
