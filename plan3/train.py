from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from glaucoma.common import exclusive_output, read_json, seed_all, sha256, versions, write_json
from glaucoma.data import AgeTransform, resize_image, worker_seed
from glaucoma.engine import autocast, save_predictions
from glaucoma.metrics import metric_key, metrics
from .data import Views, original
from .models import Classifier, Distillation, EMA, Encoder, Locator, classification_loss, dino_loss, load_backbone
from .provenance import Context, check_support, pretrained, safe_path, signature, recipe_provenance

SESSION_DEADLINE = float("inf")


def session_check():
    if time.monotonic() >= SESSION_DEADLINE-300:
        raise TimeoutError("Session publication reserve reached; retain outputs and continue in a new quota session")


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [move(v, device) for v in value]
    return value.to(device, non_blocking=True)


class PairBatches(Sampler):
    """Explicit positive/negative microbatches; accumulation does not create ranking pairs."""
    def __init__(self, labels, batch_size, seed, hard=None):
        if batch_size < 4:
            raise ValueError("Ranking microbatch must be at least 4 to include positive, random and hard negatives")
        self.p = np.flatnonzero(np.asarray(labels) == 1)
        self.n = np.flatnonzero(np.asarray(labels) == 0)
        self.batch_size, self.seed, self.hard = batch_size, seed, np.asarray(hard if hard else self.n)
        if not len(self.p) or not len(self.n):
            raise ValueError("Ranking partition requires both classes")
        if not set(self.hard).issubset(set(self.n)):
            raise ValueError("Hard-negative list contains a positive or out-of-partition index")

    def __len__(self):
        return math.ceil((len(self.p)+len(self.n))/self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        p_count = max(1, self.batch_size//4)
        h_count = max(1, (self.batch_size-p_count)//2)
        for _ in range(len(self)):
            batch = np.concatenate([rng.choice(self.p, p_count), rng.choice(self.hard, h_count),
                                    rng.choice(self.n, self.batch_size-p_count-h_count)])
            rng.shuffle(batch)
            yield batch.tolist()


class BoxData(Dataset):
    def __init__(self, rows, boxes, size):
        self.rows, self.boxes, self.size = rows, boxes, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = original(row["path"])
        # Locator sees a stretched full original, so xyxy normalization is exact and reversible.
        x = np.array(image.resize((self.size, self.size)), dtype=np.uint8, copy=True)
        annotation = self.boxes[row["name"]]
        valid = annotation["status"] == "valid"
        box = ([float(annotation[k]) for k in ("x_min", "y_min", "x_max", "y_max")] if valid else [0.]*4)
        w, h = image.size
        return torch.from_numpy(x).permute(2, 0, 1).float()/255, torch.tensor(box)/torch.tensor([w, h, w, h]), torch.tensor(float(valid)), i


def data_loader(data, batch_size, workers, seed, shuffle=False, pair=False, hard=None):
    options = dict(num_workers=workers, worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed),
                   pin_memory=torch.cuda.is_available())
    # Workers restart each epoch: saved complete-epoch recovery reproduces augmentation streams.
    if workers:
        options.update(prefetch_factor=2)
    if pair:
        options["batch_sampler"] = PairBatches([r["label"] for r in data.rows], batch_size, seed, hard)
    else:
        options.update(batch_size=batch_size, shuffle=shuffle)
    return DataLoader(data, **options)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def init_encoder(encoder, payload):
    source = payload["config"]
    if source["model"] != encoder.config["model"]:
        raise ValueError("Initialization backbone architecture changed")
    # Only transfer the encoder, including an explicit positional-grid adaptation.
    prefix = "encoder."
    state = {k[len(prefix):]: v for k, v in payload["state_dict"].items() if k.startswith(prefix)}
    if not state:
        raise ValueError("Initialization has no trained encoder")
    if hasattr(encoder, "base"):
        prefix = "base.backbone."
        backbone = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        load_backbone(encoder.base.backbone, backbone)
    else:
        load_backbone(encoder.backbone, {k[9:]: v for k, v in state.items() if k.startswith("backbone.")})


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "plan3-checkpoint-v1":
        raise ValueError("Expected a plan3 checkpoint")
    return payload


def build(payload, device):
    cfg = payload["config"]
    encoder = Encoder(cfg)
    model = {"locator": Locator, "classifier": lambda enc: Classifier(cfg, enc),
             "ssl": Distillation}[payload["kind"]](encoder)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def locate(payload, rows, device, batch_size=16):
    model = build(payload, device)
    size = payload["config"]["size"]
    result = {}
    for start in range(0, len(rows), batch_size):
        session_check()
        chunk = rows[start:start+batch_size]
        images = [original(r["path"]) for r in chunk]
        array = np.stack([np.asarray(im.resize((size, size)), dtype=np.uint8) for im in images])
        x = torch.from_numpy(array).permute(0, 3, 1, 2).float().to(device)/255
        with autocast(device):
            boxes, confidence = model(x)
        for row, image, box, score in zip(chunk, images, boxes.cpu().tolist(), confidence.sigmoid().cpu().tolist()):
            w, h = image.size
            pixels = (np.array(box)*[w, h, w, h]).tolist()
            # Conservative failure flag; never turn an invalid local into an ordinary valid view.
            valid = score >= .5 and min(pixels[2]-pixels[0], pixels[3]-pixels[1]) >= 2
            result[row["name"]] = {"box": pixels, "valid": valid, "confidence": score}
    del model
    return result


def verify_locator_review(path, locator_path, payload):
    review = read_json(path)
    allowed = set(payload["metadata"].get("review_names", []))
    checks = review.get("checks", [])
    names = [r["Image_name"] for r in checks]
    if (review.get("locator_sha256") != sha256(locator_path) or not checks or len(set(names)) != len(names)
            or set(names) != allowed or any(type(r.get("missed_disc")) is not bool for r in checks)):
        raise ValueError("Locator review must cover its entire internal review split and match the checkpoint hash")
    if sum(r["missed_disc"] for r in checks)/len(checks) > .05:
        raise ValueError("More than 5% missed discs: repair annotations/localization before ROI classification")


@torch.inference_mode()
def evaluate(model, data, device, kind, deadline=float("inf")):
    model.eval()
    pred, indices, labels, loss = [], [], [], []
    for images, ages, y, ix in data:
        if time.monotonic() >= deadline:
            raise TimeoutError("Session reserve reached during validation; resume the last complete epoch")
        with autocast(device):
            if kind == "locator":
                boxes, confidence = model(images.to(device))
                target, valid = ages.to(device), y.to(device)
                value = nn.functional.binary_cross_entropy_with_logits(confidence.float(), valid)
                if valid.bool().any():
                    value += nn.functional.smooth_l1_loss(boxes[valid.bool()].float(), target[valid.bool()])
                loss.append(float(value))
            else:
                p = model(move(images, device), ages.to(device))["logits"].double().sigmoid()
                pred.extend(p.cpu().tolist())
                labels.extend(y.tolist())
                indices.extend(ix.tolist())
    if kind == "locator":
        return None, None, {"loss": float(np.mean(loss))}
    return np.asarray(indices), np.asarray(pred), metrics(labels, pred)


@torch.inference_mode()
def evaluate_pair(model, averaged, data, device, deadline=float('inf')):
    """Two complete classifier evaluations, sharing each decode and device transfer."""
    model.eval(); averaged.eval()
    indices, labels, raw, ema = [], [], [], []
    for step, (images, ages, y, ix) in enumerate(data):
        if time.monotonic() >= deadline:
            raise TimeoutError('Session reserve reached during paired validation')
        images, ages = move(images, device), ages.to(device)
        with autocast(device):
            a = model(images, ages)['logits'].double().sigmoid()
            b = averaged(images, ages)['logits'].double().sigmoid()
        indices.extend(ix.tolist()); labels.extend(y.tolist())
        raw.extend(a.cpu().tolist()); ema.extend(b.cpu().tolist())
        if step % 1000 == 0:
            print(f'plan3 paired validation step={step+1}/{len(data)} images={len(labels)}', flush=True)
    ix = np.asarray(indices)
    return (ix, np.asarray(raw), metrics(labels, raw)), (ix, np.asarray(ema), metrics(labels, ema))


def run_training(root, config, job, device="cuda"):
    from .jobs import validate_config
    validate_config(config)
    if config.get("requires_initialization") and not job.get("initialize_from"):
        raise ValueError("This stage requires a matched trained initializer")
    if bool(config.get("ssl")) != (job["task"] == "ssl"):
        raise ValueError("SSL configs are only valid for the ssl task; supervised controls use their own configs")
    context = Context(root)
    annotation, boxes = context.annotations(safe_path(job["annotations"]), job.get("annotation_policy", "full1000"))
    fit, hold = context.split(job["fold"], job.get("inner_fold"))
    cfg = copy.deepcopy(config)
    weight, source = pretrained(cfg)
    if cfg["model"] not in ("rfm", "effnet", "green"):
        cfg["source_spec"] = {k: source[k] for k in ("architecture", "mean", "std")}
    kind = {"train": "classifier", "finalize": "classifier", "locator": "locator", "ssl": "ssl"}[job["task"]]
    seed_all(job.get("seed", 42))
    encoder = Encoder(cfg, weight, source)
    dependencies = {"pretrained": sha256(weight)}
    initialization = None
    if job.get("initialize_from"):
        path = safe_path(job["initialize_from"])
        payload = load_payload(path)
        check_support(payload["metadata"], context, fit)
        if job["fold"] == "full" and payload["kind"] == "classifier" and payload["metadata"]["fold"] != "full":
            raise ValueError("Full refit stages require a full-train supervised initializer")
        if payload["metadata"].get("seed") != job.get("seed", 42):
            raise ValueError("Stage initialization must use the same selected seed")
        init_encoder(encoder, payload)
        initialization = payload
        dependencies["initialize_from"] = sha256(path)
        dependencies["initialization_recipe"] = {"config": payload["config"], "seed": payload["metadata"].get("seed"),
            "kind": payload["kind"], "annotation_sha256": payload["metadata"].get("annotation_sha256"),
            "pretrained": payload["signature"].get("dependencies", {}).get("pretrained")}
    model = {"classifier": lambda e: Classifier(cfg, e), "locator": Locator, "ssl": Distillation}[kind](encoder).to(device)
    if initialization and initialization["kind"] == kind == "classifier":
        # Preserve the trained full-image baseline before adding a zero-initialized ROI residual.
        for head in ("global_head", "local_head"):
            values = {k[len(head)+1:]: v for k, v in initialization["state_dict"].items() if k.startswith(head+".")}
            getattr(model, head).load_state_dict(values, strict=True)
        if initialization["config"].get("view", "global") == cfg.get("view", "global"):
            # A continued joint model keeps its trained interaction weights as well.
            state = {k: v for k, v in initialization["state_dict"].items() if not k.startswith("encoder.")}
            expected = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
                raise ValueError("Joint initialization changes the trained interaction architecture")
            model.load_state_dict({**model.state_dict(), **state}, strict=True)
    locator, roi = None, {}
    if cfg.get("view", "global") != "global" or kind == "ssl":
        path = safe_path(job["locator"])
        locator = load_payload(path)
        if locator["kind"] != "locator":
            raise ValueError("Expected a locator checkpoint")
        check_support(locator["metadata"], context, fit)
        verify_locator_review(safe_path(job["locator_review"]), path, locator)
        dependencies["locator"] = sha256(path)
        dependencies["locator_review"] = sha256(safe_path(job["locator_review"]))
        # TRAIN and this internal holdout only. Never read official-valid/test in training.
        chosen = [context.rows[i] for i in np.concatenate([fit, hold])]
        roi = locate(locator, chosen, device)
    metadata = context.metadata(fit, hold, job["fold"], job.get("inner_fold"))
    metadata.update(annotation_version=annotation["version"], annotation_sha256=annotation["boxes_sha256"], dependencies=dependencies)
    if initialization and initialization["kind"] == "classifier":
        expected_age = AgeTransform.fit([context.rows[i]["age"] for i in fit]).to_dict()
        if initialization["age_transform"] != expected_age:
            raise ValueError("Supervised initialization requires the identical fold-local age transform")
    if kind == "locator":
        annotated = [i for i in fit if context.rows[i]["name"] in boxes and boxes[context.rows[i]["name"]]["status"] != "unclear"]
        # Grouped internal review split is drawn only from the outer training support.
        from sklearn.model_selection import GroupShuffleSplit
        if len(annotated) < 6:
            raise ValueError("Too few annotated training images for locator internal review")
        train_ix, review_ix = next(GroupShuffleSplit(1, test_size=.2, random_state=42).split(annotated, groups=np.asarray(context.groups)[annotated]))
        selected, reviewed = [annotated[i] for i in train_ix], [annotated[i] for i in review_ix]
        train_rows, val_rows = [context.rows[i] for i in selected], [context.rows[i] for i in reviewed]
        metadata["train_names"] = [r["name"] for r in train_rows]
        metadata["review_names"] = [r["name"] for r in val_rows]
        train_set, val_set = BoxData(train_rows, boxes, cfg["size"]), BoxData(val_rows, boxes, cfg["size"])
        age = None
    else:
        train_rows, val_rows = [context.rows[i] for i in fit], [context.rows[i] for i in hold]
        age = AgeTransform.fit([r["age"] for r in train_rows])
        train_cache = val_cache = None
        if cfg.get('pixel_cache'):
            from glaucoma.common import ARTIFACTS
            from .cache import prepare_global, source_hashes
            hashes = source_hashes(ARTIFACTS/'audit')
            folder = ARTIFACTS/'plan3_cache'
            train_cache = prepare_global(train_rows, cfg['size'], folder, context.audit['data_fingerprint'], hashes, job.get('audit_workers', 8))
            if val_rows:
                val_cache = prepare_global(val_rows, cfg['size'], folder, context.audit['data_fingerprint'], hashes, job.get('audit_workers', 8))
            dependencies['pixel_cache'] = {'train_sha256': sha256(train_cache),
                'holdout_sha256': sha256(val_cache) if val_cache else None,
                'preprocess_version': cfg['preprocess_version']}
        train_set = Views(train_rows, cfg, age, roi, True, cache=train_cache)
        val_set = Views(val_rows, cfg, age, roi, False, cache=val_cache)
    selection_config = copy.deepcopy(cfg)
    if job["task"] == "finalize":
        cfg = {**cfg, "epochs": final_epochs(context, job, config)}
        metadata["selection_config"] = selection_config
        metadata["final_weights"] = job.get("final_weights", job.get("_cv_final_weights", "ema"))
        dependencies["cv_models"] = [sha256(safe_path(p)) for p in job["cv_models"]]
        dependencies["final_weights"] = metadata["final_weights"]
    ranking = float(cfg.get("ranking_weight", 0))
    hard = None
    if cfg.get("requires_initialization") and not initialization:
        raise ValueError("This high-resolution/ranking stage requires its matched trained initializer")
    if ranking or cfg.get("paired_sampling"):
        if not job.get("hard_negatives"):
            raise ValueError("Ranking needs training-partition inner-OOF hard negatives")
        from .results import validate_hard_negatives
        hard = validate_hard_negatives(safe_path(job["hard_negatives"]), context, fit, job)
        dependencies["hard_negatives"] = sha256(safe_path(job["hard_negatives"]))
    sig = signature(cfg, context, annotation, fit, hold, job, dependencies)
    output = safe_path(job["output"])
    with exclusive_output(output / ".lock"):
        return _train(model, kind, cfg, job, sig, metadata, train_set, val_set, age, locator, output, device, hard)


def final_epochs(context, job, config):
    if job["fold"] != "full" or job.get("inner_fold") is not None:
        raise ValueError("Finalization requires full-train support")
    paths = job.get("cv_models", [])
    if len(paths) != 3:
        raise ValueError("Finalize requires three complete CV models")
    epochs, seen, selected_weights, common = [], set(), [], None
    for value in paths:
        path = safe_path(value)
        payload = load_payload(path)
        meta = payload["metadata"]
        # Runtime source_spec is only normalization metadata; training configuration is otherwise exact.
        effective = {k: v for k, v in payload["config"].items() if k != "source_spec"}
        requested = {k: v for k, v in config.items() if k != "source_spec"}
        if (effective != requested or payload["kind"] != "classifier" or meta.get("inner_fold") is not None or meta.get("seed") != job.get("seed", 42) or
                meta["fold"] not in (0, 1, 2) or meta["fold"] in seen or
                not read_json(path.parent / "status.json").get("complete")):
            raise ValueError("Finalize needs three completed same-recipe outer folds")
        fit, hold = context.split(meta["fold"])
        check_support(meta, context, fit)
        if (set(meta["holdout_names"]) != {context.rows[i]["name"] for i in hold} or
                set(meta["train_names"]) != {context.rows[i]["name"] for i in fit}):
            raise ValueError("Finalization CV support mismatch")
        current = recipe_provenance(payload)
        if common is None:
            common = current
        if common != current:
            raise ValueError("Finalization CV models have different annotation/pretrained/input provenance")
        seen.add(meta["fold"])
        epochs.append(meta["epoch"])
        selected_weights.append(meta["selected_weights"])
    job["_cv_final_weights"] = max(("raw", "ema"), key=selected_weights.count)
    return int(np.median(epochs))


def _train(model, kind, cfg, job, sig, metadata, train_set, val_set, age, locator, output, device, hard):
    output.mkdir(parents=True, exist_ok=True)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.get("backbone_lr", 1e-5)},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": cfg.get("head_lr", 1e-3)}], weight_decay=.01)
    epochs = cfg.get("epochs", 24)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    ema = EMA(model, cfg.get("ema_decay", .999))
    start, best, stale, history = 0, None, 0, []
    resume = output / "resume.pt"
    if job.get("resume"):
        if not resume.exists():
            raise ValueError("Resume requested but no matching complete-epoch checkpoint exists")
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["signature"] != sig:
            raise ValueError("Resume configuration, source code or input provenance changed")
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        ema.updates = state["ema_updates"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale, history = state["epoch"]+1, state["best"], state["stale"], state["history"]
        restore_rng(state["rng"])
        finished = start >= epochs or (kind != "ssl" and len(val_set) > 0 and stale >= cfg.get("patience", 4))
        if finished:
            write_json(output / "history.json", history)
            write_json(output / "status.json", {"complete": True, "reason": "recovered_complete_checkpoint", "last_complete_epoch": len(history)})
            return output / "model.pt"
    elif resume.exists() or (output / "model.pt").exists():
        raise ValueError("Output already contains a run; use resume or a new experiment name")
    write_json(output / "config.json", cfg)
    write_json(output / "signature.json", sig)
    write_json(output / "environment.json", versions())
    if not resume.exists():
        atomic_torch(resume, {"signature": sig, "model": cpu_state(model), "ema": cpu_state(ema.model), "ema_updates": 0,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": -1, "best": None, "stale": 0, "history": [], "rng": rng_state()})
    # Leave 5 minutes for CPU transfer and ZIP publication before the platform cutoff.
    deadline = min(SESSION_DEADLINE, time.monotonic()+job["max_hours"]*3600)-300
    complete, reason = False, "session_time_budget"
    for epoch in range(start, epochs):
        seed_all(job.get("seed", 42)+epoch)
        data = data_loader(train_set, cfg["batch_size"], job.get("workers", 0), job.get("seed", 42)+epoch,
                           True, bool(cfg.get("ranking_weight", 0) or cfg.get("paired_sampling")), hard)
        warmup = epoch < cfg.get("head_warmup_epochs", 1) and kind != "ssl"
        model.train()
        model.encoder.requires_grad_(not warmup)
        if warmup:
            model.encoder.eval()
        if kind == "ssl":
            ema.model.eval()
        optimizer.zero_grad(set_to_none=True)
        started, total, samples, interrupted, successful_steps = time.monotonic(), 0., 0, False, 0
        accumulation = cfg.get("accumulation", 8)
        for step, (images, ages, labels, ix) in enumerate(data):
            if time.monotonic() >= deadline:
                interrupted = True
                break
            images, ages, labels = move(images, device), ages.to(device), labels.to(device)
            with autocast(device):
                if kind == "locator":
                    box, confidence = model(images)
                    valid = labels.bool()
                    loss = nn.functional.binary_cross_entropy_with_logits(confidence.float(), labels)
                    if valid.any():
                        loss += nn.functional.smooth_l1_loss(box[valid].float(), ages[valid])
                elif kind == "ssl":
                    # Two global augmentations plus valid anatomical crops; no arbitrary patch equivalence.
                    global_views = [images["global"], images["global2"]]
                    student = [model(v) for v in global_views]
                    with torch.no_grad():
                        teacher = [ema.model(v) for v in global_views]
                    loss = dino_loss(student, teacher, ema.model.center)
                    valid = images["mask"].all(1)
                    if valid.any():
                        # Local-to-local consistency only, never force the disc to equal whole-image semantics.
                        local = images["local"][0][valid] if isinstance(images["local"], list) else images["local"][valid, 0]
                        local_student = [model(local), model(local.flip(-1))]
                        with torch.no_grad():
                            local_teacher = [ema.model(local), ema.model(local.flip(-1))]
                        loss += dino_loss(local_student, local_teacher, ema.model.center)
                    with torch.no_grad():
                        center = torch.cat(teacher).mean(0, keepdim=True)
                        ema.model.center.lerp_(center.float(), .1)
                        model.center.copy_(ema.model.center)
                else:
                    loss = classification_loss(model(images, ages), labels, cfg.get("ranking_weight", 0))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            group_start = step//accumulation*accumulation
            scaler.scale(loss/min(accumulation, len(data)-group_start)).backward()
            if (step+1) % accumulation == 0 or step+1 == len(data):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= before
                ema.update(model, successful)
                successful_steps += int(successful)
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())*len(labels)
            samples += len(labels)
            if step % 100 == 0:
                print(f"plan3 {kind} fold={job['fold']} epoch={epoch+1}/{epochs} step={step+1}/{len(data)} loss={total/samples:.6f}", flush=True)
                write_json(output / "progress.json", {"epoch": epoch+1, "step": step+1, "steps": len(data),
                    "loss": total/samples, "images_per_second": samples/max(time.monotonic()-started, 1e-6),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None})
        if interrupted:
            break  # Partial updates are deliberately discarded: resume.pt still ends at a full epoch.
        scheduler.step()
        selected, result, selected_prediction = model, None, None
        try:
            if kind != "ssl" and len(val_set):
                validation = data_loader(val_set, cfg.get("validation_batch_size", cfg["batch_size"]), job.get("workers", 0), 42)
                if kind == 'classifier' and cfg.get('paired_validation', False):
                    raw, averaged = evaluate_pair(model, ema.model, validation, device, deadline)
                else:
                    raw = evaluate(model, validation, device, kind, deadline)
                    averaged = evaluate(ema.model, validation, device, kind, deadline)
                if kind == 'classifier' and cfg.get('save_validation_members', False):
                    for member, prediction in [('raw', raw), ('ema', averaged)]:
                        ix, probabilities, member_metrics = prediction
                        save_predictions(output/f'epoch_{epoch+1}_{member}.csv', [val_set.rows[i] for i in ix], probabilities,
                                         [job.get('inner_fold') if job.get('inner_fold') is not None else job['fold']]*len(ix))
                        write_json(output/f'epoch_{epoch+1}_{member}.json', member_metrics)
                key = (lambda r: (-r[2]["loss"],)) if kind == "locator" else (lambda r: metric_key(r[2]))
                selected, chosen = (ema.model, averaged) if key(averaged) > key(raw) else (model, raw)
                result, selected_prediction = chosen[2], chosen
                score = key(chosen)
            else:
                selected = ema.model if kind == "ssl" or metadata.get("final_weights") == "ema" else model
                score = (epoch,)
        except TimeoutError:
            break
        improved = best is None or tuple(score) > tuple(best)
        if improved:
            best, stale = tuple(score), 0
            meta = {**metadata, "epoch": epoch+1, "holdout": result,
                    "selected_weights": "ema" if selected is ema.model else "raw", "seed": job.get("seed", 42)}
            payload = {"format": "plan3-checkpoint-v1", "kind": kind, "config": cfg,
                       "state_dict": cpu_state(selected), "metadata": meta, "signature": sig,
                       "age_transform": age.to_dict() if age else None, "locator": locator}
            atomic_torch(output / "model.pt", payload)
            if kind == "classifier" and selected_prediction is not None:
                ix, probabilities, _ = selected_prediction
                save_predictions(output / "holdout.csv", [val_set.rows[i] for i in ix], probabilities,
                                 [job.get("inner_fold") if job.get("inner_fold") is not None else job["fold"]]*len(ix))
                write_json(output / "holdout.json", {**meta, "prediction_sha256": sha256(output / "holdout.csv"),
                                                       "model_sha256": sha256(output / "model.pt")})
        else:
            stale += 1
        history.append({"epoch": epoch+1, "loss": total/max(samples, 1), "holdout": result,
                        "raw_holdout": raw[2] if result else None, "ema_holdout": averaged[2] if result else None,
                        "seconds": time.monotonic()-started, "optimizer_steps": successful_steps,
                        "ema_updates": ema.updates, "selected_weights": "ema" if selected is ema.model else "raw"})
        print(f"plan3 {kind} epoch={epoch+1} complete loss={history[-1]['loss']:.6f} holdout={result} selected={history[-1]['selected_weights']}", flush=True)
        state = {"signature": sig, "model": cpu_state(model), "ema": cpu_state(ema.model), "ema_updates": ema.updates,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": best, "stale": stale, "history": history, "rng": rng_state()}
        atomic_torch(resume, state)
        write_json(output / "history.json", history)
        complete = epoch+1 == epochs or (kind != "ssl" and len(val_set) > 0 and stale >= cfg.get("patience", 4))
        if complete:
            reason = "epochs_or_early_stop"
        write_json(output / "status.json", {"complete": complete, "reason": reason, "last_complete_epoch": epoch+1})
        if complete:
            break
    write_json(output / "status.json", {"complete": complete, "reason": reason, "last_complete_epoch": len(history),
                                         "resume_from_full_epoch": True})
    if kind == "locator" and (output / "model.pt").exists():
        saved = load_payload(output / "model.pt")
        write_json(output / "review_template.json", {"locator_sha256": sha256(output / "model.pt"),
            "checks": [{"Image_name": name, "missed_disc": None} for name in saved["metadata"]["review_names"]]})
        predictions = locate(saved, val_set.rows, device)
        write_json(output / "review_predictions.json", predictions)
    return output / "model.pt"
