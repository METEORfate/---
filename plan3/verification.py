from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from glaucoma.common import ROOT, code_fingerprint, read_json, sha256, write_json
from .provenance import pretrained, safe_path


def readiness(annotations, output):
    checks = {}
    folder = safe_path(annotations)
    report_path = folder / "acceptance.json"
    ann = read_json(report_path) if report_path.exists() else {}
    checks["annotations"] = {"ready": bool(ann.get("accepted") and ann.get("valid", 0) >= 1000 and ann.get("minimum_valid", 0) >= 1000),
                              "path": annotations}
    for model in ("rfm", "effnet", "dinov3", "retfound_dinov2", "convnext"):
        manifest = ROOT / "weights" / f"{model}_source.json"
        try:
            path, spec = pretrained({"model": model, "pretrained": manifest.relative_to(ROOT).as_posix()})
            checks[model] = {"ready": True, "sha256": sha256(path)}
        except (OSError, ValueError, KeyError) as exc:
            checks[model] = {"ready": False, "reason": str(exc)}
    result = {"formal_training_allowed": checks["annotations"]["ready"] and checks["rfm"]["ready"] and checks["effnet"]["ready"],
              "checks": checks, "legacy_glaucoma_code_fingerprint": code_fingerprint(),
              "note": "Preparation report only. No training, upload or Kaggle submission was started."}
    write_json(output, result)
    return result


def synthetic_check(output):
    """Real pretrained adapters, synthetic pixels only; no optimizer step or competition data."""
    import torch
    from .models import Classifier, Encoder, classification_loss
    from .train import atomic_torch, build, load_payload
    torch.set_num_threads(2)
    reports = {}
    for name in ("rfm", "effnet"):
        cfg = read_json(ROOT / f"configs/plan3/{'rfm448' if name == 'rfm' else 'effnet512'}.json")
        cfg.update(size=64, view="cross", local_size=80)
        weight, source = pretrained(cfg)
        model = Classifier(cfg, Encoder(cfg, weight, source)).eval()
        images = {"global": torch.rand(1, 3, 64, 64), "local": torch.rand(1, 2, 3, 80, 80),
                  "mask": torch.ones(1, 2, dtype=torch.bool), "coordinates": torch.tensor([[[.5, .5, .3, .3, 1.], [.5, .5, .5, .5, 1.]]])}
        ages = torch.zeros(1, 3)
        result = model(images, ages)
        loss = classification_loss(result, torch.ones(1))
        loss.backward()
        expected = model(images, ages)["logits"].detach()
        folder = Path(output).parent
        folder.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            path = Path(temporary) / "model.pt"
            atomic_torch(path, {"format": "plan3-checkpoint-v1", "kind": "classifier", "config": cfg,
                               "state_dict": model.state_dict()})
            restored = build(load_payload(path), "cpu")
            with torch.no_grad():
                actual = restored(images, ages)["logits"]
            torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
        reports[name] = {"pretrained_sha256": sha256(weight), "loss": float(loss.detach()),
                         "reload_max_error": float((actual-expected).abs().max()), "input_sizes": [64, 80]}
        del model, restored
    report = {"purpose": "pretrained synthetic adapter/backward/reload check, not performance or full-resolution capacity", "models": reports}
    write_json(output, report)
    return report


def compare(context, first, second, output):
    from .results import load_oof
    from glaucoma.metrics import metrics, paired_bootstrap
    a, meta_a = load_oof(first, context)
    b, meta_b = load_oof(second, context)
    labels = np.array([r["label"] for r in context.rows])
    ma, mb = metrics(labels, a), metrics(labels, b)
    delta = mb["auroc"]-ma["auroc"]
    result = {"first": ma, "second": mb, "delta_auroc": delta,
              "paired_interval": paired_bootstrap(labels, b, a), "second_seed_required": abs(delta) < .002,
              "per_fold": {str(f): {"first": metrics(labels[context.folds == f], a[context.folds == f]),
                                      "second": metrics(labels[context.folds == f], b[context.folds == f])} for f in range(3)},
              "selection_estimates": True, "first_sha256": meta_a["prediction_sha256"], "second_sha256": meta_b["prediction_sha256"]}
    write_json(output, result)
    return result


def capacity(context, config, job, output, device):
    import torch
    from .models import Classifier, Distillation, Encoder, classification_loss, dino_loss
    from .train import data_loader, move
    from .data import Views
    from glaucoma.data import AgeTransform
    from glaucoma.engine import autocast
    fit, _ = context.split(job["fold"], job.get("inner_fold"))
    rows = [context.rows[i] for i in fit]
    probe_rows = rows[:config["batch_size"]*config["accumulation"]*2]
    weight, source = pretrained(config)
    config = copy.deepcopy(config)
    if config["model"] not in ("rfm", "effnet", "green"):
        config["source_spec"] = {k: source[k] for k in ("architecture", "mean", "std")}
    roi = {}
    if config.get("view", "global") != "global" or config.get("ssl"):
        from .train import load_payload, locate, verify_locator_review
        from .provenance import check_support
        path = safe_path(job["locator"])
        locator = load_payload(path)
        check_support(locator["metadata"], context, fit)
        verify_locator_review(safe_path(job["locator_review"]), path, locator)
        roi = locate(locator, probe_rows, device)
    encoder = Encoder(config, weight, source)
    model = (Distillation(encoder) if config.get("ssl") else Classifier(config, encoder)).to(device)
    from .models import EMA
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("backbone_lr", 1e-5))
    age = AgeTransform.fit([r["age"] for r in rows])
    data = data_loader(Views(probe_rows, config, age, roi, True), config["batch_size"], job.get("workers", 0), 42)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start, n = time.monotonic(), 0
    for step, (images, ages, labels, _) in enumerate(data):
        images = move(images, device)
        with autocast(device):
            if config.get("ssl"):
                inputs = [images["global"], images["global2"]]
                student = [model(im) for im in inputs]
                with torch.no_grad():
                    teacher = [ema.model(im) for im in inputs]
                loss = dino_loss(student, teacher, ema.model.center)
                valid = images["mask"].all(1)
                if valid.any():
                    local = images["local"][0][valid] if isinstance(images["local"], list) else images["local"][valid, 0]
                    with torch.no_grad():
                        targets = [ema.model(local), ema.model(local.flip(-1))]
                    loss += dino_loss([model(local), model(local.flip(-1))], targets, ema.model.center)
            else:
                loss = classification_loss(model(images, ages.to(device)), labels.to(device))
        accumulation = config["accumulation"]
        divisor = min(accumulation, len(data)-step//accumulation*accumulation)
        (loss/divisor).backward()
        if (step+1) % accumulation == 0 or step+1 == len(data):
            optimizer.step()
            ema.update(model)
            model.zero_grad(set_to_none=True)
        n += len(labels)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    result = {"purpose": "capacity/throughput only; not model performance", "images": n,
              "seconds": elapsed, "images_per_second": n/elapsed,
              "training_pass_estimate_hours": len(rows)*elapsed/max(n, 1)/3600,
              "estimate_excludes": "full validation, EMA validation, startup and checkpoint publication",
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
              "optimizer_updates": ema.updates, "includes_adam_state_and_ema": True, "config": config}
    write_json(output / "capacity.json", result)
    print(json.dumps({"phase": "capacity_complete", **result}), flush=True)
    return result


def holdout(context, job, output, device):
    from .train import atomic_torch, load_payload
    from .results import predict_rows
    from glaucoma.engine import save_predictions
    from glaucoma.metrics import metrics
    path = safe_path(job["artifact"])
    payload = load_payload(path)
    meta = payload["metadata"]
    fit, hold = context.split(meta["fold"], meta.get("inner_fold"))
    from .provenance import check_support
    check_support(meta, context, fit)
    if meta["fold"] == "full" or not read_json(path.parent / "status.json").get("complete"):
        raise ValueError("TTA screening needs a complete held-out model")
    rows = [context.rows[i] for i in hold]
    policy = {"horizontal_tta": bool(job.get("horizontal_tta", False)), "scale_tta": job.get("scale_tta", [1.])}
    probabilities = predict_rows(path, rows, device, job.get("workers", 0), job.get("batch_size", 4),
                                 policy["horizontal_tta"], policy["scale_tta"])
    payload["config"] = {**payload["config"], "inference": policy}
    payload["metadata"] = {**meta, "holdout": metrics([r["label"] for r in rows], probabilities),
                           "tta_source_model_sha256": sha256(path)}
    atomic_torch(output / "model.pt", payload)
    save_predictions(output / "holdout.csv", rows, probabilities,
                     [meta["inner_fold"] if meta.get("inner_fold") is not None else meta["fold"]]*len(rows))
    write_json(output / "holdout.json", {**payload["metadata"], "prediction_sha256": sha256(output / "holdout.csv"),
                                        "model_sha256": sha256(output / "model.pt")})
    for name in ("status.json", "history.json"):
        shutil.copyfile(path.parent / name, output / name)
    return payload["metadata"]["holdout"]
