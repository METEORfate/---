"""Opt-in, measured execution settings; never part of a model's training signature."""
from __future__ import annotations

import math
import os
import platform
from pathlib import Path

import numpy as np
import torch

from .common import code_fingerprint, digest, read_json, versions


def environment(device):
    result = {"versions": versions(), "device": str(device), "cuda": torch.version.cuda,
              "data_volume": str(Path(__file__).resolve().anchor), "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    if str(device).startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, capability=[props.major, props.minor],
                      gpu_memory=props.total_memory, gpu_uuid=str(getattr(props, "uuid", "unavailable")),
                      cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    return result


def profile_identity(config, device):
    return {"config_sha256": digest(config), "environment": environment(device), "code_sha256": code_fingerprint()}


def load_profile(path, config, device):
    profile = read_json(path)
    if profile.get("format_version") != 1 or profile.get("identity") != profile_identity(config, device):
        raise ValueError("Performance profile hardware/software/model config mismatch; benchmark again")
    for stage, entry in profile.get("stages", {}).items():
        if stage not in ("train", "validation", "features") or not isinstance(entry.get("accepted"), bool):
            raise ValueError("Invalid performance profile stage")
        for settings in (entry.get("settings", {}), entry.get("baseline_settings", {})):
            if set(settings) - {"workers", "batch_size", "use_pixel_cache", "reduced_sync"}:
                raise ValueError("Unknown performance setting")
            for name in ("workers", "batch_size"):
                if name in settings and (type(settings[name]) is not int or settings[name] < (0 if name == "workers" else 1)):
                    raise ValueError("Invalid performance batch size/workers")
            for name in ("use_pixel_cache", "reduced_sync"):
                if name in settings and type(settings[name]) is not bool:
                    raise ValueError("Invalid performance boolean")
            if stage == "train" and settings.get("batch_size", config.get("batch_size", 4)) != config.get("batch_size", 4):
                raise ValueError("A performance profile cannot change training batch size")
        if entry.get("baseline_settings", {}).get("reduced_sync"):
            raise ValueError("Baseline settings must retain original synchronization")
    return profile


def execution_settings(config, stage, profile=None, workers=None, batch_size=None):
    settings = {"workers": 2, "batch_size": config.get("batch_size", 4) if stage != "features" else 16,
                "use_pixel_cache": stage != "features", "reduced_sync": False}
    entry = (profile or {}).get("stages", {}).get(stage, {})
    settings.update(entry.get("baseline_settings", {}))
    if entry.get("accepted"):
        settings.update(entry["settings"])
    if workers is not None:
        settings["workers"] = workers
    if batch_size is not None:
        settings["batch_size"] = batch_size
    if settings["workers"] < 0 or settings["batch_size"] <= 0:
        raise ValueError("Use workers>=0 and batch_size>0")
    return settings


def cache_payback(raw_seconds_per_image, cached_seconds_per_image, remaining_images,
                  build_seconds, required_bytes, free_bytes):
    saving = (raw_seconds_per_image - cached_seconds_per_image) * remaining_images
    return {"worth_building": bool(saving > build_seconds and free_bytes - required_bytes >= 30 * 1024**3),
            "estimated_saving_seconds": saving, "estimated_build_seconds": build_seconds,
            "estimated_net_saving_seconds": saving - build_seconds,
            "disk_reserve_ok": free_bytes - required_bytes >= 30 * 1024**3}


def numerical_check(reference, candidate):
    """Compare the same ordered predictions and true optimizer updates."""
    from .metrics import metrics
    if reference["indices"] != candidate["indices"] or reference["labels"] != candidate["labels"]:
        return {"passed": False, "reason": "prediction order/labels changed"}
    a, b = np.asarray(reference["probabilities"]), np.asarray(candidate["probabilities"])
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "reason": "invalid probabilities"}
    delta = float(np.max(np.abs(a-b), initial=0))
    scores_a, scores_b = metrics(reference["labels"], a), metrics(candidate["labels"], b)
    score_delta = {k: abs(scores_a[k]-scores_b[k]) for k in ("auroc", "auprc")}
    updates_equal = reference.get("optimizer_steps", 0) == candidate.get("optimizer_steps", 0)
    weights_ok = True
    for key, value in reference.get("parameters", {}).items():
        other = candidate.get("parameters", {}).get(key)
        if other is None or not torch.allclose(value, other, rtol=1e-3, atol=1e-5):
            weights_ok = False
            break
    feature_ok = True
    if "features" in reference:
        other = candidate.get("features")
        feature_ok = (other is not None and np.shape(reference["features"]) == np.shape(other)
                      and np.allclose(reference["features"], other, rtol=1e-3, atol=1e-3))
    passed = delta <= .001 and max(score_delta.values()) <= .001 and updates_equal and weights_ok and feature_ok
    return {"passed": bool(passed), "max_probability_difference": delta, "metric_differences": score_delta,
            "optimizer_steps_equal": updates_equal, "parameters_close": weights_ok, "features_close": bool(feature_ok)}


def recommend(baselines, candidates, checks, preparation_seconds=0., expected_images=0):
    if len(baselines) != 3 or len(candidates) != 3 or len(checks) != 3 or not all(c["passed"] for c in checks):
        return {"accepted": False, "reason": "Three complete, numerically consistent paired repeats required"}
    raw_rate = float(np.median([r["images_per_second"] for r in baselines]))
    fast_rate = float(np.median([r["images_per_second"] for r in candidates]))
    # Include worker startup, transfer, output/save and all warmup batches in the net-cost decision.
    raw_per_image = float(np.median([r["total_seconds"]/r["total_images"] for r in baselines]))
    fast_per_image = float(np.median([r["total_seconds"]/r["total_images"] for r in candidates]))
    net = (raw_per_image-fast_per_image)*expected_images-preparation_seconds
    speedup = fast_rate/raw_rate
    accepted = all(math.isfinite(v) for v in (speedup, net)) and speedup >= 1.1 and net > 0
    return {"accepted": bool(accepted), "speedup": speedup, "estimated_net_saving_seconds": net,
            "baseline_median_images_per_second": raw_rate, "candidate_median_images_per_second": fast_rate,
            "reason": "Passed speed, numerical and net-cost gates" if accepted else "Below 10% throughput or positive net-cost gate"}
