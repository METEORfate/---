"""Standard-library-only job validation, also used before any Kaggle launch."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re

TASKS = {"train", "locator", "ssl", "finalize", "collect", "hard-negatives", "fuse", "evaluate", "predict", "capacity", "holdout", "benchmark"}
TRAINING = {"train", "locator", "ssl", "finalize", "capacity", "benchmark"}


def check_annotation_threshold(report, policy="full1000"):
    if policy not in ("full1000", "pilot400"):
        raise ValueError("Unknown annotation policy")
    if policy == "full1000":
        if report.get("valid", 0) < 1000 or report.get("minimum_valid", 0) < 1000:
            raise ValueError("Formal plan3 jobs are blocked: require 1000 accepted valid boxes; use an explicitly approved pilot policy for partial annotations")
    elif (report.get("acceptance_scope") != "pilot400" or report.get("annotated", 0) < 400 or
          report.get("valid", 0) < 300 or report.get("minimum_valid", 0) < 300):
        raise ValueError("pilot400 needs an explicit pilot manifest, at least 400 annotated images and 300 clear valid boxes")


def relative(value, prefix=None):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." == value:
        raise ValueError("Path traversal is forbidden")
    if prefix and (not path.parts or path.parts[0] != prefix or len(path.parts) < 2):
        raise ValueError(f"Path must be inside {prefix}/")
    return value


def validate(job):
    if job.get("mode") != "plan3" or job.get("task") not in TASKS:
        raise ValueError("Expected a supported plan3 task")
    relative(job.get("output"), "artifacts")
    relative(job.get("annotations"))
    if job.get("annotation_policy", "full1000") not in ("full1000", "pilot400"):
        raise ValueError("Unknown annotation policy")
    if type(job.get("seed", 42)) is not int:
        raise ValueError("seed must be an integer")
    if type(job.get("preflight_capacity", False)) is not bool:
        raise ValueError("preflight_capacity must be a boolean")
    if type(job.get("workers", 0)) is not int or not 0 <= job.get("workers", 0) <= 8:
        raise ValueError("workers must be 0..8")
    if type(job.get("audit_workers", 8)) is not int or not 1 <= job.get("audit_workers", 8) <= 32:
        raise ValueError("audit_workers must be 1..32")
    if job.get('task') == 'benchmark' and (job.get('fold') != 0 or job.get('benchmark_repeats', 3) not in (2, 3)):
        raise ValueError('Benchmark uses fixed fold 0 and two or three paired repetitions')
    hours = job.get("max_hours")
    if type(hours) not in (int, float) or not math.isfinite(hours) or not 0 < hours <= 10:
        raise ValueError("max_hours must be finite in (0,10]")
    if job["task"] in TRAINING:
        relative(job.get("config"), "configs")
        if not job["config"].startswith("configs/plan3/"):
            raise ValueError("Plan3 jobs use configs/plan3/; legacy configurations keep their original entry point")
        fold = job.get("fold")
        if type(fold) is not int and fold != "full" or type(fold) is int and fold not in (0, 1, 2):
            raise ValueError("fold must be 0/1/2/full")
        if job["task"] == "finalize" and fold != "full":
            raise ValueError("finalize needs full train")
        if job["task"] == "train" and fold == "full":
            raise ValueError("Use finalize with verified CV sources for full-train classification")
        if job.get("inner_fold") is not None and (type(job["inner_fold"]) is not int or job["inner_fold"] not in (0, 1, 2) or fold == "full"):
            raise ValueError("inner_fold requires an outer fold")
    aliases, refs = set(), set()
    for item in job.get("dependencies", []):
        alias, ref = item.get("alias", ""), item.get("kernel_ref", "")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", alias) or alias in aliases:
            raise ValueError("Dependency aliases must be unique")
        if not re.fullmatch(r"[\w-]+/[\w-]+/[1-9]\d*", ref):
            raise ValueError("Every dependency must pin owner/slug/version")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("results_sha256", "")):
            raise ValueError("Dependency needs a verified results.zip SHA256")
        if not item.get("files"):
            raise ValueError("Dependency must declare required file SHA256s")
        for path, expected in item["files"].items():
            relative(path, "artifacts")
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid dependency file checksum")
        aliases.add(alias)
        refs.add(ref)
    if job.get("resume"):
        relative(job.get("resume_from"))
        if not str(job["resume_from"]).startswith("deps/"):
            raise ValueError("Cloud resume must select an isolated versioned dependency output")
    for key in ("initialize_from", "locator", "locator_review", "hard_negatives", "artifact", "ticket", "spec", "inner_oof"):
        if job.get(key):
            relative(job[key])
    for key in ("inputs", "cv_models"):
        for value in job.get(key, []):
            relative(value)
    if job["task"] == "evaluate" and not job.get("ticket"):
        raise ValueError("Official-valid requires a centrally reserved evaluation ticket")
    required = {"ssl": ("locator", "locator_review"), "collect": ("inputs",), "hard-negatives": ("inner_oof", "fold"),
                "fuse": ("spec",), "evaluate": ("artifact", "ticket"), "predict": ("artifact",),
                "holdout": ("artifact",), "finalize": ("cv_models",)}
    for key in required.get(job["task"], ()):
        if key not in job or job[key] is None or job[key] == []:
            raise ValueError(f"Task {job['task']} requires {key}")
    return job


def validate_config(config):
    if config.get('preprocess_version', 'original-augment-fov-v1') not in ('original-augment-fov-v1', 'original-fov-pad-then-augment-v2'):
        raise ValueError('Unknown preprocessing recipe')
    if config.get('preprocess_version') == 'original-fov-pad-then-augment-v2' and (config.get('view', 'global') != 'global' or config.get('ssl')):
        raise ValueError('Resized augmentation is an independent global recipe only')
    if config.get('pixel_cache', False) and config.get('preprocess_version') != 'original-fov-pad-then-augment-v2':
        raise ValueError('A resized pixel cache cannot be used with the original augmentation recipe')
    if config.get("model") not in ("rfm", "effnet", "green", "dinov3", "retfound_dinov2", "convnext"):
        raise ValueError("Unsupported plan3 backbone")
    for key in ("size", "batch_size", "accumulation", "epochs", "patience"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["batch_size"]*config["accumulation"] != 32:
        raise ValueError("Plan3 comparisons use effective batch 32")
    if config.get("ranking_weight", 0) not in (0, .1, .3):
        raise ValueError("Ranking sweep is 0/.1/.3")
    if config.get("paired_sampling") and config["batch_size"] < 4:
        raise ValueError("Explicit paired sampling needs microbatch >=4")
    if not 0 <= config.get("ema_decay", .999) < 1:
        raise ValueError("EMA decay must be in [0,1)")
    for scale in config.get("roi_scales", [3, 5]):
        if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("ROI scales must be positive finite values")
    if config.get("local_sizes") and (len(config["local_sizes"]) != len(config.get("roi_scales", [3, 5])) or
            any(type(s) is not int or s <= 0 for s in config["local_sizes"])):
        raise ValueError("local_sizes must specify one positive size per ROI scale")
    relative(config.get("pretrained"), "weights")
    return config
