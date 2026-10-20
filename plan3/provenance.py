from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS, ROOT, digest, read_json, sha256, versions
from . import INPUT_VERSION
from glaucoma.data import load_rows
from .data import partition, require_annotations
from .jobs import check_annotation_threshold


def code_hash():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def safe_path(value, base=ROOT):
    base = Path(base).resolve()
    path = (base / value).resolve()
    if not isinstance(value, str) or "\\" in value or Path(value).is_absolute() or not path.is_relative_to(base):
        raise ValueError(f"Unsafe relative project path: {value}")
    return path


class Context:
    def __init__(self, root):
        self.root = Path(root)
        self.audit = require_audit(root)
        self.rows = load_rows(root, "train")
        with (ARTIFACTS / "audit/folds.csv").open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        mapping = {r["Image_name"]: r for r in records}
        if len(mapping) != len(records) or set(mapping) != {r["name"] for r in self.rows}:
            raise ValueError("Fixed folds do not match train")
        self.folds = np.array([int(mapping[r["name"]]["fold"]) for r in self.rows])
        self.groups = [mapping[r["name"]]["group"] for r in self.rows]
        if set(self.folds) != {0, 1, 2} or sha256(ARTIFACTS / "audit/folds.csv") != self.audit["fold_fingerprint"]:
            raise ValueError("Fixed fold fingerprint changed")

    def split(self, fold, inner_fold=None):
        return partition(self.rows, self.folds, self.groups, fold, inner_fold)

    def annotations(self, folder, policy="full1000"):
        report, boxes = require_annotations(folder, self.audit)
        check_annotation_threshold(report, policy)
        by_name = {r["name"]: r for r in self.rows}
        for name, expected in report["image_sha256"].items():
            if name not in by_name or sha256(by_name[name]["path"]) != expected:
                raise ValueError("Annotated image bytes changed")
        return report, boxes

    def metadata(self, fit, hold, fold, inner_fold):
        return {"data_fingerprint": self.audit["data_fingerprint"], "fold_fingerprint": self.audit["fold_fingerprint"],
                "fold": fold, "inner_fold": inner_fold, "train_names": [self.rows[i]["name"] for i in fit],
                "holdout_names": [self.rows[i]["name"] for i in hold]}


def check_support(metadata, context, fit):
    names = {context.rows[i]["name"] for i in fit}
    if (metadata.get("data_fingerprint") != context.audit["data_fingerprint"] or
            metadata.get("fold_fingerprint") != context.audit["fold_fingerprint"] or
            not metadata.get("train_names") or not (set(metadata["train_names"]) | set(metadata.get("review_names", []))).issubset(names)):
        raise ValueError("Initialization/locator includes forbidden holdout support or different provenance")


def recipe_provenance(payload):
    meta = payload["metadata"]
    return {"annotations": meta.get("annotation_sha256"), "annotation_version": meta.get("annotation_version"),
            "pretrained": meta.get("dependencies", {}).get("pretrained"),
            "initialization_recipe": meta.get("dependencies", {}).get("initialization_recipe"),
            "code": payload["signature"].get("code_sha256"), "input": payload["signature"].get("input_version")}


def pretrained(config):
    manifest_path = safe_path(config["pretrained"])
    spec = read_json(manifest_path)
    if not all(spec.get(k) for k in ("sha256", "source", "license")):
        raise ValueError("Pretrained manifest needs SHA256, source and license")
    defaults = {"rfm": "rfm-1.3.pt", "effnet": "efficientnetv2_s_in1k.pth", "green": "retfoundgreen_statedict.pth"}
    filename = spec.get("filename", defaults.get(config["model"], ""))
    if not filename or Path(filename).name != filename:
        raise ValueError("Pretrained filename must be a basename")
    path = manifest_path.parent / filename
    if sha256(path) != spec["sha256"]:
        raise ValueError("Pretrained weight hash mismatch")
    if config["model"] not in defaults:
        if len(spec.get("mean", [])) != 3 or len(spec.get("std", [])) != 3 or not all(float(v) > 0 for v in spec["std"]):
            raise ValueError("Challenge model requires explicit matched mean/std")
    return path, spec


def signature(config, context, annotations, fit, hold, job, dependencies):
    return {"config": config, **context.metadata(fit, hold, job["fold"], job.get("inner_fold")),
            "seed": job.get("seed", 42), "annotations_sha256": annotations["boxes_sha256"],
            "annotation_policy": job.get("annotation_policy", "full1000"),
            "annotation_version": annotations["version"], "task": job["task"],
            "dependencies": dependencies, "code_sha256": code_hash(), "input_version": INPUT_VERSION,
            "runtime": {k: v for k, v in versions().items() if k != "platform"}}
