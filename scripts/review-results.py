"""Recheck saved train-only predictions against labels, folds and checkpoint provenance."""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS, read_json, sha256, write_json
from glaucoma.data import AgeTransform, load_rows
from glaucoma.engine import folds_for
from glaucoma.metrics import metric_key, metrics


def predictions(path, rows, folds):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    actual = [(r["Image_name"], int(r["True_Label"]), int(r["fold"])) for r in records]
    expected = [(r["name"], r["label"], int(f)) for r, f in zip(rows, folds)]
    if actual != expected:
        raise ValueError(f"Prediction names/order/labels/folds differ: {path}")
    p = np.array([float(r["Probability"]) for r in records])
    y = np.array([r["label"] for r in rows])
    result = metrics(y, p)
    # Independent Mann-Whitney rank calculation, with average ranks for ties.
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    positive = int(y.sum())
    rank_auc = (average_ranks[inverse][y == 1].sum() - positive * (positive + 1) / 2) / (positive * (len(y) - positive))
    if abs(rank_auc - result["auroc"]) > 1e-12:
        raise ValueError("Independent rank AUROC differs from reported implementation")
    return p, result


def same_metrics(actual, saved):
    if any(not np.isfinite(saved[key]) or abs(actual[key] - saved[key]) > 1e-12 for key in actual):
        raise ValueError("Saved metrics do not match actual predictions")


def review():
    audit = require_audit(ROOT / "glaucoma_first")
    rows = load_rows(ROOT / "glaucoma_first", "train")
    folds = folds_for(rows)
    fingerprint = sha256(ARTIFACTS / "audit" / "folds.csv")
    report = {"checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "data_fingerprint": audit["data_fingerprint"], "fold_fingerprint": fingerprint,
              "scope": "TRAIN-only development estimates; epoch/C selection is not nested; patient identity unavailable",
              "models": {}, "input_sha256": {}}
    values = {}
    for name in ("green_probe_age", "green_probe_image", "rfm_probe_age", "rfm_probe_image"):
        path = ARTIFACTS / name / "oof.csv"
        meta = read_json(path.with_suffix(".json"))
        if (meta["prediction_sha256"] != sha256(path) or meta["data_fingerprint"] != audit["data_fingerprint"]
                or meta["fold_fingerprint"] != fingerprint):
            raise ValueError(f"OOF provenance mismatch: {name}")
        p, result = predictions(path, rows, folds)
        same_metrics(result, read_json(path.parent / "report.json")["selected"])
        values[name] = p
        report["models"][name] = {"pooled": result, "by_fold": {
            str(f): metrics([r["label"] for r, k in zip(rows, folds) if k == f], p[folds == f]) for f in range(3)}}
        for item in (path, path.with_suffix(".json"), path.parent / "report.json"):
            report["input_sha256"][str(item.relative_to(ROOT))] = sha256(item)
    folder = ARTIFACTS / "green_ft2" / "fold0"
    if not read_json(folder / "status.json")["complete"]:
        raise ValueError("Green fold 0 is incomplete")
    payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    history = read_json(folder / "history.json")
    best = max(history, key=lambda r: metric_key(r["holdout"]))
    if (payload["config"] != read_json(ROOT / "configs" / "green_ft2.json") or meta["fold"] != 0
            or meta["seed"] != 42 or meta["epoch"] != best["epoch"]
            or meta["data_fingerprint"] != audit["data_fingerprint"] or meta["fold_fingerprint"] != fingerprint
            or payload["age_transform"] != AgeTransform.fit([r["age"] for r, f in zip(rows, folds) if f != 0]).to_dict()):
        raise ValueError("Green checkpoint config/epoch/provenance/training-only age transform mismatch")
    hold_rows = [r for r, f in zip(rows, folds) if f == 0]
    _, tuned = predictions(folder / "holdout.csv", hold_rows, folds[folds == 0])
    same_metrics(tuned, meta["holdout"])
    same_metrics(tuned, best["holdout"])
    report["models"]["green_ft2_fold0"] = {"metrics": tuned, "best_epoch": best["epoch"], "completed_epochs": len(history)}
    for name in ("model.pt", "history.json", "holdout.csv", "status.json"):
        path = folder / name
        report["input_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    frozen = report["models"]["green_probe_age"]["by_fold"]["0"]
    rfm = report["models"]["rfm_probe_age"]["by_fold"]["0"]
    report["same_fold_deltas"] = {"green_ft2_minus_green_frozen": tuned["auroc"] - frozen["auroc"],
                                  "green_ft2_minus_rfm_frozen": tuned["auroc"] - rfm["auroc"]}
    for filename, first, second in (
        ("green_ft2_vs_frozen_fold0.json", frozen, tuned),
        ("rfm_age_vs_green_oof.json", report["models"]["green_probe_age"]["pooled"], report["models"]["rfm_probe_age"]["pooled"]),
    ):
        path = ARTIFACTS / "comparisons" / filename
        saved = read_json(path)
        same_metrics(first, saved["first"])
        same_metrics(second, saved["second"])
        report["input_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    report["bootstrap_note"] = "Existing intervals linked to matching recomputed point estimates; intervals not rerun"
    report["rfm_budget_ledger"] = read_json(ARTIFACTS / "rfm_budget.json")
    report["decision"] = "Complete unchanged Green folds 1/2 before comparing pooled OOF; request CNN artifacts; no new valid evaluation"
    output = ARTIFACTS / "verification" / "results_review_20261018.json"
    write_json(output, report)
    print(output, flush=True)
    print(report["same_fold_deltas"], flush=True)
    print("All prediction/metric/fold/checkpoint checks passed", flush=True)
    return report


if __name__ == "__main__":
    review()
