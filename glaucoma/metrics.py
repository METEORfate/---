from __future__ import annotations

import numpy as np


def metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Predictions must be finite probabilities aligned with labels")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes are required to calculate AUROC")
    fpr, tpr, _ = roc_curve(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "sensitivity_at_95_specificity": float(tpr[fpr <= 0.05].max()),
            "n": len(y), "positive": int(y.sum())}


def metric_key(result):
    return result["auroc"], result["auprc"]


def paired_bootstrap(y, p, q=None, repeats=1000, seed=42):
    """Image-level stratified paired intervals, not a claim of patient independence."""
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    q = np.asarray(q) if q is not None else None
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes required")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        ix = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        value = roc_auc_score(y[ix], p[ix])
        if q is not None:
            value -= roc_auc_score(y[ix], q[ix])
        values.append(value)
    result = {"lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975)),
              "resampling_unit": "image; unknown patient clustering is not controlled", "repeats": repeats}
    if q is not None:
        result["fraction_delta_positive"] = float(np.mean(np.array(values) > 0))
    return result

