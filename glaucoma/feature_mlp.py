"""Bounded CPU ablation on cached features, using the existing inference MLP head."""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .audit import require_audit
from .common import ARTIFACTS, ROOT, exclusive_output, read_json, sha256, versions, write_json
from .data import AgeTransform, load_rows
from .engine import folds_for, save_predictions
from .metrics import metric_key, metrics
from .models import FundusModel, download_weights, save_checkpoint


def make_head(dimension):
    return nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 1))


def fit_head(x_fit, y_fit, x_hold, y_hold, config, seed, deadline, epochs=None):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(x_fit)
    fit = torch.from_numpy(scaler.transform(x_fit).astype(np.float32))
    hold = torch.from_numpy(scaler.transform(x_hold).astype(np.float32)) if x_hold is not None else None
    y = torch.tensor(y_fit, dtype=torch.float32)
    head = make_head(fit.shape[1])
    with torch.no_grad():
        head[-1].bias.fill_(math.log(float(y.sum()) / float(len(y)-y.sum())))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    limit = int(epochs or config["epochs"])
    best, stale, history, chosen = (-1., -1.), 0, [], None
    for epoch in range(1, limit+1):
        head.train()
        order = torch.randperm(len(fit))
        loss_sum = 0.
        for indices in order.split(config["batch_size"]):
            if time.time() >= deadline:
                raise TimeoutError("CPU head trial time limit reached; completed folds remain resumable")
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(head(fit[indices]).squeeze(1), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MLP head loss")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        head.eval()
        result, predictions = None, None
        if hold is not None:
            with torch.inference_mode():
                predictions = torch.cat([torch.sigmoid(head(b).squeeze(1).double()) for b in hold.split(4096)]).numpy()
            result = metrics(y_hold, predictions)
        improved = result is None or metric_key(result) > best
        history.append({"epoch": epoch, "loss": loss_sum/len(fit), "holdout": result})
        print(f"CPU MLP seed={seed} epoch={epoch}/{limit} loss={loss_sum/len(fit):.6f} holdout={result}", flush=True)
        if improved:
            chosen = {"head": copy.deepcopy(head.state_dict()), "epoch": epoch,
                      "predictions": predictions, "metrics": result}
            if result is not None:
                best = metric_key(result)
            stale = 0
        else:
            stale += 1
        if hold is not None and stale >= config["patience"]:
            break
    chosen.update(mean=torch.tensor(scaler.mean_, dtype=torch.float32),
                  scale=torch.tensor(scaler.scale_, dtype=torch.float32), history=history)
    return chosen


def run_trial(root, features_path, config, output, seed=42, minutes=20, threads=2):
    torch.set_num_threads(threads)
    if (config.get("model") != "green" or config.get("head") != "mlp"
        or not config.get("head_standardize") or config.get("tail_blocks") != 0):
        raise ValueError("This ablation requires a standardized MLP and a frozen backbone")
    output, features_path = Path(output), Path(features_path)
    with exclusive_output(output / "run.lock"):
        audit = require_audit(root)
        rows = load_rows(root, "train")
        names = [r["name"] for r in rows]
        meta = read_json(features_path.with_suffix(".json"))
        if (not meta["complete"] or meta["names"] != names or meta["signature"]["model"] != config["model"]
            or meta["signature"]["size"] != config["size"] or meta["signature"]["data"] != audit["data_fingerprint"]):
            raise ValueError("Feature cache provenance differs from this data/config")
        signature = {"config": config, "seed": seed, "data_fingerprint": audit["data_fingerprint"],
                     "fold_fingerprint": sha256(ARTIFACTS / "audit" / "folds.csv"),
                     "features_sha256": sha256(features_path)}
        signature_path = output / "signature.json"
        if signature_path.exists() and read_json(signature_path) != signature:
            raise ValueError("Choose a new output: trial configuration or provenance changed")
        write_json(signature_path, signature)
        if (output / "report.json").exists():
            return read_json(output / "report.json")
        deadline = time.time() + minutes * 60
        x_image = np.load(features_path, mmap_mode="r", allow_pickle=False)
        y = np.array([r["label"] for r in rows])
        ages = np.array([r["age"] for r in rows])
        folds = folds_for(rows)
        oof = np.full(len(rows), np.nan)
        fold_reports = []
        try:
            for fold in range(3):
                fit, hold = folds != fold, folds == fold
                age = AgeTransform.fit(ages[fit])
                x = np.concatenate([x_image, age.transform(ages)], 1) if config.get("use_age", True) else x_image
                path = output / f"fold{fold}_head.pt"
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["signature"] != signature or saved["fold"] != fold:
                        raise ValueError("Saved fold provenance changed")
                    result = saved["result"]
                    result["predictions"] = result["predictions"].numpy()
                else:
                    write_json(output / "status.json", {"complete": False, "phase": "fold_fit", "fold": fold})
                    result = fit_head(x[fit], y[fit], x[hold], y[hold], config, seed+fold, deadline)
                    serial = {**result, "predictions": torch.tensor(result["predictions"])}
                    temp = path.with_suffix(".tmp")
                    torch.save({"signature": signature, "fold": fold, "age_transform": age.to_dict(), "result": serial}, temp)
                    temp.replace(path)
                oof[hold] = result["predictions"]
                fold_reports.append({"fold": fold, "fit_rows": int(fit.sum()), "heldout_rows": int(hold.sum()),
                                     "epoch": result["epoch"], "metrics": result["metrics"], "age_transform": age.to_dict()})
                write_json(output / "fold_reports.json", fold_reports)
            selected = metrics(y, oof)
            save_predictions(output / "oof.csv", rows, oof, folds)
            write_json(output / "oof.json", {"config": config, "data_fingerprint": signature["data_fingerprint"],
                       "fold_fingerprint": signature["fold_fingerprint"], "prediction_sha256": sha256(output / "oof.csv")})
            epochs = max(1, int(np.median([r["epoch"] for r in fold_reports])))
            age = AgeTransform.fit(ages)
            x = np.concatenate([x_image, age.transform(ages)], 1) if config.get("use_age", True) else x_image
            write_json(output / "status.json", {"complete": False, "phase": "full_refit", "epochs": epochs})
            refit = fit_head(x, y, None, None, config, seed, deadline, epochs)
            model = FundusModel(config, download_weights(config["model"]))
            model.head.load_state_dict(refit["head"])
            model.head_mean.copy_(refit["mean"])
            model.head_scale.copy_(refit["scale"])
            metadata = {"mode": "frozen_mlp_full_refit", "fold": "full", "seed": seed, "epochs": epochs,
                        "data_fingerprint": signature["data_fingerprint"], "fold_fingerprint": signature["fold_fingerprint"],
                        "selection": "Heldout epochs selected within train three-fold CV; OOF scores are development estimates",
                        "versions": versions()}
            save_checkpoint(output / "model.pt", model, config, age, metadata)
            report = {"selected": selected, "folds": fold_reports, "full_refit_epochs": epochs, "metadata": metadata}
            write_json(output / "report.json", report)
            write_json(output / "status.json", {"complete": True})
            print(f"CPU MLP complete: {selected}", flush=True)
            return report
        except BaseException as exc:
            write_json(output / "status.json", {"complete": False, "phase": "stopped", "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "green_mlp.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.minutes <= 0 or args.threads < 1:
        parser.error("Use positive minutes and threads")
    run_trial(args.data_root, args.features, read_json(args.config), args.output, args.seed, args.minutes, args.threads)
