from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .audit import require_audit
from .common import ARTIFACTS, ROOT, code_fingerprint, read_json, sha256, versions, write_csv, write_json
from .data import ImageDataset, PREPROCESS_VERSION, decode_image, load_rows, read_image, resize_image
from .engine import autocast, device_for, loader, save_predictions, synchronize
from .metrics import metric_key, metrics, paired_bootstrap
from .models import load_checkpoint


class Predictor:
    """Offline inference; no pretrained downloads or automatic remote code."""
    def __init__(self, artifact, device="auto"):
        self.device = device_for(device)
        artifact = Path(artifact).resolve()
        self.entries, self.weights = [], []
        if artifact.suffix == ".json":
            ensemble = read_json(artifact)
            if ensemble.get("format") != "glaucoma-ensemble-v1":
                raise ValueError("Unknown ensemble format")
            if not 1 <= len(ensemble["members"]) <= 2:
                raise ValueError("Deployment permits at most two members")
            for member in ensemble["members"]:
                path = (artifact.parent / member["path"]).resolve()
                if sha256(path) != member["sha256"]:
                    raise ValueError("Ensemble checkpoint checksum mismatch")
                self.entries.append(load_checkpoint(path, self.device))
                self.weights.append(float(member["weight"]))
            self.tta = ensemble.get("horizontal_tta", False)
        else:
            self.entries = [load_checkpoint(artifact, self.device)]
            self.weights = [1.0]
            self.tta = False
        if any(w < 0 or not math.isfinite(w) for w in self.weights) or abs(sum(self.weights)-1) > 1e-8:
            raise ValueError("Fusion weights must be nonnegative and sum to one")

    @torch.inference_mode()
    def predict_tensor(self, images, raw_ages):
        output = torch.zeros(len(images), device=self.device)
        for (model, age, _), weight in zip(self.entries, self.weights):
            x = images.to(self.device)
            if x.shape[-2:] != (model.size, model.size):
                x = F.interpolate(x, size=(model.size, model.size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            ages = torch.from_numpy(age.transform(raw_ages)).to(self.device)
            with autocast(self.device):
                p = torch.sigmoid(model(x, ages).float())
                if self.tta:
                    p = .5 * (p + torch.sigmoid(model(x.flip(-1), ages).float()))
            output += weight * p
        return output

    @torch.inference_mode()
    def predict(self, image_path, age):
        # Decode/crop once, then resize each branch exactly as during training.
        output = 0.0
        decoded = decode_image(image_path, max(item[0].size for item in self.entries))
        for (model, transform, _), weight in zip(self.entries, self.weights):
            image = torch.from_numpy(resize_image(decoded, model.size)).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
            ages = torch.from_numpy(transform.transform([age])).to(self.device)
            with autocast(self.device):
                p = torch.sigmoid(model(image, ages).double())
                if self.tta:
                    p = .5 * (p + torch.sigmoid(model(image.flip(-1), ages).double()))
            output += weight * float(p.item())
        return {"Probability": output, "Pred_Label": int(output >= .5)}


def write_submission(path, rows, probabilities):
    if len(rows) != len(probabilities):
        raise ValueError("Submission row count mismatch")
    names, records = set(), []
    for row, probability in zip(rows, probabilities):
        name, p = row["name"], float(probability)
        if name in names or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Duplicate name or invalid probability")
        names.add(name)
        records.append({"Image_name": name, "Probability": format(p, ".17g"), "Pred_Label": int(p >= .5)})
    write_csv(path, records, ["Image_name", "Probability", "Pred_Label"])


def check_submission(path, rows):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["Image_name", "Probability", "Pred_Label"]:
            raise ValueError("Incorrect submission columns")
        records = list(reader)
    if len(records) != len(rows) or [r["Image_name"] for r in records] != [r["name"] for r in rows]:
        raise ValueError("Submission names/order/count differ from test.csv")
    for r in records:
        p = float(r["Probability"])
        if not math.isfinite(p) or not 0 <= p <= 1 or int(r["Pred_Label"]) != int(p >= .5):
            raise ValueError("Invalid probability or threshold label")
    return {"rows": len(records), "valid": True}


def predict_split(root, artifact, split, output, batch_size=16, workers=2, device="auto"):
    predictor = Predictor(artifact, device)
    rows = load_rows(root, split)
    if split == "test" and len(rows) != 5855:
        raise ValueError("Competition test.csv must contain exactly 5,855 images")
    total = np.zeros(len(rows), dtype=np.float64)
    # Separate resolution-specific loaders preserve exact training preprocessing.
    for (model, age, _), weight in zip(predictor.entries, predictor.weights):
        data = loader(ImageDataset(rows, model.size, age), batch_size, workers)
        with torch.inference_mode():
            for step, (images, ages, _, indices) in enumerate(data):
                images, ages = images.to(predictor.device, non_blocking=True), ages.to(predictor.device, non_blocking=True)
                with autocast(predictor.device):
                    p = torch.sigmoid(model(images, ages).double())
                    if predictor.tta:
                        p = .5 * (p + torch.sigmoid(model(images.flip(-1), ages).double()))
                total[indices.numpy()] += weight * p.cpu().numpy()
                if (step+1) % 50 == 0:
                    print(f"predict {split} {min((step+1)*batch_size, len(rows))}/{len(rows)}", flush=True)
        del data
    if split == "test":
        write_submission(output, rows, total)
        check_submission(output, rows)
    else:
        save_predictions(output, rows, total)
    return rows, total


def evaluate_valid(root, artifact, output, milestone, batch_size=16, workers=2, device="auto", bootstrap=1000):
    report = require_audit(root)
    ledger_path = ARTIFACTS / "valid_milestones.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {}
    signature = sha256(artifact)
    if milestone in ledger and ledger[milestone]["artifact_sha256"] != signature:
        raise ValueError("This official-valid milestone was already used for another artifact; use train folds for tuning")
    # Reserve before reading valid labels; failed/repeated attempts cannot silently reset usage.
    ledger[milestone] = {"artifact_sha256": signature, "status": "running"}
    write_json(ledger_path, ledger)
    rows, p = predict_split(root, artifact, "valid", Path(output) / "valid.csv", batch_size, workers, device)
    y = np.array([r["label"] for r in rows])
    result = {"full": metrics(y, p), "bootstrap": paired_bootstrap(y, p, repeats=bootstrap)}
    clean = np.array([r["name"] not in set(report["valid_train_similarity_overlap"]) for r in rows])
    if clean.any() and len(np.unique(y[clean])) == 2:
        result["without_train_similarity_overlap"] = metrics(y[clean], p[clean])
    journal = {}
    for line in (ARTIFACTS / "audit" / "images.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = __import__("json").loads(line)
            if r["split"] == "valid" and r["ok"]:
                journal[r["name"]] = r
        except ValueError:
            pass
    masks = {"age_missing": [not math.isfinite(r["age"]) for r in rows],
             "age_at_least_70": [math.isfinite(r["age"]) and r["age"] >= 70 for r in rows],
             "low_resolution": [min(journal[r["name"]]["width"], journal[r["name"]]["height"]) < 1000 for r in rows],
             "dark": [journal[r["name"]]["mean_brightness"] < 40 for r in rows]}
    result["subgroups"] = {}
    for key, mask in masks.items():
        mask = np.array(mask)
        result["subgroups"][key] = metrics(y[mask], p[mask]) if mask.any() and len(np.unique(y[mask])) == 2 else {"n": int(mask.sum()), "metrics": None}
    write_json(Path(output) / "valid_report.json", result)
    ledger[milestone]["status"] = "complete"
    ledger[milestone]["metrics"] = result["full"]
    write_json(ledger_path, ledger)
    print(result["full"], flush=True)
    return result


def collect_oof(root, paths, output):
    rows = load_rows(root, "train")
    from .engine import folds_for
    folds = folds_for(rows)
    mapping, seen_folds, reference_config = {}, set(), None
    for path in paths:
        if not read_json(Path(path).parent / "status.json").get("complete"):
            raise ValueError("Finish or resume every fold before collecting OOF")
        checkpoint = Path(path).parent / "model.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        meta = payload["metadata"]
        if reference_config is None:
            reference_config = payload["config"]
        if (payload["config"] != reference_config or meta["fold_fingerprint"] != sha256(ARTIFACTS / "audit" / "folds.csv")
            or meta["data_fingerprint"] != require_audit(root)["data_fingerprint"]):
            raise ValueError("OOF checkpoints have different configs, folds or data provenance")
        fold = meta["fold"]
        if fold not in (0, 1, 2) or fold in seen_folds:
            raise ValueError("OOF inputs must be three distinct held-out folds")
        seen_folds.add(fold)
        with Path(path).open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        expected = {r["name"] for i, r in enumerate(rows) if folds[i] == fold}
        if {r["Image_name"] for r in records} != expected:
            raise ValueError("Holdout predictions are incomplete or contain training-fold rows")
        for record in records:
            if record["Image_name"] in mapping or int(record["fold"]) != fold:
                raise ValueError("OOF duplication or mislabeled fold")
            mapping[record["Image_name"]] = float(record["Probability"])
    if seen_folds != {0, 1, 2}:
        raise ValueError("Missing OOF fold")
    p = np.array([mapping[r["name"]] for r in rows])
    save_predictions(output, rows, p, folds)
    write_json(Path(output).with_suffix(".json"), {"metrics": metrics([r["label"] for r in rows], p),
               "config": reference_config, "data_fingerprint": require_audit(root)["data_fingerprint"],
               "fold_fingerprint": sha256(ARTIFACTS / "audit" / "folds.csv"), "prediction_sha256": sha256(output)})


def fuse(root, first_oof, second_oof, first_model, second_model, output):
    from .engine import folds_for
    rows = load_rows(root, "train")
    folds = folds_for(rows)
    def read_oof(path):
        meta = read_json(Path(path).with_suffix(".json"))
        if (meta["prediction_sha256"] != sha256(path)
            or meta["fold_fingerprint"] != sha256(ARTIFACTS / "audit" / "folds.csv")
            or meta["data_fingerprint"] != require_audit(root)["data_fingerprint"]):
            raise ValueError("OOF predictions/folds/data changed since generation")
        with Path(path).open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        if ([r["Image_name"] for r in records] != [r["name"] for r in rows]
            or [int(r["fold"]) for r in records] != folds.tolist()
            or [int(r["True_Label"]) for r in records] != [r["label"] for r in rows]):
            raise ValueError("Fusion requires aligned TRAIN-only OOF predictions")
        return np.array([float(r["Probability"]) for r in records]), meta
    (first, first_meta), (second, second_meta) = read_oof(first_oof), read_oof(second_oof)
    y = np.array([r["label"] for r in rows])
    candidates = [{"first_weight": w, **metrics(y, w*first+(1-w)*second)} for w in (0, .25, .5, .75, 1)]
    best = max(candidates, key=metric_key)
    w = best["first_weight"]
    output = Path(output).resolve()
    members = []
    for model, weight, oof_meta in ((first_model, w, first_meta), (second_model, 1-w, second_meta)):
        if weight == 0:
            continue
        path = Path(model).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"]["fold"] != "full" or checkpoint["config"] != oof_meta["config"]:
            raise ValueError("Fusion deployment checkpoints must be full-train refits")
        members.append({"path": __import__("os").path.relpath(path, output.parent), "sha256": sha256(path), "weight": weight})
    write_json(output, {"format": "glaucoma-ensemble-v1", "members": members, "horizontal_tta": False,
                        "selection": "train-only 3-fold OOF; grid-selection metrics are optimistic", "grid": candidates,
                        "paired_delta_vs_first": paired_bootstrap(y, w*first+(1-w)*second, first),
                        "t4_latency_verified": False})


def benchmark(root, artifact, output, device="auto", count=200, warmup=20):
    predictor = Predictor(artifact, device)
    rows = load_rows(root, "test")
    # Include byte-size extremes and a deterministic sample, rather than only CSV prefix.
    rng = np.random.default_rng(42)
    ix = set(rng.choice(len(rows), min(count, len(rows)), replace=False).tolist())
    ix.update(sorted(range(len(rows)), key=lambda i: rows[i]["file_size"])[:5])
    ix.update(sorted(range(len(rows)), key=lambda i: rows[i]["file_size"])[-5:])
    for _ in range(warmup):
        predictor.predict(rows[0]["path"], rows[0]["age"])
    timing = []
    for i in sorted(ix):
        synchronize(predictor.device)
        start = time.perf_counter()
        predictor.predict(rows[i]["path"], rows[i]["age"])
        synchronize(predictor.device)
        timing.append((time.perf_counter()-start)*1000)
    gpu = torch.cuda.get_device_name() if predictor.device.startswith("cuda") else None
    t4 = gpu is not None and "T4" in gpu.upper()
    result = {"gpu": gpu, "device": predictor.device, "n": len(timing), "batch_size": 1,
              "scope": "warm model; raw file read, JPEG decode, crop, resize, transfers, forward and output",
              "p50_ms": float(np.percentile(timing, 50)), "p95_ms": float(np.percentile(timing, 95)),
              "p99_ms": float(np.percentile(timing, 99)), "max_ms": max(timing),
              "official_t4": t4, "full_test_coverage": len(ix) == len(rows),
              "t4_latency_verified": t4 and len(ix) == len(rows) and max(timing) <= 100,
              "artifact_sha256": sha256(artifact), "code_fingerprint": code_fingerprint(),
              "test_csv_sha256": sha256(Path(root) / "test.csv"), "versions": versions()}
    write_json(output, result)
    print(result, flush=True)
    return result


def export_bundle(artifact, destination, benchmark_report=None, allow_unverified=False):
    import os
    artifact, destination = Path(artifact).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Export destination must be a new directory")
    if not allow_unverified:
        if benchmark_report is None:
            raise ValueError("Final export requires a passing T4 report; use --allow-unverified only for a clearly marked candidate")
        report = read_json(benchmark_report)
        if (not report["t4_latency_verified"] or report["artifact_sha256"] != sha256(artifact)
            or report["code_fingerprint"] != code_fingerprint()):
            raise ValueError("Missing matching T4 timing evidence")
    destination.mkdir(parents=True)
    if artifact.suffix == ".json":
        bundle = read_json(artifact)
        for i, member in enumerate(bundle["members"]):
            source = (artifact.parent / member["path"]).resolve()
            if sha256(source) != member["sha256"]:
                raise ValueError("Checkpoint changed since fusion")
            target = destination / f"member_{i}.pt"
            shutil.copy2(source, target)
            member["path"] = target.name
        write_json(destination / "ensemble.json", bundle)
        entry = "ensemble.json"
    else:
        shutil.copy2(artifact, destination / "model.pt")
        entry = "model.pt"
    shutil.copytree(ROOT / "glaucoma", destination / "glaucoma", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "run.py", destination / "run.py")
    shutil.copy2(ROOT / "requirements-inference.txt", destination / "requirements.txt")
    shutil.copy2(ROOT / "requirements-inference.txt", destination / "requirements-inference.txt")
    shutil.copy2(ROOT / "SOURCES.md", destination / "SOURCES.md")
    if (ROOT / "requirements.lock.txt").exists():
        shutil.copy2(ROOT / "requirements.lock.txt", destination / "requirements.lock.txt")
    if benchmark_report:
        shutil.copy2(benchmark_report, destination / "timing_evidence.json")
    license_dir = ROOT / "weights"
    for path in license_dir.glob("*LICENSE*"):
        shutil.copy2(path, destination / path.name)
    for path in license_dir.glob("*_source.json"):
        shutil.copy2(path, destination / path.name)
    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if total > 500_000_000:
        raise ValueError(f"Uncompressed bundle is oversized: {total}")
    write_json(destination / "bundle.json", {"entry": entry, "bytes_before_manifest": total,
               "source_artifact_sha256": sha256(artifact), "code_fingerprint": code_fingerprint(),
               "t4_latency_verified": not allow_unverified, "candidate_only": allow_unverified,
               "command": f"python run.py --data-root /path/to/glaucoma_first predict --model {entry} --output submission.csv"})
    # Check final size including the manifest.
    if sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) > 500_000_000:
        raise ValueError("Manifest pushes bundle above 500,000,000 bytes")
