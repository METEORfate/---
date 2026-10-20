from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
from scipy.special import expit, logit

from glaucoma.common import digest, read_json, sha256, write_csv, write_json
from glaucoma.data import AgeTransform, load_rows
from glaucoma.deploy import check_submission, write_submission
from glaucoma.metrics import metrics, paired_bootstrap
from .data import Views
from .provenance import Context, check_support, safe_path, recipe_provenance


def read_predictions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summary(rows, probabilities):
    y = np.array([r["label"] for r in rows])
    result = {"full": metrics(y, probabilities), "subgroups": {}}
    masks = {"age_missing": [not np.isfinite(r["age"]) for r in rows],
             "age_at_least_70": [np.isfinite(r["age"]) and r["age"] >= 70 for r in rows]}
    from PIL import Image
    sizes, brightness = [], []
    for row in rows:
        with Image.open(row["path"]) as im:
            sizes.append(min(im.size) < 1000)
            im.thumbnail((32, 32))
            brightness.append(float(np.asarray(im.convert("RGB")).mean()) < 40)
    masks.update(low_resolution=sizes, dark=brightness)
    for name, values in masks.items():
        mask = np.array(values, dtype=bool)
        result["subgroups"][name] = metrics(y[mask], probabilities[mask]) if len(np.unique(y[mask])) == 2 else {"n": int(mask.sum())}
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    result["misordered_pairs_tie_half"] = (1-result["full"]["auroc"])*n_pos*n_neg
    result["uncertainty"] = paired_bootstrap(y, probabilities)
    return result


def collect(context, paths, output, outer_fold=None):
    from .train import load_payload
    from glaucoma.engine import save_predictions
    mapping, seen, reference, seed, sources, common = {}, set(), None, None, [], None
    if len(paths) != 3:
        raise ValueError("OOF collection requires exactly three models")
    for value in paths:
        path = Path(value)
        model = load_payload(path / "model.pt")
        meta, cfg = model["metadata"], model["config"]
        pred_meta = read_json(path / "holdout.json")
        if not read_json(path / "status.json").get("complete"):
            raise ValueError("Finish or resume every fold before collecting OOF")
        if (pred_meta["prediction_sha256"] != sha256(path / "holdout.csv") or
                pred_meta["model_sha256"] != sha256(path / "model.pt")):
            raise ValueError("Holdout predictions/model changed")
        fold = meta["fold"] if outer_fold is None else meta.get("inner_fold")
        if fold not in (0, 1, 2) or fold in seen or model["kind"] != "classifier":
            raise ValueError("Need distinct classifier folds 0/1/2")
        if outer_fold is None and meta.get("inner_fold") is not None:
            raise ValueError("Nested predictions are not outer OOF")
        if outer_fold is not None and meta["fold"] != outer_fold:
            raise ValueError("Inner OOF excludes the wrong outer fold")
        if reference is None:
            reference, seed = cfg, meta["seed"]
        if cfg != reference or meta["seed"] != seed:
            raise ValueError("OOF configurations/seeds differ")
        current = recipe_provenance(model)
        if common is None:
            common = current
        if current != common:
            raise ValueError("OOF annotation, initialization or pretrained/input provenance differs")
        fit, hold = context.split(meta["fold"], meta.get("inner_fold"))
        check_support(meta, context, fit)
        expected = {context.rows[i]["name"] for i in hold}
        if set(meta["holdout_names"]) != expected or set(meta["train_names"]) != {context.rows[i]["name"] for i in fit}:
            raise ValueError("Checkpoint support does not match its declared partition")
        records = read_predictions(path / "holdout.csv")
        if len(records) != len(expected) or {r["Image_name"] for r in records} != expected:
            raise ValueError("Incomplete or duplicate held-out predictions")
        truth = {r["name"]: r["label"] for r in context.rows}
        for record in records:
            name, probability = record["Image_name"], float(record["Probability"])
            if (name in mapping or int(record["True_Label"]) != truth[name] or int(record["fold"]) != fold or
                    not np.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("OOF label/fold/probability mismatch")
            mapping[name] = probability
        seen.add(fold)
        sources.append({"path": str(path), "model_sha256": sha256(path / "model.pt"),
                        "holdout_sha256": sha256(path / "holdout.csv"), "signature": model["signature"]})
    selection = np.arange(len(context.rows)) if outer_fold is None else np.flatnonzero(context.folds != outer_fold)
    rows = [context.rows[i] for i in selection]
    if set(mapping) != {r["name"] for r in rows}:
        raise ValueError("OOF coverage is incomplete")
    probabilities = np.array([mapping[r["name"]] for r in rows])
    save_predictions(output, rows, probabilities, context.folds[selection])
    report = {"format": "plan3-oof-v1", "outer_fold": outer_fold, "config": reference, "seed": seed,
              "data_fingerprint": context.audit["data_fingerprint"], "fold_fingerprint": context.audit["fold_fingerprint"],
              "prediction_sha256": sha256(output), "sources": sources, "recipe_provenance": common, "metrics": metrics([r["label"] for r in rows], probabilities),
              "evaluation_scope": "configuration/epoch selection estimate; patient grouping unverified"}
    if outer_fold is None:
        report["report"] = summary(rows, probabilities)
        report["per_fold"] = {str(f): metrics(np.array([r["label"] for r in rows])[context.folds == f], probabilities[context.folds == f]) for f in range(3)}
    write_json(Path(output).with_suffix(".json"), report)
    return report


def load_oof(path, context, outer_fold=None):
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    selection = np.arange(len(context.rows)) if outer_fold is None else np.flatnonzero(context.folds != outer_fold)
    rows = [context.rows[i] for i in selection]
    records = read_predictions(path)
    if (meta.get("format") != "plan3-oof-v1" or meta.get("outer_fold") != outer_fold or
            meta["prediction_sha256"] != sha256(path) or
            any(meta[k] != context.audit[k] for k in ("data_fingerprint", "fold_fingerprint")) or
            [r["Image_name"] for r in records] != [r["name"] for r in rows] or
            [int(r["True_Label"]) for r in records] != [r["label"] for r in rows] or
            [int(r["fold"]) for r in records] != context.folds[selection].tolist()):
        raise ValueError("Expected aligned, untampered train-only OOF with matching outer exclusion")
    p = np.array([float(r["Probability"]) for r in records])
    metrics([r["label"] for r in rows], p)
    return p, meta


def make_hard_negatives(context, inner_oof, outer_fold, output):
    fit, _ = context.split(outer_fold)
    probabilities, meta = load_oof(inner_oof, context, None if outer_fold == "full" else outer_fold)
    negatives = [j for j, i in enumerate(fit) if context.rows[i]["label"] == 0]
    chosen = sorted(negatives, key=lambda j: probabilities[j], reverse=True)[:max(1, len(negatives)//10)]
    result = {"format": "plan3-hard-negatives-v1", "fold": outer_fold,
              "train_names": [context.rows[i]["name"] for i in fit],
              "names": [context.rows[fit[j]]["name"] for j in chosen],
              "inner_oof_path": str(Path(inner_oof).resolve()), "inner_oof_sha256": sha256(inner_oof),
              "inner_oof_metadata_sha256": sha256(Path(inner_oof).with_suffix(".json")),
              "data_fingerprint": meta["data_fingerprint"], "fold_fingerprint": meta["fold_fingerprint"], "source_report": meta}
    write_json(output, result)
    return result


def validate_hard_negatives(path, context, fit, job):
    item = read_json(path)
    names = [context.rows[i]["name"] for i in fit]
    if (item.get("format") != "plan3-hard-negatives-v1" or item["fold"] != job["fold"] or
            set(item["train_names"]) != set(names) or job.get("inner_fold") is not None or
            any(item[k] != context.audit[k] for k in ("data_fingerprint", "fold_fingerprint"))):
        raise ValueError("Hard negatives were not generated inside this training partition")
    # Embed the validated source report in the artifact when exporting across machines.
    proof = item.get("source_report")
    expected_outer = None if job["fold"] == "full" else job["fold"]
    if not proof or proof.get("outer_fold") != expected_outer or proof.get("prediction_sha256") != item["inner_oof_sha256"]:
        raise ValueError("Hard-negative artifact needs embedded inner-OOF provenance")
    expected = {r["name"]: (j, r["label"]) for j, r in enumerate(context.rows[i] for i in fit)}
    if not item["names"] or any(n not in expected or expected[n][1] != 0 for n in item["names"]):
        raise ValueError("Invalid hard-negative membership")
    return [expected[n][0] for n in item["names"]]


def fuse_values(matrix, mode, coefficient=None, intercept=0.):
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Fusion expects finite aligned probabilities")
    if mode == "probability":
        return matrix.mean(1)
    features = logit(np.clip(matrix, 1e-6, 1-1e-6))
    if mode == "logit":
        return expit(features.mean(1))
    if mode == "logistic":
        return expit(features @ coefficient + intercept)
    raise ValueError("Unknown fusion mode")


def fit_meta(matrix, labels):
    from sklearn.linear_model import LogisticRegression
    fitted = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(logit(np.clip(matrix, 1e-6, 1-1e-6)), labels)
    return fitted.coef_[0], float(fitted.intercept_[0])


def fit_ensemble(context, spec, output):
    from .train import load_payload
    sources = [load_oof(safe_path(entry["oof"]), context) for entry in spec["experts"]]
    matrix = np.column_stack([s[0] for s in sources])
    labels = np.array([r["label"] for r in context.rows])
    mode = spec["fusion"]
    coefficient, intercept = None, 0.
    report = {"probability_mean": metrics(labels, fuse_values(matrix, "probability")),
              "logit_mean": metrics(labels, fuse_values(matrix, "logit"))}
    if mode == "logistic":
        nested = np.full(len(labels), np.nan)
        for fold in range(3):
            inside = []
            for expert, (_, source_meta) in zip(spec["experts"], sources):
                values, meta = load_oof(safe_path(expert["inner_oof"][str(fold)]), context, fold)
                if meta["config"] != source_meta["config"] or meta["seed"] != source_meta["seed"]:
                    raise ValueError("Nested and outer OOF experts have different recipes")
                inside.append(values)
            fit = context.folds != fold
            c, b = fit_meta(np.column_stack(inside), labels[fit])
            nested[~fit] = fuse_values(matrix[~fit], "logistic", c, b)
        report["nested_logistic"] = metrics(labels, nested)
        coefficient, intercept = fit_meta(matrix, labels)
        report["logistic_fit_only"] = metrics(labels, fuse_values(matrix, mode, coefficient, intercept))
    elif mode not in ("probability", "logit"):
        raise ValueError("Unknown fusion")
    artifact = {"format": "plan3-ensemble-v1", "fusion": mode, "coefficient": coefficient.tolist() if coefficient is not None else None,
                "intercept": intercept, "experts": [], "data_fingerprint": context.audit["data_fingerprint"],
                "fold_fingerprint": context.audit["fold_fingerprint"], "report": report}
    for e, (entry, (_, source_meta)) in enumerate(zip(spec["experts"], sources)):
        models = []
        for j, value in enumerate(entry["models"]):
            path = safe_path(value)
            model = load_payload(path)
            if (model["kind"] != "classifier" or model["metadata"].get("selection_config", model["config"]) != source_meta["config"] or
                    model["metadata"]["seed"] != source_meta["seed"] or
                    not read_json(path.parent / "status.json").get("complete") or
                    any(model["metadata"][k] != context.audit[k] for k in ("data_fingerprint", "fold_fingerprint"))):
                raise ValueError("Ensemble member does not match its OOF recipe")
            target = Path(output).parent / "members" / f"expert{e}_model{j}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            models.append({"path": target.relative_to(Path(output).parent).as_posix(), "sha256": sha256(target), "fold": model["metadata"]["fold"]})
        folds = [m["fold"] for m in models]
        if folds != ["full"] and sorted(folds) != [0, 1, 2]:
            raise ValueError("Each expert must be a full-train refit or exactly three outer-fold models")
        policy = source_meta["config"].get("inference", {"horizontal_tta": False, "scale_tta": [1.]})
        if ("horizontal_tta" in spec and bool(spec["horizontal_tta"]) != policy["horizontal_tta"] or
                "scale_tta" in spec and spec["scale_tta"] != policy["scale_tta"]):
            raise ValueError("Ensemble TTA must match each expert's verified OOF inference policy")
        artifact["experts"].append({"models": models, "oof_sha256": source_meta["prediction_sha256"], "inference": policy})
    write_json(output, artifact)
    return artifact


def artifact_identity(path):
    path = Path(path)
    if path.suffix != ".json":
        return sha256(path)
    artifact = read_json(path)
    if artifact.get("format") != "plan3-ensemble-v1":
        raise ValueError("Expected an ensemble artifact")
    for expert in artifact["experts"]:
        for model in expert["models"]:
            if sha256(safe_path(model["path"], path.parent)) != model["sha256"]:
                raise ValueError("Ensemble member hash changed")
    return digest({"manifest_sha256": sha256(path), "members": [[m["sha256"] for m in e["models"]] for e in artifact["experts"]]})


def verify_artifact(context, artifact, require_full=False):
    from .train import load_payload
    artifact = Path(artifact)
    artifact_identity(artifact)
    if artifact.suffix == ".json":
        manifest = read_json(artifact)
        paths = [safe_path(m["path"], artifact.parent) for e in manifest["experts"] for m in e["models"]]
        if any(manifest[k] != context.audit[k] for k in ("data_fingerprint", "fold_fingerprint")):
            raise ValueError("Ensemble data/fold provenance differs")
    else:
        if not read_json(artifact.parent / "status.json").get("complete"):
            raise ValueError("Evaluation/prediction requires a completed model")
        paths = [artifact]
    for path in paths:
        payload = load_payload(path)
        meta = payload["metadata"]
        if payload["kind"] != "classifier" or meta.get("inner_fold") is not None or require_full and meta["fold"] != "full":
            raise ValueError("Final delivery requires full-train classifiers; nested models cannot be deployed")
        fit, _ = context.split(meta["fold"])
        check_support(meta, context, fit)
        names = [context.rows[i]["name"] for i in fit]
        expected_age = AgeTransform.fit([context.rows[i]["age"] for i in fit]).to_dict()
        if set(meta["train_names"]) != set(names) or payload["age_transform"] != expected_age:
            raise ValueError("Deployment support/age transform does not match the training partition")


def predict_rows(path, rows, device="cuda", workers=0, batch_size=4, horizontal=None, scales=None):
    import torch
    from glaucoma.engine import autocast
    from .train import build, data_loader, load_payload, locate, move, session_check
    path = Path(path)
    if path.suffix == ".json":
        artifact_identity(path)
        artifact = read_json(path)
        experts = []
        for expert in artifact["experts"]:
            values = [predict_rows(safe_path(m["path"], path.parent), rows, device, workers, batch_size,
                                   expert["inference"]["horizontal_tta"], expert["inference"]["scale_tta"]) for m in expert["models"]]
            experts.append(np.mean(values, axis=0))
        return fuse_values(np.column_stack(experts), artifact["fusion"], artifact["coefficient"], artifact["intercept"])
    payload = load_payload(path)
    if payload["kind"] != "classifier":
        raise ValueError("Prediction requires classifier checkpoint")
    model = build(payload, device)
    cfg = payload["config"]
    horizontal = cfg.get("inference", {}).get("horizontal_tta", False) if horizontal is None else horizontal
    scales = cfg.get("inference", {}).get("scale_tta", [1.]) if scales is None else scales
    if not scales or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("TTA scales must be finite positive values")
    roi = locate(payload["locator"], rows, device) if payload.get("locator") else {}
    age = AgeTransform(**payload["age_transform"])
    total = np.zeros(len(rows))
    for scale in scales:
        data = data_loader(Views(rows, cfg, age, roi, False, scale), batch_size, workers, 42)
        with torch.inference_mode():
            for images, ages, _, ix in data:
                session_check()
                images, ages = move(images, device), ages.to(device)
                with autocast(device):
                    probability = model(images, ages)["logits"].double().sigmoid()
                    if horizontal:
                        flipped_local = [im.flip(-1) for im in images["local"]] if isinstance(images["local"], list) else images["local"].flip(-1)
                        flipped = {**images, "global": images["global"].flip(-1), "local": flipped_local,
                                   "coordinates": images["coordinates"].clone()}
                        flipped["coordinates"][..., 0] = 1-flipped["coordinates"][..., 0]
                        probability = (probability+model(flipped, ages)["logits"].double().sigmoid())/2
                total[ix.numpy()] += probability.cpu().numpy()/len(scales)
    return total


def export_predictions(context, artifact, split, output, device, workers=0, batch_size=4):
    if split not in ("test", "valid"):
        raise ValueError("Use held-out OOF tools for train predictions")
    verify_artifact(context, artifact, require_full=split == "test")
    rows = load_rows(context.root, split)
    probabilities = predict_rows(artifact, rows, device, workers, batch_size)
    if split == "test":
        write_submission(output, rows, probabilities)
        check_submission(output, rows)
        if len(rows) != 5855:
            raise ValueError("Official test delivery requires 5855 rows")
    else:
        from glaucoma.engine import save_predictions
        save_predictions(output, rows, probabilities)
    return rows, probabilities
