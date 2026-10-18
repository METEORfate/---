"""Bounded P1 -> P2 development-fold study; never reads official valid/test."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch
from PIL import Image, ImageDraw

from glaucoma import engine
from glaucoma.audit import require_audit
from glaucoma.common import (ARTIFACTS, Budget, code_fingerprint, exclusive_output,
                             read_json, sha256, versions, write_json)
from glaucoma.data import decode_image, load_rows
from glaucoma.metrics import metrics, paired_bootstrap
from glaucoma.models import load_checkpoint
from glaucoma.multiview import image_dataset, native_views, view_boxes


def patch_preview(rows, output, size):
    """Source boxes plus exact deterministic network views, on TRAIN only."""
    output.mkdir(parents=True, exist_ok=True)
    selected = [next(row for row in rows if row["label"] == label) for label in (0, 1)]
    records = []
    for number, row in enumerate(selected):
        image = decode_image(row["path"], size, full_decode=True)
        views = native_views(image, size)
        preview = image.copy()
        preview.thumbnail((600, 600))
        canvas = Image.new("RGB", (960, 720), "#202020")
        canvas.paste(preview, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{row['name']} | image label={row['label']} | source FOV={image.size}", fill="white")
        for i, box in enumerate(view_boxes(*image.size)):
            scaled = [int(v * preview.size[j % 2] / image.size[j % 2]) for j, v in enumerate(box)]
            scaled[1] += 28
            scaled[3] += 28
            draw.rectangle(scaled, outline=(255, 220 - i * 30, 50 + i * 40), width=2)
            draw.text((scaled[0]+3, scaled[1]+3), str(i+1), fill="white")
        tiles = [views["global"], *views["local"]]
        for i, tile in enumerate(tiles):
            tile = Image.fromarray((tile.permute(1, 2, 0).numpy()*255).round().astype(np.uint8)).resize((150, 150))
            x, y = 630 + (i % 2)*160, 38 + (i // 2)*220
            canvas.paste(tile, (x, y))
            label = "global" if i == 0 else f"patch {i}, valid={bool(views['mask'][i-1])}"
            draw.text((x, y+155), label, fill="white")
        canvas.save(output / f"views_{number}.png")
        records.append({"name": row["name"], "source_sha256": sha256(row["path"]),
                        "source_fov_size": image.size, "coordinates": views["coordinates"].tolist(),
                        "valid": views["mask"].tolist(), "local_status": views["status"].tolist()})
    write_json(output / "views.json", records)


def comparison(reference, candidate):
    def read(path):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        return {r["Image_name"]: r for r in records}
    a, b = read(reference), read(candidate)
    if not set(b).issubset(a):
        raise ValueError("Reference predictions do not cover the development fold")
    names = list(b)
    if any(a[n]["True_Label"] != b[n]["True_Label"] or a[n]["fold"] != b[n]["fold"] for n in names):
        raise ValueError("Comparison labels or folds differ")
    y = [int(b[n]["True_Label"]) for n in names]
    p, q = [float(a[n]["Probability"]) for n in names], [float(b[n]["Probability"]) for n in names]
    first, second = metrics(y, p), metrics(y, q)
    return {"reference": str(reference), "candidate": str(candidate), "first": first, "second": second,
            "delta_auroc": second["auroc"]-first["auroc"],
            "delta_auprc": second["auprc"]-first["auprc"],
            "paired_image_bootstrap": paired_bootstrap(y, q, p, repeats=500),
            "limitation": "Same development fold used for epoch selection; patient grouping unknown; not final OOF evidence",
            "requires_second_seed_if_small_positive_delta": 0 < second["auroc"]-first["auroc"] < .002}


def protected_files():
    paths = [ARTIFACTS / "green_ft2" / "oof.csv", ARTIFACTS / "green_ft2" / "oof.json",
             ARTIFACTS / "rfm_probe_age" / "model.pt", ARTIFACTS / "rfm_probe_age" / "oof.csv",
             ARTIFACTS / "baseline" / "submission.csv", ARTIFACTS / "valid_milestones.json",
             ROOT / "weights" / "rfm-1.3.pt"]
    paths.extend(ARTIFACTS / "green_ft2" / f"fold{i}" / "model.pt" for i in range(3))
    return {str(path.relative_to(ROOT)): sha256(path) if path.exists() else None for path in paths}


def session(hours=7, workers=4):
    if not torch.cuda.is_available():
        raise RuntimeError("P1/P2 study requires the existing CUDA runtime")
    torch.set_num_threads(4)
    output = ARTIFACTS / "p1p2"
    with ExitStack() as stack:
        for name in ("p1p2", "main_study", "nightly", "acceleration", "green_cv"):
            stack.enter_context(exclusive_output(ARTIFACTS / name / "run.lock"))
        with Budget(True) as budget:
            allowed = min(hours*3600, budget.limit-budget.used-180)
            if allowed < 600:
                raise TimeoutError("Insufficient existing RFM budget for a bounded development run")
            deadline = time.monotonic() + allowed
            state = {"status": "running", "current": "preflight", "started_epoch": time.time(),
                     "deadline_epoch": time.time()+allowed, "fold": 0, "seed": 42, "workers": workers,
                     "rfm_used_seconds_at_start": budget.used, "rfm_limit_seconds": budget.limit,
                     "scope": "P1/P2 development fold only; no official valid/test, full refit, or extra folds",
                     "stages": {}, "comparisons": {}}

            def save():
                state["last_update_epoch"] = time.time()
                state["rfm_used_seconds_estimate"] = budget.used + time.monotonic()-budget.started
                write_json(output / "state.json", state)

            save()
            initial = protected_files()
            baseline_path = output / "protected_sources.json"
            if baseline_path.exists() and read_json(baseline_path) != initial:
                raise ValueError("A protected result/checkpoint changed since the previous P1/P2 session")
            write_json(baseline_path, initial)
            try:
                audit = require_audit(ROOT / "glaucoma_first")
                configs = {name: read_json(ROOT / "configs" / f"rfm_{name}.json")
                           for name in ("p1_global", "p2_multiview")}
                write_json(output / "manifest.json", {
                    "configs": configs, "versions": versions(), "code_fingerprint": code_fingerprint(),
                    "entry_sha256": sha256(__file__), "data_fingerprint": audit["data_fingerprint"],
                    "fold_sha256": sha256(ARTIFACTS / "audit" / "folds.csv"),
                    "budget_policy": "Existing cumulative 8h RFM ledger, never reset; reserve 180s; P1 <=35% session, P2 remainder",
                    "cache_policy": "Native fixed-view caches only with a measured, accepted execution profile; raw fallback otherwise"})
                rows = load_rows(ROOT / "glaucoma_first", "train")
                patch_preview(rows, output / "views", configs["p2_multiview"]["size"])
                for name in ("p1_global", "p2_multiview"):
                    budget.check()
                    state["current"] = name
                    save()
                    target = output / name / "fold0"
                    remaining = deadline - time.monotonic()
                    cap = min(remaining, allowed*.35) if name == "p1_global" else remaining
                    if cap < 300:
                        state.update(status="budget_limited", reason="Insufficient time to start the next stage")
                        return
                    source = output / "p1_global" / "fold0" / "model.pt" if name == "p2_multiview" else None
                    if source and not source.exists():
                        state.update(status="needs_resume", reason="P1 has not saved a complete validated epoch")
                        return
                    profile = output / "acceleration" / f"{name}_profile.json"
                    use_profile = profile if profile.exists() else None
                    engine.fine_tune(ROOT / "glaucoma_first", configs[name], target, fold=0,
                                     workers=workers, device="cuda", resume=(target / "resume.pt").exists(),
                                     budget=budget, max_hours=cap/3600, initialize_from=source,
                                     performance_profile=use_profile,
                                     validation_batch_size=4, validation_workers=workers)
                    status = read_json(target / "status.json")
                    stage = {"status": status, "checkpoint": str(target / "model.pt")}
                    state["stages"][name] = stage
                    if (target / "model.pt").exists():
                        payload = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
                        stage.update(best_epoch=payload["metadata"]["epoch"], metrics=payload["metadata"]["holdout"],
                                     checkpoint_sha256=sha256(target / "model.pt"))
                        del payload
                        for reference in (ARTIFACTS / "green_ft2" / "fold0" / "holdout.csv",
                                          ARTIFACTS / "rfm_probe_age" / "oof.csv"):
                            key = f"{name}_vs_{reference.parent.name}"
                            result = comparison(reference, target / "holdout.csv")
                            state["comparisons"][key] = result
                            write_json(output / "comparisons" / f"{key}.json", result)
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
                p1 = output / "p1_global" / "fold0" / "holdout.csv"
                p2 = output / "p2_multiview" / "fold0" / "holdout.csv"
                if p1.exists() and p2.exists():
                    state["comparisons"]["p2_vs_p1"] = comparison(p1, p2)
                    write_json(output / "comparisons" / "p2_vs_p1.json", state["comparisons"]["p2_vs_p1"])
                state.update(status="complete" if all(s["status"]["complete"] for s in state["stages"].values()) else "budget_limited",
                             current=None, decision="Development evidence only; review before P3 or additional folds")
            except BaseException as exc:
                state.update(status="budget_limited" if isinstance(exc, TimeoutError) else "failed",
                             error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                state["protected_sources_unchanged"] = protected_files() == initial
                save()
                write_json(output / "report.json", state)
                if not state["protected_sources_unchanged"]:
                    raise RuntimeError("Protected source/result checksums changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.hours <= 8 or args.workers < 0:
        parser.error("Use 0<hours<=8 and workers>=0")
    session(args.hours, args.workers)
