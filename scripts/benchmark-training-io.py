"""Paired train/validation/features benchmarks on TRAIN only; opt-in execution profiles."""
from __future__ import annotations

import argparse
import gc
import os
import random
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch
from torch.utils.data import Subset

from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS, Budget, OutputBusy, exclusive_output, read_json, seed_all, sha256, write_json
from glaucoma.data import AgeTransform, ImageDataset, load_rows
from glaucoma.engine import autocast, device_for, folds_for, loader, synchronize, train_batches
from glaucoma.models import FundusModel, SPECS
from glaucoma.performance import cache_payback, numerical_check, profile_identity, recommend
from glaucoma.pixel_cache import build_pixel_cache, find_pixel_cache


class SessionBudget:
    def __init__(self, minutes, cumulative):
        self.deadline = time.monotonic()+minutes*60
        self.cumulative = cumulative

    def check(self):
        self.cumulative.check()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Performance session budget reached")


@contextmanager
def idle_slot(wait=False):
    while True:
        stack = ExitStack()
        try:
            for name in ("green_cv", "nightly", "main_study", "acceleration"):
                stack.enter_context(exclusive_output(ARTIFACTS / name / "run.lock"))
        except OutputBusy:
            stack.close()
            if not wait:
                raise
            print("Waiting for the existing training queue; no GPU benchmark started", flush=True)
            time.sleep(15)
            continue
        break
    try:
        cv_state = ARTIFACTS / "green_cv/state.json"
        if cv_state.exists() and read_json(cv_state).get("status") != "complete":
            raise RuntimeError("Current Green CV must finish successfully before benchmarking; resume/resolve it first")
        yield
    finally:
        stack.close()


def protected_files(checkpoint):
    names = {"model.pt", "resume.pt", "history.json", "holdout.csv", "oof.csv", "oof.json",
             "submission.csv", "folds.csv", "report.json", "status.json"}
    paths = {checkpoint.resolve()}
    for path in ARTIFACTS.rglob("*"):
        if path.is_file() and path.name in names and "acceleration" not in path.relative_to(ARTIFACTS).parts:
            paths.add(path.resolve())
    return {str(p): sha256(p) for p in sorted(paths)}


def sample_indices(rows, eligible, count, seed):
    rng = np.random.default_rng(seed)
    count = min(count, len(eligible))
    pos = np.array([i for i in eligible if rows[i]["label"] == 1], dtype=int)
    neg = np.array([i for i in eligible if rows[i]["label"] == 0], dtype=int)
    npos = min(len(pos), max(2, round(count*len(pos)/len(eligible))))
    nneg = min(len(neg), count-npos)
    if npos < 1 or nneg < 1:
        raise ValueError("Benchmark needs both classes in each TRAIN subset")
    selected = np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]
    rng.shuffle(selected)
    return selected.tolist()


def make_optimizer(model, config):
    return torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.get("backbone_lr", 1e-5)},
        {"params": model.head.parameters(), "lr": config.get("head_lr", 1e-3)}], weight_decay=.01)


def run_workload(stage, settings, config, payload, dataset, indices, budget, device, scratch):
    budget.check()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    setup_started = time.monotonic()
    seed_all(42)
    model = FundusModel(config)
    model.load_state_dict(payload["state_dict"])
    model.configure_trainable(config.get("tail_blocks", 2)).to(device)
    optimizer = make_optimizer(model, config) if stage == "train" else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    data = loader(Subset(dataset, indices), settings["batch_size"], settings["workers"], stage == "train", 42)
    synchronize(device)
    setup_seconds = time.monotonic()-setup_started
    if len(data) <= 8:
        raise ValueError("Use more than 8 batches per workload")
    seed_all(42)
    started = time.monotonic()
    values = done = None
    try:
        if stage == "train":
            result = train_batches(model, data, optimizer, scaler, device, config,
                                   reduced_sync=settings["reduced_sync"], budget=budget, warmup_batches=8)
            save_started = time.monotonic()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
                       scratch / "training_state.pt")
            save_seconds = time.monotonic()-save_started
            total_seconds = time.monotonic()-started
            parameters = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
            timing = {"measured_images": result["measured_images"], "seconds": result["measured_seconds"],
                      "startup_seconds": result["startup_seconds"], "compute_and_transfer_seconds": result["seconds"],
                      "save_seconds": save_seconds, "optimizer_steps": result["optimizer_steps"],
                      "total_images": result["count"], "total_seconds": total_seconds}
            check_data = loader(Subset(ImageDataset(dataset.rows, dataset.size, cache=dataset.cache_path), indices), 16, 0)
            order, labels, probabilities = [], [], []
            model.eval()
            with torch.inference_mode():
                for images, _, y, ix in check_data:
                    budget.check()
                    ages = torch.from_numpy(dataset.ages[ix.numpy()]).to(device)
                    with autocast(device):
                        logits = model(images.to(device), ages)
                    probabilities.extend(torch.sigmoid(logits.double()).cpu().tolist())
                    order.extend(ix.tolist())
                    labels.extend(y.tolist())
            comparison = {"indices": order, "labels": labels, "probabilities": probabilities,
                          "parameters": parameters, "optimizer_steps": result["optimizer_steps"]}
        else:
            model.eval()
            order, labels, probabilities, features = [], [], [], []
            first_arrival, measured_start, measured_count, save_seconds = None, None, 0, 0.
            shape = (len(dataset), SPECS[config["model"]]["dim"])
            values = np.lib.format.open_memmap(scratch / "features.npy", mode="w+", dtype=np.float32, shape=shape) if stage == "features" else None
            done = np.lib.format.open_memmap(scratch / "done.npy", mode="w+", dtype=np.bool_, shape=(len(dataset),)) if stage == "features" else None
            if done is not None:
                done[:] = False
            with torch.inference_mode():
                for step, (images, ages, y, ix) in enumerate(data):
                    budget.check()
                    if step == 0:
                        first_arrival = time.monotonic()-started
                    if step == 8:
                        synchronize(device)
                        measured_start = time.monotonic()
                    with autocast(device):
                        if stage == "features":
                            feature = model.features(images.to(device, non_blocking=True))
                        else:
                            logits = model(images.to(device, non_blocking=True), ages.to(device, non_blocking=True))
                    if stage == "features":
                        array = feature.float().cpu().numpy()
                        if not np.isfinite(array).all():
                            raise FloatingPointError("Non-finite benchmark features")
                        save_started = time.monotonic()
                        values[ix.numpy()] = array
                        values.flush()
                        done[ix.numpy()] = True
                        done.flush()
                        save_seconds += time.monotonic()-save_started
                        features.append(array)
                    else:
                        probabilities.extend(torch.sigmoid(logits.double()).cpu().tolist())
                    order.extend(ix.tolist())
                    labels.extend(y.tolist())
                    if step >= 8:
                        measured_count += len(images)
            synchronize(device)
            ended = time.monotonic()
            timing = {"measured_images": measured_count, "seconds": ended-measured_start,
                      "startup_seconds": first_arrival, "save_seconds": save_seconds,
                      "compute_and_transfer_seconds": ended-started-save_seconds,
                      "optimizer_steps": 0, "total_images": len(order), "total_seconds": ended-started}
            comparison = {"indices": order, "labels": labels, "probabilities": probabilities}
            if stage == "features":
                feature_array = np.concatenate(features)
                with torch.inference_mode():
                    logits = model.classify(torch.from_numpy(feature_array).to(device), torch.from_numpy(dataset.ages[order]).to(device))
                    comparison["probabilities"] = torch.sigmoid(logits.double()).cpu().tolist()
                comparison["features"] = feature_array
            values = done = None
        timing.update(images_per_second=timing["measured_images"]/max(timing["seconds"], 1e-9), setup_seconds=setup_seconds,
                      settings=settings, peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None)
        return timing, comparison

    finally:
        # Release Windows memmaps and persistent workers even on OOM or budget expiry.
        del values, done, data, model, optimizer, scaler


def candidate_settings(stage, config, cache_available):
    if stage == "train":
        return [{"batch_size": config.get("batch_size", 4), "workers": 4,
                 "use_pixel_cache": cache_available, "reduced_sync": True}]
    return [{"batch_size": batch, "workers": workers, "use_pixel_cache": cache_available, "reduced_sync": False}
            for workers in (4, 2, 0) for batch in (16, 32, 8, 4)]


def best_trial(result):
    passing = [t for t in result["candidates"] if t.get("decision", {}).get("accepted")]
    return max(passing, key=lambda t: t["decision"]["estimated_net_saving_seconds"], default=None)


def write_summary(report, profile, path):
    lines = ["# 可复用训练加速实测报告", "", f"运行状态：`{report['status']}`。",
             "本报告只比较 train 及内部留出样本；不构成模型效果提升或 T4 推理验收。",
             "历史像素缓存收益保留在原 training_io_benchmark.json，不计作本轮新增收益。", "",
             "| 阶段 | 基准张/秒 | 候选张/秒 | 吞吐倍数 | 推荐采用 |", "|---|---:|---:|---:|---|"]
    for stage in ("train", "validation", "features"):
        result = report["stages"].get(stage)
        best = best_trial(result) if result else None
        if best:
            decision = best["decision"]
            accepted = profile["stages"].get(stage, {}).get("accepted", False)
            lines.append(f"| {stage} | {decision['baseline_median_images_per_second']:.2f} | "
                         f"{decision['candidate_median_images_per_second']:.2f} | {decision['speedup']:.3f} | {'是' if accepted else '否'} |")
        else:
            lines.append(f"| {stage} | — | — | — | {'未通过门槛' if result else '未测'} |")
    integrity = "尚未执行" if "protected_files_unchanged" not in report else ("全部未变" if report["protected_files_unchanged"] else "失败，禁止采用")
    lines.extend(["", f"测速自身耗时：{report.get('benchmark_seconds', 0):.1f} 秒。",
                  f"正式检查点和结果校验：{integrity}。",
                  "", "每个候选需要三次配对测量、一致性检查、至少10%中位吞吐提升和正的预计净节省。",
                  "表中为通过门槛候选中的最佳已测配置；未通过与中断的候选详见 report.json。",
                  "净节省计入启动、保存及缓存准备；测速本身另列为一次性成本，需要后续任务复用抵偿。",
                  "短样本结果不能直接视为完整 epoch 倍数；未测模型不能继承本机结果。",
                  "", "启用方法和显式参数优先规则见项目 README 的“可复用性能配置”。",
                  "profile.json 为显式启用配置，不自动重启或修改正式训练。"])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def benchmark(args):
    output = args.output.resolve()
    if not output.is_relative_to((ARTIFACTS / "acceleration").resolve()):
        raise ValueError("Benchmark outputs must be inside artifacts/acceleration")
    with exclusive_output(output / "run.lock"):
        try:
            return _benchmark(args)
        except BaseException as exc:
            report_path = output / "report.json"
            if report_path.exists():
                report = read_json(report_path)
                if report.get("status") == "waiting":
                    report.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
                    write_json(report_path, report)
                    write_summary(report, {"stages": {}}, output / "report.md")
            raise


def _benchmark(args):
    config, checkpoint, output = read_json(args.config), args.checkpoint.resolve(), args.output.resolve()
    if not output.is_relative_to((ARTIFACTS / "acceleration").resolve()):
        raise ValueError("Benchmark outputs must be inside artifacts/acceleration")
    output.mkdir(parents=True, exist_ok=True)
    report_path, profile_path = output / "report.json", output / "profile.json"
    report = {"status": "waiting", "scope": "TRAIN fit/internal holdout only; never official valid/test", "config": config,
              "checkpoint": str(checkpoint), "stages": {}, "budget_minutes": args.max_minutes, "warmup_batches": 8,
              "unverified_models": [m for m in ("green", "effnet", "rfm") if m != config["model"]],
              "method": "3 alternating paired repeats, fixed images/initial states; train includes accumulation/optimizer/save"}
    write_json(report_path, report)
    write_summary(report, {"stages": {}}, output / "report.md")
    with idle_slot(args.wait_for_idle), Budget(config["model"] == "rfm") as cumulative:
        budget = SessionBudget(args.max_minutes, cumulative)
        device = device_for(args.device)
        torch.set_num_threads(4)
        report.update(status="running", started_epoch=time.time(), identity=profile_identity(config, device))
        write_json(report_path, report)
        profile = {"format_version": 1, "identity": report["identity"], "stages": {}, "report": str(report_path)}
        before = protected_files(checkpoint)
        full_cache, sample_cache = None, None
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["config"] != config or payload["metadata"].get("fold") != args.fold:
                raise ValueError("Use a matching-config, matching-fold checkpoint")
            audit = require_audit(ROOT / "glaucoma_first")
            rows = load_rows(ROOT / "glaucoma_first", "train")
            fold_ids = folds_for(rows)
            fit, hold = np.flatnonzero(fold_ids != args.fold), np.flatnonzero(fold_ids == args.fold)
            age = AgeTransform.fit([rows[i]["age"] for i in fit])
            selected = {"train": sample_indices(rows, fit, args.count, 42), "validation": sample_indices(rows, hold, args.count, 43)}
            selected["features"] = selected["train"]
            full_cache = find_pixel_cache(rows, config["size"], ARTIFACTS, audit["data_fingerprint"])
            report.update(sample_names={k: [rows[i]["name"] for i in v] for k, v in selected.items()},
                          data_fingerprint=audit["data_fingerprint"], fold_sha256=sha256(ARTIFACTS / "audit/folds.csv"),
                          checkpoint_sha256=before[str(checkpoint)], benchmark_script_sha256=sha256(Path(__file__)),
                          existing_pixel_cache=str(full_cache) if full_cache else None)
            profile.update(data_fingerprint=audit["data_fingerprint"], source_checkpoint_sha256=before[str(checkpoint)])
            preparation_seconds, sample_rows, remap = 0., None, None
            with tempfile.TemporaryDirectory(prefix="benchmark-", dir=output) as tmp:
                scratch = Path(tmp)
                if not full_cache and args.build_cache:
                    sample_ix = sorted(set(selected["train"]+selected["validation"]))
                    sample_rows = [rows[i] for i in sample_ix]
                    remap = {original: i for i, original in enumerate(sample_ix)}
                    sample_cache = scratch / "sample_pixels.npy"
                    build_started = time.monotonic()
                    build_pixel_cache(sample_rows, config["size"], sample_cache, budget=budget)
                    preparation_seconds = time.monotonic()-build_started
                    report["cache_build_estimate_seconds"] = preparation_seconds*len(rows)/len(sample_rows)
                for stage in args.stages:
                    budget.check()
                    cache_available = bool(full_cache or sample_cache)
                    stage_rows = sample_rows if sample_cache else rows
                    stage_ix = [remap[i] for i in selected[stage]] if sample_cache else selected[stage]
                    raw = ImageDataset(stage_rows, config["size"], age, stage == "train")
                    cached = ImageDataset(stage_rows, config["size"], age, stage == "train", full_cache or sample_cache) if cache_available else raw
                    if cache_available:
                        for i in stage_ix[:32]:
                            random.seed(2026+i)
                            a = raw[i][0]
                            random.seed(2026+i)
                            if not torch.equal(a, cached[i][0]):
                                raise AssertionError("Caching changed pixels/seeded augmentations")
                    baseline = {"batch_size": config.get("batch_size", 4) if stage != "features" else 16,
                                "workers": 4 if stage != "features" else 2,
                                "use_pixel_cache": bool(full_cache) and stage != "features", "reduced_sync": False}
                    expected = (len(hold) if stage == "validation" else len(fit)) * (args.expected_passes or (1 if stage == "features" else config.get("epochs", 8)))
                    result = {"baseline": baseline, "candidates": [], "expected_images": expected,
                              "search_space": candidate_settings(stage, config, cache_available),
                              "pixel_equality": "passed on up to 32 samples" if cache_available else "no cache",
                              "selection_estimate": "short subset benchmark; not a full-epoch speed claim"}
                    report["stages"][stage] = result
                    for candidate in candidate_settings(stage, config, cache_available):
                        if candidate == baseline:
                            continue
                        trial = {"settings": candidate, "baselines": [], "candidates": [], "checks": []}
                        result["candidates"].append(trial)
                        try:
                            for repeat in range(3):
                                pair = {}
                                for label in (("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")):
                                    settings = baseline if label == "baseline" else candidate
                                    dataset = cached if settings["use_pixel_cache"] else raw
                                    pair[label] = run_workload(stage, settings, config, payload, dataset, stage_ix, budget, device, scratch)
                                    print(f"{stage} repeat={repeat+1} {label} {settings}: {pair[label][0]['images_per_second']:.2f} images/s", flush=True)
                                trial["checks"].append(numerical_check(pair["baseline"][1], pair["candidate"][1]))
                                trial["baselines"].append(pair["baseline"][0])
                                trial["candidates"].append(pair["candidate"][0])
                                del pair
                            estimated_build = report.get("cache_build_estimate_seconds", 0.) if candidate["use_pixel_cache"] and not full_cache else 0.
                            trial["decision"] = recommend(trial["baselines"], trial["candidates"], trial["checks"], preparation_seconds+estimated_build, expected)
                        except torch.cuda.OutOfMemoryError:
                            trial["decision"] = {"accepted": False, "reason": "CUDA out of memory; candidate discarded"}
                            pair = {}
                            gc.collect()
                            torch.cuda.empty_cache()
                        finally:
                            write_json(report_path, report)
                    best = best_trial(result)
                    result["recommendation"] = best["decision"] if best else {"accepted": False, "reason": "No candidate passed all gates"}
                    profile["stages"][stage] = {"accepted": bool(best), "settings": best["settings"] if best else baseline,
                                               "baseline_settings": baseline}
                if sample_cache:
                    best = max((best_trial(s) for s in report["stages"].values() if best_trial(s)),
                               key=lambda t: t["decision"]["estimated_net_saving_seconds"], default=None)
                    if best:
                        raw_seconds = float(np.median([r["total_seconds"]/r["total_images"] for r in best["baselines"]]))
                        fast_seconds = float(np.median([r["total_seconds"]/r["total_images"] for r in best["candidates"]]))
                        payback = cache_payback(raw_seconds, fast_seconds, max(s["expected_images"] for s in report["stages"].values()),
                                               report["cache_build_estimate_seconds"], len(rows)*config["size"]**2*3+4096, shutil.disk_usage(ARTIFACTS).free)
                        report["cache_payback"] = payback
                        if payback["worth_building"] and time.monotonic()+report["cache_build_estimate_seconds"] < budget.deadline:
                            target = ARTIFACTS / "pixel_cache" / f"train_{config['size']}_{audit['data_fingerprint'][:10]}.npy"
                            build_started = time.monotonic()
                            full_cache = build_pixel_cache(rows, config["size"], target, budget=budget)
                            report["cache_build_actual_seconds"] = time.monotonic()-build_started
                            report["built_pixel_cache"] = str(full_cache)
                report["status"] = "complete"
        except TimeoutError as exc:
            report.update(status="budget_reached", reason=str(exc))
        except BaseException as exc:
            report.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for stage, result in report["stages"].items():
                if stage not in profile["stages"]:
                    best = best_trial(result)
                    profile["stages"][stage] = {"accepted": bool(best), "settings": best["settings"] if best else result["baseline"],
                                               "baseline_settings": result["baseline"]}
                if profile["stages"][stage]["settings"]["use_pixel_cache"] and not full_cache:
                    profile["stages"][stage].update(accepted=False, reason="Full cache unavailable within payback/disk/time gates")
            after = protected_files(checkpoint)
            report.update(protected_files_unchanged=before == after, protected_file_sha256=before,
                          source_and_formal_results_unchanged=before == after, finished_epoch=time.time())
            report["benchmark_seconds"] = report["finished_epoch"]-report["started_epoch"]
            report["cost_note"] = "Per-stage net savings include startup/save/cache preparation; profiling is an additional upfront cost. Short-subset extrapolation only."
            if before != after or report["status"] == "failed":
                for entry in profile["stages"].values():
                    entry["accepted"] = False
                if before != after:
                    report.update(status="failed", reason="Protected checkpoints/results changed")
            write_json(profile_path, profile)
            write_json(report_path, report)
            write_summary(report, profile, output / "report.md")
            print(f"Performance report: {report_path}", flush=True)
            print(f"Opt-in profile: {profile_path}", flush=True)
            if before != after:
                raise RuntimeError("Protected checkpoints/results changed")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/green_ft2.json")
    p.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "green_ft2/fold0/model.pt")
    p.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--stages", nargs="+", choices=("train", "validation", "features"), default=["train", "validation", "features"])
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--max-minutes", type=float, default=30)
    p.add_argument("--expected-passes", type=int)
    p.add_argument("--build-cache", action="store_true", help="Build only after sample payback/space/time checks")
    p.add_argument("--wait-for-idle", action="store_true")
    p.add_argument("--output", type=Path, default=ARTIFACTS / "acceleration/performance")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.count < 512 or not 0 < args.max_minutes <= 30 or (args.expected_passes is not None and args.expected_passes <= 0):
        p.error("Use count>=512, 0<max-minutes<=30, expected-passes>0")
    benchmark(args)
