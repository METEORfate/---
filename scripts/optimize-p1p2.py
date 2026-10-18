"""Bounded native-view cache qualification, construction, then automatic P1 -> P2."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import random
import shutil
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
from torch.utils.data import Subset

from glaucoma import engine
from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS, Budget, exclusive_output, read_json, seed_all, sha256, write_json
from glaucoma.data import AgeTransform, load_rows
from glaucoma.models import FundusModel, download_weights
from glaucoma.multiview import image_dataset
from glaucoma.performance import numerical_check, profile_identity
from glaucoma.view_cache import build_view_cache, estimated_bytes, find_view_cache


class LimitedBudget:
    def __init__(self, budget, seconds):
        self.budget, self.deadline = budget, time.monotonic()+seconds

    def check(self):
        self.budget.check()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Native cache qualification/construction time cap reached")


class FixedSamples:
    """Benchmark only: identical per-image random augmentations across persistent workers."""
    def __init__(self, dataset, validation):
        self.dataset, self.validation = dataset, validation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        state = random.getstate()
        try:
            random.seed(42+index)
            return (self.dataset if index < 64 else self.validation)[index]
        finally:
            random.setstate(state)


class PhaseSampler:
    offset = 0

    def __iter__(self):
        return iter(range(self.offset, self.offset+64))

    def __len__(self):
        return 64


class WorkloadLoader:
    def __init__(self, data, sampler, validation):
        self.data, self.sampler, self.validation = data, sampler, validation
        self.rows = data.dataset.dataset.rows

    def __iter__(self):
        self.sampler.offset = 64 if self.validation else 0
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class TimedLoader:
    def __init__(self, data):
        self.data, self.waiting, self.startup = data, 0., 0.

    def __iter__(self):
        self.waiting = self.startup = 0.
        started = time.monotonic()
        iterator = iter(self.data)
        for step in range(len(self.data)):
            before = time.monotonic()
            batch = next(iterator)
            after = time.monotonic()
            self.waiting += after-before
            if step == 0:
                self.startup = after-started
            yield batch

    def __len__(self):
        return len(self.data)


def shutdown(data):
    iterator = getattr(data, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        data._iterator = None


def required_check(check):
    """Use the requested output contract; stricter per-parameter allclose is diagnostic."""
    return bool(check.get("max_probability_difference", float("inf")) <= .001
                and all(check.get("metric_differences", {}).get(k, float("inf")) <= .001 for k in ("auroc", "auprc"))
                and check.get("optimizer_steps_equal") and check.get("training_order_equal"))


def sample(rows, indices, count, seed):
    rng = np.random.default_rng(seed)
    pos = [i for i in indices if rows[i]["label"] == 1]
    neg = [i for i in indices if rows[i]["label"] == 0]
    chosen = np.r_[rng.choice(pos, min(len(pos), max(4, count//16)), replace=False),
                   rng.choice(neg, count-min(len(pos), max(4, count//16)), replace=False)]
    rng.shuffle(chosen)
    return chosen.tolist()


def workload(config, initial, train, valid, age, budget, scratch, packing=0):
    budget.check()
    seed_all(42)
    setup = time.monotonic()
    model = FundusModel(config)
    model.load_state_dict(initial)
    model.configure_trainable(config["tail_blocks"]).to("cuda")
    model.performance_view_batch_size = packing
    backbone = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config["backbone_lr"]},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone], "lr": config["head_lr"]}],
        weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    engine.synchronize("cuda")
    result = {"setup_seconds": time.monotonic()-setup}
    updates = 0
    for phase, warmup in (("warmup", True), ("fine_tune", False)):
        seed_all(42)
        record = engine.train_batches(model, train, optimizer, scaler, "cuda", config,
                                      warmup=warmup, budget=budget, warmup_batches=8, record_order=True)
        record["images_per_second"] = record["measured_images"]/record["measured_seconds"]
        record["data_preparation_seconds"] = max(0., record["waiting"]-record["startup_seconds"])
        record["compute_transfer_update_seconds"] = max(0., record["seconds"]-record["waiting"])
        result[phase] = record
        updates += record["optimizer_steps"]
    valid = TimedLoader(valid)
    before = time.monotonic()
    ix, probabilities, scores = engine.evaluate_loader(model, valid, "cuda", budget=budget)
    seconds = time.monotonic()-before
    result["validation"] = {"seconds": seconds, "startup_seconds": valid.startup,
        "data_preparation_seconds": max(0., valid.waiting-valid.startup),
        "compute_transfer_seconds": max(0., seconds-valid.waiting),
        "images_per_second": len(ix)/max(seconds-valid.startup, 1e-6), "metrics": scores}
    before = time.monotonic()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, scratch)
    result["save_seconds"] = time.monotonic()-before
    check = {"indices": ix.tolist(), "labels": [int(valid.data.rows[i]["label"]) for i in ix],
             "probabilities": probabilities.tolist(), "optimizer_steps": updates,
             "parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}}
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result, check


def summarize(trials, epochs, fit_count, val_count):
    output = {}
    for path in ("raw", "cache"):
        rows = [t[path] for t in trials]
        rates = {phase: float(np.median([r[phase]["images_per_second"] for r in rows]))
                 for phase in ("warmup", "fine_tune", "validation")}
        cold = rows[0]["warmup"]["startup_seconds"] * 2
        saves = float(np.median([r["save_seconds"] for r in rows]))
        total = fit_count/rates["warmup"] + (epochs-1)*fit_count/rates["fine_tune"] + epochs*val_count/rates["validation"] + epochs*(cold+saves)
        output[path] = {"median_rates": rates, "cold_loader_startup_seconds": cold,
                        "estimated_remaining_seconds": total}
    output["speedups"] = {p: output["cache"]["median_rates"][p]/output["raw"]["median_rates"][p]
                          for p in ("warmup", "fine_tune", "validation")}
    output["estimated_saving_seconds"] = output["raw"]["estimated_remaining_seconds"]-output["cache"]["estimated_remaining_seconds"]
    output["numerical_passed"] = all(t["check"]["passed"] for t in trials)
    output["throughput_passed"] = output["speedups"]["fine_tune"] >= 1.1 and output["speedups"]["validation"] >= 1.1 and output["speedups"]["warmup"] >= 1.1
    return output


def run(minutes=12, build_minutes=45, resume_qualification=False, pack_views=False):
    torch.set_num_threads(4)
    folder = ARTIFACTS / "p1p2/acceleration"
    folder.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("p1p2_training", ROOT / "scripts/run-p1p2.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    report = read_json(folder / "report.json") if resume_qualification else {"status": "qualifying", "benchmark_limit_minutes": minutes, "cache_build_limit_minutes": build_minutes,
              "models": {}, "adopted": [], "scope": "TRAIN-only fold0, fixed samples; production model/seed/augmentation/batch unchanged"}
    if pack_views and not report.get("packing_candidate"):
        report["cache_only_p2"] = report["models"].pop("p2_multiview", None)
        report["packing_candidate"] = True
        report["extra_qualification_minutes"] = 6
        report["cache_only_qualification_seconds"] = report.get("qualification_seconds", 0.)
        report["packing_qualification_started_epoch"] = time.time()
        report["packing_qualification_deadline_epoch"] = time.time()+360
        prior_build = report.pop("full_cache_build_started_epoch", None)
        if prior_build is not None:
            report["prior_cache_preparation_seconds"] = max(0., (folder / "packing_pause.json").stat().st_mtime-prior_build)
    report["cache_build_limit_minutes"] = build_minutes
    protected = queue.protected_files()
    started = report.get("started_epoch", time.time())
    data_loaders = []
    with ExitStack() as stack:
        for name in ("p1p2", "main_study", "nightly", "acceleration", "green_cv"):
            stack.enter_context(exclusive_output(ARTIFACTS / name / "run.lock"))
        with Budget(True) as cumulative:
            report.update(status="qualifying", started_epoch=started,
                          initial_used_seconds=report.get("initial_used_seconds", cumulative.used))
            report["parameter_comparison_policy"] = "Strict per-parameter allclose retained as diagnostic, not an additional adoption gate. Required probabilities/metrics <=0.001, exact pixels/augmentation/order, equal real updates remain mandatory."
            write_json(folder / "report.json", report)
            qualification_deadline = report.get("packing_qualification_deadline_epoch", started+minutes*60) if pack_views else started+minutes*60
            qualification = LimitedBudget(cumulative, max(0., qualification_deadline-time.time()))
            try:
                rows = load_rows(ROOT / "glaucoma_first", "train")
                audit = require_audit(ROOT / "glaucoma_first")
                folds = engine.folds_for(rows)
                fit, hold = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
                age = AgeTransform.fit([rows[i]["age"] for i in fit])
                selected = sample(rows, fit, 64, 42)+sample(rows, hold, 64, 2026)
                subset = [rows[i] for i in selected]
                if resume_qualification and report.get("sampled_names") != [r["name"] for r in subset]:
                    raise ValueError("Resumed benchmark samples changed")
                configs = {name: read_json(ROOT / "configs" / f"rfm_{name}.json") for name in ("p1_global", "p2_multiview")}
                before = time.monotonic()
                component_costs = {}
                cache = build_view_cache(subset, configs["p2_multiview"], folder / "sample", audit["data_fingerprint"], qualification, timings=component_costs)
                preparation = time.monotonic()-before
                if resume_qualification and "sample_component_seconds" in report:
                    component_costs = report["sample_component_seconds"]
                    preparation = report["sample_cache_prepare_seconds"]
                report.update(sample_cache_prepare_seconds=preparation, sample_component_seconds=component_costs,
                              sample_cache_bytes=sum(Path(p).stat().st_size for p in cache.values()),
                              sampled_names=[r["name"] for r in subset],
                              estimated_full_cache_bytes=estimated_bytes(len(rows), configs["p1_global"]["size"]))
                report["startup_policy"] = "One four-worker pool per path reused across train/validation/repeats; cold worker startup measured once per path. Completion forecast assumes two such startups per production epoch."
                # Keep all train/validation workers fixed at 4. Persistent pools amortize
                # startup between paired repeats, while cold startup is reported separately.
                for name, config in configs.items():
                    previous = report["models"].get(name, {})
                    if len(previous.get("trials", [])) == 3 and previous.get("pixels_and_seeded_augmentation_exact"):
                        for trial in previous["trials"]:
                            trial["check"]["passed"] = required_check(trial["check"])
                            trial["check"]["parameter_comparison_is_diagnostic"] = True
                        previous["summary"] = summarize(previous["trials"], config["epochs"], len(fit), len(hold))
                        write_json(folder / "report.json", report)
                        print(f"Reusing {name}: three complete paired repeats, original deadline retained", flush=True)
                        continue
                    seed_all(42)
                    initial_model = FundusModel(config, download_weights("rfm"))
                    initial = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
                    del initial_model
                    raw = image_dataset(subset, config, age, True)
                    fast = image_dataset(subset, config, age, True, cache=cache if config.get("multiview") else {"global": cache["global"]})
                    for augment in (False, True):
                        raw.augment = fast.augment = augment
                        for i in range(len(subset)):
                            qualification.check()
                            random.seed(100+i)
                            a = raw[i]
                            random.seed(100+i)
                            b = fast[i]
                            xa, xb = a[0], b[0]
                            if isinstance(xa, dict):
                                for key in xa:
                                    torch.testing.assert_close(xa[key], xb[key], rtol=0, atol=0)
                            else:
                                torch.testing.assert_close(xa, xb, rtol=0, atol=0)
                    pools = {}
                    for path, cache_paths in (("raw", None), ("cache", cache if config.get("multiview") else {"global": cache["global"]})):
                        dataset = FixedSamples(image_dataset(subset, config, age, True, cache_paths),
                                               image_dataset(subset, config, age, False, cache_paths))
                        sampler = PhaseSampler()
                        data = engine.loader(dataset, 4, 4, sampler=sampler)
                        pools[path] = [WorkloadLoader(data, sampler, mode) for mode in (False, True)]
                        data_loaders.append(data)
                    trials = []
                    report["models"][name] = {"trials": trials, "pixels_and_seeded_augmentation_exact": True}
                    for repeat in range(3):
                        qualification.check()
                        pair, checks = {}, {}
                        for path in (("raw", "cache") if repeat % 2 == 0 else ("cache", "raw")):
                            print(f"qualify {name} repeat={repeat+1} path={path}", flush=True)
                            packing = 24 if pack_views and config.get("multiview") and path == "cache" else 0
                            pair[path], checks[path] = workload(config, initial, *pools[path], age, qualification, folder / "scratch.pt", packing)
                        pair["check"] = numerical_check(checks["raw"], checks["cache"])
                        order_equal = all(pair["raw"][p]["sample_order"] == pair["cache"][p]["sample_order"] for p in ("warmup", "fine_tune"))
                        pair["check"]["training_order_equal"] = order_equal
                        pair["check"]["parameter_comparison_is_diagnostic"] = True
                        pair["check"]["passed"] = required_check(pair["check"]) and checks["raw"]["optimizer_steps"] > 0
                        trials.append(pair)
                        print(f"check {name} repeat={repeat+1}: {pair['check']}", flush=True)
                        write_json(folder / "report.json", report)
                        del checks
                    report["models"][name]["summary"] = summarize(trials, config["epochs"], len(fit), len(hold))
                    for loaders in pools.values():
                        shutdown(loaders[0].data)
                    data_loaders.clear()
                    del pools, raw, fast, initial
                report["qualification_seconds"] = (report["cache_only_qualification_seconds"] + time.time()-report["packing_qualification_started_epoch"]) if pack_views else time.time()-started
                build_estimate = preparation*len(rows)/len(subset)*1.25
                report["estimated_full_build_seconds_conservative"] = build_estimate
                savings = sum(m["summary"]["estimated_saving_seconds"] for m in report["models"].values())
                qualified = [name for name, m in report["models"].items() if m["summary"]["numerical_passed"] and m["summary"]["throughput_passed"]]
                passed = bool(qualified)
                report.update(estimated_net_saving_seconds=savings-build_estimate,
                              qualification_passed=passed, status="qualified" if passed else "rejected")
                write_json(folder / "report.json", report)
                global_estimate = component_costs["global"]*len(rows)/len(subset)*1.25
                p1_ok = "p1_global" in qualified and report["models"]["p1_global"]["summary"]["estimated_saving_seconds"] > global_estimate
                p2_cost = build_estimate - (global_estimate if p1_ok else 0)
                p2_ok = "p2_multiview" in qualified and report["models"]["p2_multiview"]["summary"]["estimated_saving_seconds"] > p2_cost
                selected_models = (["p1_global"] if p1_ok else []) + (["p2_multiview"] if p2_ok else [])
                report["selected_models_before_build"] = selected_models
                selected_saving = sum(report["models"][name]["summary"]["estimated_saving_seconds"] for name in selected_models)
                selected_cost = build_estimate if p2_ok else global_estimate
                report["estimated_net_saving_seconds"] = selected_saving-selected_cost-report["qualification_seconds"]
                if selected_models and selected_saving > selected_cost+report["qualification_seconds"]:
                    before = time.monotonic()
                    report["full_cache_build_started_epoch"] = time.time()
                    report["status"] = "building_full_cache"
                    write_json(folder / "report.json", report)
                    forecast_training = sum(report["models"][name]["summary"]["cache" if name in selected_models else "raw"]["estimated_remaining_seconds"] for name in configs)
                    remaining_budget = cumulative.limit-cumulative.used-(time.monotonic()-cumulative.started)
                    preparation_cap = min(build_minutes*60, max(0., remaining_budget-min(forecast_training, 3*3600)-900))
                    report["cache_preparation_cap_seconds"] = preparation_cap
                    report["forecast_training_seconds_after_adoption"] = forecast_training
                    report["forecast_fits_remaining_budget"] = forecast_training+selected_cost+900 <= remaining_budget
                    if preparation_cap < selected_cost:
                        raise RuntimeError("Cache preparation cannot preserve the reserved training window")
                    construction = LimitedBudget(cumulative, preparation_cap)
                    # Publish the independently qualified global profile first. If the
                    # local build reaches its cap, P1 can still use its completed cache.
                    for name in selected_models:
                        config = configs[name]
                        build_view_cache(rows, config, ARTIFACTS, audit["data_fingerprint"], construction)
                        if find_view_cache(rows, config, ARTIFACTS, audit["data_fingerprint"]):
                            profile = {"format_version": 1, "identity": profile_identity(config, "cuda"),
                                       "view_encode_batch_size": 24 if pack_views and config.get("multiview") else 0,
                                       "native_view_cache_verified": True, "stages": {phase: {"accepted": True,
                                       "settings": {"workers": 4, "batch_size": 4, "use_pixel_cache": True, "reduced_sync": False}}
                                       for phase in ("train", "validation")}}
                            write_json(folder / f"{name}_profile.json", profile)
                            report["adopted"].append(name)
                            report["full_cache_build_seconds"] = time.monotonic()-before
                            write_json(folder / "report.json", report)
                    report["status"] = "adopted"
                elif passed:
                    report["status"] = "not_built_cost_gate"
            except (TimeoutError, torch.OutOfMemoryError, RuntimeError, AssertionError, ValueError, OSError) as exc:
                report.update(status="partially_adopted" if report["adopted"] else "fallback_raw", error=f"{type(exc).__name__}: {exc}")
                print(report["error"], flush=True)
            finally:
                for data in data_loaders:
                    shutdown(data)
                scratch = folder / "scratch.pt"
                if scratch.exists():
                    scratch.unlink()
                if "full_cache_build_started_epoch" in report:
                    report["full_cache_build_seconds"] = time.time()-report["full_cache_build_started_epoch"]
                    report["total_cache_preparation_seconds"] = report["full_cache_build_seconds"]+report.get("prior_cache_preparation_seconds", 0.)
                report.update(finished_epoch=time.time(), elapsed_seconds=time.time()-started,
                              used_seconds_at_finish=cumulative.used+time.monotonic()-cumulative.started,
                              disk_free_bytes=shutil.disk_usage(ARTIFACTS).free,
                              protected_sources_unchanged=queue.protected_files() == protected)
                write_json(folder / "report.json", report)
                if not report["protected_sources_unchanged"]:
                    raise RuntimeError("Protected source/result checksums changed")
    print("Qualification/preparation finished; automatically restarting P1 -> P2", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    queue.session(hours=7, workers=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=12)
    parser.add_argument("--build-minutes", type=float, default=45)
    parser.add_argument("--resume-qualification", action="store_true")
    parser.add_argument("--pack-views", action="store_true")
    args = parser.parse_args()
    run(args.minutes, args.build_minutes, args.resume_qualification, args.pack_views)
