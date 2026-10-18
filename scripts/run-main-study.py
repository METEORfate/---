"""One bounded local experiment session; resume with the same command after review."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from glaucoma.common import ARTIFACTS, Budget, OutputBusy, exclusive_output, read_json, write_csv, write_json


def run_rfm(deadline, workers):
    import torch
    from glaucoma.engine import extract_features, train_probe
    torch.set_num_threads(4)
    config = read_json(ROOT / "configs" / "rfm_probe.json")
    with Budget(True) as cumulative:
        class CombinedBudget:
            def check(self):
                cumulative.check()
                if time.time() >= deadline:
                    raise TimeoutError("Local session time reached; RFM features can resume")
        budget = CombinedBudget()
        budget.check()
        features = extract_features(ROOT / "glaucoma_first", config, workers=workers,
                                    device="cuda", budget=budget)
        for use_age in (False, True):
            budget.check()
            output = ARTIFACTS / ("rfm_probe_age" if use_age else "rfm_probe_image")
            if not (output / "report.json").exists():
                train_probe(ROOT / "glaucoma_first", {**config, "use_age": use_age}, features, output, budget)


def session(hours, workers):
    output = ARTIFACTS / "main_study"
    with exclusive_output(output / "run.lock"):
        state_path = output / "state.json"
        state = read_json(state_path) if state_path.exists() else {"completed": []}
        deadline = time.time() + hours * 3600
        state.pop("error", None)
        state.update(status="running", current=None, session_started_epoch=time.time(), deadline_epoch=deadline)
        write_json(state_path, state)

        def run(*args, script="run.py"):
            subprocess.run([sys.executable, "-u", str(ROOT / script), *map(str, args)], cwd=ROOT,
                           check=True, stdout=sys.stdout, stderr=sys.stderr,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def step(name, function):
            if name in state["completed"]:
                return
            if time.time() >= deadline:
                raise TimeoutError("This session ended; rerun to continue saved work")
            state["current"] = name
            write_json(state_path, state)
            print(f"MAIN STUDY START {name}", flush=True)
            function()
            state["completed"].append(name)
            write_json(state_path, state)

        try:
            def wait_baseline():
                while True:
                    if time.time() >= deadline:
                        raise TimeoutError("Session expired while waiting for the baseline")
                    try:
                        with exclusive_output(ARTIFACTS / "nightly" / "run.lock"):
                            previous = read_json(ARTIFACTS / "nightly" / "state.json")
                            if previous["status"] != "complete":
                                raise RuntimeError(f"Baseline queue requires attention: {previous}")
                            return
                    except OutputBusy:
                        time.sleep(5)
            step("baseline_finished", wait_baseline)
            step("submission_checked", lambda: run("check-submission", "--submission", ARTIFACTS / "baseline" / "submission.csv"))
            def export_baseline():
                target = ARTIFACTS / "baseline_candidate"
                if not (target / "bundle.json").exists():
                    run("export", "--model", ARTIFACTS / "green_probe_age" / "model.pt", "--output", target,
                        "--benchmark", ARTIFACTS / "baseline" / "local_timing.json", "--allow-unverified")
            step("baseline_candidate_saved", export_baseline)
            comparisons = ARTIFACTS / "comparisons"
            step("green_age_oof_comparison", lambda: run("compare", "--first", ARTIFACTS / "green_probe_image" / "oof.csv",
                 "--second", ARTIFACTS / "green_probe_age" / "oof.csv", "--output", comparisons / "green_age_oof.json"))
            def green_ft():
                target = ARTIFACTS / "green_ft2" / "fold0"
                args = ["--device", "cuda", "train", "--config", "configs/green_ft2.json", "--fold", "0",
                        "--seed", "42", "--output", target, "--workers", str(workers),
                        "--max-hours", str(max(.01, min(6, (deadline-time.time()) / 3600)))]
                if (target / "resume.pt").exists():
                    args.append("--resume")
                run(*args)
                if not read_json(target / "status.json").get("complete"):
                    state["green_needs_resume"] = True
                    return
                state["green_needs_resume"] = False
                # Same images, labels and held-out fold as fine-tuning; no official valid.
                with (ARTIFACTS / "green_probe_age" / "oof.csv").open(encoding="utf-8", newline="") as f:
                    records = [r for r in csv.DictReader(f) if int(r["fold"]) == 0]
                baseline = comparisons / "green_frozen_fold0.csv"
                write_csv(baseline, records, ["Image_name", "Probability", "True_Label", "fold"])
                run("compare", "--first", baseline, "--second", target / "holdout.csv",
                    "--output", comparisons / "green_ft2_vs_frozen_fold0.json")
            if state.get("green_needs_resume") and "green_ft2_fold0" in state["completed"]:
                state["completed"].remove("green_ft2_fold0")
            step("green_ft2_fold0", green_ft)
            step("rfm_frozen_challenge", lambda: run("--rfm-worker", "--deadline", deadline, "--workers", workers,
                                                   script="scripts/run-main-study.py"))
            for variant in ("image", "age"):
                step(f"rfm_{variant}_oof_comparison", lambda v=variant: run("compare", "--first",
                     ARTIFACTS / "green_probe_age" / "oof.csv", "--second", ARTIFACTS / f"rfm_probe_{v}" / "oof.csv",
                     "--output", comparisons / f"rfm_{v}_vs_green_oof.json"))
            state.update(status="session_complete", current=None)
        except BaseException as exc:
            state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            write_json(state_path, state)
            raise
        finally:
            state["last_update_epoch"] = time.time()
            write_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rfm-worker", action="store_true")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if not 0 < args.hours <= 12 or args.workers < 0:
        parser.error("Use 0<hours<=12 and workers>=0")
    if args.rfm_worker:
        if args.deadline is None:
            parser.error("The RFM worker needs an absolute session deadline")
        run_rfm(args.deadline, args.workers)
    else:
        session(args.hours, args.workers)
