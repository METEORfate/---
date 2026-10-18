"""Complete Green's existing three folds in one bounded session, then compare TRAIN OOF."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.common import ARTIFACTS, exclusive_output, read_json, write_json


def session(hours=3, workers=4):
    output = ARTIFACTS / "green_cv"
    with ExitStack() as stack:
        for name in ("green_cv", "main_study", "nightly", "acceleration"):
            stack.enter_context(exclusive_output(ARTIFACTS / name / "run.lock"))
        state_path = output / "state.json"
        state = {"status": "running", "current": None, "completed_folds": [],
                 "started_epoch": time.time(), "deadline_epoch": time.time() + hours * 3600,
                 "workers": workers, "config": "configs/green_ft2.json", "seed": 42,
                 "scope": "Green folds 1/2, pooled OOF and comparisons only; no official valid/test or full refit"}

        def save():
            state["last_update_epoch"] = time.time()
            write_json(state_path, state)

        def remaining():
            value = (state["deadline_epoch"] - time.time()) / 3600
            if value <= 0:
                raise TimeoutError("Green CV session budget reached; rerun to resume complete epochs")
            return value

        def run(*args, script="run.py"):
            remaining()
            subprocess.run([sys.executable, "-u", str(ROOT / script), *map(str, args)], cwd=ROOT, check=True,
                           stdout=sys.stdout, stderr=sys.stderr,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        save()
        try:
            state["current"] = "verify_saved_results"
            save()
            run(script="scripts/review-results.py")
            for fold in (0, 1, 2):
                target = ARTIFACTS / "green_ft2" / f"fold{fold}"
                complete = (target / "status.json").exists() and read_json(target / "status.json").get("complete")
                if not complete:
                    if fold == 0:
                        raise RuntimeError("Original development fold must already be complete")
                    state["current"] = f"green_ft2_fold{fold}"
                    save()
                    args = ["--device", "cuda", "train", "--config", ROOT / "configs" / "green_ft2.json",
                            "--fold", fold, "--seed", 42, "--output", target, "--workers", workers,
                            "--max-hours", remaining()]
                    if (target / "resume.pt").exists():
                        args.append("--resume")
                    run(*args)
                    if not read_json(target / "status.json").get("complete"):
                        state.update(status="needs_resume", current=f"green_ft2_fold{fold}")
                        save()
                        return
                state["completed_folds"].append(fold)
                save()
            state["current"] = "collect_oof"
            save()
            oof = ARTIFACTS / "green_ft2" / "oof.csv"
            run("collect-oof", "--inputs", *[ARTIFACTS / "green_ft2" / f"fold{f}" / "holdout.csv" for f in range(3)], "--output", oof)
            for baseline in ("green_probe_age", "rfm_probe_age"):
                state["current"] = f"compare_{baseline}"
                save()
                run("compare", "--first", ARTIFACTS / baseline / "oof.csv", "--second", oof,
                    "--output", ARTIFACTS / "comparisons" / f"green_ft2_vs_{baseline}_oof.json")
            state.update(status="complete", current=None, metrics=read_json(oof.with_suffix(".json"))["metrics"],
                         next_action="Review full OOF together with returned CNN evidence before candidate selection")
        except BaseException as exc:
            state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.hours <= 12 or args.workers < 0:
        parser.error("Use 0<hours<=12 and workers>=0")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    session(args.hours, args.workers)
