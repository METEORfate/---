"""Day-one queue. Later experiments are explicit commands, never selected on public score."""
from __future__ import annotations

import time
from pathlib import Path

from .common import ARTIFACTS, OutputBusy, ROOT, read_json, write_json
from .engine import exclusive_output


def run_nightly(root, hours=10, workers=4, device="auto"):
    with exclusive_output(ARTIFACTS / "nightly" / "run.lock"):
        return _run_nightly(root, hours, workers, device)


def _run_nightly(root, hours, workers, device):
    from .audit import run_audit, require_audit
    from .engine import age_baseline, extract_features, train_probe
    from .deploy import evaluate_valid, predict_split, benchmark
    started = time.monotonic()
    output = ARTIFACTS / "nightly"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    state = read_json(state_path) if state_path.exists() else {"completed": []}
    state.update(status="running", started_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
    write_json(state_path, state)
    def step(name, function):
        if name in state["completed"]:
            return
        if time.monotonic()-started >= hours*3600:
            raise TimeoutError("Nightly session budget reached; rerun nightly to resume")
        state["current"] = name
        write_json(state_path, state)
        print(f"NIGHTLY START {name}", flush=True)
        function()
        state["completed"].append(name)
        write_json(state_path, state)
    try:
        def finish_audit():
            while True:
                try:
                    return run_audit(root, workers=workers)
                except OutputBusy:
                    if time.monotonic()-started >= hours*3600:
                        raise TimeoutError("Session expired while waiting for the existing audit")
                    time.sleep(5)
        step("full_audit", finish_audit)
        require_audit(root)
        from .handoff import package_handoff
        step("colleague_handoff", lambda: package_handoff(root))
        step("age_baseline", lambda: age_baseline(root, ARTIFACTS / "age_baseline"))
        config = read_json(ROOT / "configs" / "green_probe.json")
        feature_holder = {}
        def extract():
            path = extract_features(root, config, workers=2, device=device)
            feature_holder["path"] = path
            state["green_features"] = str(path)
        step("green_features", extract)
        features = Path(state["green_features"])
        for use_age in (False, True):
            variant = {**config, "use_age": use_age}
            folder = ARTIFACTS / ("green_probe_age" if use_age else "green_probe_image")
            step(f"green_probe_{use_age}", lambda c=variant, f=folder: train_probe(root, c, features, f))
        def select():
            from .metrics import metric_key
            candidates = [ARTIFACTS / "green_probe_image", ARTIFACTS / "green_probe_age"]
            best = max(candidates, key=lambda f: metric_key(read_json(f / "report.json")["selected"]))
            state["baseline_model"] = str(best / "model.pt")
        step("select_train_oof_baseline", select)
        model = Path(state["baseline_model"])
        step("baseline_valid", lambda: evaluate_valid(root, model, ARTIFACTS / "baseline", "baseline", workers=2, device=device))
        step("baseline_submission", lambda: predict_split(root, model, "test", ARTIFACTS / "baseline" / "submission.csv", workers=2, device=device))
        step("local_timing", lambda: benchmark(root, model, ARTIFACTS / "baseline" / "local_timing.json", device, count=50))
        state.update(status="complete", current=None)
    except BaseException as exc:
        state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        write_json(state_path, state)
        raise
    finally:
        state["session_seconds"] = time.monotonic()-started
        write_json(state_path, state)

