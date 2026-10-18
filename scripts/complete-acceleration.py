"""Build cache, measure speed/equality, then resume the saved study within its deadline."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.common import ARTIFACTS, exclusive_output, read_json, sha256, write_json


def main():
    output = ARTIFACTS / "acceleration"
    with exclusive_output(output / "run.lock"):
        state = {"status": "running", "phase": "backup_checkpoint"}
        def phase(name):
            state["phase"] = name
            write_json(output / "state.json", state)
            print(f"ACCELERATION {name}", flush=True)
        def run(script, *args):
            subprocess.run([sys.executable, "-u", str(ROOT / script), *map(str, args)], cwd=ROOT,
                           check=True, stdout=sys.stdout, stderr=sys.stderr,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            main_state = read_json(ARTIFACTS / "main_study" / "state.json")
            if main_state["status"] == "running":
                raise RuntimeError("Pause the original study at a completed checkpoint first")
            deadline = main_state["deadline_epoch"]
            backup = output / "checkpoint_before"
            backup.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for name in ("model.pt", "resume.pt", "history.json", "holdout.csv"):
                source = ARTIFACTS / "green_ft2" / "fold0" / name
                shutil.copy2(source, backup / name)
                hashes[name] = sha256(source)
                if hashes[name] != sha256(backup / name):
                    raise RuntimeError("Checkpoint backup verification failed")
            write_json(output / "checkpoint_hashes.json", hashes)
            phase("build_lossless_cache")
            run("scripts/build-pixel-cache.py", "--workers", 6)
            phase("benchmark_raw_vs_cache")
            run("scripts/benchmark-training-io.py", "--count", 512)
            benchmark = read_json(output / "training_io_benchmark.json")
            cached = [r for r in benchmark["results"] if r["input"] == "cache"]
            best = max(cached, key=lambda r: r["images_per_second"])
            state.update(selected_workers=best["workers"], speedup=benchmark["cache_speedup_vs_raw"],
                         pixel_equality=benchmark["pixel_and_seeded_augmentation_equality"])
            for name in ("model.pt", "resume.pt"):
                if sha256(ARTIFACTS / "green_ft2" / "fold0" / name) != hashes[name]:
                    raise RuntimeError("A benchmark modified the original checkpoint")
            remaining = (deadline-time.time())/3600
            phase("resume_training")
            if remaining > .01:
                logs = ARTIFACTS / "main_study"
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for name in ("stdout.log", "stderr.log"):
                    if (logs / name).exists():
                        shutil.copy2(logs / name, logs / f"{stamp}_{name}")
                with (logs / "stdout.log").open("w", encoding="utf-8") as out, (logs / "stderr.log").open("w", encoding="utf-8") as err:
                    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "run-main-study.py"),
                                 "--hours", str(min(12, remaining)), "--workers", str(best["workers"])], cwd=ROOT,
                                 stdout=out, stderr=err, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                write_json(logs / "process.json", {"pid": process.pid, "started_epoch": time.time(),
                           "workers": best["workers"], "session_hours": remaining, "resumed_after_acceleration": True})
                state["resumed_pid"] = process.pid
            else:
                state["resume_reason"] = "Original session deadline reached; cache ready for the next session"
            state.update(status="complete", phase=None)
        except BaseException as exc:
            state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(output / "state.json", state)


if __name__ == "__main__":
    main()
