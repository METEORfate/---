"""A 20-minute CPU head comparison; no official valid or test predictions."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.common import ARTIFACTS, exclusive_output, read_json, write_json


def main():
    output = ARTIFACTS / "head_ablation"
    with exclusive_output(output / "run.lock"):
        deadline = time.monotonic() + 20*60
        features = read_json(ARTIFACTS / "nightly" / "state.json")["green_features"]
        state = {"status": "running", "seeds_completed": [], "official_valid_used": False}
        def run(args):
            subprocess.run([sys.executable, "-u", *map(str, args)], cwd=ROOT, check=True,
                           stdout=sys.stdout, stderr=sys.stderr,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for seed in (42, 2026):
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU ablation session budget reached")
                state["current_seed"] = seed
                write_json(output / "state.json", state)
                target = ARTIFACTS / f"green_mlp_age_seed{seed}"
                run(["-m", "glaucoma.feature_mlp", "--features", features, "--output", target, "--seed", seed,
                     "--threads", 2, "--minutes", max(.01, (deadline-time.monotonic())/60)])
                compare = output / f"seed{seed}_vs_linear.json"
                run([ROOT / "run.py", "compare", "--first", ARTIFACTS / "green_probe_age" / "oof.csv",
                     "--second", target / "oof.csv", "--output", compare])
                result = read_json(compare)
                delta = result["second"]["auroc"] - result["first"]["auroc"]
                state["seeds_completed"].append(seed)
                state[f"seed{seed}_auroc_delta"] = delta
                write_json(output / "state.json", state)
                if seed == 42 and delta <= 0:
                    state["decision"] = "No AUROC gain; retain the linear head and do not spend a repeat seed"
                    break
            else:
                state["decision"] = "Two-seed CPU development evidence available; review before any official-valid evaluation"
            state.update(status="complete", current_seed=None)
        except BaseException as exc:
            state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(output / "state.json", state)


if __name__ == "__main__":
    main()
