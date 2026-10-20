from __future__ import annotations

import argparse
from pathlib import Path
import time

from glaucoma.common import ROOT, read_json, write_json
from .jobs import validate, validate_config
from .provenance import Context, safe_path


def execute(root, job, device="cuda"):
    validate(job)
    from . import train
    train.SESSION_DEADLINE = min(job.get("_session_deadline", float("inf")), time.monotonic()+job["max_hours"]*3600)
    context = Context(root)
    # All formal plan3 tasks, including capacity probes, share the annotation gate.
    context.annotations(safe_path(job["annotations"]), job.get("annotation_policy", "full1000"))
    output = safe_path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    task = job["task"]
    if task == 'benchmark':
        from .benchmark import run
        config = validate_config(read_json(safe_path(job['config'])))
        return run(context, config, job, output, device)
    if task in ("train", "locator", "ssl", "finalize", "capacity"):
        config = validate_config(read_json(safe_path(job["config"])))
        from .train import run_training
        if task == "capacity":
            from .verification import capacity
            return capacity(context, config, job, output, device)
        if task == "train" and job.get("preflight_capacity", False):
            from .verification import capacity
            import math
            probe = capacity(context, config, job, output / "capacity", device)
            if probe["optimizer_updates"] < 1 or not math.isfinite(probe["images_per_second"]) or probe["images_per_second"] <= 0:
                raise ValueError("Capacity preflight did not complete a valid optimizer update; classification training is blocked")
            print("plan3 capacity preflight passed; starting classifier training", flush=True)
        return run_training(root, config, job, device)
    from . import results
    if task == "collect":
        return results.collect(context, [safe_path(p) for p in job["inputs"]], output / "oof.csv", job.get("outer_fold"))
    if task == "hard-negatives":
        return results.make_hard_negatives(context, safe_path(job["inner_oof"]), job["fold"], output / "hard_negatives.json")
    if task == "fuse":
        return results.fit_ensemble(context, read_json(safe_path(job["spec"])), output / "ensemble.json")
    if task == "evaluate":
        from .ledger import evaluate
        return evaluate(context, safe_path(job["artifact"]), safe_path(job["ticket"]), output, device, job.get("workers", 0), job.get("batch_size", 4))
    if task == "predict":
        return results.export_predictions(context, safe_path(job["artifact"]), "test", output / "submission.csv", device, job.get("workers", 0), job.get("batch_size", 4))
    if task == "holdout":
        from .verification import holdout
        return holdout(context, job, output, device)
    raise ValueError("Unsupported task")


def main():
    parser = argparse.ArgumentParser(description="方案3：接收标注、准备、训练与结果核验")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-annotations")
    imp.add_argument("--source", required=True)
    imp.add_argument("--spec", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--minimum", type=int, default=1000)
    imp.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    labelme = sub.add_parser("import-labelme")
    labelme.add_argument("--source", required=True)
    labelme.add_argument("--output", required=True)
    labelme.add_argument("--policy", choices=("full1000", "pilot400"), default="full1000")
    labelme.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    run = sub.add_parser("run")
    run.add_argument("--job", required=True)
    run.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    run.add_argument("--device", default="cuda")
    reserve = sub.add_parser("reserve-evaluation")
    reserve.add_argument("--artifact", required=True)
    reserve.add_argument("--milestone", choices=("candidate", "delivery"), required=True)
    reserve.add_argument("--output", required=True)
    reserve.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    accept = sub.add_parser("accept-evaluation")
    accept.add_argument("--result", required=True)
    ready = sub.add_parser("readiness")
    ready.add_argument("--annotations", default="artifacts/plan3_inputs/annotations")
    ready.add_argument("--output", default="artifacts/plan3/readiness.json")
    synthetic = sub.add_parser("synthetic-check")
    synthetic.add_argument("--output", default="artifacts/plan3/synthetic_check.json")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--first", required=True)
    comparison.add_argument("--second", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    args = parser.parse_args()
    if args.command == "synthetic-check":
        from .verification import synthetic_check
        print(synthetic_check(args.output))
    elif args.command == "compare":
        from .verification import compare
        print(compare(Context(args.data_root), args.first, args.second, args.output))
    elif args.command == "readiness":
        from .verification import readiness
        print(readiness(args.annotations, args.output))
    elif args.command == "run":
        execute(args.data_root, read_json(args.job), args.device)
    elif args.command == "import-labelme":
        from .data import convert_labelme, import_annotations
        context = Context(args.data_root)
        source, spec, fingerprint = convert_labelme(args.source, context.rows, args.output)
        report = import_annotations(source, spec, context.rows, context.folds, context.audit, args.output,
                                    300 if args.policy == "pilot400" else 1000, args.policy)
        report["source_json_fingerprint"] = fingerprint
        write_json(Path(args.output) / "acceptance.json", report)
        print({k: report[k] for k in ("accepted", "acceptance_scope", "valid", "annotated", "folds", "version")})
    elif args.command == "import-annotations":
        from .data import import_annotations
        context = Context(args.data_root)
        print(import_annotations(args.source, args.spec, context.rows, context.folds, context.audit, args.output, args.minimum))
    elif args.command == "reserve-evaluation":
        from .ledger import reserve
        reserve(Context(args.data_root), args.artifact, args.milestone, args.output)
    else:
        from .ledger import accept_result
        print(accept_result(args.result))


if __name__ == "__main__":
    main()
