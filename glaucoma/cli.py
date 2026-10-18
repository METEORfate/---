from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import ARTIFACTS, Budget, ROOT, read_json, versions, write_json


def config_at(path):
    config = read_json(path)
    if config["model"] not in ("green", "effnet", "rfm") or config["size"] <= 0:
        raise ValueError("Unsupported model config")
    return config


def main(argv=None):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    parser = argparse.ArgumentParser(description="Glaucoma competition: audit, train, validate, predict")
    parser.add_argument("--data-root", type=Path, default=ROOT / "glaucoma_first")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("environment")
    p = subs.add_parser("smoke")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "green_ft2.json")
    p = subs.add_parser("audit")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, default=ARTIFACTS / "audit")
    p = subs.add_parser("import-audit")
    p.add_argument("--source", type=Path, required=True, help="Unpacked central audit folder")
    p.add_argument("--workers", type=int, default=4)
    p = subs.add_parser("handoff")
    p.add_argument("--code-only", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "handoff")
    p = subs.add_parser("download")
    p.add_argument("--model", choices=["green", "effnet", "rfm"], required=True)
    p = subs.add_parser("age-baseline")
    p.add_argument("--output", type=Path, default=ARTIFACTS / "age_baseline")
    p = subs.add_parser("features")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--performance-profile", type=Path)
    p = subs.add_parser("probe")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", choices=["0", "1", "2", "full"], default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--performance-profile", type=Path)
    p.add_argument("--validation-batch-size", type=int)
    p.add_argument("--validation-workers", type=int)
    p.add_argument("--max-hours", type=float, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--teacher", type=Path, help="Matching-fold model/ensemble for distillation")
    p.add_argument("--initialize-from", type=Path, help="Same-fold global checkpoint for multi-view warm start")
    p = subs.add_parser("finalize")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cv-dirs", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--performance-profile", type=Path)
    p.add_argument("--validation-batch-size", type=int)
    p.add_argument("--validation-workers", type=int)
    p.add_argument("--max-hours", type=float, default=10)
    p.add_argument("--resume", action="store_true")
    for command in ("predict", "evaluate"):
        p = subs.add_parser(command)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--workers", type=int, default=2)
        if command == "predict":
            p.add_argument("--split", choices=["train", "valid", "test"], default="test")
        else:
            p.add_argument("--milestone", choices=["baseline", "candidate", "delivery"], required=True)
            p.add_argument("--bootstrap", type=int, default=1000)
    p = subs.add_parser("collect-oof")
    p.add_argument("--inputs", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("fuse")
    for name in ("first-oof", "second-oof", "first-model", "second-model", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p = subs.add_parser("compare")
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("benchmark")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=200, help="Use 5855 for full-test certification")
    p = subs.add_parser("export")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--allow-unverified", action="store_true")
    p = subs.add_parser("check-submission")
    p.add_argument("--submission", type=Path, required=True)
    p = subs.add_parser("nightly")
    p.add_argument("--hours", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "environment":
        import torch
        info = versions()
        info.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
        write_json(ARTIFACTS / "environment.json", info)
        print(info)
        return
    if args.command == "audit":
        if args.limit and args.output.resolve() == (ARTIFACTS / "audit").resolve():
            parser.error("A bounded sample must use a separate --output; it must not overwrite the full audit")
        from .audit import run_audit
        run_audit(args.data_root, args.output, args.workers, args.limit)
        return
    if args.command == "download":
        from .models import download_weights
        with Budget(args.model == "rfm"):
            print(download_weights(args.model))
        return
    if args.command in ("import-audit", "handoff"):
        from .handoff import import_audit, package_handoff
        if args.command == "import-audit":
            import_audit(args.data_root, args.source, workers=args.workers)
        else:
            print(package_handoff(args.data_root, args.output, args.code_only))
        return
    from . import engine
    import torch
    torch.set_num_threads(4)
    if args.command == "age-baseline":
        engine.age_baseline(args.data_root, args.output)
    elif args.command == "smoke":
        config = config_at(args.config)
        with Budget(config["model"] == "rfm"):
            engine.smoke(args.data_root, config, args.device)
    elif args.command in ("features", "probe", "train", "finalize"):
        config = config_at(args.config)
        with Budget(config["model"] == "rfm") as budget:
            if args.command == "features":
                engine.extract_features(args.data_root, config, args.batch_size, args.workers, args.device, args.split, budget,
                                        performance_profile=args.performance_profile)
            elif args.command == "probe":
                engine.train_probe(args.data_root, config, args.features, args.output, budget)
            elif args.command == "train":
                if args.fold == "full" and args.epochs is None:
                    parser.error("Full refits need explicit --epochs selected on train folds; prefer finalize")
                fold = "full" if args.fold == "full" else int(args.fold)
                engine.fine_tune(args.data_root, config, args.output, fold, args.seed, args.epochs,
                                 args.workers, args.device, args.resume, args.teacher, budget, args.max_hours,
                                 performance_profile=args.performance_profile,
                                 validation_batch_size=args.validation_batch_size, validation_workers=args.validation_workers,
                                 initialize_from=args.initialize_from)
            else:
                import numpy as np
                if any(not read_json(d / "status.json").get("complete") for d in args.cv_dirs):
                    raise ValueError("Finish or resume all CV runs before the full refit")
                metas = [torch.load(d / "model.pt", map_location="cpu", weights_only=True) for d in args.cv_dirs]
                if {m["metadata"]["fold"] for m in metas} != {0, 1, 2} or any(m["config"] != config for m in metas):
                    raise ValueError("Finalize requires three matching-config CV models")
                from .audit import require_audit
                from .common import sha256
                current = require_audit(args.data_root)["data_fingerprint"]
                if any(m["metadata"]["data_fingerprint"] != current or
                       m["metadata"]["fold_fingerprint"] != sha256(ARTIFACTS / "audit" / "folds.csv") for m in metas):
                    raise ValueError("CV model data/fold provenance no longer matches")
                epochs = max(1, int(np.median([m["metadata"]["epoch"] for m in metas])))
                engine.fine_tune(args.data_root, config, args.output, "full", 42, epochs,
                                 args.workers, args.device, args.resume, None, budget, args.max_hours,
                                 performance_profile=args.performance_profile,
                                 validation_batch_size=args.validation_batch_size, validation_workers=args.validation_workers)
    elif args.command == "nightly":
        from .nightly import run_nightly
        run_nightly(args.data_root, args.hours, args.workers, args.device)
    else:
        from . import deploy
        if args.command == "predict":
            if args.split == "valid":
                parser.error("Use evaluate --milestone to keep official-valid access accountable")
            deploy.predict_split(args.data_root, args.model, args.split, args.output, args.batch_size, args.workers, args.device)
        elif args.command == "evaluate":
            deploy.evaluate_valid(args.data_root, args.model, args.output, args.milestone, args.batch_size, args.workers, args.device, args.bootstrap)
        elif args.command == "collect-oof":
            deploy.collect_oof(args.data_root, args.inputs, args.output)
        elif args.command == "fuse":
            deploy.fuse(args.data_root, args.first_oof, args.second_oof, args.first_model, args.second_model, args.output)
        elif args.command == "benchmark":
            deploy.benchmark(args.data_root, args.model, args.output, args.device, args.count)
        elif args.command == "export":
            deploy.export_bundle(args.model, args.output, args.benchmark, args.allow_unverified)
        elif args.command == "check-submission":
            from .data import load_rows
            print(deploy.check_submission(args.submission, load_rows(args.data_root, "test")))
        elif args.command == "compare":
            import csv
            from .metrics import metrics, paired_bootstrap
            def records(path):
                with path.open(encoding="utf-8", newline="") as f:
                    return list(csv.DictReader(f))
            first, second = records(args.first), records(args.second)
            if [(r["Image_name"], r["True_Label"]) for r in first] != [(r["Image_name"], r["True_Label"]) for r in second]:
                raise ValueError("Paired comparison requires identical names/order/labels")
            y = [int(r["True_Label"]) for r in first]
            a, b = [float(r["Probability"]) for r in first], [float(r["Probability"]) for r in second]
            result = {"first": metrics(y, a), "second": metrics(y, b), "delta_second_minus_first": paired_bootstrap(y, b, a)}
            delta = result["second"]["auroc"] - result["first"]["auroc"]
            result["requires_other_seed_confirmation"] = 0 < delta < .002
            write_json(args.output, result)

