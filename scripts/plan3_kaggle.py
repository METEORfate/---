"""Private Kaggle plan3 jobs. Preparation is local; push is annotation-gated."""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plan3.jobs import TRAINING, relative, validate, validate_config, check_annotation_threshold

STATE = ROOT / "artifacts/kaggle_plan3"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def push_kernel(cfg, job, folder, wait_minutes=0):
    """Freeze one submission and retry only the known temporary GPU-slot rejection."""
    if type(wait_minutes) is not int or not 0 <= wait_minutes <= 180:
        raise ValueError("wait_minutes must be 0..180")
    gpu = read(folder / "kernel-metadata.json")["enable_gpu"]
    cmd = ["kaggle", "kernels", "push", "-p", str(folder), "--timeout", str(int((job["max_hours"]+1)*3600))]
    if gpu:
        cmd += ["--accelerator", cfg["accelerator"]]
    deadline = time.monotonic()+wait_minutes*60
    receipt = folder / "submission.json"
    while True:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        message = (result.stdout or "")+(result.stderr or "")
        print(message.strip(), flush=True)
        success = re.search(r"Kernel version (\d+) successfully pushed", message)
        if success:
            version = int(success.group(1))
            write(receipt, {"status": "submitted", "kernel_ref": f"{job['run_ref']}/{version}",
                            "package": read(folder / "package.json")})
            return version
        # Kaggle CLI 2.2.4 can exit zero even when the service rejects a push.
        if "Maximum batch GPU session count" not in message:
            write(receipt, {"status": "failed", "message": message.strip()})
            raise RuntimeError("Kaggle did not acknowledge a successful kernel submission")
        if time.monotonic() >= deadline:
            write(receipt, {"status": "gpu_slot_unavailable", "message": message.strip()})
            raise RuntimeError("No free Kaggle GPU session; the training job was not submitted")
        write(receipt, {"status": "waiting_for_gpu_slot", "run_ref": job["run_ref"],
                        "package": read(folder / "package.json")})
        print("Waiting 45 seconds for a free GPU session; no training submission has been accepted yet", flush=True)
        time.sleep(min(45, max(0, deadline-time.monotonic())))
        preflight(cfg, job)


def payload():
    paths = [ROOT / n for n in ("requirements-kaggle.txt", "kaggle/runner.py")]
    for folder, pattern in (("glaucoma", "*.py"), ("plan3", "*.py"), ("configs/plan3", "*.json")):
        paths.extend(sorted((ROOT / folder).rglob(pattern)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


BOOTSTRAP = '''import base64, io, sys, zipfile
from pathlib import Path
project = Path('/kaggle/working/project')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_PAYLOAD))) as z:
    for item in z.infolist():
        target = (project / item.filename).resolve()
        if not target.is_relative_to(project.resolve()) or chr(92) in item.filename or ((item.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError('Unsafe source archive')
    z.extractall(project)
sys.path.insert(0, str(project))
from plan3.cloud import main
main(JOB)
'''


def prepare(cfg, job, name):
    validate(job)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Use a lowercase hyphenated run name")
    job = copy.deepcopy(job)
    job["run_ref"] = f"{cfg['owner']}/{name}"
    job["data_dataset"] = cfg["data_dataset"]
    job["resources_dataset"] = cfg.get("plan3_resources_dataset", f"{cfg['owner']}/glaucoma-plan3-resources")
    snapshot = STATE / "resources/resources.zip"
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as z:
            manifest_bytes = z.read("resources-manifest.json")
            annotation = json.loads(z.read(job["annotations"]+"/acceptance.json"))
        job["resource_expectation"] = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "boxes_sha256": annotation["boxes_sha256"], "annotation_version": annotation["version"]}
    if job["task"] in TRAINING:
        validate_config(read(ROOT / job["config"]))
    folder = STATE / "kernels" / name
    folder.mkdir(parents=True, exist_ok=True)
    source = payload()
    (folder / "run_kaggle.py").write_text("SOURCE_PAYLOAD = " + repr(source) + "\nJOB = " + repr(job) + "\n" + BOOTSTRAP, encoding="utf-8")
    gpu = job["task"] in {"train", "locator", "ssl", "finalize", "capacity", "predict", "evaluate", "holdout", "benchmark"}
    metadata = {"id": job["run_ref"], "title": name.replace("-", " "), "code_file": "run_kaggle.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
                "dataset_sources": [cfg["data_dataset"], cfg.get("plan3_resources_dataset", f"{cfg['owner']}/glaucoma-plan3-resources")],
                "competition_sources": [], "kernel_sources": sorted({d["kernel_ref"] for d in job.get("dependencies", [])})}
    write(folder / "kernel-metadata.json", metadata)
    write(folder / "job.json", job)
    write(folder / "package.json", {"runner_sha256": sha(folder / "run_kaggle.py"),
                                   "source_sha256": hashlib.sha256(base64.b64decode(source)).hexdigest()})
    return folder


def gate(job):
    report = read(ROOT / relative(job["annotations"]) / "acceptance.json")
    check_annotation_threshold(report, job.get("annotation_policy", "full1000"))
    if (not report.get("accepted") or sha(ROOT / job["annotations"] / "boxes.csv") != report.get("boxes_sha256") or
            report.get("input_version") != "exif-original-xyxy-roi-v1" or
            report.get("coordinate_space") != "exif_transposed_original_pixels_xyxy_half_open"):
        raise ValueError("Formal Kaggle jobs are blocked until annotation acceptance passes")
    audit = read(ROOT / "artifacts/audit/report.json")
    if any(report[k] != audit[k] for k in ("data_fingerprint", "fold_fingerprint")):
        raise ValueError("Annotation data/fold fingerprint mismatch")
    if job["task"] == "evaluate":
        ticket = job.get("evaluation_ticket", {})
        ledger = read(ROOT / "artifacts/valid_milestones.json")
        milestone = ticket.get("milestone")
        if "baseline" not in ledger or ledger.get(milestone, {}).get("artifact_sha256") != ticket.get("artifact_sha256"):
            raise ValueError("Evaluation ticket does not match the current central reservation")


def preflight(cfg, job):
    """Read-only checks before any GPU launch; no download of the 33GB dataset."""
    gate(job)
    snapshot = STATE / "resources/resources.zip"
    if not snapshot.exists():
        raise ValueError("Prepare and upload the separate plan3 resource snapshot before launching")
    with zipfile.ZipFile(snapshot) as z:
        manifest = json.loads(z.read("resources-manifest.json"))
        available = set(z.namelist())
        staged_annotation = json.loads(z.read(job["annotations"]+"/acceptance.json"))
        local_annotation = read(ROOT / job["annotations"] / "acceptance.json")
        if any(staged_annotation.get(k) != local_annotation.get(k) for k in ("boxes_sha256", "version", "data_fingerprint", "fold_fingerprint")):
            raise ValueError("Resource snapshot contains a different annotation version; rebuild and upload it first")
        if job["task"] in TRAINING:
            config = read(ROOT / job["config"])
            relative(config["pretrained"], "weights")
            if config["pretrained"] not in manifest["sha256"]:
                raise ValueError("This task's pretrained manifest is missing from the resource snapshot")
            source = json.loads(z.read(config["pretrained"]))
            if sha(ROOT / config["pretrained"]) != manifest["sha256"][config["pretrained"]]:
                raise ValueError("Local pretrained manifest changed after resource preparation")
            if config.get("view", "global") != "global" or config.get("ssl"):
                if not job.get("locator") or not job.get("locator_review"):
                    raise ValueError("ROI/SSL task requires a matched locator and its accepted internal review")
    dependencies = {d["alias"]: d for d in job.get("dependencies", [])}
    required = [job[k] for k in ("initialize_from", "locator", "locator_review", "hard_negatives", "artifact", "spec", "inner_oof", "resume_from") if job.get(k)]
    required += job.get("inputs", []) + job.get("cv_models", [])
    for path in required:
        relative(path)
        if path.startswith("deps/"):
            _, alias, name = path.split("/", 2)
            dep = dependencies.get(alias)
            if not dep or not any(p == name or p.startswith(name.rstrip("/")+"/") for p in dep["files"]):
                raise ValueError(f"Task input is not supplied by a pinned dependency: {path}")
        elif path not in available:
            raise ValueError(f"Task input is not included in the frozen resources: {path}")
    # The supported API authenticates locally; credentials are never included in the payload.
    original_path = sys.path[:]
    try:
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "scripts"]
        from kaggle.api.kaggle_api_extended import KaggleApi
    finally:
        sys.path[:] = original_path
    api = KaggleApi()
    api.authenticate()
    owned = {item.ref: item for item in api.dataset_list(mine=True)}
    for reference in (cfg["data_dataset"], cfg.get("plan3_resources_dataset", f"{cfg['owner']}/glaucoma-plan3-resources")):
        item = owned.get(reference)
        if item is None or not item.is_private:
            raise ValueError(f"Dataset must exist, belong to this account and be private: {reference}")
        raw_status = api.dataset_status(reference)
        status = str(getattr(raw_status, "status", raw_status)).split(".")[-1].lower()
        if status != "ready":
            raise ValueError(f"Dataset is not ready: {reference}")
    quota = subprocess.run(["kaggle", "quota", "--csv"], capture_output=True, text=True, check=True)
    print(quota.stdout.strip())
    values = list(csv.DictReader(io.StringIO(quota.stdout)))
    gpu_task = job["task"] in {"train", "locator", "ssl", "finalize", "capacity", "predict", "evaluate", "holdout", "benchmark"}
    if gpu_task:
        remaining = next((float(r["remaining"].removesuffix("h")) for r in values if r["resource"] == "GPU"), None)
        if remaining is None or remaining < job["max_hours"]+1:
            raise ValueError("Insufficient free GPU quota for this bounded session plus publication reserve; shorten max_hours or wait for refresh")


def resources(cfg, annotations, extras=(), annotation_policy="full1000"):
    """Create a new private resource snapshot; never modify/upload the old P1/P2 dataset."""
    annotation = read(ROOT / relative(annotations) / "acceptance.json")
    gate({"annotations": annotations, "task": "train", "annotation_policy": annotation_policy})
    audit = read(ROOT / "artifacts/audit/report.json")
    if not audit.get("complete") or not audit.get("folds_ready") or audit.get("failures"):
        raise ValueError("A successful central audit is required")
    files = [(p, f"shared_audit/{p.name}") for p in (ROOT / "artifacts/audit").iterdir()
             if p.name in {"report.json", "images.jsonl", "folds.csv", "groups.csv", "failures.csv", "exact_duplicates.csv", "near_duplicates.csv"}]
    files += [(ROOT / annotations / name, f"{annotations}/{name}") for name in ("boxes.csv", "acceptance.json")]
    for name in extras:
        relative(name, "artifacts")
        if not name.startswith("artifacts/plan3_inputs/") or Path(name).suffix not in (".json", ".csv"):
            raise ValueError("Extra resources must be explicit JSON/CSV under artifacts/plan3_inputs/")
        files.append((ROOT / name, name))
    models = set()
    for config in (ROOT / "configs/plan3").glob("*.json"):
        value = read(config)
        if "pretrained" not in value:
            continue
        manifest = ROOT / relative(value["pretrained"], "weights")
        if not manifest.exists():
            continue  # Explicitly absent challenge weights stay blocked by readiness/run checks.
        source = read(manifest)
        defaults = {"rfm": "rfm-1.3.pt", "effnet": "efficientnetv2_s_in1k.pth", "green": "retfoundgreen_statedict.pth"}
        filename = source.get("filename", defaults.get(value["model"]))
        if not filename or Path(filename).name != filename:
            raise ValueError("Invalid pretrained filename")
        weight = manifest.parent / filename
        if sha(weight) != source["sha256"] or not source.get("source") or not source.get("license"):
            raise ValueError("Pretrained source/hash/license missing")
        for p in (manifest, weight):
            models.add(p)
    files += [(p, p.relative_to(ROOT).as_posix()) for p in sorted(models)]
    folder = STATE / "resources"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"sha256": {}, "data_fingerprint": audit["data_fingerprint"], "fold_fingerprint": audit["fold_fingerprint"]}
    with zipfile.ZipFile(folder / "resources.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path, name in files:
            manifest["sha256"][name] = sha(path)
            z.write(path, name)
        z.writestr("resources-manifest.json", json.dumps(manifest, indent=2))
    write(folder / "dataset-metadata.json", {"id": cfg.get("plan3_resources_dataset", f"{cfg['owner']}/glaucoma-plan3-resources"),
          "title": "Glaucoma plan3 private resources", "licenses": [{"name": "other"}]})
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "push"):
        s = sub.add_parser(command)
        s.add_argument("--job", required=True)
        s.add_argument("--name", required=True)
        if command == "push":
            s.add_argument("--wait-minutes", type=int, default=0)
    s = sub.add_parser("prepare-resources")
    s.add_argument("--annotations", default="artifacts/plan3_inputs/annotations")
    s.add_argument("--extra", action="append", default=[])
    s.add_argument("--annotation-policy", choices=("full1000", "pilot400"), default="full1000")
    s = sub.add_parser("dependency")
    s.add_argument("--archive", required=True)
    s.add_argument("--kernel", required=True)
    s.add_argument("--alias", required=True)
    s.add_argument("--output", required=True)
    for command in ("status", "pull"):
        s = sub.add_parser(command)
        s.add_argument("--name", required=True)
        s.add_argument("--version", type=int, required=True)
    args = p.parse_args()
    cfg = read(ROOT / "kaggle.local.json")
    if args.command in ("prepare", "push"):
        job = read(args.job)
        if job["task"] == "evaluate":
            job["evaluation_ticket"] = read(ROOT / relative(job["ticket"]))
        if args.command == "push":
            preflight(cfg, job)
        folder = prepare(cfg, job, args.name)
        if args.command == "push":
            push_kernel(cfg, read(folder / "job.json"), folder, args.wait_minutes)
        print(folder)
    elif args.command == "prepare-resources":
        print(resources(cfg, args.annotations, args.extra, args.annotation_policy))
    elif args.command == "dependency":
        with zipfile.ZipFile(args.archive) as z:
            manifest = json.loads(z.read("plan3-result-manifest.json"))
        if manifest["run_ref"] != args.kernel.rsplit("/", 1)[0]:
            raise ValueError("Kernel identity mismatch")
        result = {"alias": args.alias, "kernel_ref": args.kernel, "results_sha256": sha(args.archive), "files": manifest["files"]}
        write(args.output, result)
        print(args.output)
    else:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.name) or args.version < 1:
            raise ValueError("Invalid name/version")
        ref = f"{cfg['owner']}/{args.name}/{args.version}"
        if args.command == "status":
            subprocess.run(["kaggle", "kernels", "status", ref], check=True)
        else:
            output = STATE / "downloads" / args.name / str(args.version)
            if output.exists() and any(output.iterdir()):
                raise ValueError("Refusing to overwrite existing downloaded evidence")
            subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(output), "--file-pattern", r"^(results\.zip|kaggle_run\.json)$"], check=True)
            print(output)


if __name__ == "__main__":
    main()
