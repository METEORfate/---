from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from .jobs import relative, validate


def legacy_helpers(project):
    spec = importlib.util.spec_from_file_location("legacy_cloud_helpers", project / "kaggle/runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4*1024**2), b""):
            h.update(part)
    return h.hexdigest()


def dataset_mount(inputs, reference):
    owner, slug = reference.split("/")
    candidates = [p for p in (inputs / slug, inputs / "datasets" / owner / slug) if p.is_dir()]
    if not candidates:
        candidates = [p for p in inputs.rglob(slug) if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify a unique dataset mount: {reference}")
    return candidates[0]


def stage_dependencies(inputs, project, job, extract):
    archives = list(Path(inputs).rglob("results.zip"))
    hashes = {p: file_sha(p) for p in archives}
    for entry in job.get("dependencies", []):
        matches = [p for p, value in hashes.items() if value == entry["results_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous fixed-version dependency: {entry['alias']}")
        destination = project / "deps" / entry["alias"]
        if destination.exists():
            raise ValueError("Dependency alias already exists")
        with zipfile.ZipFile(matches[0]) as z:
            if "plan3-result-manifest.json" not in z.namelist():
                raise ValueError("Dependency lacks a plan3 identity manifest")
            manifest = json.loads(z.read("plan3-result-manifest.json"))
            if manifest.get("run_ref") != entry["kernel_ref"].rsplit("/", 1)[0]:
                raise ValueError("Dependency run identity does not match the pinned kernel")
        extract(matches[0], destination)
        for name, expected in entry["files"].items():
            relative(name, "artifacts")
            if file_sha(destination / name) != expected:
                raise ValueError("Dependency file checksum mismatch")
    if job.get("resume"):
        source = project / relative(job["resume_from"])
        target = project / relative(job["output"], "artifacts")
        if not (source / "resume.pt").is_file() or target.exists():
            raise ValueError("Resume requires an isolated previous complete-epoch result")
        shutil.copytree(source, target)


def package(project, work, job, report):
    output = project / relative(job["output"], "artifacts")
    files = [p for p in output.rglob("*") if p.is_file() and not p.name.endswith((".tmp", ".lock"))] if output.exists() else []
    manifest = {"format": "plan3-result-manifest-v1", "run_ref": job["run_ref"], "job": job,
                "status": report["status"], "files": {p.relative_to(project).as_posix(): file_sha(p) for p in files}}
    with zipfile.ZipFile(work / "results.zip", "w", zipfile.ZIP_STORED) as z:
        for path in files:
            z.write(path, path.relative_to(project).as_posix())
        z.writestr("plan3-result-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("kaggle_run.json", json.dumps(report, ensure_ascii=False, indent=2))


def main(job):
    validate(job)
    started = time.monotonic()
    work, inputs = Path("/kaggle/working"), Path("/kaggle/input")
    project = Path(__file__).resolve().parents[1]
    data_existed = (work / "data").exists()
    os.chdir(project)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    report = {"job": job, "status": "started"}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", "requirements-kaggle.txt"], check=True)
        import torch
        from glaucoma.common import versions, write_json
        from glaucoma.handoff import import_audit
        from .provenance import Context, safe_path
        from .__main__ import execute
        job["_session_deadline"] = started+job["max_hours"]*3600
        helpers = legacy_helpers(project)
        resource_mount = dataset_mount(inputs, job["resources_dataset"])
        data_mount = dataset_mount(inputs, job["data_dataset"])
        manifest = helpers.stage_resources(resource_mount, project)
        expectation = job.get("resource_expectation")
        if not expectation or file_sha(project / "resources-manifest.json") != expectation["manifest_sha256"]:
            raise ValueError("Mounted resources differ from the frozen local resource snapshot")
        stage_dependencies(inputs, project, job, helpers.extract_checked)
        root = helpers.find_data(data_mount, work)
        import_audit(root, project / "shared_audit", workers=job.get("audit_workers", 8))
        context = Context(root)
        accepted, _ = context.annotations(safe_path(job["annotations"]), job.get("annotation_policy", "full1000"))
        if accepted["boxes_sha256"] != expectation["boxes_sha256"] or accepted["version"] != expectation["annotation_version"]:
            raise ValueError("Mounted annotations are not the accepted version pinned by this job")
        if context.audit["data_fingerprint"] != manifest["data_fingerprint"]:
            raise ValueError("Resource/data identity mismatch")
        torch.set_num_threads(4)
        report.update(versions=versions(), gpu_count=torch.cuda.device_count(),
                      gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        print(json.dumps({"phase": "inputs_verified", "task": job["task"],
                          "annotation_policy": job.get("annotation_policy", "full1000"),
                          "gpu_count": report["gpu_count"], "gpus": report["gpus"]}), flush=True)
        gpu_tasks = {"train", "locator", "ssl", "finalize", "capacity", "predict", "evaluate", "holdout", "benchmark"}
        device = "cuda" if job["task"] in gpu_tasks else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Kaggle did not allocate a CUDA GPU")
        if job["task"] == "evaluate":
            ticket = job["evaluation_ticket"]
            # The local launcher reserves centrally and embeds the immutable ticket, not credentials.
            write_json(safe_path(job["ticket"]), ticket)
        execute(root, job, device)
        print(json.dumps({"phase": "task_finished", "task": job["task"]}), flush=True)
        status = project / job["output"] / "status.json"
        report.update(status="complete", training_complete=json.loads(status.read_text()).get("complete") if status.exists() else None)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        work.mkdir(parents=True, exist_ok=True)
        (work / "kaggle_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        package(project, work, job, report)
        # Publish only the explicit result archive and report. Raw data/weights must not become kernel outputs.
        if project.resolve() == (work / "project").resolve() and project.resolve().is_relative_to(work.resolve()):
            shutil.rmtree(project)
        extracted = work / "data"
        if not data_existed and extracted.is_dir() and extracted.resolve().parent == work.resolve():
            shutil.rmtree(extracted)
