"""Kaggle entry point. The local launcher prepends an embedded source archive."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in item.filename:
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        z.extractall(destination)


def find_data(inputs, work):
    candidates = [p.parent for p in inputs.rglob("train.csv")
                  if all((p.parent / f"{s}.csv").is_file() and
                         (p.parent / s).is_dir() for s in ("train", "valid", "test"))]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError("Multiple data roots mounted; attach only one data dataset")
    archives = list(inputs.rglob("glaucoma_first.zip"))
    if len(archives) != 1:
        raise RuntimeError("Attach the private glaucoma data dataset first")
    with zipfile.ZipFile(archives[0]) as z:
        needed = sum(i.file_size for i in z.infolist())
    if shutil.disk_usage(work).free < needed + 10 * 1024**3:
        raise RuntimeError("Insufficient disk to extract data and retain 10 GiB for outputs")
    extract_checked(archives[0], work / "data")
    return find_data(work / "data", work)


def stage_resources(inputs, project):
    # Kaggle may expose an uploaded ZIP as expanded dataset files.
    manifests = list(inputs.rglob("resources-manifest.json"))
    archives = list(inputs.rglob("resources.zip"))
    if len(manifests) == 1:
        source = manifests[0].parent
        manifest = json.loads(manifests[0].read_text())
        for name in manifest["sha256"]:
            target = (project / name).resolve()
            origin = (source / name).resolve()
            if (not target.is_relative_to(project.resolve()) or
                    not origin.is_relative_to(source.resolve()) or "\\" in name):
                raise ValueError("Unsafe resource manifest path")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        shutil.copyfile(manifests[0], project / "resources-manifest.json")
    elif not manifests and len(archives) == 1:
        extract_checked(archives[0], project)
        manifest = json.loads((project / "resources-manifest.json").read_text())
    else:
        raise RuntimeError("Attach exactly one project resources dataset")
    for name, expected in manifest["sha256"].items():
        target = (project / name).resolve()
        if not target.is_relative_to(project.resolve()) or "\\" in name:
            raise ValueError("Unsafe resource manifest path")
        h = hashlib.sha256()
        with target.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024**2), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise RuntimeError(f"Resource hash mismatch: {name}")
    return manifest


def package_results(project, work, job, report):
    # Explicit outputs only: no raw images, pretrained weights or large pixel caches.
    paths = [project / "artifacts" / n for n in
             ("environment.json", "kaggle_run.json")]
    if job.get("initialize_from"):
        paths.append(project / job["initialize_from"])
    if job.get("output"):
        folder = (project / job["output"]).resolve()
        if not folder.is_relative_to(project / "artifacts"):
            raise ValueError("Job output must stay within artifacts/")
        paths += [p for p in folder.rglob("*") if p.is_file()]
    (project / "artifacts").mkdir(exist_ok=True)
    (project / "artifacts/kaggle_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(work / "results.zip", "w", zipfile.ZIP_STORED) as z:
        for path in paths:
            if path.is_file() and not path.name.endswith((".lock", ".tmp")):
                z.write(path, path.relative_to(project).as_posix())


def main(payload, job):
    started = time.monotonic()
    work = Path("/kaggle/working")
    project = work / "project"
    inputs = Path("/kaggle/input")
    project.mkdir(parents=True, exist_ok=True)
    extract_checked(io.BytesIO(base64.b64decode(payload)), project)
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    report = {"job": job, "status": "started"}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-q", "-r", "requirements-kaggle.txt"], check=True)
        import torch
        import torchvision
        from glaucoma.common import versions, write_json
        from glaucoma.models import FundusModel
        torch.set_num_threads(4)
        report.update(versions=versions(), gpu_count=torch.cuda.device_count(),
                      gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                      cuda_runtime=torch.version.cuda)
        write_json(project / "artifacts/environment.json", report)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU job received no CUDA device; check Kaggle quota/account settings")
        if job["mode"] == "smoke":
            # Bounded synthetic forward/backward: no competition images or training scores.
            config = json.loads((project / "configs/rfm_p1_global.json").read_text())
            weight = None
            if job.get("with_resources"):
                stage_resources(inputs, project)
                weight = project / "weights/rfm-1.3.pt"
            model = FundusModel(config, weight).configure_trainable(-1).cuda()
            x = torch.rand(1, 3, config["size"], config["size"], device="cuda")
            age = torch.zeros(1, 3, device="cuda")
            loss = model(x, age).float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            report.update(synthetic_loss=float(loss.detach()), pretrained_weights_loaded=weight is not None,
                          purpose="connectivity and implementation only")
        else:
            manifest = stage_resources(inputs, project)
            if job.get("resume_kernel"):
                prior = list(inputs.rglob("results.zip"))
                if len(prior) != 1:
                    raise RuntimeError("Attach exactly one prior kernel with results.zip")
                extract_checked(prior[0], project)
            if job.get("resume") and not (project / job["output"] / "resume.pt").is_file():
                raise RuntimeError("Resume requested but the prior kernel lacks the matching resume.pt")
            if job.get("initialize_from") and not (project / job["initialize_from"]).is_file():
                raise RuntimeError("Initialization checkpoint is absent from the prior kernel")
            data = find_data(inputs, work)
            from glaucoma.handoff import import_audit
            import_audit(data, project / "shared_audit", workers=job["workers"])
            config = json.loads((project / job["config"]).read_text())
            config["auto_pixel_cache"] = job["auto_pixel_cache"]
            write_json(project / "configs/kaggle_effective.json", config)
            # The user retired the historical cumulative RFM cap. Only this
            # session's limit and Kaggle's platform quota apply to cloud work.
            from glaucoma.engine import fine_tune
            fine_tune(data, config, project / job["output"], fold=job["fold"],
                      seed=job["seed"], workers=job["workers"], device="cuda",
                      resume=job.get("resume", False), budget=None, max_hours=job["max_hours"],
                      initialize_from=job.get("initialize_from"))
            state = json.loads((project / job["output"] / "status.json").read_text())
            report["training_complete"] = state["complete"]
        report["status"] = "complete"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (work / "kaggle_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        package_results(project, work, job, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main(SOURCE_PAYLOAD, JOB)  # Injected by scripts/kaggle.py; credentials are never embedded.
