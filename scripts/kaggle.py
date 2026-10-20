"""Standard-library-only Kaggle packaging, launch, status and isolated output retrieval."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/kaggle"
LOCAL = ROOT / "kaggle.local.json"
AUDIT_FILES = ("report.json", "images.jsonl", "folds.csv", "groups.csv", "failures.csv",
               "exact_duplicates.csv", "near_duplicates.csv")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def cli(*args, capture=False):
    result = subprocess.run(["kaggle", *map(str, args)], check=True, text=True,
                            encoding="utf-8", capture_output=capture)
    return result.stdout if capture else None


def settings():
    if not LOCAL.exists():
        raise RuntimeError("Run scripts/kaggle.ps1 init --owner YOUR_USERNAME first")
    return read(LOCAL)


def validate_job(job):
    if job.get("mode") not in ("smoke", "train"):
        raise ValueError("mode must be smoke or train")
    if job["mode"] == "smoke":
        return
    if job.get("fold") not in (0, 1, 2):
        raise ValueError("Cloud training accepts train-only CV folds 0/1/2")
    hours = job.get("max_hours", 0)
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or not 0 < hours <= 10:
        raise ValueError("max_hours must be finite and in (0, 10]")
    if not isinstance(job.get("workers"), int) or not 0 <= job["workers"] <= 8:
        raise ValueError("workers must be 0..8")
    for key in ("config", "output"):
        value = job.get(key, "")
        p = (ROOT / value).resolve()
        prefix = ROOT / ("configs" if key == "config" else "artifacts")
        if (not value or Path(value).is_absolute() or "\\" in value or
                not p.is_relative_to(prefix) or p == prefix):
            raise ValueError(f"{key} must be a relative path within {prefix.name}/")
    if not (ROOT / job["config"]).is_file():
        raise ValueError("Config file does not exist")
    if job.get("resume") and not job.get("resume_kernel"):
        raise ValueError("resume needs a resume_kernel containing checkpoints")
    if not isinstance(job.get("seed"), int):
        raise ValueError("seed must be an integer")
    if not isinstance(job.get("auto_pixel_cache"), bool):
        raise ValueError("auto_pixel_cache must be a boolean")
    for key in ("resume_kernel",):
        if job.get(key) and not re.fullmatch(r"[\w-]+/[\w-]+(?:/\d+)?", job[key]):
            raise ValueError("Invalid kernel source")
    if job.get("initialize_from"):
        p = (ROOT / job["initialize_from"]).resolve()
        if not p.is_relative_to(ROOT / "artifacts") or "\\" in job["initialize_from"]:
            raise ValueError("initialize_from must stay within artifacts/")
        if not job.get("resume_kernel"):
            raise ValueError("initialize_from needs a prior kernel containing the checkpoint")


def source_payload():
    # Allowlist only; no .env, OAuth credentials, runtime.local.json or local artifacts.
    files = [ROOT / n for n in ("run.py", "requirements-kaggle.txt")]
    for folder, pattern in (("glaucoma", "*.py"), ("configs", "*.json")):
        files.extend(sorted((ROOT / folder).glob(pattern)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def prepare_kernel(cfg, job, name):
    validate_job(job)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Use lowercase letters, digits and hyphens for the run name")
    folder = STATE / "kernels" / name
    folder.mkdir(parents=True, exist_ok=True)
    payload = source_payload()
    preamble = "SOURCE_PAYLOAD = " + repr(payload) + "\nJOB = " + repr(job) + "\n"
    runner = (ROOT / "kaggle/runner.py").read_text(encoding="utf-8")
    # Future imports must precede the injected constants.
    runner = runner.replace("from __future__ import annotations\n", "")
    (folder / "run_kaggle.py").write_text(preamble + runner, encoding="utf-8")
    metadata = {"id": f"{cfg['owner']}/{name}", "title": name.replace("-", " "),
                "code_file": "run_kaggle.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": ([cfg["resources_dataset"]] if job.get("with_resources") else [])
                    if job["mode"] == "smoke" else [cfg["data_dataset"], cfg["resources_dataset"]],
                "competition_sources": [], "kernel_sources": [job["resume_kernel"]]
                    if job.get("resume_kernel") else []}
    write(folder / "kernel-metadata.json", metadata)
    write(folder / "job.json", job)
    write(folder / "package.json", {"runner_sha256": sha(folder / "run_kaggle.py"),
                                     "payload_sha256": hashlib.sha256(base64.b64decode(payload)).hexdigest()})
    return folder


def prepare_resources(cfg):
    folder = STATE / "datasets/resources"
    folder.mkdir(parents=True, exist_ok=True)
    report = read(ROOT / "artifacts/audit/report.json")
    if not report.get("complete") or not report.get("folds_ready") or report["failures"]:
        raise RuntimeError("A complete successful central audit is required")
    if sha(ROOT / "artifacts/audit/folds.csv") != report["fold_fingerprint"]:
        raise RuntimeError("Central fold fingerprint differs")
    files = [(ROOT / "artifacts/audit" / n, f"shared_audit/{n}") for n in AUDIT_FILES]
    weight_names = ("rfm-1.3.pt", "rfm_source.json", "rfm_LICENSE.txt")
    files += [(ROOT / "weights" / n, f"weights/{n}") for n in weight_names]
    for model in ("rfm",):
        source = read(ROOT / "weights" / f"{model}_source.json")
        filename = source.get("filename", "efficientnetv2_s_in1k.pth" if model == "effnet" else "rfm-1.3.pt")
        if sha(ROOT / "weights" / filename) != source["sha256"]:
            raise RuntimeError(f"Pretrained weight hash mismatch: {model}")
    manifest = {"sha256": {}, "data_fingerprint": report["data_fingerprint"],
                "fold_fingerprint": report["fold_fingerprint"]}
    target = folder / "resources.zip"
    temp = target.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path, name in files:
            # Hash the exact bytes written into the snapshot.
            content = path.read_bytes()
            manifest["sha256"][name] = hashlib.sha256(content).hexdigest()
            z.writestr(name, content)
        z.writestr("resources-manifest.json", json.dumps(manifest, indent=2))
    temp.replace(target)
    write(folder / "dataset-metadata.json", {"id": cfg["resources_dataset"],
          "title": "Glaucoma project resources", "licenses": [{"name": "other"}]})
    return folder


def prepare_data(cfg):
    folder = STATE / "datasets/data"
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / "glaucoma_first.zip"
    target = folder / source.name
    # One-time 33 GB copy on exFAT; raw data and source archive remain untouched.
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        if shutil.disk_usage(folder).free < source.stat().st_size + 5 * 1024**3:
            raise RuntimeError("Not enough free space for the one-time data staging copy")
        temp = target.with_suffix(".tmp")
        shutil.copyfile(source, temp)
        temp.replace(target)
    if sha(source) != sha(target):
        raise RuntimeError("Staged data archive differs; remove the staged copy and retry")
    write(folder / "dataset-metadata.json", {"id": cfg["data_dataset"],
          "title": "Glaucoma competition private data", "licenses": [{"name": "other"}]})
    return folder


def completed_upload(folder, cache_root=None):
    """Recover a completed blob reference without sending any file bytes."""
    folder = Path(folder).resolve()
    archives = list(folder.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError("Recovery requires exactly one staged ZIP archive")
    archive = archives[0]
    cache_root = Path(cache_root) if cache_root else Path(tempfile.gettempdir()) / ".kaggle/uploads"
    matches = []
    for path in cache_root.glob("*.json"):
        item = read(path)
        if (Path(item.get("path", "")).resolve() == archive and
                item.get("upload_complete") is True):
            request = item.get("start_blob_upload_request", {})
            stat = archive.stat()
            if (request.get("contentLength") != stat.st_size or
                    request.get("lastModifiedEpochSeconds") != int(stat.st_mtime)):
                raise RuntimeError("Staged archive changed after upload; refusing cached token")
            if time.time() - item.get("timestamp", 0) >= 7 * 24 * 3600:
                raise RuntimeError("Completed upload record is outside Kaggle's seven-day resume window")
            if not item.get("start_blob_upload_response", {}).get("token"):
                raise RuntimeError("Completed upload record lacks its blob reference")
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError("No unique completed upload record; no files were retransmitted")
    return matches[0]


def finalize_upload(folder, kind):
    """Fresh authentication plus metadata finalization; never uploads file bytes."""
    # This entry point shares its basename with the installed kaggle package.
    import_path = sys.path[:]
    try:
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
        from kaggle.api.kaggle_api_extended import (
            KaggleApi, KaggleCredentials, ApiCreateDatasetRequest, ApiDatasetNewFile,
        )
    finally:
        sys.path[:] = import_path
    from requests.exceptions import HTTPError
    folder = Path(folder).resolve()
    metadata = read(folder / "dataset-metadata.json")
    api = KaggleApi()
    # CLI 2.2.4 can keep an expired OAuth token through a long transfer.
    # Refresh through the SDK, without printing tokens or changing global packages.
    with api.build_kaggle_client() as client:
        credentials = KaggleCredentials.load(client=client)
        if credentials is not None:
            credentials.refresh_access_token()
    api.authenticate()
    owner, slug = metadata["id"].split("/")
    if api.get_config_value(api.CONFIG_NAME_USER) != owner:
        raise RuntimeError("Authenticated account differs from the dataset owner")
    try:
        existing = api.dataset_status(metadata["id"])
    except HTTPError as exc:
        if exc.response.status_code not in (403, 404):
            raise
        existing = None
    if existing is not None:
        status = str(getattr(existing, "status", existing)).split(".")[-1].lower()
    else:
        item = completed_upload(folder)
        uploaded = ApiDatasetNewFile()
        uploaded.token = item["start_blob_upload_response"]["token"]
        request = ApiCreateDatasetRequest()
        request.title, request.owner_slug, request.slug = metadata["title"], owner, slug
        request.license_name = metadata["licenses"][0]["name"]
        request.is_private, request.category_ids, request.files = True, [], [uploaded]
        with api.build_kaggle_client() as client:
            response = api.with_retry(client.datasets.dataset_api_client.create_dataset)(request)
        if getattr(response, "error", None):
            raise RuntimeError(response.error)
        status = "accepted_pending_processing"
    receipt = {"dataset": metadata["id"], "status": status,
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "file_bytes_retransmitted": 0}
    write(STATE / f"{kind}_upload_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def upload_dataset(folder, kind, version=False):
    if version:
        cli("datasets", "version", "-p", folder, "-m", "Project resource snapshot", "--keep-tabular")
    else:
        try:
            cli("datasets", "create", "-p", folder, "--keep-tabular")
        except subprocess.CalledProcessError:
            # Refuse to restart a multi-hour transfer. Only a completed upload
            # can be finalized, using fresh credentials and a private request.
            completed_upload(folder)
            print("File upload completed; retrying private dataset creation with fresh authentication.")
            finalize_upload(folder, kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--owner", required=True)
    sub.add_parser("check")
    for command in ("prepare", "push"):
        s = sub.add_parser(command)
        s.add_argument("--job", type=Path, default=ROOT / "kaggle/job.json")
        s.add_argument("--name", default="glaucoma-rfm-p1-fold1")
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--with-resources", action="store_true", help="Use uploaded RFM weights in the smoke check")
    for command in ("prepare-dataset", "upload", "finalize-upload"):
        s = sub.add_parser(command)
        s.add_argument("kind", choices=("data", "resources"))
        if command == "upload":
            s.add_argument("--version", action="store_true", help="Update an existing private dataset")
    for command in ("status", "pull"):
        s = sub.add_parser(command)
        s.add_argument("--name", default="glaucoma-rfm-p1-fold1")
        s.add_argument("--version", type=int)
    args = p.parse_args()
    if args.command == "init":
        if LOCAL.exists():
            print(f"Keeping existing configuration: {LOCAL}")
            return
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
            raise ValueError("Invalid Kaggle owner")
        write(LOCAL, {"owner": args.owner, "accelerator": "NvidiaTeslaT4",
                     "data_dataset": f"{args.owner}/glaucoma-competition-private-data",
                     "resources_dataset": f"{args.owner}/glaucoma-project-resources"})
        print(LOCAL)
        return
    cfg = settings()
    if args.command == "check":
        cli("--version")
        cli("datasets", "list", "--mine", "--csv")
        print(json.dumps(cfg, indent=2))
    elif args.command in ("prepare", "push"):
        if args.with_resources and not args.smoke:
            p.error("--with-resources requires --smoke")
        job = {"mode": "smoke", "with_resources": args.with_resources} if args.smoke else read(args.job)
        folder = prepare_kernel(cfg, job, args.name)
        if args.command == "push":
            if job["mode"] == "train":
                for ref in (cfg["data_dataset"], cfg["resources_dataset"]):
                    cli("datasets", "files", ref, "--page-size", "1")
            elif job.get("with_resources"):
                cli("datasets", "files", cfg["resources_dataset"], "--page-size", "1")
            # Account quota is a server decision. Limit this single bounded session.
            timeout = 600 if job["mode"] == "smoke" else int((job["max_hours"] + 1) * 3600)
            cli("kernels", "push", "-p", folder, "--accelerator", cfg["accelerator"], "--timeout", timeout)
        print(folder)
    elif args.command == "finalize-upload":
        finalize_upload(STATE / "datasets" / args.kind, args.kind)
    elif args.command in ("prepare-dataset", "upload"):
        folder = prepare_data(cfg) if args.kind == "data" else prepare_resources(cfg)
        if args.command == "upload":
            upload_dataset(folder, args.kind, args.version)
        print(folder)
    else:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.name):
            raise ValueError("Invalid run name")
        ref = f"{cfg['owner']}/{args.name}"
        if args.version:
            ref += f"/{args.version}"
        if args.command == "status":
            cli("kernels", "status", ref)
        else:
            destination = STATE / "downloads" / args.name / (str(args.version) if args.version else "latest")
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError("Download target exists; use a new run name/version or move the old download first")
            cli("kernels", "output", ref, "-p", destination,
                "--file-pattern", r"^(results\.zip|kaggle_run\.json)$")
            print(f"Isolated results (not merged into local training): {destination}")


if __name__ == "__main__":
    main()
