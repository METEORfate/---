"""Portable audit identity and a small, explicit colleague handoff package."""
from __future__ import annotations

import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ARTIFACTS, ROOT, exclusive_output, read_json, sha256, write_json
from .data import content_fingerprint, csv_fingerprints, load_rows, local_fingerprint


def import_audit(root, source, output=ARTIFACTS / "audit", workers=4):
    """Verify copied pixels by SHA256; reuse the central decoding and grouping work."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Import source must be a separate unpacked central audit folder")
    report = read_json(source / "report.json")
    if (not report.get("complete") or report["failures"] or report["exact_label_conflicts"]
            or not report.get("folds_ready") or not report.get("fold_fingerprint")):
        raise ValueError("Source audit must be complete, successful and contain finalized folds")
    if csv_fingerprints(root) != report["csv_sha256"]:
        raise ValueError("CSV bytes differ from central audit; use the original CSV files")
    if sha256(source / "folds.csv") != report["fold_fingerprint"]:
        raise ValueError("Shared fold manifest is damaged or modified")
    rows = sum((load_rows(root, s) for s in ("train", "valid", "test")), [])
    prior = {}
    with (source / "images.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                prior[item["key"]] = item
            except json.JSONDecodeError:
                continue
    if set(prior) != {r["key"] for r in rows} or len(rows) != report["total"]:
        raise ValueError("Image manifest coverage differs from local data")
    records = [prior[r["key"]] for r in rows]
    if not all(r["ok"] for r in records) or content_fingerprint(root, records) != report["data_fingerprint"]:
        raise ValueError("Source image manifest is damaged or incomplete")
    def verify(row):
        expected = prior[row["key"]]
        if row["file_size"] != expected["size"] or sha256(row["path"]) != expected["sha256"]:
            raise ValueError(f"Copied image differs: {row['key']}")
        return {**expected, "size": row["file_size"], "mtime_ns": row["file_mtime_ns"]}
    with exclusive_output(output / "audit.lock"):
        # Invalidate an older local certificate before touching its supporting files.
        write_json(output / "report.json", {**report, "complete": False, "folds_ready": False,
                                           "import_status": "verifying_all_image_bytes"})
        temp = output / "images.jsonl.tmp"
        with temp.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, record in enumerate(pool.map(verify, rows), 1):
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if i % 2000 == 0:
                    print(f"import-audit verified {i}/{len(rows)}", flush=True)
        temp.replace(output / "images.jsonl")
        for name in ("folds.csv", "groups.csv", "failures.csv", "exact_duplicates.csv", "near_duplicates.csv"):
            shutil.copyfile(source / name, output / name)
        report = {**report, "local_fingerprint": local_fingerprint(root, rows),
                  "import_status": "all_image_sha256_verified", "import_source": str(source)}
        write_json(output / "report.json", report)
        print(f"Imported identical data and folds: {report['data_fingerprint']}", flush=True)
    return report


def package_handoff(root, output=ROOT / "handoff", code_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in ("run.py", "README.md", "同事任务书.md", "SOURCES.md",
             "requirements.txt", "requirements-inference.txt", "requirements.lock.txt", ".gitignore")]
    for folder, pattern in (("glaucoma", "*.py"), ("configs", "*.json"), ("scripts", "*.ps1"), ("scripts", "*.py"), ("tests", "*.py")):
        files.extend(sorted((ROOT / folder).glob(pattern)))
    # No raw data, local runtime paths, tokens, caches or unrelated model weights.
    files.extend(p for p in sorted((ROOT / "weights").glob("effnet*")) if p.is_file())
    cnn_weights = ROOT / "weights" / "efficientnetv2_s_in1k.pth"
    if cnn_weights.exists():
        files.append(cnn_weights)
    result = {}
    def archive(name, inputs, relative):
        target = output / name
        temp = target.with_suffix(".tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for path in inputs:
                z.write(path, path.relative_to(relative).as_posix())
        temp.replace(target)
        result[name] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    archive("cnn_colleague.zip", files, ROOT)
    if not code_only:
        from .audit import require_audit
        report = require_audit(root)
        audit = ARTIFACTS / "audit"
        names = ("report.json", "images.jsonl", "folds.csv", "groups.csv", "failures.csv",
                 "exact_duplicates.csv", "near_duplicates.csv")
        archive("shared_audit.zip", [audit / name for name in names], audit)
        result["data_fingerprint"] = report["data_fingerprint"]
        result["fold_fingerprint"] = report["fold_fingerprint"]
    write_json(output / "manifest.json", result)
    return result
