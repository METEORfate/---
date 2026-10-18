from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import ARTIFACTS, digest, exclusive_output, read_json, sha256, write_csv, write_json
from .data import crop_fov, content_fingerprint, csv_fingerprints, local_fingerprint, load_rows


def inspect(row):
    path = Path(row["path"])
    record = {"key": row["key"], "split": row["split"], "name": row["name"],
              "size": row["file_size"], "mtime_ns": row["file_mtime_ns"]}
    try:
        raw = path.read_bytes()
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            record.update(width=image.width, height=image.height, format=image.format, mode=image.mode)
            # Full native-resolution decoding: draft decoding is deliberately disabled here.
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            image, cropped = crop_fov(image)
            thumb = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS).convert("L"))
        record["thumb"] = base64.b64encode(thumb.tobytes()).decode()
        coords = np.arange(32)
        basis = np.cos(np.pi * np.arange(8)[:, None] * (coords[None, :] + 0.5) / 32)
        coeff = basis @ thumb.astype(np.float32) @ basis.T
        bits = coeff.ravel()[1:] > np.median(coeff.ravel()[1:])
        record["phash"] = f"{int(''.join('1' if x else '0' for x in bits), 2):016x}"
        record.update(mean_brightness=float(thumb.mean()), cropped=cropped, ok=True)
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return record


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def make_groups(records, rows):
    good = [r for r in records if r["ok"]]
    uf = UnionFind(r["key"] for r in records)
    exact, buckets = {}, defaultdict(list)
    candidates, exact_pairs = [], []
    hashes = np.array([int(r["phash"], 16) for r in good], dtype=np.uint64)
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    for inserted, r in enumerate(good):
        h = r["sha256"]
        if h in exact:
            other = exact[h]
            uf.union(r["key"], other["key"])
            exact_pairs.append({"a": other["key"], "b": r["key"], "type": "byte_identical"})
        else:
            exact[h] = r
        value = int(r["phash"], 16)
        nearby = set()
        # Five disjoint bands ensure <=4 changed bits share at least one band.
        for band, start in enumerate((0, 13, 26, 39, 52)):
            width = 12 if band == 4 else 13
            token = (band, (value >> start) & ((1 << width) - 1))
            nearby.update(buckets[token])
        thumb = np.frombuffer(base64.b64decode(r["thumb"]), np.uint8).astype(np.int16)
        indices = np.fromiter(nearby, dtype=np.int64, count=len(nearby))
        different = hashes[indices] ^ np.uint64(value)
        if hasattr(np, "bitwise_count"):
            distances = np.bitwise_count(different)
        else:  # NumPy 1.26 compatibility without a quadratic Python inner loop.
            distances = bit_counts[different.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for j, distance in zip(indices[distances <= 4], distances[distances <= 4]):
            j, distance = int(j), int(distance)
            other = good[j]
            if other["sha256"] == h:
                continue
            old = np.frombuffer(base64.b64decode(other["thumb"]), np.uint8).astype(np.int16)
            mad = float(np.abs(thumb - old).mean())
            # Conservative similarity groups are not claimed to identify patients.
            grouped = distance <= 2 and mad <= 1.0
            if grouped:
                uf.union(r["key"], other["key"])
            candidates.append({"a": other["key"], "b": r["key"], "hamming": distance,
                               "thumbnail_mad": mad, "grouped": grouped,
                               "review": "similarity_group_not_patient_identity" if grouped else "manual_review"})
        for band, start in enumerate((0, 13, 26, 39, 52)):
            width = 12 if band == 4 else 13
            buckets[(band, (value >> start) & ((1 << width) - 1))].append(inserted)
        if inserted and inserted % 5000 == 0:
            print(f"similarity grouped {inserted}/{len(good)}; {len(candidates)} candidate pairs", flush=True)
    labels = {r["key"]: r["label"] for r in rows if r["split"] != "test"}
    conflicts = [p for p in exact_pairs if p["a"] in labels and p["b"] in labels
                 and labels[p["a"]] != labels[p["b"]]]
    return {r["key"]: uf.find(r["key"]) for r in records}, exact_pairs, candidates, conflicts


def run_audit(root, output=ARTIFACTS / "audit", workers=2, limit=None, build_folds=True):
    with exclusive_output(Path(output) / "audit.lock"):
        return _run_audit(root, output, workers, limit, build_folds)


def _run_audit(root, output, workers, limit, build_folds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = sum((load_rows(root, s) for s in ("train", "valid", "test")), [])
    selected = rows[:limit] if limit else rows
    journal = output / "images.jsonl"
    prior = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                prior[record["key"]] = record
            except json.JSONDecodeError:
                continue  # Interrupted final line is ignored, never a successful audit result.
    pending, records = [], []
    for row in selected:
        old = prior.get(row["key"])
        if old and old.get("ok") and old.get("size") == row["file_size"] and old.get("mtime_ns") == row["file_mtime_ns"]:
            records.append(old)
        else:
            pending.append(row)
    print(f"audit: {len(rows)} files, {len(records)} resumable, {len(pending)} to decode", flush=True)
    started = time.monotonic()
    with journal.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, record in enumerate(pool.map(inspect, pending), 1):
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if i % 100 == 0:
                f.flush()
                if i % 1000 == 0:
                    print(f"audit decoded {i}/{len(pending)}; {time.monotonic()-started:.1f}s", flush=True)
                    write_json(output / "progress.json", {"status": "running", "decoded": len(records),
                               "total": len(rows), "new_decodes": i, "seconds": time.monotonic()-started})
    by_key = {r["key"]: r for r in records}
    records = [by_key[r["key"]] for r in selected]
    failures = [r for r in records if not r["ok"]]
    groups, exact, nearby, conflicts = make_groups(records, selected)
    write_csv(output / "failures.csv", failures, ["key", "error"])
    write_csv(output / "exact_duplicates.csv", exact, ["a", "b", "type"])
    write_csv(output / "near_duplicates.csv", nearby, ["a", "b", "hamming", "thumbnail_mad", "grouped", "review"])
    write_csv(output / "groups.csv", [{"key": k, "group": v} for k, v in groups.items()], ["key", "group"])
    train_groups = {groups[r["key"]] for r in selected if r["split"] == "train"}
    overlap = [r["name"] for r in selected if r["split"] == "valid" and groups[r["key"]] in train_groups]
    report = {"complete": len(selected) == len(rows), "full_pixel_decode": True,
              "data_fingerprint": content_fingerprint(root, records),
              "local_fingerprint": local_fingerprint(root, rows), "csv_sha256": csv_fingerprints(root),
              "scanned": len(records), "total": len(rows),
              "failures": len(failures), "exact_label_conflicts": conflicts,
              "near_candidate_pairs": len(nearby), "exact_duplicate_pairs": len(exact),
              "valid_train_similarity_overlap": overlap,
              "patient_independence_verified": False, "folds_ready": False,
              "near_duplicate_review_complete": not any(not p["grouped"] for p in nearby),
              "group_rule": "byte equality or phash<=2 and 32x32 FOV grayscale MAD<=1; not patient identity",
              "splits": {s: {"rows": len([r for r in rows if r["split"] == s]),
                                "positive": sum(r["label"] == 1 for r in rows if r["split"] == s) if s != "test" else None,
                                "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in records if r["split"] == s and r["ok"]))}
                         for s in ("train", "valid", "test")}}
    write_json(output / "report.json", report)
    if build_folds and report["complete"] and not failures and not conflicts:
        from sklearn.model_selection import StratifiedGroupKFold
        train = [r for r in rows if r["split"] == "train"]
        y = np.array([r["label"] for r in train])
        g = np.array([groups[r["key"]] for r in train])
        folds = np.full(len(train), -1, dtype=int)
        for fold, (_, holdout) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g)):
            folds[holdout] = fold
        write_csv(output / "folds.csv", [{"Image_name": r["name"], "fold": int(f), "group": group}
                                        for r, f, group in zip(train, folds, g)], ["Image_name", "fold", "group"])
        report["folds_ready"] = True
        report["fold_fingerprint"] = sha256(output / "folds.csv")
        write_json(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("complete", "scanned", "failures", "exact_duplicate_pairs", "near_candidate_pairs")}), flush=True)
    write_json(output / "progress.json", {"status": "complete" if report["complete"] else "partial",
               "decoded": len(records), "total": len(rows), "failures": len(failures), "folds_ready": report["folds_ready"]})
    return report


def require_audit(root, audit_dir=ARTIFACTS / "audit"):
    report = read_json(Path(audit_dir) / "report.json")
    if not report["complete"] or report["failures"] or report["exact_label_conflicts"]:
        raise RuntimeError("Complete the full audit and resolve decode/label failures before training")
    if not report.get("folds_ready"):
        raise RuntimeError("Rerun audit in the training environment to finalize the fold manifest")
    if report.get("local_fingerprint") != local_fingerprint(root):
        raise RuntimeError("Local files/CSV changed or audit format is old; rerun audit or import-audit")
    if report.get("fold_fingerprint") != sha256(Path(audit_dir) / "folds.csv"):
        raise RuntimeError("Fold manifest changed since audit; restore the shared manifest")
    return report
