"""S1: 元数据可预测性测试 (无图像内容).

输入只用: 年龄 / 图像尺寸 / 文件字节数 / 32x32 灰度缩略图统计 / 裁边标记 / 文件序号.
分组交叉验证使用 A-1 的患者代理分组 (年龄连续块), 避免同患者跨折.

输出: WB/audit/s1_metadata.json
"""
from __future__ import annotations

import base64
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "glaucoma_first"
OUT = Path(__file__).resolve().parent


def patient_groups(rows):
    ages = [float(r["Age"]) if r["Age"].strip() else float("nan") for r in rows]
    blocks, start = [], 0
    for i in range(1, len(rows)):
        same = (ages[i] == ages[i - 1]) or (np.isnan(ages[i]) and np.isnan(ages[i - 1]))
        if not same:
            blocks.append((start, i))
            start = i
    blocks.append((start, len(rows)))
    g = {}
    gid = 0
    for a, b in blocks:
        for s in range(a, b, 2):
            for m in range(s, min(s + 2, b)):
                g[rows[m]["Image_name"]] = gid
            gid += 1
    return g


def load_meta():
    recs = {}
    with (ROOT / "artifacts/audit/images.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["key"]] = r
    return recs


def features_for(name, row, rec, res_rank):
    a = float(row["Age"]) if row["Age"].strip() else np.nan
    age = 56.8 if np.isnan(a) else a
    thumb = np.frombuffer(base64.b64decode(rec["thumb"]), np.uint8).astype(np.float32)
    w, h = rec["width"], rec["height"]
    size = rec["size"]
    return {
        "age": age,
        "age_missing": float(np.isnan(a)),
        "width": w,
        "height": h,
        "area": w * h / 1e6,
        "aspect": w / h,
        "log_bytes": np.log(size),
        "bytes_per_px": size / (w * h),
        "brightness": float(thumb.mean()),
        "bright_std": float(thumb.std()),
        "bright_p10": float(np.percentile(thumb, 10)),
        "bright_p90": float(np.percentile(thumb, 90)),
        "bright_range": float(np.percentile(thumb, 90) - np.percentile(thumb, 10)),
        "dark_frac": float((thumb < 10).mean()),
        "cropped": float(rec.get("cropped", False)),
        "res_code": float(res_rank.get(f"{w}x{h}", -1)),
        "idx": float(int(name.split("_")[1].split(".")[0])),
    }


def evaluate(X, y, g, label, cols):
    cv = GroupKFold(5)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, g):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=40, random_state=42,
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return {"name": label, "features": cols,
            "auroc": float(roc_auc_score(y, oof)), "auprc": float(average_precision_score(y, oof)),
            "n": int(len(y)), "positive": int(y.sum())}, oof


def main():
    train_rows = list(csv.DictReader((DATA / "train.csv").open(encoding="utf-8-sig", newline="")))
    y = np.array([int(r["True_Label"]) for r in train_rows])
    gmap = patient_groups(train_rows)
    g = np.array([gmap[r["Image_name"]] for r in train_rows])
    recs = load_meta()
    res_count = Counter(f"{recs[f'train:{r['Image_name']}']['width']}x{recs[f'train:{r['Image_name']}']['height']}"
                        for r in train_rows)
    order = [k for k, _ in res_count.most_common()]
    res_rank = {k: i for i, k in enumerate(order)}

    keys, feats = [], []
    for r in train_rows:
        rec = recs[f"train:{r['Image_name']}"]
        feats.append(features_for(r["Image_name"], r, rec, res_rank))
        keys.append(r["Image_name"])
    cols = list(feats[0])
    X = np.array([[f[c] for c in cols] for f in feats], dtype=np.float64)

    results = {}
    oofs = {}
    subsets = {
        "S1-A_age": ["age", "age_missing"],
        "S1-B_image_stats": ["width", "height", "area", "aspect", "log_bytes",
                             "bytes_per_px", "brightness", "bright_std", "bright_p10",
                             "bright_p90", "bright_range", "dark_frac", "cropped", "res_code"],
        "S1-C_index": ["idx"],
        "S1-D_all": cols,
    }
    for name, sub in subsets.items():
        ix = [cols.index(c) for c in sub]
        res, oof = evaluate(X[:, ix], y, g, name, sub)
        results[name] = res
        oofs[name] = oof
        print(name, round(res["auroc"], 4), round(res["auprc"], 4), flush=True)

    # 单变量 AUROC (方向不取绝对值)
    uni = {}
    for c in cols:
        v = X[:, cols.index(c)]
        try:
            a = roc_auc_score(y, v)
        except ValueError:
            a = float("nan")
        uni[c] = float(a)
    results["univariate_auroc"] = uni

    # 分辨率族的阳性率
    res_rate = {}
    for k in order:
        m = np.array([f"{recs[f'train:{n}']['width']}x{recs[f'train:{n}']['height']}" == k for n in keys])
        if m.sum() >= 100:
            res_rate[k] = {"n": int(m.sum()), "pos_rate": float(y[m].mean())}
    results["resolution_positive_rate"] = dict(sorted(res_rate.items(), key=lambda kv: -kv[1]["n"]))
    results["n_resolution_families"] = len(order)

    np.save(OUT / "s1_oof_all.npy", oofs["S1-D_all"])
    with (OUT / "s1_keys.json").open("w", encoding="utf-8") as f:
        json.dump(keys, f)
    (OUT / "s1_metadata.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("auroc", "auprc")}
                      for k, v in results.items() if isinstance(v, dict) and "auroc" in v}, indent=2))


if __name__ == "__main__":
    main()
