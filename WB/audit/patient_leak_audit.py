"""A-1: 患者代理分组构建 + 现有 OOF 的跨折患者泄漏定量.

假设: train.csv/valid.csv 按患者排序, 同一患者贡献 1-2 张图, 且同患者年龄完全相同.
      因此 "连续且年龄相同的块" 可作为患者代理 (patient proxy).
检验:
  1. 块内标签一致率 vs 随机基线 (患者级标签相关性)
  2. 现有三折 (pHash 分组) 下, holdout 图像有多少比例其同患者伙伴落在训练折
  3. 泄漏子集 vs 干净子集 的 AUROC 差 (这是泄漏对分数贡献的直接估计)
输出: WB/audit/patient_leak_audit.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "glaucoma_first"
OUT = Path(__file__).resolve().parent


def load_csv(split):
    with (DATA / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def age_blocks(rows):
    """连续且年龄相同的行 -> 块; 块内按相邻两两配对为患者代理."""
    ages = [float(r["Age"]) if r["Age"].strip() else float("nan") for r in rows]
    blocks, start = [], 0
    for i in range(1, len(rows)):
        same = (ages[i] == ages[i - 1]) or (
            np.isnan(ages[i]) and np.isnan(ages[i - 1])
        )
        if not same:
            blocks.append((start, i))
            start = i
    blocks.append((start, len(rows)))
    group_of, pair_of = {}, {}
    gid = 0
    for a, b in blocks:
        for s in range(a, b, 2):
            members = list(range(s, min(s + 2, b)))
            for m in members:
                group_of[rows[m]["Image_name"]] = gid
                pair_of[rows[m]["Image_name"]] = [
                    rows[k]["Image_name"] for k in members
                ]
            gid += 1
    return blocks, group_of, pair_of


def main():
    result = {}
    train_rows = load_csv("train")
    blocks, group_of, pair_of = age_blocks(train_rows)
    y = np.array([int(r["True_Label"]) for r in train_rows])
    name_to_i = {r["Image_name"]: i for i, r in enumerate(train_rows)}

    # --- 1. 块内标签一致性 ---
    agree = n = 0
    for a, b in blocks:
        for s in range(a, b - 1, 2):
            n += 1
            agree += int(y[s] == y[s + 1])
    p = y.mean()
    base = p * p + (1 - p) ** 2
    # 二项标准误
    se = (base * (1 - base) / n) ** 0.5
    result["patient_proxy"] = {
        "n_rows": len(train_rows),
        "n_blocks": len(blocks),
        "n_proxy_patients": len(set(group_of.values())),
        "block_len_hist": {str(k): sum(1 for a, b in blocks if b - a == k)
                           for k in sorted({b - a for a, b in blocks})},
        "pair_label_agreement": agree / n,
        "random_baseline_agreement": float(base),
        "z_score": float((agree / n - base) / se),
        "pairs": n,
    }

    # --- 2/3. 现有折的患者泄漏 ---
    folds = {}
    with (ROOT / "artifacts/audit/folds.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            folds[r["Image_name"]] = int(r["fold"])

    oof = {}
    with (ROOT / "artifacts/green_ft2/oof.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            oof[r["Image_name"]] = (float(r["Probability"]), int(r["True_Label"]), int(r["fold"]))

    names = list(oof)
    yy = np.array([oof[k][1] for k in names])
    pp = np.array([oof[k][0] for k in names])
    ff = np.array([folds.get(k, -1) for k in names])
    partner_in_train = np.zeros(len(names), dtype=bool)
    has_partner = np.zeros(len(names), dtype=bool)
    for i, k in enumerate(names):
        mates = [m for m in pair_of.get(k, []) if m != k]
        if mates:
            has_partner[i] = True
            partner_in_train[i] = any(folds.get(m) != ff[i] for m in mates)

    def auc(mask):
        if mask.sum() < 50 or len(np.unique(yy[mask])) < 2:
            return None
        return {
            "auroc": float(roc_auc_score(yy[mask], pp[mask])),
            "auprc": float(average_precision_score(yy[mask], pp[mask])),
            "n": int(mask.sum()),
            "positive": int(yy[mask].sum()),
        }

    result["green_ft2_leakage"] = {
        "overall": auc(np.ones(len(names), dtype=bool)),
        "has_fellow_eye_in_train": auc(partner_in_train),
        "fellow_eye_also_holdout": auc(has_partner & ~partner_in_train),
        "no_partner": auc(~has_partner),
        "fraction_holdout_with_partner_in_train": float(partner_in_train.mean()),
        "fraction_holdout_with_partner": float(has_partner.mean()),
    }
    per_fold = {}
    for f in sorted(set(ff.tolist())):
        m = ff == f
        per_fold[str(f)] = {
            "all": auc(m),
            "leaked": auc(m & partner_in_train),
            "clean": auc(m & has_partner & ~partner_in_train),
        }
    result["green_ft2_leakage_per_fold"] = per_fold

    # --- P1 fold0 同样的检查 (单折模型) ---
    p1 = {}
    with (ROOT / "artifacts/p1p2/p1_global/fold0/holdout.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            p1[r["Image_name"]] = (float(r["Probability"]), int(r["True_Label"]))
    n1 = list(p1)
    y1 = np.array([p1[k][1] for k in n1])
    s1 = np.array([p1[k][0] for k in n1])
    leak1 = np.zeros(len(n1), dtype=bool)
    has1 = np.zeros(len(n1), dtype=bool)
    for i, k in enumerate(n1):
        mates = [m for m in pair_of.get(k, []) if m != k]
        if mates:
            has1[i] = True
            leak1[i] = any(folds.get(m) != 0 for m in mates)

    def auc1(mask, ys, ss):
        if mask.sum() < 50 or len(np.unique(ys[mask])) < 2:
            return None
        return {"auroc": float(roc_auc_score(ys[mask], ss[mask])),
                "auprc": float(average_precision_score(ys[mask], ss[mask])),
                "n": int(mask.sum()), "positive": int(ys[mask].sum())}

    result["p1_fold0_leakage"] = {
        "all": auc1(np.ones(len(n1), bool), y1, s1),
        "leaked": auc1(leak1, y1, s1),
        "clean": auc1(has1 & ~leak1, y1, s1),
        "fraction_with_partner_in_train": float(leak1.mean()),
    }

    (OUT / "patient_leak_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
