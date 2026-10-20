"""R1: 排序错误分析器.

对已有 OOF / holdout 预测计算:
  - 逆序对总数与 AUROC 一致性 (自校验)
  - 每个阳性被多少阴性排在其前 (难度分布)
  - 每个阴性错误超过多少阳性
  - 错误在不同分辨率族 / 年龄区间 / 亮度分箱的集中程度
  - 模型间错误样本重叠 (Green OOF vs P1 fold0, 同一子集)

输出: WB/audit/r1_ranking.json + 难例清单 csv
"""
from __future__ import annotations

import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "glaucoma_first"
OUT = Path(__file__).resolve().parent


def load_pred(path):
    d = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d[r["Image_name"]] = float(r["Probability"])
    return d


def load_labels():
    rows = list(csv.DictReader((DATA / "train.csv").open(encoding="utf-8-sig", newline="")))
    return ({r["Image_name"]: int(r["True_Label"]) for r in rows},
            {r["Image_name"]: (float(r["Age"]) if r["Age"].strip() else np.nan) for r in rows})


def load_meta():
    recs = {}
    with (ROOT / "artifacts/audit/images.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "train":
                recs[r["name"]] = r
    return recs


def inversion_profile(y, s):
    """对每个阳性统计排在它前面的阴性数量 (排序+前缀和, 避免 n_pos*n_neg 矩阵)."""
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    pos_ranks = np.flatnonzero(ys == 1)
    # 每个阳性位置之前 (rank 更小 = 分数更低) 的阴性数 = 位置 - 之前阳性数
    n_neg_below = pos_ranks - np.arange(len(pos_ranks))
    n_neg_total = int((ys == 0).sum())
    inversions = int((n_neg_total - n_neg_below).sum())
    auroc_from_inv = 1 - inversions / (len(pos_ranks) * n_neg_total)
    # 每个阴性: 排在其后的阳性数 (= 被该阴性错误超过的阳性数)
    neg_ranks = np.flatnonzero(ys == 0)
    pos_below_neg = len(pos_ranks) - np.searchsorted(pos_ranks, neg_ranks, side="right")
    return {
        "n_pos": int(len(pos_ranks)), "n_neg": n_neg_total,
        "inversions": inversions,
        "auroc_from_inversions": float(auroc_from_inv),
        "pos_neg_below_frac": (n_neg_below / n_neg_total).tolist(),
        "neg_pos_below": pos_below_neg.tolist(),
        "pos_order": pos_ranks, "neg_order": neg_ranks,
    }


def group_report(y, s, groups, name):
    out = {}
    for gval in sorted(set(groups)):
        m = groups == gval
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        prof = inversion_profile(y[m], s[m])
        out[str(gval)] = {
            "n": int(m.sum()), "positive": int(y[m].sum()),
            "auroc": float(roc_auc_score(y[m], s[m])),
            "auprc": float(average_precision_score(y[m], s[m])),
            "share_of_all_inversions": float(prof["inversions"]),
        }
    return out


def main():
    lab, age = load_labels()
    meta = load_meta()
    res = {}

    green = load_pred(ROOT / "artifacts/green_ft2/oof.csv")
    names = [n for n in green if n in lab]
    y = np.array([lab[n] for n in names])
    s = np.array([green[n] for n in names])
    prof = inversion_profile(y, s)

    res["green_ft2_oof"] = {
        "n": len(y), "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
        "inversions": prof["inversions"],
        "auroc_from_inversions": prof["auroc_from_inversions"],
        "inversion_check_delta": abs(prof["auroc_from_inversions"] - roc_auc_score(y, s)),
    }
    pnb = np.array(prof["pos_neg_below_frac"])
    res["green_ft2_oof"]["positive_difficulty"] = {
        "median_frac_neg_above": float(np.median(pnb)),
        "p90_frac_neg_above": float(np.percentile(pnb, 90)),
        "frac_pos_with_more_than_5pct_neg_above": float((pnb > 0.05).mean()),
        "top1pct_positives_carry_inversion_share": float(
            np.sort(pnb)[::-1][: max(1, len(pnb) // 100)].sum() / pnb.sum()),
    }
    npb = np.array(prof["neg_pos_below"]) / prof["n_pos"]
    res["green_ft2_oof"]["negative_difficulty"] = {
        "median_frac_pos_below": float(np.median(npb)),
        "frac_neg_above_all_positives": float((npb == 1).mean()),
        "top1pct_negatives_carry_inversion_share": float(
            np.sort(npb)[::-1][: max(1, len(npb) // 100)].sum() / npb.sum()),
    }

    # 分组: 分辨率族 / 年龄区间 / 亮度四分位
    res_key = np.array([f"{meta[n]['width']}x{meta[n]['height']}" for n in names])
    big = defaultdict(int)
    for k in res_key:
        big[k] += 1
    res_key = np.array([k if big[k] >= 800 else "other" for k in res_key])
    res["by_resolution"] = group_report(y, s, res_key, "resolution")

    ages = np.array([age[n] for n in names])
    bins = np.digitize(np.nan_to_num(ages, nan=56.8), [40, 55, 65, 75])
    res["by_age_bin"] = group_report(y, s, bins, "age")

    bright = np.array([np.frombuffer(base64.b64decode(meta[n]["thumb"]), np.uint8).mean() for n in names])
    q = np.quantile(bright, [0.25, 0.5, 0.75])
    res["by_brightness_quartile"] = group_report(y, s, np.digitize(bright, q), "brightness")

    # 模型间错误重叠: Green OOF vs P1 fold0 (同一批 fold0 holdout)
    p1 = load_pred(ROOT / "artifacts/p1p2/p1_global/fold0/holdout.csv")
    common = [n for n in p1 if n in green and n in lab]
    yc = np.array([lab[n] for n in common])
    sg = np.array([green[n] for n in common])
    sp = np.array([p1[n] for n in common])
    res["green_vs_p1_same_subset"] = {
        "n": len(yc), "positive": int(yc.sum()),
        "green_auroc": float(roc_auc_score(yc, sg)), "p1_auroc": float(roc_auc_score(yc, sp)),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(sg)), np.argsort(np.argsort(sp)))[0, 1]),
    }
    # 难例重叠: 各自最难的阳性 (被最多阴性超过的 10%)
    def hard_pos(sc, yy):
        pr = inversion_profile(yy, sc)
        frac = np.array(pr["pos_neg_below_frac"])
        idx = np.array(common)[np.array(yy == 1)][np.argsort(-frac)][: max(1, int((yy == 1).sum() * .1))]
        return set(idx.tolist())
    hg, hp = hard_pos(sg, yc), hard_pos(sp, yc)
    res["green_vs_p1_same_subset"]["hard_positive_overlap"] = len(hg & hp) / max(1, len(hg))

    # 导出最难阳性 / 最高分阴性
    pos_idx = np.array(names)[y == 1]
    frac_pos = np.array(prof["pos_neg_below_frac"])
    order = np.argsort(-frac_pos)[:50]
    rows = [{"Image_name": pos_idx[i], "score": float(s[y == 1][i]),
             "frac_neg_above": float(frac_pos[i]), "age": float(ages[y == 1][i]),
             "resolution": f"{meta[pos_idx[i]]['width']}x{meta[pos_idx[i]]['height']}"}
            for i in order]
    with (OUT / "r1_hard_positives.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    neg_idx = np.array(names)[y == 0]
    frac_neg = np.array(prof["neg_pos_below"]) / prof["n_pos"]
    order = np.argsort(-frac_neg)[:50]
    rows = [{"Image_name": neg_idx[i], "score": float(s[y == 0][i]),
             "frac_pos_below": float(frac_neg[i]), "age": float(ages[y == 0][i]),
             "resolution": f"{meta[neg_idx[i]]['width']}x{meta[neg_idx[i]]['height']}"}
            for i in order]
    with (OUT / "r1_false_positive_negatives.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (OUT / "r1_ranking.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k in
                      ("green_ft2_oof", "green_vs_p1_same_subset")}, indent=2, ensure_ascii=False))
    print("by_resolution:")
    for k, v in sorted(res["by_resolution"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {k:14s} n={v['n']:6d} pos={v['positive']:5d} auroc={v['auroc']:.4f} inv_share={v['share_of_all_inversions']:.3f}")


if __name__ == "__main__":
    main()
