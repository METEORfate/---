# -*- coding: utf-8 -*-
"""评估可行方案1：用已有 fold0 预测做定量诊断（不改动任何原始产物）。"""
import csv, os, sys, json
import numpy as np
from scipy.stats import rankdata

ROOT = r"E:/全国高校计算机能力挑战赛-大数据挑战赛"

def load(path, fold=None):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if fold is not None and int(r["fold"]) != fold:
                continue
            d[r["Image_name"]] = (float(r["Probability"]), int(r["True_Label"]))
    return d

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    r = rankdata(s)  # average ranks => ties count 0.5
    P = int((y == 1).sum()); N = int((y == 0).sum())
    D = P * N - (r[y == 1].sum() - P * (P + 1) / 2.0)
    return 1.0 - D / (P * N), D, P, N

p1 = load(os.path.join(ROOT, "artifacts/p1p2/p1_global/fold0/holdout.csv"))
green_all = load(os.path.join(ROOT, "artifacts/green_ft2/oof.csv"))
rfm_all = load(os.path.join(ROOT, "artifacts/rfm_probe_age/oof.csv"))
green = {k: v for k, v in green_all.items() if k in p1}
rfm = {k: v for k, v in rfm_all.items() if k in p1}

names = sorted(p1)
y = np.array([p1[n][1] for n in names])
P1 = np.array([p1[n][0] for n in names])
G = np.array([green[n][0] for n in names])
R = np.array([rfm[n][0] for n in names])
out = {}
out["n"] = len(names); out["P"] = int(y.sum()); out["N"] = int((1 - y).sum())

for tag, s in [("green_fold0", G), ("rfm_frozen_fold0", R), ("p1_fold0", P1)]:
    a, D, P, N = auroc(y, s)
    out[tag] = dict(auroc=round(a, 10), D=int(round(D)))

# ---------- 1) 秩平均集成：互补性经验上界 ----------
def z(s):
    return (rankdata(s) / len(s))

def blend(sigs, w=None):
    w = np.ones(len(sigs)) / len(sigs) if w is None else np.asarray(w, float)
    return sum(wi * z(s) for wi, s in zip(w, sigs))

ens = {}
ens["G+R"] = blend([G, R])
ens["G+P1"] = blend([G, P1])
ens["R+P1"] = blend([R, P1])
ens["G+R+P1"] = blend([G, R, P1])
for k, s in ens.items():
    a, D, _, _ = auroc(y, s)
    ens[k] = dict(auroc=round(a, 10), D=int(round(D)))
out["ensemble_rank_avg"] = ens

# 最优权重（在 fold0 上拟合，偏乐观）
best = None
for wg in np.arange(0, 1.01, 0.1):
    for wr in np.arange(0, 1.01 - wg + 1e-9, 0.1):
        wp = 1 - wg - wr
        a, D, _, _ = auroc(y, wg * z(G) + wr * z(R) + wp * z(P1))
        if best is None or a > best[0]:
            best = (a, (wg, wr, wp), D)
out["best_weight_G_R_P1"] = dict(auroc=round(best[0], 10), weights=best[1], D=int(round(best[2])))

# ---------- 2) 错排集中度复算 ----------
P_, N_ = out["P"], out["N"]
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
rP1 = rankdata(P1)
# 阳性错排量 = 分数高于它的阴性数（并列 0.5）；阴性错排量 = 分数低于它的阳性数（并列 0.5）
def pos_errors(s):
    ns = np.sort(s[neg_idx]); vs = s[pos_idx]
    below = np.searchsorted(ns, vs, side="left")
    equal = np.searchsorted(ns, vs, side="right") - below
    return (len(ns) - below) - 0.5 * equal
def neg_errors(s):
    ps = np.sort(s[pos_idx]); vs = s[neg_idx]
    below = np.searchsorted(ps, vs, side="left")
    equal = np.searchsorted(ps, vs, side="right") - below
    return below + 0.5 * equal
pe = pos_errors(P1)
ne = neg_errors(P1)
Dtot = pe.sum()          # 每个错排对在 pe 与 ne 各出现一次，故 D = pe.sum() = ne.sum()
assert abs(ne.sum() - Dtot) < 1e-6
order = np.argsort(-pe)
cum = np.cumsum(pe[order])
conc = {}
for k in [10, 20, 30, 50, 100]:
    conc[f"top{k}_pos"] = round(float(cum[k - 1] / Dtot) * 100, 2)
neg_order = np.argsort(-ne)
cumn = np.cumsum(ne[neg_order])
conc["top100_neg"] = round(float(cumn[99] / Dtot) * 100, 2)
out["concentration_pct"] = conc
out["D_check"] = int(round(Dtot))

# ---------- 3) "完美修正 top-k 阳性" 能到多少 ----------
hyp = {}
base_D = Dtot
for k in [10, 20, 30, 50, 80, 100, 150, 200]:
    hyp[f"fix_top{k}_pos"] = round(1 - (base_D - cum[k - 1]) / (P_ * N_), 5)
out["hypothetical_auroc_fix_top_positives"] = dict(hyp)
# 同时完美修正 top-k 阳性 + top-m 阴性（阴性需修掉其上方阳性）
for k, m in [(50, 100), (100, 200), (150, 300), (200, 400)]:
    hyp[f"fix_pos{k}_neg{m}"] = round(1 - (base_D - cum[k - 1] - cumn[m - 1]) / (P_ * N_), 5)
out["hypothetical_auroc_fix_pos_and_neg"] = dict(hyp)

# 需要净修正多少错排才到 0.99
target = 0.99
need = base_D - (1 - target) * P_ * N_
out["need_fix_pairs_for_099"] = int(round(need))
out["need_fix_pct_of_errors"] = round(float(need / base_D) * 100, 2)
out["equivalent_bottom_positives"] = round(float(need / N_), 2)
# Green -> P1 已经修正了多少
out["green_to_p1_fixed_pairs"] = int(round(out["green_fold0"]["D"] - base_D))
out["green_to_p1_fixed_pct_of_green_errors"] = round(
    float((out["green_fold0"]["D"] - base_D) / out["green_fold0"]["D"]) * 100, 2)

# ---------- 4) 难阳性是否跨模型一致难（关键机制检验）----------
# 定义 fneg(x) = 分数高于 x 的阴性占比（含并列 0.5）。对阳性即"错排比例"，跨模型可比。
def fneg_of(s):
    ns = np.sort(s[neg_idx])
    below = np.searchsorted(ns, s, side="left")
    equal = np.searchsorted(ns, s, side="right") - below
    return ((len(ns) - below) - 0.5 * equal) / len(ns)

FN = {"P1": fneg_of(P1), "G": fneg_of(G), "R": fneg_of(R)}
q80 = np.quantile(FN["P1"][pos_idx], 0.80)
hard = pos_idx[FN["P1"][pos_idx] >= q80]        # P1 最差的 20% 阳性
easy = pos_idx[FN["P1"][pos_idx] <= np.quantile(FN["P1"][pos_idx], 0.20)]
hard_stats = {"n_hard_pos": int(len(hard)), "n_easy_pos": int(len(easy)),
              "n_pos": int(len(pos_idx)),
              "hard_threshold_fneg": round(float(q80), 4)}
for k in ["P1", "G", "R"]:
    hard_stats[f"hard_pos_median_fneg_{k}"] = round(float(np.median(FN[k][hard])), 4)
for k in ["P1", "G", "R"]:
    hard_stats[f"easy_pos_median_fneg_{k}"] = round(float(np.median(FN[k][easy])), 4)
# 三个模型都判为"落在 40% 阴性之下"的阳性比例
alllow = np.mean([(FN["P1"][i] > 0.4) and (FN["G"][i] > 0.4) and (FN["R"][i] > 0.4) for i in pos_idx])
hard_stats["frac_pos_fneg>0.4_in_all3"] = round(float(alllow), 4)
# 至少一个模型把它排到阴性中位数之上
anyhigh = np.mean([(FN["P1"][i] < 0.5) or (FN["G"][i] < 0.5) or (FN["R"][i] < 0.5) for i in pos_idx])
hard_stats["frac_pos_better_than_negmedian_in_any"] = round(float(anyhigh), 4)
# 最差 100 阳性在三模型下的中位 fneg（越大越差）
worst100 = pos_idx[np.argsort(-FN["P1"][pos_idx])[:100]]
for k in ["P1", "G", "R"]:
    hard_stats[f"worst100_median_fneg_{k}"] = round(float(np.median(FN[k][worst100])), 4)
# 最差 100 阳性的跨模型重叠（各自按 fneg 排序取最差 100）
sets = {k: set(pos_idx[np.argsort(-FN[k][pos_idx])[:100]].tolist()) for k in ["P1", "G", "R"]}
hard_stats["worst100_overlap_P1_G"] = len(sets["P1"] & sets["G"])
hard_stats["worst100_overlap_P1_R"] = len(sets["P1"] & sets["R"])
hard_stats["worst100_overlap_all3"] = len(sets["P1"] & sets["G"] & sets["R"])
# 三模型两两 Spearman（阳性内部）
from scipy.stats import spearmanr as _sp
for a, b in [("P1", "G"), ("P1", "R"), ("G", "R")]:
    rho = _sp(FN[a][pos_idx], FN[b][pos_idx]).statistic
    hard_stats[f"spearman_pos_{a}_vs_{b}"] = round(float(rho), 4)
out["hard_positive_consistency"] = hard_stats

# ---------- 5) 图像分辨率/文件大小 与 难易 的关系（抽样，控制耗时）----------
from PIL import Image
rng = np.random.default_rng(0)
q80 = np.quantile(FN["P1"][pos_idx], 0.80)
q20 = np.quantile(FN["P1"][pos_idx], 0.20)
sample_pos = pos_idx
sample_neg = rng.choice(neg_idx, size=min(1200, len(neg_idx)), replace=False)
sample_idx = np.concatenate([sample_pos, sample_neg])
size_info = {}
for j in sample_idx:
    p = os.path.join(ROOT, "glaucoma_first/train", names[j])
    try:
        with Image.open(p) as im:
            size_info[j] = (im.size[0], im.size[1], os.path.getsize(p))
    except Exception:
        size_info[j] = (0, 0, 0)
ok_idx = np.array([j for j in sample_idx if size_info[j][0] > 0])
px = np.array([size_info[j][0] * size_info[j][1] for j in ok_idx], float)
fs = np.array([size_info[j][2] for j in ok_idx], float)
FNv = {"P1": FN["P1"], "G": FN["G"], "R": FN["R"]}
res = {}
for tag, sel in [("hard_pos(fneg>=q80)", [j for j in ok_idx if y[j] == 1 and FNv["P1"][j] >= q80]),
                 ("mid_pos", [j for j in ok_idx if y[j] == 1 and q20 < FNv["P1"][j] < q80]),
                 ("easy_pos(fneg<=q20)", [j for j in ok_idx if y[j] == 1 and FNv["P1"][j] <= q20]),
                 ("all_neg", [j for j in ok_idx if y[j] == 0])]:
    if len(sel) == 0:
        res[tag] = dict(n=0); continue
    a = np.array(sel)
    res[tag] = dict(n=int(len(sel)),
                    med_megapixel=round(float(np.median(px[np.isin(ok_idx, a)])) / 1e6, 2),
                    med_filesize_KB=round(float(np.median(fs[np.isin(ok_idx, a)])) / 1024, 1),
                    med_width=int(np.median([size_info[j][0] for j in sel])),
                    med_height=int(np.median([size_info[j][1] for j in sel])))
out["image_quality_vs_difficulty"] = res

# 分辨率是否与 P1 分数相关（阳性内部，抽样）
from scipy.stats import spearmanr
pp = [j for j in ok_idx if y[j] == 1]
nn = [j for j in ok_idx if y[j] == 0]
if len(pp) > 10:
    r1 = spearmanr([size_info[j][0] * size_info[j][1] for j in pp], P1[pp])
    out["spearman_megapixel_vs_p1score_within_pos"] = dict(rho=round(float(r1.statistic), 4), p=float(r1.pvalue))
r2 = spearmanr([size_info[j][0] * size_info[j][1] for j in nn], P1[nn])
out["spearman_megapixel_vs_p1score_within_neg"] = dict(rho=round(float(r2.statistic), 4), p=float(r2.pvalue))

from scipy.stats import mannwhitneyu
hp = [size_info[j][0] * size_info[j][1] for j in ok_idx if y[j] == 1 and FNv["P1"][j] >= q80]
ep = [size_info[j][0] * size_info[j][1] for j in ok_idx if y[j] == 1 and FNv["P1"][j] <= q20]
if hp and ep:
    u = mannwhitneyu(hp, ep, alternative="two-sided")
    out["mannwhitney_hardpos_vs_easypos_megapixel"] = dict(
        n_hard=len(hp), n_easy=len(ep),
        med_hard=round(float(np.median(hp)) / 1e6, 2),
        med_easy=round(float(np.median(ep)) / 1e6, 2),
        p=round(float(u.pvalue), 5))

with open(os.path.join(ROOT, "WB/assess_plan1_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
