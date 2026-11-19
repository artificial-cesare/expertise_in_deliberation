"""Do experts speak longer, and do they score higher than non-experts at equal length? (2026-11-19, exploratory)

Substantial utterances (>= 5 content words). Three comparisons, the first two within one setting:
  A. Eighth Amendment session: assembly experts (n = 86) vs the citizens of the same session (n = 58);
  B. UK parliamentary hearings: witnesses vs committee members (hearing_roles.py, duplicated hearing dropped, chairs out);
  C. pooled: assembly experts + witnesses vs citizens of all assemblies (participants + gender-equality table
     reports). Setting and group are confounded here, so C is descriptive only.
Length: medians and quintiles per group, Mann-Whitney, Kolmogorov-Smirnov. Score gap at equal length:
  1. OLS of the LLM score on expert + natural cubic spline of log(1 + tokens) (df = 4), HC1 SEs; in B also
     clustered by speaker;
  2. ordered logit of the score on expert + spline(log length): odds ratio for a higher score;
  3. length matching: each expert turn paired with the non-expert turn closest in log length (with replacement,
     caliper 0.1 log units, ~10% of length); mean paired difference with bootstrap CI and Wilcoxon signed-rank;
  4. stratified permutation test: strata = quintiles of length in the pooled comparison, group labels permuted
     within strata (10,000 permutations), statistic = n-weighted mean of within-stratum differences;
  5. in A, each expert speaker's mean length-adjusted residual (score residualised on spline(log length) within A).
Run from the repo root (CL env):  python reproduce/expert_gap_length_v2.py   -> outputs/paper_v2/expert_gap_length_v2.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from patsy import dmatrix
from scipy.stats import mannwhitneyu, ks_2samp, wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load
from hearing_roles import expert_corpus

OUT = Path("outputs/paper_v2/expert_gap_length_v2.json")
Y = "expertise_llm_score"; SPL = "cr(loglen, df=4, constraints='center')"
rng = np.random.default_rng(20261119)

D = load(); D = D[D.num_non_function_words >= 5]
H = expert_corpus(); H = H[~H.duplicate & H.role.isin(["witness", "member"]) & (H.num_non_function_words >= 5)].copy()
H["loglen"] = np.log1p(H.length_tokens)
cols = ["speaker", Y, "length_tokens", "loglen"]
ae = D[D.role == "expert"][cols].assign(expert=1)
c8 = D[(D.folder == "8th_amendment") & (D.role == "participant")][cols].assign(expert=0)
call = D[D.role.isin(["participant", "rapporteur"])][cols].assign(expert=0)
wit = H[H.role == "witness"][cols].assign(expert=1); mem = H[H.role == "member"][cols].assign(expert=0)
COMP = {"A_eighth_amendment": pd.concat([ae, c8]), "B_hearings": pd.concat([wit, mem]),
        "C_pooled": pd.concat([ae, wit, call])}


def length_summary(d):
    e, c = d[d.expert == 1].length_tokens, d[d.expert == 0].length_tokens
    u = mannwhitneyu(e, c)
    qs = [.1, .25, .5, .75, .9]
    return {"n_expert": len(e), "n_nonexpert": len(c), "median_expert": float(e.median()), "median_nonexpert": float(c.median()),
            "quantiles_expert": e.quantile(qs).round(1).tolist(), "quantiles_nonexpert": c.quantile(qs).round(1).tolist(),
            "mwu_p": float(u.pvalue), "rank_biserial": round(2 * u.statistic / (len(e) * len(c)) - 1, 3),  # > 0: experts longer
            "ks_p": float(ks_2samp(e, c).pvalue)}


def ols(d, cluster=False):
    f = smf.ols(f"{Y} ~ expert + {SPL}", d)
    fit = f.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.speaker)[0]}) if cluster else f.fit(cov_type="HC1")
    lo, hi = fit.conf_int().loc["expert"]
    return {"b": round(fit.params.expert, 3), "ci": [round(lo, 3), round(hi, 3)], "p": float(fit.pvalues.expert)}


def ordinal(d):
    X = dmatrix(f"{SPL} - 1", d, return_type="dataframe"); X.columns = [f"s{i}" for i in range(X.shape[1])]
    X["expert"] = d.expert.values
    fit = OrderedModel(d[Y].astype(int).values, X, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
    b, se = fit.params["expert"], fit.bse["expert"]
    return {"log_odds": round(b, 3), "odds_ratio": round(np.exp(b), 2), "or_ci": [round(np.exp(b - 1.96 * se), 2), round(np.exp(b + 1.96 * se), 2)],
            "p": float(fit.pvalues["expert"])}


def matched(d, caliper=0.1):
    e, c = d[d.expert == 1], d[d.expert == 0]
    cl = c.loglen.values; diffs, gaps = [], []
    for _, row in e.iterrows():
        j = np.argmin(np.abs(cl - row.loglen))
        if abs(cl[j] - row.loglen) <= caliper:
            diffs.append(row[Y] - c[Y].values[j]); gaps.append(row.loglen - cl[j])
    diffs = np.array(diffs)
    boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
    return {"matched": len(diffs), "of": len(e), "mean_diff": round(diffs.mean(), 3),
            "ci": [round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)],
            "wilcoxon_p": float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else None,
            "mean_abs_loglen_gap": round(float(np.mean(np.abs(gaps))), 3)}


def stratified(d, k=5, n_perm=10000):
    d = d.reset_index(drop=True); d["stratum"] = pd.qcut(d.loglen.rank(method="first"), k, labels=False)
    grp = {s: g.assign(index_pos=g.index.values) for s, g in d.groupby("stratum")}  # row positions per stratum
    lab = d.expert.values.astype(bool)
    def stat2(lab):
        tot, w = 0.0, 0
        for s, g in grp.items():
            m = lab[g.index_pos.values]; y = g[Y].values
            if m.all() or (~m).all(): continue
            tot += len(g) * (y[m].mean() - y[~m].mean()); w += len(g)
        return tot / w
    obs = stat2(lab); perm = []
    for _ in range(n_perm):
        l2 = lab.copy()
        for g in grp.values():
            idx = g.index_pos.values; l2[idx] = rng.permutation(l2[idx])
        perm.append(stat2(l2))
    per = d.groupby("stratum").apply(lambda g: pd.Series({
        "tokens": f"{int(g.length_tokens.min())}-{int(g.length_tokens.max())}", "n_expert": int(g.expert.sum()),
        "n_nonexpert": int((1 - g.expert).sum()), "llm_expert": round(g[g.expert == 1][Y].mean(), 2),
        "llm_nonexpert": round(g[g.expert == 0][Y].mean(), 2),
        "mwu_p": float(mannwhitneyu(g[g.expert == 1][Y], g[g.expert == 0][Y]).pvalue) if 0 < g.expert.sum() < len(g) else None}),
        include_groups=False)
    return {"weighted_diff": round(obs, 3), "perm_p": (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1),
            "strata": per.to_dict("index")}


S = {}
for name, d in COMP.items():
    d = d.dropna(subset=[Y]).reset_index(drop=True)
    S[name] = {"length": length_summary(d), "ols": ols(d), "ordinal": ordinal(d), "matched": matched(d),
               "stratified": stratified(d)}
    if name == "B_hearings": S[name]["ols_cluster_speaker"] = ols(d, cluster=True)
    if name == "A_eighth_amendment":
        res = smf.ols(f"{Y} ~ {SPL}", d).fit().resid
        cit = res[d.expert == 0].mean()
        S[name]["per_expert_speaker"] = {s: {"n": int(len(g)), "adj_diff_vs_citizens": round(res[g.index].mean() - cit, 2)}
                                         for s, g in d[d.expert == 1].groupby("speaker")}

OUT.write_text(json.dumps(S, indent=1, default=str))
print(json.dumps(S, indent=1, default=str))
print("->", OUT)
