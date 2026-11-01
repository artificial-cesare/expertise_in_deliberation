"""Every statistic quoted in the revised paper, on the revised corpus.

Revised corpus: the ageing-population transcript (data/assemblies/processed_files/ageing_population/)
is no longer part of "Constitutional Reform". All three assemblies' LLM scores are used.
Writes outputs/paper_v2/stats.json and prints a readable summary.
Run from the repo root:  python reproduce/paper_v2_stats.py
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import cohen_kappa_score

ROOT = Path("data/assemblies/processed_files")
ASM = {"8th_amendment": "Constitutional Reform", "gendereq": "Gender Equality", "scotland": "Public Policy"}
FEATS = ["jargon_density", "lexical_sophistication", "syntactic_complexity", "reasoning_markers", "evidential_markers",
         "length_tokens", "num_non_function_words", "hedge_density", "booster_density", "epistemic_balance",
         "specificity_length_norm", "specificity_mean_word_len_norm", "specificity_numbers_norm",
         "specificity_content_prop", "overall_score", "distance_base", "similarity_base",
         "distance_finetuned", "similarity_finetuned", "expertise_llm_score"]
EXPERTS = {"DR. WALSH", "DR. DOYLE", "PROFESSOR MADDEN", "DR. DUNN", "DR. FORAN"}  # Expert Advisory Group / medical experts


def role(label: str) -> str:
    x = str(label).upper().strip()
    if x in EXPERTS: return "expert"
    if x == "MS. FINEGAN": return "secretariat"
    if re.search(r"^CHAIR|CATHERINE DAY|GENERAL ANNOUNCEMENT", x): return "chair"
    if re.search(r"FACILITATOR|^F ?\d*$|^F\d$", x): return "facilitator"
    if re.search(r"^(BREAKOUT GROUP|TABLE) ?\d+$", x): return "table spokesperson"
    if re.search(r"CITIZEN|PARTICIPANT|ASSEMBLY MEMBER|^P ?\d*$|^R$|^TABLE \d+ - (?!FACILITATOR)|^(JORDAN|DAVE|FRANK|FIONNUALA|WILLIAM O'HALLORAN)$", x):
        return "participant"
    return "unknown"


def load() -> pd.DataFrame:
    parts = []
    for folder, name in ASM.items():
        for f in sorted((ROOT / folder).glob("*_speeches.csv")):
            d = pd.read_csv(f)
            k = len(d) // 3
            d = d.assign(assembly=name, file=f.name, pos=np.arange(len(d)) / len(d),
                         window=["start"] * k + ["middle"] * k + ["end"] * (len(d) - 2 * k))
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["role"] = d.speaker.map(role)
    d["words"] = d.text.astype(str).str.split().str.len()
    d["loglen"] = np.log1p(d.length_tokens)
    return d


def r_p(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 4 or x[m].nunique() < 2: return None
    r, p = pearsonr(x[m], y[m]); return {"r": round(r, 3), "p": float(p), "n": int(m.sum())}


def residual(y, covars):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covars])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b


def partial(df, a, b, with_assembly=True):
    x = df[[a, b, "loglen", "assembly"]].dropna()
    cov = [x.loglen] + ([x.assembly == v for v in sorted(x.assembly.unique())[1:]] if with_assembly else [])
    r, p = pearsonr(residual(x[a].values, cov), residual(x[b].values, cov))
    return {"r": round(r, 3), "p": float(p), "n": len(x)}


def main():
    D = load(); S = {}
    sub = {thr: D[D.num_non_function_words >= thr] for thr in (5, 6)}  # >=5 and >5
    s5 = sub[5]
    # Tables 1, 3, 6
    S["table1"] = {a: {"total": int((D.assembly == a).sum()), "substantial": int((s5.assembly == a).sum()),
                       "pct": round(100 * (s5.assembly == a).sum() / (D.assembly == a).sum(), 1),
                       "avg_words": round(s5.loc[s5.assembly == a, "words"].mean(), 1)} for a in ASM.values()}
    S["table1"]["total"] = {"total": len(D), "substantial": len(s5)}
    S["table3_6"] = {a: {f: f"{g[f].mean():.3f} ± {g[f].std():.3f}" for f in FEATS} for a, g in s5.groupby("assembly")}
    # Correlations (pooled), both thresholds
    for thr, lab in [(5, "ge5"), (6, "gt5")]:
        s = sub[thr]
        S[f"corr_{lab}"] = {t: {f: r_p(s[t], s[f]) for f in FEATS if f != t}
                            for t in ["expertise_llm_score", "overall_score", "similarity_base", "similarity_finetuned"]}
        S[f"corr_{lab}"]["dist_sim_base"] = r_p(s.distance_base, s.similarity_base)
        S[f"corr_{lab}"]["dist_sim_finetuned"] = r_p(s.distance_finetuned, s.similarity_finetuned)
        S[f"corr_{lab}"]["loglen"] = {t: r_p(s[t], s.loglen) for t in ["expertise_llm_score", "overall_score"]}
        S[f"partial_{lab}"] = {f: {"len": partial(s, "expertise_llm_score", f, False),
                                   "len_asm": partial(s, "expertise_llm_score", f, True)}
                               for f in FEATS if f not in ("expertise_llm_score", "length_tokens")}
        S[f"partial_argqa_{lab}"] = {f: partial(s, "overall_score", f, True) for f in
                                     ["specificity_mean_word_len_norm", "jargon_density", "syntactic_complexity", "lexical_sophistication"]}
        S[f"within_{lab}"] = {a: {f: r_p(g.expertise_llm_score, g[f]) for f in FEATS if f != "expertise_llm_score"}
                              for a, g in s.groupby("assembly")}
        # Variance explained
        x = s[["expertise_llm_score", "loglen", "assembly"]].dropna()
        def r2(cols):
            X = np.column_stack([np.ones(len(x))] + cols); y = x.expertise_llm_score.values
            b, *_ = np.linalg.lstsq(X, y, rcond=None); return round(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3)
        dums = [(x.assembly == v).astype(float).values for v in sorted(x.assembly.unique())[1:]]
        S[f"r2_{lab}"] = {"assembly": r2(dums), "loglen": r2([x.loglen.values]), "both": r2(dums + [x.loglen.values])}
    # Speaker roles (>=5)
    g = s5.groupby(["assembly", "role"])
    S["roles"] = g.agg(n=("expertise_llm_score", "size"), llm_mean=("expertise_llm_score", "mean"),
                       argqa_mean=("overall_score", "mean"), median_tokens=("length_tokens", "median")).round(2).reset_index().to_dict("records")
    c = s5[s5.assembly == "Constitutional Reform"]
    e, p_ = c[c.role == "expert"], c[c.role == "participant"]
    u = mannwhitneyu(e.expertise_llm_score, p_.expertise_llm_score)
    xx = c[c.role.isin(["expert", "participant"])]
    res_y = residual(xx.expertise_llm_score.values, [xx.loglen])
    S["known_groups"] = {"expert_n": len(e), "participant_n": len(p_), "expert_mean": round(e.expertise_llm_score.mean(), 2),
                         "participant_mean": round(p_.expertise_llm_score.mean(), 2), "mwu_p": float(u.pvalue),
                         "rank_biserial": round(1 - 2 * u.statistic / (len(e) * len(p_)), 3),
                         "length_adjusted_gap": round(res_y[(xx.role == "expert").values].mean() - res_y[(xx.role == "participant").values].mean(), 2),
                         "expert_argqa": round(e.overall_score.mean(), 2), "participant_argqa": round(p_.overall_score.mean(), 2)}
    # Score distributions
    S["llm_dist"] = s5.groupby("assembly").expertise_llm_score.value_counts(normalize=True).round(3).unstack().fillna(0).to_dict("index")
    # Temporal (per-transcript thirds, pooled), with and without filter, BH-FDR within each
    for thr, lab in [(0, "nofilter"), (5, "ge5")]:
        rows = []
        for a, gg in D[D.num_non_function_words >= thr].groupby("assembly"):
            for f in FEATS:
                grp = [gg.loc[gg.window == w, f].dropna() for w in ("start", "middle", "end")]
                if pd.concat(grp).nunique() < 2: continue
                H, p = kruskal(*grp); n = sum(map(len, grp)); m = [x.mean() for x in grp]
                rows.append({"assembly": a, "feature": f, "n": n, "H": H, "p": p, "eps2": H / (n - 1),
                             "mean_start": m[0], "mean_middle": m[1], "mean_end": m[2],
                             "trend": "up" if m[0] < m[1] < m[2] else "down" if m[0] > m[1] > m[2] else "mixed"})
        T = pd.DataFrame(rows); T["q"] = multipletests(T.p, method="fdr_bh")[1]
        S[f"temporal_{lab}"] = T.round(4).to_dict("records")
    # Kappa (Table 5) without the ageing-population item
    R = pd.read_csv("data/validation/ratings_matrix.csv", index_col=0)
    excl = {"experts/future/committees.parliament.uk_speeches.csv/row_69", "assemblies/scotland/8992int020_speeches.csv/row_67",
            "experts/constitution/committees.parliament.uk_speeches.csv/row_0"}
    for lab, drop in [("published", excl), ("revised", excl | {i for i in R.index if "9-July" in i})]:
        M = R.drop(index=[i for i in drop if i in R.index]); hum = [c for c in M.columns if c != "llm"]
        k = lambda a, b: cohen_kappa_score(M[a], M[b], weights="quadratic")
        ku = lambda a, b: cohen_kappa_score(M[a], M[b])
        lh = [k("llm", h) for h in hum]; hh = [k(a, b) for i, a in enumerate(hum) for b in hum[i + 1:]]
        S[f"kappa_{lab}"] = {"n": len(M), "llm_vs_human": [round(v, 3) for v in lh], "human_pairs": [round(v, 3) for v in hh],
                             "llm_mean": round(np.mean(lh), 3), "llm_sd": round(np.std(lh), 3),
                             "human_mean": round(np.mean(hh), 3), "human_sd": round(np.std(hh), 3),
                             "unweighted_llm": round(np.mean([ku("llm", h) for h in hum]), 3),
                             "unweighted_human": round(np.mean([ku(a, b) for i, a in enumerate(hum) for b in hum[i + 1:]]), 3)}
    out = Path("outputs/paper_v2"); out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ["table1", "r2_gt5", "r2_ge5", "known_groups", "kappa_published", "kappa_revised", "llm_dist"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
