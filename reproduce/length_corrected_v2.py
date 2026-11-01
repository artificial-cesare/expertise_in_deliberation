"""Length-corrected scores: every comparison of the paper, redone at equal utterance length (2026-11-01).

Corrected value = residual of an OLS regression of the measure on [1, log(1 + tokens)], fitted once on all
3,413 substantial utterances (pooled, so assemblies stay comparable), plus the pooled mean so the scale is kept.
Comparisons: assembly means, speaker roles, temporal Kruskal-Wallis (BH over the tests run here), and the
score distribution across assemblies. Raw results are recomputed alongside for a direct comparison.

Run from the repo root:  python reproduce/length_corrected_v2.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load

D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True)
MEAS = {"expertise_llm_score": "LLM score", "overall_score": "ArgQA", "lexical_sophistication": "Lexical sophistication",
        "similarity_base": "Similarity (base)", "similarity_finetuned": "Similarity (fine-tuned)",
        "specificity_numbers_norm": "Numerical specificity", "specificity_content_prop": "Content-word proportion",
        "specificity_mean_word_len_norm": "Mean word length"}
X = np.column_stack([np.ones(len(S)), S.loglen])
for m in MEAS:
    ok = S[m].notna().values
    b, *_ = np.linalg.lstsq(X[ok], S.loc[ok, m].values, rcond=None)
    S[m + "_lc"] = np.where(ok, S[m] - X @ b + S.loc[ok, m].mean(), np.nan)
    S.attrs[m] = {"slope_per_log_token": round(float(b[1]), 3)}
OUT = {"slopes": {m: S.attrs[m] for m in MEAS}}

# assemblies
OUT["assembly_means"] = {a: {m: {"raw": round(g[m].mean(), 3), "corrected": round(g[m + "_lc"].mean(), 3)}
                             for m in ("expertise_llm_score", "overall_score")} for a, g in S.groupby("assembly")}
# share of low scores at equal length: corrected LLM score below the pooled 2.5
OUT["roles"] = (S.groupby(["assembly", "role_group"])
                .agg(n=("expertise_llm_score", "size"), llm_raw=("expertise_llm_score", "mean"),
                     llm_corrected=("expertise_llm_score_lc", "mean"), argqa_raw=("overall_score", "mean"),
                     argqa_corrected=("overall_score_lc", "mean"))
                .round(2).reset_index().to_dict("records"))
def mw(a, b, col):
    u = mannwhitneyu(a[col], b[col]); return {"p": float(u.pvalue), "rank_biserial": round(float(2 * u.statistic / (len(a) * len(b)) - 1), 3)}
c = S[S.assembly == "Constitutional Reform"]; g = S[S.assembly == "Gender Equality"]; pp = S[S.assembly == "Public Policy"]
OUT["role_tests"] = {
    "constitutional_expert_vs_citizen": {k: mw(c[c.role == "expert"], c[c.role == "participant"], k) for k in ("expertise_llm_score", "expertise_llm_score_lc")},
    "constitutional_secretariat_vs_citizen": {k: mw(c[c.role == "secretariat"], c[c.role == "participant"], k) for k in ("expertise_llm_score", "expertise_llm_score_lc")},
    "gender_chair_vs_facilitators": {k: mw(g[g.role == "chair"], g[g.role == "facilitator"], k) for k in ("expertise_llm_score", "expertise_llm_score_lc")},
    "gender_rapporteur_vs_facilitation": {k: mw(g[g.role_group == "rapporteur"], g[g.role_group == "facilitation"], k) for k in ("expertise_llm_score", "expertise_llm_score_lc")},
    "pp_facilitation_vs_participant": {k: mw(pp[pp.role_group == "facilitation"], pp[pp.role_group == "participant"], k) for k in ("expertise_llm_score", "expertise_llm_score_lc")},
}
OUT["role_means_detail"] = {a: gg.groupby("role")[["expertise_llm_score", "expertise_llm_score_lc"]].mean().round(2).to_dict("index")
                            for a, gg in S.groupby("assembly")}

# temporal
rows = []
for a, ga in S.groupby("assembly"):
    for m in MEAS:
        for kind, col in (("raw", m), ("corrected", m + "_lc")):
            x = ga[[col, "window"]].dropna()
            grp = [x.loc[x.window == w, col] for w in ("start", "middle", "end")]
            H, p = kruskal(*grp); n = len(x)
            means = [g_.mean() for g_ in grp]
            rows.append({"assembly": a, "measure": MEAS[m], "kind": kind, "means": [round(v, 3) for v in means],
                         "H": round(H, 2), "p": p, "eps2": round(H / (n - 1), 3),
                         "monotonic": bool((np.diff(means) > 0).all() or (np.diff(means) < 0).all())})
T = pd.DataFrame(rows)
for kind in ("raw", "corrected"):
    i = T.kind == kind; T.loc[i, "q"] = multipletests(T.loc[i, "p"], method="fh" if False else "fdr_bh")[1]
T["trend"] = np.where(T.q >= .05, "n.s.", np.where(T.monotonic, np.where([m[-1] > m[0] for m in T.means], "up", "down"), "mixed"))
OUT["temporal"] = T.assign(p=T.p.round(4), q=T.q.round(4)).to_dict("records")
# length itself over time, for interpretation
OUT["length_by_third"] = {a: ga.groupby("window").length_tokens.median().to_dict() for a, ga in S.groupby("assembly")}

Path("outputs/paper_v2/length_corrected_v2.json").write_text(json.dumps(OUT, indent=1, default=str))
pd.set_option("display.width", 220)
print("slopes:", OUT["slopes"])
print("\nAssembly means:", json.dumps(OUT["assembly_means"], indent=0))
print("\nRoles:"); print(pd.DataFrame(OUT["roles"]).to_string(index=False))
print("\nRole tests:"); [print(" ", k, {kk: (round(v['p'], 4), v['rank_biserial']) for kk, v in d.items()}) for k, d in OUT["role_tests"].items()]
print("\nRole means detail:", OUT["role_means_detail"])
print("\nTemporal (raw vs corrected):")
W = T.pivot_table(index=["assembly", "measure"], columns="kind", values=["q", "eps2", "trend"], aggfunc="first")
M = T.pivot_table(index=["assembly", "measure"], columns="kind", values="means", aggfunc="first")
print(pd.concat([W, M], axis=1).to_string())
print("\nMedian tokens by third:", OUT["length_by_third"])
