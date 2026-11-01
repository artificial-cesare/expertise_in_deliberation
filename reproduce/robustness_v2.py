"""Robustness checks for the revised paper (2026-11-01).

1. Length and mean word length by speaker role (do experts speak longer?), within the constitutional-reform
   session where experts and citizens speak in the same discussion.
2. Content-word threshold sweep (5, 10, 20), the comparison the original corr.py ran but whose outputs are lost:
   sample size, LLM score mean and SD, and the main raw and length-adjusted correlations at each threshold.

Run from the repo root after features_v2.py:  python reproduce/robustness_v2.py
Writes outputs/paper_v2/robustness_v2.json and prints a summary.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, r_p, partial, residual

D = load()
WORD = re.compile(r"[A-Za-z]+")
D["mean_word_len_raw"] = [np.mean([len(w) for w in WORD.findall(str(t))]) if WORD.search(str(t)) else np.nan for t in D.text]
OUT = {}

# ---- 1. experts vs citizens: length and mean word length --------------------------------------------
c = D[(D.assembly == "Constitutional Reform") & (D.num_non_function_words >= 5)]
e, p = c[c.role == "expert"], c[c.role == "participant"]
def compare(col):
    u = mannwhitneyu(e[col], p[col])
    return {"expert_median": round(float(e[col].median()), 2), "citizen_median": round(float(p[col].median()), 2),
            "expert_mean": round(float(e[col].mean()), 2), "citizen_mean": round(float(p[col].mean()), 2),
            "mwu_p": float(u.pvalue), "rank_biserial": round(1 - 2 * u.statistic / (len(e) * len(p)), 3)}
OUT["experts_vs_citizens"] = {"n_expert": len(e), "n_citizen": len(p),
                              "length_tokens": compare("length_tokens"), "words": compare("words"),
                              "mean_word_len_raw": compare("mean_word_len_raw"),
                              "mean_word_len_norm": compare("specificity_mean_word_len_norm")}
# mean word length gap at equal length (word length and length are correlated)
xx = pd_ = c[c.role.isin(["expert", "participant"])].dropna(subset=["mean_word_len_raw"])
res = residual(xx.mean_word_len_raw.values, [xx.loglen])
OUT["experts_vs_citizens"]["mean_word_len_gap_at_equal_length"] = round(
    float(res[(xx.role == "expert").values].mean() - res[(xx.role == "participant").values].mean()), 3)
# all substantial utterances, all roles: length per role
OUT["median_tokens_by_role"] = (D[D.num_non_function_words >= 5].groupby(["assembly", "role_group"]).length_tokens
                                .median().round(1).reset_index().to_dict("records"))

# ---- 2. threshold sweep ----------------------------------------------------------------------------
FEATS = ["overall_score", "specificity_length_norm", "specificity_mean_word_len_norm", "lexical_sophistication",
         "jargon_density", "syntactic_complexity", "specificity_numbers_norm", "epistemic_balance",
         "reasoning_markers", "evidential_markers", "similarity_base", "similarity_finetuned"]
sweep = {}
profiles = {}
for thr in (5, 10, 20):
    s = D[D.num_non_function_words >= thr]
    row = {"n": len(s), "n_by_assembly": s.assembly.value_counts().to_dict(),
           "share_of_all": round(len(s) / len(D), 3),
           "llm": {a: {"mean": round(g.expertise_llm_score.mean(), 2), "sd": round(g.expertise_llm_score.std(), 2)}
                   for a, g in s.groupby("assembly")},
           "llm_pooled": {"mean": round(s.expertise_llm_score.mean(), 2), "sd": round(s.expertise_llm_score.std(), 2)},
           "r_loglen": r_p(s.expertise_llm_score, s.loglen),
           "raw": {f: r_p(s.expertise_llm_score, s[f]) for f in FEATS},
           "partial": {f: partial(s, "expertise_llm_score", f, True) for f in FEATS if f != "specificity_length_norm"}}
    k = s[(s.assembly == "Constitutional Reform") & s.role.isin(["expert", "participant"])]
    rr = residual(k.expertise_llm_score.values, [k.loglen])
    row["known_groups"] = {"n_expert": int((k.role == "expert").sum()), "n_citizen": int((k.role == "participant").sum()),
                           "gap_raw": round(k[k.role == "expert"].expertise_llm_score.mean() - k[k.role == "participant"].expertise_llm_score.mean(), 2),
                           "gap_equal_length": round(rr[(k.role == "expert").values].mean() - rr[(k.role == "participant").values].mean(), 2)}
    sweep[thr] = row
    profiles[thr] = [row["raw"][f]["r"] for f in FEATS]
sweep["profile_spearman"] = {f"{a}_vs_{b}": round(spearmanr(profiles[a], profiles[b])[0], 3) for a, b in [(5, 10), (5, 20), (10, 20)]}
OUT["threshold_sweep"] = sweep

out = Path("outputs/paper_v2/robustness_v2.json"); out.write_text(json.dumps(OUT, indent=1, default=str))
ev = OUT["experts_vs_citizens"]
print("Experts vs citizens (Constitutional, >=5):", ev["n_expert"], ev["n_citizen"])
for k_ in ("length_tokens", "words", "mean_word_len_raw"):
    print(f"  {k_:18s}", ev[k_])
print("  mean word length gap at equal length:", ev["mean_word_len_gap_at_equal_length"])
print("\nThreshold sweep")
for thr in (5, 10, 20):
    r = sweep[thr]
    print(f" >= {thr:2d}: n={r['n']} {r['n_by_assembly']} | LLM pooled {r['llm_pooled']} | r(loglen)={r['r_loglen']['r']}")
    print("   LLM by assembly:", r["llm"])
    print("   raw r:", {f: r["raw"][f]["r"] for f in FEATS})
    print("   partial r:", {f: r["partial"][f]["r"] for f in r["partial"]})
    print("   known groups:", r["known_groups"])
print(" correlation-profile agreement (Spearman):", sweep["profile_spearman"])
print("saved", out)
