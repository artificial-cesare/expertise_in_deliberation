"""Length and the LLM score in expert and citizen speech (paper subsection "expert_length", 2026-11-19).

Substantial utterances (>= 5 content words), revised data (paper_v2_stats.load).
1. Eighth Amendment session (paper table "expert_length"): assembly experts (role "expert", n = 86) and the citizens
   of the same session (role "participant", n = 58). Per group: LLM score by length quarter (quarters formed within
   the group), share of turns scored <= 2, Pearson r of the score with log(1 + tokens) for all turns and for the
   halves below / at or above the group's median length; expert turns scored <= 2 (share in the shorter half, share
   containing "?"); mean base and fine-tuned distance to the witness centroids by quarter and their r with the score.
2. Aggregated over all sessions: citizens of every assembly (participants plus the gender-equality table reports,
   as in role_correlations_v2.py) against the assembly experts, and against the assembly experts plus the UK
   parliamentary witnesses (hearing_roles.py, duplicated gender hearing dropped); participants alone as a check.
   Score by length quintile (formed within each group), by common token bins, OLS
   slope of the score on log length (all turns; < 100 and >= 100 tokens), and the expert x log-length interaction
   (HC1 standard errors).
Run from the repo root (CL env):  python reproduce/expert_length_v2.py   -> outputs/paper_v2/expert_length_v2.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load
from hearing_roles import expert_corpus

OUT = Path("outputs/paper_v2/expert_length_v2.json")
Y = "expertise_llm_score"
BINS = [0, 25, 50, 100, 200, np.inf]; BIN_LABELS = ["<25", "25-49", "50-99", "100-199", "200+"]


def r(d):
    d = d[[Y, "loglen"]].dropna()
    if len(d) < 4: return None
    x, p = pearsonr(d.loglen, d[Y]); return {"r": round(x, 3), "p": float(p), "n": len(d)}


def quarters(g):
    q = pd.qcut(g.length_tokens.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    t = g.groupby(q, observed=True).agg(n=(Y, "size"), median_tokens=("length_tokens", "median"), llm=(Y, "mean"),
                                        share_le2=(Y, lambda s: (s <= 2).mean()), share_ge4=(Y, lambda s: (s >= 4).mean()),
                                        distance_base=("distance_base", "mean"), distance_finetuned=("distance_finetuned", "mean"))
    return t.round(3).to_dict("index")


def halves(g):
    m = g.length_tokens.median()
    return {"all": r(g), "shorter_half": r(g[g.length_tokens < m]), "longer_half": r(g[g.length_tokens >= m]),
            "sd": round(g[Y].std(), 3), "mean": round(g[Y].mean(), 3), "median_tokens": float(m)}


def slopes(g):
    fit = lambda d: smf.ols(f"{Y} ~ loglen", d).fit(cov_type="HC1")
    res = {}
    for lab, d in [("all", g), ("lt100", g[g.length_tokens < 100]), ("ge100", g[g.length_tokens >= 100])]:
        if len(d) < 10: res[lab] = None; continue
        f = fit(d); lo, hi = f.conf_int().loc["loglen"]
        res[lab] = {"b": round(f.params.loglen, 3), "ci": [round(lo, 3), round(hi, 3)], "p": float(f.pvalues.loglen), "n": len(d)}
    return res


D = load(); D = D[D.num_non_function_words >= 5]
S = {}

# 1. Eighth Amendment session
E8 = D[D.role == "expert"]; C8 = D[(D.folder == "8th_amendment") & (D.role == "participant")]
S["session"] = {"experts": {"quarters": quarters(E8), **halves(E8)}, "citizens": {"quarters": quarters(C8), **halves(C8)}}
lo = E8[E8[Y] <= 2]; hi = E8[E8[Y] >= 3]
S["session"]["expert_low_turns"] = {
    "n": len(lo), "n_scored_1": int((lo[Y] == 1).sum()), "median_tokens": float(lo.length_tokens.median()),
    "share_in_shorter_half": round((lo.length_tokens < E8.length_tokens.median()).mean(), 3),
    "share_question": round(lo.text.str.contains(r"\?").mean(), 3), "share_question_ge3": round(hi.text.str.contains(r"\?").mean(), 3)}
S["session"]["expert_distance_vs_llm"] = {c: r(E8.assign(loglen=E8[c])) for c in
                                          ("distance_base", "distance_finetuned", "similarity_base", "similarity_finetuned")}

# 2. Aggregated over all sessions
cit = D[D.role.isin(["participant", "rapporteur"])].assign(group="citizens")
ae = E8.assign(group="assembly experts")
H = expert_corpus(); H = H[~H.duplicate & (H.role == "witness")].copy()
H = H.assign(group="parliamentary witnesses", loglen=np.log1p(H.length_tokens), assembly="UK hearings")
cols = ["group", "assembly", Y, "length_tokens", "loglen"]
X = pd.concat([cit[cols], ae[cols], H[cols]], ignore_index=True).dropna(subset=[Y])
X["bin"] = pd.cut(X.length_tokens, BINS, right=False, labels=BIN_LABELS)
S["aggregated"] = {"citizens_by_assembly": cit.assembly.value_counts().to_dict()}
def quintiles(d):
    q = pd.qcut(d.length_tokens.rank(method="first"), 5, labels=["P0-20", "P20-40", "P40-60", "P60-80", "P80-100"])
    return d.groupby(q, observed=True).agg(n=(Y, "size"), min_tokens=("length_tokens", "min"), median_tokens=("length_tokens", "median"),
                                           max_tokens=("length_tokens", "max"), llm=(Y, "mean"),
                                           share_le2=(Y, lambda s: (s <= 2).mean()), share_ge4=(Y, lambda s: (s >= 4).mean())).round(3).to_dict("index")


GROUPS = {g: d for g, d in X.groupby("group")}
P = D[D.role == "participant"].assign(group="participants only")  # citizens without the gender-equality table reports
GROUPS["participants only"] = P[cols].dropna(subset=[Y]).assign(bin=lambda z: pd.cut(z.length_tokens, BINS, right=False, labels=BIN_LABELS))
for g, d in GROUPS.items():
    S["aggregated"][g] = {"bins": d.groupby("bin", observed=False).agg(n=(Y, "size"), llm=(Y, "mean")).round(3).to_dict("index"),
                          "quintiles": quintiles(d), "slopes": slopes(d), **halves(d)}
for lab, ex in [("assembly_experts", ["assembly experts"]), ("all_experts", ["assembly experts", "parliamentary witnesses"])]:
    d = X[X.group.isin(["citizens"] + ex)].assign(expert=lambda z: (z.group != "citizens").astype(int))
    f = smf.ols(f"{Y} ~ loglen * expert", d).fit(cov_type="HC1"); lo_, hi_ = f.conf_int().loc["loglen:expert"]
    S["aggregated"][f"interaction_{lab}"] = {"b": round(f.params["loglen:expert"], 3), "ci": [round(lo_, 3), round(hi_, 3)],
                                             "p": float(f.pvalues["loglen:expert"]), "n": len(d)}

OUT.write_text(json.dumps(S, indent=1, default=str))
print(json.dumps(S, indent=1, default=str))
print("->", OUT)
