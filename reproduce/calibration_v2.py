"""Context-size calibration: every number of the paper's calibration paragraphs and table (2026-11-19).

Reads the three calibration runs with the Appendix A prompt (outputs/paper_v2/calibration_appendixA/ge{5,10,20}/
context_calibration_results.csv, written by code/LLM_expertise/context_size_calibration.py) and leaves out the items of
the ageing-population voting session, which is not part of the corpus (paper_v2_stats.load()). --include-ageing keeps
them and reproduces the numbers of the paper before 2026-11-19.
Conditions: no context (0), budgets of 100, 500 and 1,000 tokens, and the 20 preceding utterances (the window).
  - per sample: n, mean score per condition, Friedman test, ICC(2,1) across the five conditions (pingouin, as in
    analyze_calibration.py), Spearman rho of each condition with the window, mean absolute and mean signed difference
    window minus condition (Wilcoxon signed-rank), share of utterances whose score rose / fell from no context to the
    window, and r of that rise with log(1 + tokens);
  - the distinct utterances of the three samples by content-word band (5-9, 10-19, 20+): repeat (window vs the stored
    score, which had the same input), rho with the window, rise from no context to the window, mean score.
Run from the repo root (CL env):  python reproduce/calibration_v2.py [--include-ageing]
  -> outputs/paper_v2/calibration_v2.json (calibration_v2_with_ageing.json with the flag)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pingouin as pg
from scipy.stats import friedmanchisquare, spearmanr, pearsonr, wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load

ap = argparse.ArgumentParser(); ap.add_argument("--include-ageing", action="store_true"); a = ap.parse_args()
BASE = Path("outputs/paper_v2/calibration_appendixA")
OUT = Path(f"outputs/paper_v2/calibration_v2{'_with_ageing' if a.include_ageing else ''}.json")
C = ["0", "100", "500", "1000", "20_utterances"]; W = "20_utterances"

# stored scores (same input as the window condition); _v1 = as first scored, which is what the calibration repeated
D = load(include_ageing=True)
STORED = D.set_index(["file", "row"]).expertise_llm_score_v1


def sample(thr):
    d = pd.read_csv(BASE / thr / "context_calibration_results.csv")
    if not a.include_ageing: d = d[~d.file_name.str.contains("9-July")]
    P = d.pivot_table(index="utterance_id", columns="budget_condition", values="score")[C]
    meta = d.drop_duplicates("utterance_id").set_index("utterance_id")[["file_name", "row_index", "assembly", "target_content_words", "target_token_count"]]
    return P.join(meta)


def icc(P):
    L = P[C].reset_index().melt(id_vars="utterance_id", var_name="cond", value_name="score")
    t = pg.intraclass_corr(L, targets="utterance_id", raters="cond", ratings="score"); r = t[t.Type == "ICC2"].iloc[0]
    return {"icc": round(float(r.ICC), 3), "ci": [float(x) for x in r["CI95%"]]}


S = {"include_ageing": a.include_ageing, "samples": {}}
for thr in ("ge5", "ge10", "ge20"):
    P = sample(thr); rise = P[W] - P["0"]
    S["samples"][thr] = {
        "n": len(P), "by_assembly": P.assembly.value_counts().to_dict(), "means": P[C].mean().round(3).to_dict(),
        "friedman_chi2": round(float(friedmanchisquare(*[P[c] for c in C]).statistic), 1), **icc(P),
        "rho_with_window": {c: round(float(spearmanr(P[c], P[W])[0]), 3) for c in C if c != W},
        "abs_diff_window_minus": {c: round(float((P[W] - P[c]).abs().mean()), 3) for c in C if c != W},
        "signed_diff_window_minus": {c: round(float((P[W] - P[c]).mean()), 3) for c in C if c != W},
        "wilcoxon_p_window_vs": {c: float(wilcoxon(P[W] - P[c]).pvalue) for c in C if c != W},
        "share_rose_from_no_context": round(float((rise > 0).mean()), 3), "share_fell_from_no_context": round(float((rise < 0).mean()), 3),
        "rise_vs_loglen": dict(zip(["r", "p"], map(float, pearsonr(rise, np.log1p(P.target_token_count)))))}

# distinct utterances of the three samples
U = pd.concat([sample(t) for t in ("ge5", "ge10", "ge20")]); U = U[~U.index.duplicated()].copy()
U["stored"] = [STORED.get((f, r), np.nan) for f, r in zip(U.file_name, U.row_index)]
assert U.stored.notna().all(), "calibration item without a stored score"
U["band"] = pd.cut(U.target_content_words, [4, 9, 19, np.inf], labels=["5-9", "10-19", "20+"])


def band(g):
    rise = g[W] - g["0"]
    return {"n": len(g), "repeat_rho": round(float(spearmanr(g[W], g.stored)[0]), 3), "repeat_identical": round(float((g[W] == g.stored).mean()), 4),
            "repeat_abs_diff": round(float((g[W] - g.stored).abs().mean()), 3), "repeat_signed_diff": round(float((g[W] - g.stored).mean()), 3),
            "rho_1000_window": round(float(spearmanr(g["1000"], g[W])[0]), 3), "rho_0_window": round(float(spearmanr(g["0"], g[W])[0]), 4),
            "rise": round(float(rise.mean()), 3), "share_rose": round(float((rise > 0).mean()), 3), "share_fell": round(float((rise < 0).mean()), 3),
            "mean_window_score": round(float(g[W].mean()), 3), "share_public_policy": round(float((g.assembly == "scotland").mean()), 3)}


S["distinct"] = {"n": len(U), **{b: band(g) for b, g in U.groupby("band", observed=True)}, "all": band(U)}
OUT.write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))
print("->", OUT)
