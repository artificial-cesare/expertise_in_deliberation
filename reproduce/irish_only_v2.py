"""Correlations with the LLM score, all assemblies vs the two Irish assemblies (paper table "irish_only", 2026-11-19).

Substantial utterances (>= 5 content words), revised corpus (paper_v2_stats.load(), ageing session left out).
For each feature: raw Pearson r and partial r (both variables residualised on log(1 + tokens) and assembly dummies,
paper_v2_stats.partial), for all assemblies and for Constitutional Reform + Gender Equality.
Features in raw units (features_v2.py): numerical specificity is numbers per 100 words; "contains a number" (has_num_v2)
is reported alongside. Until 2026-11-22 numerical specificity was the per-transcript capped count, with "contains a
number" and numbers per 100 spaCy tokens as length-free versions.
Also: share of the public-policy material, SD of the score, and share of utterances with a hedge or booster.
Run from the repo root (CL env):  python reproduce/irish_only_v2.py   -> outputs/paper_v2/irish_only_v2.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, r_p, partial

OUT = Path("outputs/paper_v2/irish_only_v2.json")
Y = "expertise_llm_score"
FEATS = ["overall_score", "mean_word_length", "lexical_sophistication", "syntactic_complexity", "jargon_density",
         "reasoning_markers", "epistemic_balance", "numerical_specificity", "has_num"]

D = load(); S = D[D.num_non_function_words >= 5].copy()
S["has_num"] = S.has_num_v2
assert S.numerical_specificity.notna().all()
IR = S[S.assembly != "Public Policy"]

out = {"n_all": len(S), "n_irish": len(IR),
       "n_epistemic_all": int(S.epistemic_balance.notna().sum()), "n_epistemic_irish": int(IR.epistemic_balance.notna().sum()),
       "share_public_policy": round((S.assembly == "Public Policy").mean(), 3),
       "sd_public_policy": round(S.loc[S.assembly == "Public Policy", Y].std(), 3), "sd_irish": round(IR[Y].std(), 3),
       "share_le2_public_policy": round((S.loc[S.assembly == "Public Policy", Y] <= 2).mean(), 3),
       "median_tokens_public_policy": float(S.loc[S.assembly == "Public Policy", "length_tokens"].median()),
       "share_hedge_or_booster": round(S.epistemic_balance.notna().mean(), 3), "features": {}}
for f in FEATS:
    out["features"][f] = {"all": {"r": r_p(S[Y], S[f]), "partial": partial(S, Y, f)},
                          "irish": {"r": r_p(IR[Y], IR[f]), "partial": partial(IR, Y, f)}}
OUT.write_text(json.dumps(out, indent=1))
star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
fmt = lambda d: f"{d['r']:+.2f}{star(d['p'])}".replace("+0", "").replace("-0", "-")
print({k: v for k, v in out.items() if k != "features"})
for f, v in out["features"].items():
    print(f"{f:32s} all {fmt(v['all']['r']):>8} {fmt(v['all']['partial']):>8}   irish {fmt(v['irish']['r']):>8} {fmt(v['irish']['partial']):>8}")
print("->", OUT)
