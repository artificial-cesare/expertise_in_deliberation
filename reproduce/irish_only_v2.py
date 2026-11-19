"""Correlations with the LLM score, all assemblies vs the two Irish assemblies (paper table "irish_only", 2026-11-19).

Substantial utterances (>= 5 content words), revised corpus (paper_v2_stats.load(), ageing session left out).
For each feature: raw Pearson r and partial r (both variables residualised on log(1 + tokens) and assembly dummies,
paper_v2_stats.partial), for all assemblies and for Constitutional Reform + Gender Equality.
Two length-free versions of numerical specificity: "contains a number" and numbers per 100 tokens, where numbers are
spaCy tokens with like_num and tokens are all spaCy tokens (en_core_web_sm); they are read from
outputs/paper_v2/numbers_rate_v2.csv, which this script writes when it is missing.
Also: share of the public-policy material, SD of the score, and share of utterances with a hedge or booster.
Run from the repo root (CL env):  python reproduce/irish_only_v2.py   -> outputs/paper_v2/irish_only_v2.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, r_p, partial

OUT = Path("outputs/paper_v2/irish_only_v2.json"); NUM = Path("outputs/paper_v2/numbers_rate_v2.csv")
Y = "expertise_llm_score"
FEATS = ["overall_score", "specificity_length_norm", "specificity_mean_word_len_norm", "lexical_sophistication",
         "syntactic_complexity", "jargon_density", "reasoning_markers", "epistemic_balance", "specificity_numbers_norm",
         "has_num", "num_rate"]

D = load(); S = D[D.num_non_function_words >= 5].copy()
if not NUM.exists():
    import spacy
    nlp = spacy.load("en_core_web_sm"); rows = []
    for f, r, t in zip(S.file, S.row, S.text.astype(str)):
        doc = nlp(t); n = sum(tok.like_num for tok in doc)
        rows.append({"file": f, "row": r, "n_num": n, "n_tok": len(doc), "num_rate": 100 * n / len(doc), "has_num": float(n > 0)})
    pd.DataFrame(rows).to_csv(NUM, index=False)
S = S.merge(pd.read_csv(NUM)[["file", "row", "num_rate", "has_num"]], on=["file", "row"], how="left")
assert S.num_rate.notna().all()
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
