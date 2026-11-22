"""Correlation tables by speaker role (exploratory, 2026-11-13).

Same features, plotting code and global z-scoring as reproduce/figures_v2.py (>= 5 content words, revised data),
on two subsets of the assembly utterances, with roles from paper_v2_stats.role():
  - citizens: role "participant" plus the Gender Equality table reports ("BREAKOUT GROUP n", "TABLE n", role
    "rapporteur"), which are citizens reporting their table and the only member turns in that assembly
    (chair, facilitators, secretariat, experts and unlabelled turns out);
  - experts: role == "expert", the Expert Advisory Group / medical experts of the Eighth Amendment session.
Writes heatmaps and correlations.json to outputs/paper_v2/correlations_roles/<subset>/<model>/ and a long table of
every pair (r, p, n) for the full corpus and each subset to outputs/paper_v2/correlations_roles/pairs.csv.
Run from the repo root after features_v2.py:  python reproduce/role_correlations_v2.py
"""
import contextlib, io, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_corr import ComprehensiveCorrelationAnalyzer
from paper_v2_stats import load, FEATS as PAPER_COLS  # the 19 columns of figures_v2.PAPER_COLS

OUT = Path("outputs/paper_v2/correlations_roles")
D = load(); D = D[D.num_non_function_words >= 5]
SUBSETS = {"all": D,
           "citizens": D[D.role.isin(["participant", "rapporteur"])],
           "experts": D[D.role == "expert"]}
a = ComprehensiveCorrelationAnalyzer(norm_type="global", scaler_type="zscore", include_scotland="yes")
rows = []
for name, sub in SUBSETS.items():
    S = sub[PAPER_COLS].reset_index(drop=True)
    comp = sub.assembly.value_counts().to_dict()
    for model in ("base", "finetuned"):
        feats = a.get_dynamic_features(S, model)
        with contextlib.redirect_stdout(io.StringIO()):
            corr, p = a.compute_correlations(a.normalize_features(S, feats), feats)
            if name != "all":
                out = OUT / name / model; out.mkdir(parents=True, exist_ok=True)
                a.create_correlation_heatmaps(corr, p, f"{name.title()} - {model.title()} Model", out)
                a.save_correlation_json(corr, p, out / "correlations.json",
                                        {"n": len(S), "model": model, "threshold": ">=5", "subset": name, "assemblies": comp})
        for i, x in enumerate(feats):
            for j, y in enumerate(feats[i + 1:], i + 1):
                rows.append({"subset": name, "x": x, "y": y, "r": corr.iat[i, j], "p": p.iat[i, j],
                             "n": int(S[[x, y]].dropna().shape[0])})
    print(f"{name}: n = {len(S)}", comp)
P = pd.DataFrame(rows).drop_duplicates(["subset", "x", "y"])  # pairs without embedding columns appear in both models
P.to_csv(OUT / "pairs.csv", index=False)
print("->", OUT)
