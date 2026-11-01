"""Figures 3-4 and Appendix B (revised): correlation and p-value heatmaps.

Same plotting code and global z-scoring as the published figures (reproduce/paper_corr.py), on the revised
data: ageing-population transcript removed, all three assemblies' LLM scores, >= 5 content words,
corrected lexical sophistication and broader evidential markers.
Run from the repo root after features_v2.py:  python reproduce/figures_v2.py
"""
import contextlib, io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_corr import ComprehensiveCorrelationAnalyzer
from paper_v2_stats import load

D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True)
a = ComprehensiveCorrelationAnalyzer(norm_type="global", scaler_type="zscore", include_scotland="yes")
for model in ("base", "finetuned"):
    out = Path("outputs/paper_v2/correlations") / model; out.mkdir(parents=True, exist_ok=True)
    feats = a.get_dynamic_features(S, model)
    with contextlib.redirect_stdout(io.StringIO()):
        corr, p = a.compute_correlations(a.normalize_features(S, feats), feats)
        a.create_correlation_heatmaps(corr, p, f"Assemblies - {model.title()} Model", out)
        a.save_correlation_json(corr, p, out / "correlations.json", {"n": len(S), "model": model, "threshold": ">=5"})
    print(model, "n =", len(S), "| LLM vs ArgQA r =", round(corr.loc["Expertise Llm Score", "Overall Score"], 3), "->", out)
