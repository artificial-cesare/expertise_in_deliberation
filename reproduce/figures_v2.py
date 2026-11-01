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

# The paper's 14 features plus the four scores. The CSVs also hold experimental columns (new_*, mrc_*, row
# indices, the v1 copies) that the published figures never showed, so select explicitly.
PAPER_COLS = ["jargon_density", "lexical_sophistication", "syntactic_complexity", "reasoning_markers",
              "evidential_markers", "length_tokens", "num_non_function_words", "hedge_density", "booster_density",
              "epistemic_balance", "specificity_length_norm", "specificity_mean_word_len_norm",
              "specificity_numbers_norm", "specificity_content_prop", "overall_score",
              "distance_base", "similarity_base", "distance_finetuned", "similarity_finetuned", "expertise_llm_score"]
D = load(); S = D.loc[D.num_non_function_words >= 5, PAPER_COLS].reset_index(drop=True)
a = ComprehensiveCorrelationAnalyzer(norm_type="global", scaler_type="zscore", include_scotland="yes")
for model in ("base", "finetuned"):
    out = Path("outputs/paper_v2/correlations") / model; out.mkdir(parents=True, exist_ok=True)
    feats = a.get_dynamic_features(S, model)
    with contextlib.redirect_stdout(io.StringIO()):
        corr, p = a.compute_correlations(a.normalize_features(S, feats), feats)
        a.create_correlation_heatmaps(corr, p, f"Assemblies - {model.title()} Model", out)
        a.save_correlation_json(corr, p, out / "correlations.json", {"n": len(S), "model": model, "threshold": ">=5"})
    print(model, "n =", len(S), "| LLM vs ArgQA r =", round(corr.loc["Expertise Llm Score", "Overall Score"], 3), "->", out)
