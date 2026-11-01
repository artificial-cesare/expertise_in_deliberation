"""Rebuild the correlation heatmaps (Figs 3-4, Appendix B) and temporal plots (Appendix C).

Uses the thesis-era ComprehensiveCorrelationAnalyzer (reproduce/paper_corr.py) with
the paper's settings: assemblies only, all three assemblies, >5 content words, global z-score.

--scotland-llm drop   reproduces the published figures. They were made before Scotland had
                      LLM scores, so every LLM-score cell there uses the Irish assemblies only.
--scotland-llm keep   (default) uses Scotland's LLM scores too.

Run from the repo root:  python reproduce/correlations.py [--scotland-llm drop]
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_corr import ComprehensiveCorrelationAnalyzer  # noqa: E402

LLM = "Expertise Llm Score"


def load_assemblies(data_dir: Path, scotland_llm: str) -> tuple[ComprehensiveCorrelationAnalyzer, pd.DataFrame]:
    a = ComprehensiveCorrelationAnalyzer(base_dir=data_dir, norm_type="global", scaler_type="zscore",
                                         include_scotland="yes")
    with contextlib.redirect_stdout(io.StringIO()):
        data = a.load_and_combine_data(a.assemblies_path, a.assembly_topics)
    if scotland_llm == "drop":
        data["Public Policy"]["expertise_llm_score"] = np.nan
    return a, pd.concat(list(data.values()), ignore_index=True)


def assembly_correlations(data_dir: Path, scotland_llm: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson r and p-value matrices, as in the paper's heatmaps."""
    a, df = load_assemblies(data_dir, scotland_llm)
    feats = a.get_dynamic_features(df, model)
    with contextlib.redirect_stdout(io.StringIO()):
        return a.compute_correlations(a.normalize_features(df, feats), feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scotland-llm", choices=["keep", "drop"], default="keep")
    ap.add_argument("--data-dir", type=Path, default=Path("data/assemblies/processed_files"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/correlations"))
    ap.add_argument("--no-temporal", action="store_true", help="skip the Appendix C plots")
    args = ap.parse_args()

    a, df = load_assemblies(args.data_dir, args.scotland_llm)
    n_llm = int(df["expertise_llm_score"].notna().sum())
    print(f"n = {len(df)} utterances, {n_llm} with an LLM score (scotland-llm={args.scotland_llm})")
    for model in ("base", "finetuned"):
        out = args.out_dir / f"scotland_llm_{args.scotland_llm}" / model
        out.mkdir(parents=True, exist_ok=True)
        feats = a.get_dynamic_features(df, model)
        title = f"Assemblies - {model.title()} Model"
        with contextlib.redirect_stdout(io.StringIO()):
            norm = a.normalize_features(df, feats)
            corr, p = a.compute_correlations(norm, feats)
            a.create_correlation_heatmaps(corr, p, title, out)
            a.save_correlation_summary(corr, p, out / "correlation_summary.txt", title)
            a.save_correlation_json(corr, p, out / "correlations.json",
                                    {"dataset": "assemblies", "model_type": model, "n_samples": len(df),
                                     "n_llm_scored": n_llm, "scotland_llm": args.scotland_llm})
            if not args.no_temporal:
                a.temporal_analysis(norm, feats, out / "temporal", "assemblies", model)
        row = pd.DataFrame({"r": corr.loc[LLM], "p": p.loc[LLM]}).drop(LLM).round(3)
        print(f"\n{model}: LLM score vs each feature  ->  {out}\n{row.to_string()}")


if __name__ == "__main__":
    main()
