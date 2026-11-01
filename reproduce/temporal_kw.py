"""Kruskal-Wallis tests behind Section 5.4 and Table 7 (temporal trends).

Each transcript is split into thirds by row order (start / middle / end), the thirds are
pooled per assembly, and a Kruskal-Wallis test compares them; epsilon^2 = H / (n - 1).
Windows are set before any filtering, so they always reflect position in the session.

The published values use no content-word filter (--min-words 0, the default), although the
paper says >=5 content words throughout. Pass --min-words 5 to apply that filter.

Run from the repo root:  python reproduce/temporal_kw.py [--min-words 5]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_corr import ComprehensiveCorrelationAnalyzer  # noqa: E402

WINDOWS = ("start", "middle", "end")
ASSEMBLIES = ComprehensiveCorrelationAnalyzer(include_scotland="yes").assembly_topics


def thirds(n: int) -> list[str]:
    """Window label per row; the last third takes the remainder (as corr.py does)."""
    k = n // 3
    return ["start"] * k + ["middle"] * k + ["end"] * (n - 2 * k)


def load_assembly(folder: Path, min_words: int) -> pd.DataFrame:
    frames = []
    for csv in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv)
        frames.append(df.assign(window=thirds(len(df)), source_file=csv.name))
    df = pd.concat(frames, ignore_index=True)
    return df[df["num_non_function_words"] >= min_words].reset_index(drop=True)


def kruskal_by_window(df: pd.DataFrame, feature: str) -> dict:
    groups = [df.loc[df["window"] == w, feature].dropna() for w in WINDOWS]
    h, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {"n": n, "H": h, "p": p, "eps2": h / (n - 1),
            **{f"median_{w}": g.median() for w, g in zip(WINDOWS, groups)},
            **{f"mean_{w}": g.mean() for w, g in zip(WINDOWS, groups)}}


def features(df: pd.DataFrame) -> list[str]:
    """Same feature set as the correlation analysis (base + finetuned embedding columns)."""
    a = ComprehensiveCorrelationAnalyzer()
    base = a.get_dynamic_features(df, "base")
    return base + [f for f in a.get_dynamic_features(df, "finetuned") if f not in base]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-words", type=int, default=0, help="keep utterances with >= this many content words")
    ap.add_argument("--data-dir", type=Path, default=Path("data/assemblies/processed_files"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/temporal"))
    args = ap.parse_args()

    rows = []
    for folder, topic in ASSEMBLIES.items():
        df = load_assembly(args.data_dir / folder, args.min_words)
        for feat in features(df):
            if df[feat].nunique() > 1:  # Kruskal-Wallis is undefined for a constant feature
                rows.append({"assembly": topic, "feature": feat, **kruskal_by_window(df, feat)})
            else:
                print(f"skipped {topic} / {feat}: constant")
    res = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"kruskal_wallis_minwords{args.min_words}.csv"
    res.to_csv(out, index=False)
    cols = ["assembly", "feature", "n", "p", "eps2", "mean_start", "mean_middle", "mean_end"]
    print(res[cols].round(4).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
