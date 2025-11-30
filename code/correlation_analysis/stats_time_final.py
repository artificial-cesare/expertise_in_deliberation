"""
Generate descriptive statistics (mean, sd, min, max) for each feature,
both overall and by temporal segment (start, middle, end), for each CSV file.
Outputs one .txt report per file.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# -------------------------- CONFIGURABLE DEFAULTS -------------------------- #
INPUT_DIR = Path("data/assemblies/processed_files")
OUTPUT_DIR = Path("code/correlation_analysis/results/feature_stats")

ASSEMBLY_TOPICS = {
    "8th_amendment": "Constitutional Reform",
    "gendereq": "Gender Equality",
    "scotland": "Public Policy",
}
EXPERT_TOPICS = {
    "constitution": "Constitutional Reform",
    "future": "Public Policy",
    "gender": "Gender Equality",
}

FEATURE_COLUMNS = [
    "length_tokens",
    "num_non_function_words",
    "jargon_density",
    "lexical_sophistication",
    "syntactic_complexity",
    "hedge_density",
    "booster_density",
    "epistemic_balance",
    "evidential_markers",
    "reasoning_markers",
    "specificity_length_norm",
    "specificity_mean_word_len_norm",
    "specificity_numbers_norm",
    "specificity_content_prop",
    "distance_base", 
    "similarity_base", 
    "cluster_finetuned", 
    "distance_finetuned", 
    "similarity_finetuned", 
    "expertise_llm_score",
    "overall_score",
]

TIME_WINDOWS = ["start", "middle", "end"]


# ----------------------------- HELPER FUNCTIONS --------------------------- #
def discover_csv_files(base_path: Path, topic_map: Dict[str, str]) -> List[Path]:
    """Locate CSV files in topic subfolders, with _csv fallback."""
    csv_paths: List[Path] = []
    if not base_path.exists():
        return csv_paths

    for folder in topic_map.keys():
        folder_path = base_path / folder
        if not folder_path.exists():
            alt = base_path / f"{folder}_csv"
            if alt.exists():
                folder_path = alt
            else:
                continue
        csv_paths.extend(sorted(folder_path.glob("*.csv")))
    return csv_paths


def load_data(file_path: Path) -> pd.DataFrame:
    """Load CSV and add row order index."""
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=False).rename(columns={"index": "_row_order"})
    return df


def assign_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Split dataframe into start/middle/end based on row order."""
    df = df.sort_values("_row_order").copy()
    n = len(df)
    chunk_size = n // 3

    df["time_window"] = "end"  # default
    df.iloc[:chunk_size, df.columns.get_loc("time_window")] = "start"
    df.iloc[chunk_size : 2 * chunk_size, df.columns.get_loc("time_window")] = "middle"
    return df


def compute_stats(values: pd.Series) -> Dict[str, float]:
    """Compute mean, sd, min, max for a numeric series."""
    clean = values.dropna()
    if clean.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def format_stats(stats: Dict[str, float]) -> str:
    """Format stats dict as a readable string."""
    if stats["n"] == 0:
        return "n = 0 (no data)"
    return (
        f"n = {stats['n']}, "
        f"mean = {stats['mean']:.4f}, "
        f"sd = {stats['sd']:.4f}, "
        f"min = {stats['min']:.4f}, "
        f"max = {stats['max']:.4f}"
    )


def process_file(file_path: Path, output_dir: Path) -> None:
    """Generate stats report for a single CSV file."""
    df = load_data(file_path)
    df = assign_time_windows(df)

    lines = [
        f"FILE: {file_path.name}",
        "=" * max(50, len(file_path.name) + 6),
        f"Total rows: {len(df)}",
        "",
    ]

    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue

        lines.append(f"Feature: {feature}")
        lines.append("-" * 40)

        # Overall stats
        overall = compute_stats(df[feature])
        lines.append(f"  OVERALL : {format_stats(overall)}")

        # Stats by time window
        for window in TIME_WINDOWS:
            subset = df.loc[df["time_window"] == window, feature]
            window_stats = compute_stats(subset)
            lines.append(f"  {window.upper():<7} : {format_stats(window_stats)}")

        lines.append("")

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_path.stem}_feature_stats.txt"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_file}")


# ------------------------------- MAIN ------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature statistics by file and time window."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DIR),
        help="Base directory containing assembly CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to write TXT reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    datasets = [
        ("assemblies", input_dir, ASSEMBLY_TOPICS),
        ("experts", input_dir / "experts", EXPERT_TOPICS),
    ]

    any_files = False
    for dataset_name, base_path, topics in datasets:
        csv_files = discover_csv_files(base_path, topics)
        if not csv_files:
            continue

        any_files = True
        dataset_output = output_dir / dataset_name
        print(f"\nProcessing {dataset_name}: {len(csv_files)} files")

        for csv_file in csv_files:
            process_file(csv_file, dataset_output)

    if not any_files:
        print(f"No CSV files found under {input_dir}")


if __name__ == "__main__":
    main()