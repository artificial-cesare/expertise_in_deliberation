"""
Generate descriptive statistics (mean, sd, min, max) for each feature,
both overall and by temporal segment (start, middle, end), for each CSV file.
Outputs one .txt report per file, plus a summary report by topic.
Also performs Kruskal-Wallis tests across time segments and generates LaTeX tables.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.topic_maps import ASSEMBLY_TOPICS, EXPERT_TOPICS

# -------------------------- CONFIGURABLE DEFAULTS -------------------------- #
INPUT_DIR = Path("data/assemblies/new_feats/")
OUTPUT_DIR = Path("code/correlation_analysis/time_results/feature_stats")

# Content word thresholds to filter utterances
CONTENT_WORD_THRESHOLDS = [5, 10, 20]

FEATURE_COLUMNS = [
    "expertise_llm_score",
    "overall_score",
    "distance_base",
    "similarity_base",
    "distance_finetuned",
    "similarity_finetuned",
    "jargon_density",
    "lexical_sophistication",
    "syntactic_complexity",
    "hedge_density",
    "booster_density",
    "epistemic_balance",
    "reasoning_markers",
    "evidential_markers",
    "specificity_numbers",
    "specificity_word_length",
    "subjectivity_score",
    "mrc_familiarity",
    "mrc_concreteness",
    "mrc_imageability",
    "length_tokens",
    "num_non_function_words",
]

# Display names for LaTeX table
FEATURE_DISPLAY_NAMES = {
    "length_tokens": "Length (Tokens)",
    "num_non_function_words": "Vocabulary Breadth",
    "jargon_density": "Jargon Density",
    "lexical_sophistication": "Lexical Sophistication",
    "syntactic_complexity": "Syntactic Complexity",
    "hedge_density": "Hedge Density",
    "booster_density": "Booster Density",
    "epistemic_balance": "Epistemic Balance",
    "evidential_markers": "Evidential Markers",
    "reasoning_markers": "Reasoning Markers",
    "specificity_length_norm": "Specificity (Length)",
    "specificity_mean_word_len_norm": "Specificity (Word Len)",
    "specificity_numbers_norm": "Specificity (Numbers)",
    "specificity_content_prop": "Specificity (Content Prop.)",
    "distance_base": "Distance (Base)",
    "similarity_base": "Similarity (Base)",
    "cluster_finetuned": "Cluster (Finetuned)",
    "distance_finetuned": "Distance (Finetuned)",
    "similarity_finetuned": "Similarity (Finetuned)",
    "expertise_llm_score": "LLM Expertise Score",
    "overall_score": "Overall Expertise Score",
}

TIME_WINDOWS = ["start", "middle", "end"]


# ----------------------------- HELPER FUNCTIONS --------------------------- #
def discover_csv_files(base_path: Path, topic_map: Dict[str, str]) -> Dict[str, List[Path]]:
    """Locate CSV files in topic subfolders, grouped by topic name."""
    csv_by_topic: Dict[str, List[Path]] = {}
    if not base_path.exists():
        return csv_by_topic

    for folder, topic_name in topic_map.items():
        folder_path = base_path / folder
        if not folder_path.exists():
            alt = base_path / f"{folder}_csv"
            if alt.exists():
                folder_path = alt
            else:
                continue
        files = sorted(folder_path.glob("*.csv"))
        if files:
            csv_by_topic[topic_name] = files
    return csv_by_topic


def load_data(file_path: Path, min_content_words: Optional[int] = None) -> pd.DataFrame:
    """Load CSV, add row order index, and optionally filter by content word threshold."""
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=False).rename(columns={"index": "_row_order"})
    
    # Filter by content word threshold if specified
    if min_content_words is not None and "num_non_function_words" in df.columns:
        df = df[df["num_non_function_words"] >= min_content_words].copy()
        df = df.reset_index(drop=True)
    
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


def perform_kruskal_wallis(df: pd.DataFrame, feature: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Perform Kruskal-Wallis test across time windows for a feature.
    Returns (H-statistic, p-value, trend_direction).
    trend_direction is 'up', 'down', or 'mixed' based on median comparison.
    """
    groups = []
    medians = {}
    
    for window in TIME_WINDOWS:
        subset = df.loc[df["time_window"] == window, feature].dropna()
        if len(subset) >= 2:  # Need at least 2 samples per group
            groups.append(subset.values)
            medians[window] = subset.median()
        else:
            return None, None, None
    
    if len(groups) < 3:
        return None, None, None
    
    try:
        h_stat, p_value = kruskal(*groups)
    except Exception:
        return None, None, None
    
    # Determine trend direction based on median changes
    if medians["start"] < medians["middle"] < medians["end"]:
        trend = "up"
    elif medians["start"] > medians["middle"] > medians["end"]:
        trend = "down"
    elif medians["start"] < medians["end"]:
        trend = "up"
    elif medians["start"] > medians["end"]:
        trend = "down"
    else:
        trend = "mixed"
    
    return h_stat, p_value, trend


def process_file(file_path: Path, output_dir: Path, min_content_words: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Generate stats report for a single CSV file. 
    Returns the processed df and Kruskal-Wallis results.
    """
    df = load_data(file_path, min_content_words=min_content_words)
    
    if len(df) == 0:
        return df, {}
    
    df = assign_time_windows(df)

    threshold_info = f" (min {min_content_words} content words)" if min_content_words else ""
    lines = [
        f"FILE: {file_path.name}{threshold_info}",
        "=" * max(50, len(file_path.name) + 6 + len(threshold_info)),
        f"Total rows: {len(df)}",
        "",
    ]

    kruskal_results = {}

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

        # Kruskal-Wallis test
        h_stat, p_value, trend = perform_kruskal_wallis(df, feature)
        if h_stat is not None:
            sig_marker = "*" if p_value < 0.05 else ""
            lines.append(f"  KRUSKAL-WALLIS: H = {h_stat:.4f}, p = {p_value:.4f} {sig_marker}")
            lines.append(f"  TREND: {trend}")
            kruskal_results[feature] = {
                "h_stat": h_stat,
                "p_value": p_value,
                "trend": trend,
                "significant": p_value < 0.05
            }
        else:
            lines.append("  KRUSKAL-WALLIS: insufficient data")

        lines.append("")

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_thresh{min_content_words}" if min_content_words else ""
    output_file = output_dir / f"{file_path.stem}_feature_stats{suffix}.txt"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_file}")
    
    return df, kruskal_results


def generate_topic_summary(
    topic_name: str, 
    topic_df: pd.DataFrame, 
    lines: List[str]
) -> None:
    """Append feature stats for a topic (overall + by time window) to lines."""
    lines.append(f"\n{'='*70}")
    lines.append(f"TOPIC: {topic_name}")
    lines.append(f"{'='*70}")
    lines.append(f"Total rows: {len(topic_df)}")
    lines.append("")

    for feature in FEATURE_COLUMNS:
        if feature not in topic_df.columns:
            continue

        lines.append(f"Feature: {feature}")
        lines.append("-" * 50)

        # Overall stats for topic
        overall = compute_stats(topic_df[feature])
        lines.append(f"    OVERALL : {format_stats(overall)}")

        # Stats by time window
        for window in TIME_WINDOWS:
            subset = topic_df.loc[topic_df["time_window"] == window, feature]
            window_stats = compute_stats(subset)
            lines.append(f"    {window.upper():<7} : {format_stats(window_stats)}")

        # Kruskal-Wallis test for aggregated topic data
        h_stat, p_value, trend = perform_kruskal_wallis(topic_df, feature)
        if h_stat is not None:
            sig_marker = "*" if p_value < 0.05 else ""
            lines.append(f"    KRUSKAL-WALLIS: H = {h_stat:.4f}, p = {p_value:.4f} {sig_marker}")
            lines.append(f"    TREND: {trend}")
        else:
            lines.append("    KRUSKAL-WALLIS: insufficient data")

        lines.append("")


def generate_dataset_summary(
    dataset_name: str,
    csv_by_topic: Dict[str, List[Path]],
    output_dir: Path,
    min_content_words: Optional[int] = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Generate a single summary .txt with stats grouped by topic.
    Returns aggregated Kruskal-Wallis results by topic for LaTeX generation.
    """
    threshold_info = f" (min {min_content_words} content words)" if min_content_words else ""
    lines = [
        f"SUMMARY REPORT: {dataset_name.upper()}{threshold_info}",
        "=" * 70,
        f"Generated for {len(csv_by_topic)} topics",
        "",
    ]

    # Store results for LaTeX table: {topic: {feature: [results per file]}}
    topic_kruskal_results: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for topic_name, csv_files in sorted(csv_by_topic.items()):
        # Load and combine all files for this topic
        topic_dfs = []
        for csv_file in csv_files:
            df = load_data(csv_file, min_content_words=min_content_words)
            if len(df) == 0:
                continue
            df = assign_time_windows(df)
            topic_dfs.append(df)
            
            # Also compute per-file Kruskal-Wallis for LaTeX table
            for feature in FEATURE_COLUMNS:
                if feature not in df.columns:
                    continue
                h_stat, p_value, trend = perform_kruskal_wallis(df, feature)
                if h_stat is not None:
                    topic_kruskal_results[topic_name][feature].append({
                        "file": csv_file.name,
                        "h_stat": h_stat,
                        "p_value": p_value,
                        "trend": trend,
                        "significant": p_value < 0.05
                    })
        
        if not topic_dfs:
            continue
        
        topic_df = pd.concat(topic_dfs, ignore_index=True)
        generate_topic_summary(topic_name, topic_df, lines)

    # Write summary report
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_thresh{min_content_words}" if min_content_words else ""
    summary_file = output_dir / f"{dataset_name}_summary_by_topic{suffix}.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated summary: {summary_file}")
    
    return dict(topic_kruskal_results)


def generate_latex_table(
    dataset_name: str,
    csv_by_topic: Dict[str, List[Path]],
    topic_kruskal_results: Dict[str, Dict[str, List[Dict]]],
    output_dir: Path,
    min_content_words: Optional[int] = None
) -> None:
    """Generate LaTeX table summarizing temporal trends by topic."""
    
    threshold_info = f" (min {min_content_words} content words)" if min_content_words else ""
    suffix = f"_thresh{min_content_words}" if min_content_words else ""
    
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        f"    \\caption{{Aggregated Temporal Trends for {dataset_name.title()}{threshold_info}. \\textbf{{Note:}} Arrows ($\\nearrow, \\searrow$) indicate the dominant direction of significant changes ($p < 0.05$). Percentages represent the proportion of files in that topic showing this specific trend.}}",
        f"    \\label{{tab:temporal_trends_{dataset_name}{suffix}}}",
        r"    \begin{tabular}{llcl}",
        r"        \toprule",
        r"        \textbf{Topic} & \textbf{Feature} & \textbf{Trend} & \textbf{\% of Sessions} \\",
        r"        \midrule",
    ]
    
    for topic_name in sorted(topic_kruskal_results.keys()):
        feature_results = topic_kruskal_results[topic_name]
        n_files = len(csv_by_topic.get(topic_name, []))
        
        # Aggregate trends per feature
        feature_trends = []
        for feature, results in feature_results.items():
            sig_results = [r for r in results if r["significant"]]
            if not sig_results:
                continue
            
            up_count = sum(1 for r in sig_results if r["trend"] == "up")
            down_count = sum(1 for r in sig_results if r["trend"] == "down")
            mixed_count = sum(1 for r in sig_results if r["trend"] == "mixed")
            
            total_sig = len(sig_results)
            percentage = (total_sig / n_files) * 100 if n_files > 0 else 0
            
            # Determine dominant trend
            if up_count > down_count and up_count > mixed_count:
                trend_symbol = r"$\nearrow$"
                trend_color = r"\textcolor{blue}"
            elif down_count > up_count and down_count > mixed_count:
                trend_symbol = r"$\searrow$"
                trend_color = r"\textcolor{red}"
            else:
                trend_symbol = "Mixed"
                trend_color = ""
            
            if trend_color and trend_symbol != "Mixed":
                trend_display = f"{trend_color}{{{trend_symbol}}}"
            else:
                trend_display = trend_symbol
            
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
            feature_trends.append({
                "feature": display_name,
                "trend": trend_display,
                "percentage": percentage,
                "count": total_sig,
                "total": n_files
            })
        
        # Sort by percentage descending
        feature_trends.sort(key=lambda x: x["percentage"], reverse=True)
        
        # Take top 5 trends for each topic
        top_trends = feature_trends[:5]
        
        if not top_trends:
            continue
        
        # First row includes topic name
        first = top_trends[0]
        lines.append(
            f"        \\textbf{{{topic_name}}} & {first['feature']} & {first['trend']} & "
            f"{first['percentage']:.1f}\\% ({first['count']}/{first['total']}) \\\\"
        )
        lines.append(f"        ($N={n_files}$ sessions) & & & \\\\")
        
        # Subsequent rows
        for trend in top_trends[1:]:
            lines.append(
                f"         & {trend['feature']} & {trend['trend']} & "
                f"{trend['percentage']:.1f}\\% ({trend['count']}/{trend['total']}) \\\\"
            )
        
        # Check for stable features
        stable_count = 0
        for feature in FEATURE_COLUMNS:
            if feature in feature_results:
                sig_count = sum(1 for r in feature_results[feature] if r["significant"])
                if sig_count == 0:
                    stable_count += 1
            else:
                stable_count += 1
        
        if stable_count > len(FEATURE_COLUMNS) * 0.5:
            stable_pct = (stable_count / len(FEATURE_COLUMNS)) * 100
            lines.append(
                f"         & \\textit{{*Most features stable}} & -- & {stable_pct:.1f}\\% \\\\"
            )
        
        lines.append(r"        \addlinespace")
    
    # Remove last \addlinespace
    if lines[-1] == r"        \addlinespace":
        lines.pop()
    
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    
    # Write LaTeX file
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_file = output_dir / f"{dataset_name}_temporal_trends{suffix}.tex"
    latex_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated LaTeX table: {latex_file}")


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
        csv_by_topic = discover_csv_files(base_path, topics)
        if not csv_by_topic:
            continue

        any_files = True
        total_files = sum(len(files) for files in csv_by_topic.values())
        
        # Process for each content word threshold
        for threshold in CONTENT_WORD_THRESHOLDS:
            print(f"\n{'='*70}")
            print(f"Processing {dataset_name}: {total_files} files across {len(csv_by_topic)} topics")
            print(f"Content word threshold: >= {threshold}")
            print(f"{'='*70}")
            
            dataset_output = output_dir / dataset_name / f"threshold_{threshold}"

            # Process individual files
            for topic_name, csv_files in csv_by_topic.items():
                for csv_file in csv_files:
                    process_file(csv_file, dataset_output, min_content_words=threshold)

            # Generate topic summary and get Kruskal-Wallis results
            topic_kruskal_results = generate_dataset_summary(
                dataset_name, csv_by_topic, dataset_output, min_content_words=threshold
            )
            
            # Generate LaTeX table
            generate_latex_table(
                dataset_name, csv_by_topic, topic_kruskal_results, dataset_output, min_content_words=threshold
            )

    if not any_files:
        print(f"No CSV files found under {input_dir}")


if __name__ == "__main__":
    main()