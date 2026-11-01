#!/usr/bin/env python3
"""
Analyze Context Size Calibration Results
Computes ICC, Spearman correlations, systematic bias tests, and generates visualizations.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, friedmanchisquare, pearsonr
import pingouin as pg

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUDGET_ORDER: List[Union[int, str]] = [0, 500, 1000, 1500, 2000, "20_utterances"]
TOKEN_BUDGETS: List[int] = [0, 500, 1000, 1500, 2000]
CI95_Z: float = 1.96           # z-score for 95% confidence interval
ALPHA: float = 0.05            # significance threshold
ICC_EXCELLENT: float = 0.80    # ICC >= 0.80 → excellent reliability
ICC_GOOD: float = 0.70         # ICC >= 0.70 → good reliability
ICC_MODERATE: float = 0.60     # ICC >= 0.60 → moderate reliability
DEFAULT_MAD_THRESHOLD: float = 1.0
MIN_ASSEMBLY_UTTERANCES: int = 10
PLOT_DPI: int = 300
SCORE_MIN: int = 1
SCORE_MAX: int = 5


def load_results(results_path: Path) -> pd.DataFrame:
    """Load calibration results CSV."""
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    return df


def pivot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape data: rows = utterances, columns = budget conditions, values = scores.

    Args:
        df: Results dataframe with one row per (utterance, condition)

    Returns:
        Pivoted dataframe with utterances as rows, conditions as columns
    """
    pivot = df.pivot_table(
        index='utterance_id',
        columns='budget_condition',
        values='score',
        aggfunc='first'  # Should be only one score per (utterance, condition)
    )

    for col in pivot.columns:
        pivot[col] = pd.to_numeric(pivot[col], errors='coerce')

    return pivot


def compute_icc(pivot_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute Intraclass Correlation Coefficient (ICC) across all conditions.
    Uses ICC(2,1) - two-way random effects, single rater.

    Args:
        pivot_df: Pivoted scores (utterances × conditions)

    Returns:
        Tuple of (ICC value, lower 95% CI, upper 95% CI)
    """
    long_df = pivot_df.reset_index().melt(
        id_vars='utterance_id',
        var_name='condition',
        value_name='score'
    )
    long_df = long_df.dropna(subset=['score'])

    icc_result = pg.intraclass_corr(
        data=long_df,
        targets='utterance_id',
        raters='condition',
        ratings='score'
    )

    # Get ICC(2,1) - "ICC2" in pingouin terminology
    icc2_row = icc_result[icc_result['Type'] == 'ICC2']

    if len(icc2_row) == 0:
        print("Warning: Could not compute ICC2, trying ICC3")
        icc2_row = icc_result[icc_result['Type'] == 'ICC3']

    icc_val = icc2_row['ICC'].values[0]
    ci_lower = icc2_row['CI95%'].values[0][0]
    ci_upper = icc2_row['CI95%'].values[0][1]

    return icc_val, ci_lower, ci_upper


def compute_pairwise_spearman(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Spearman correlations between all conditions.

    Args:
        pivot_df: Pivoted scores (utterances × conditions)

    Returns:
        Correlation matrix as DataFrame
    """
    conditions = pivot_df.columns.tolist()
    n_conditions = len(conditions)

    corr_matrix = np.zeros((n_conditions, n_conditions))

    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                valid_mask = pivot_df[cond1].notna() & pivot_df[cond2].notna()
                if valid_mask.sum() > 1:
                    rho, _ = spearmanr(
                        pivot_df.loc[valid_mask, cond1],
                        pivot_df.loc[valid_mask, cond2]
                    )
                    corr_matrix[i, j] = rho
                else:
                    corr_matrix[i, j] = np.nan

    return pd.DataFrame(corr_matrix, index=conditions, columns=conditions)


def compute_score_range_per_utterance(pivot_df: pd.DataFrame) -> pd.Series:
    """
    Compute score range (max - min) across conditions for each utterance.

    Args:
        pivot_df: Pivoted scores

    Returns:
        Series with utterance_id as index, score_range as values
    """
    return pivot_df.max(axis=1) - pivot_df.min(axis=1)


def compute_mad_per_utterance(pivot_df: pd.DataFrame) -> pd.Series:
    """
    Compute Mean Absolute Deviation from mean across conditions for each utterance.

    Args:
        pivot_df: Pivoted scores

    Returns:
        Series with MAD values
    """
    mean_score = pivot_df.mean(axis=1)
    return pivot_df.sub(mean_score, axis=0).abs().mean(axis=1)


def friedman_test(pivot_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Friedman test for systematic differences across conditions.
    Non-parametric alternative to repeated measures ANOVA.

    Args:
        pivot_df: Pivoted scores

    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    complete_df = pivot_df.dropna()
    conditions_data = [complete_df[col].values for col in complete_df.columns]

    if len(conditions_data) < 2:
        return np.nan, np.nan

    stat, pval = friedmanchisquare(*conditions_data)
    return stat, pval


def analyze_context_sensitivity_by_length(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    Test if score variability (range) correlates with utterance length.

    Hypothesis: shorter utterances are more context-sensitive.

    Args:
        df: Original results dataframe
        pivot_df: Pivoted scores

    Returns:
        Tuple of (Pearson r, p-value)
    """
    score_range = compute_score_range_per_utterance(pivot_df)
    token_counts = df.groupby('utterance_id')['target_token_count'].first()

    common_idx = score_range.index.intersection(token_counts.index)
    r, pval = pearsonr(token_counts.loc[common_idx], score_range.loc[common_idx])
    return r, pval


def analyze_by_assembly(df: pd.DataFrame, pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC for each assembly separately.

    Args:
        df: Original results dataframe
        pivot_df: Pivoted scores

    Returns:
        DataFrame with assembly-specific ICC values
    """
    assembly_map = df.groupby('utterance_id')['assembly'].first()
    results = []

    for assembly in df['assembly'].unique():
        utt_ids = assembly_map[assembly_map == assembly].index
        pivot_subset = pivot_df.loc[pivot_df.index.isin(utt_ids)]

        if len(pivot_subset) < MIN_ASSEMBLY_UTTERANCES:
            print(f"  Warning: Only {len(pivot_subset)} utterances for {assembly}, skipping ICC")
            continue

        try:
            icc, ci_low, ci_high = compute_icc(pivot_subset)
            results.append({
                'assembly': assembly,
                'n_utterances': len(pivot_subset),
                'icc': icc,
                'ci_lower': ci_low,
                'ci_upper': ci_high
            })
        except Exception as e:
            print(f"  Error computing ICC for {assembly}: {e}")

    return pd.DataFrame(results)


def plot_mean_scores_by_budget(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot mean expertise score ± 95% CI by context budget.
    Shows if there's systematic bias (scores trending with context size).

    Args:
        df: Results dataframe
        output_dir: Directory to save plot
    """
    summary = df.groupby('budget_condition')['score'].agg([
        'mean', 'std', 'count', 'sem'
    ]).reset_index()

    summary['ci95'] = CI95_Z * summary['sem']
    summary['budget_condition'] = pd.Categorical(
        summary['budget_condition'],
        categories=BUDGET_ORDER,
        ordered=True
    )
    summary = summary.sort_values('budget_condition')

    plt.figure(figsize=(10, 6))
    x = range(len(summary))
    plt.errorbar(
        x,
        summary['mean'],
        yerr=summary['ci95'],
        marker='o',
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        label='Mean ± 95% CI'
    )
    plt.xticks(x, summary['budget_condition'], rotation=45)
    plt.xlabel('Context Budget (tokens)', fontsize=12)
    plt.ylabel('Mean Expertise Score', fontsize=12)
    plt.title('Mean Expertise Score by Context Budget', fontsize=14, fontweight='bold')
    plt.ylim(SCORE_MIN, SCORE_MAX)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'mean_score_by_budget.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_spearman_heatmap(corr_matrix: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot heatmap of pairwise Spearman correlations.

    Args:
        corr_matrix: Pairwise correlation matrix
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0.8,
        vmin=0.6,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Spearman ρ'}
    )
    plt.title('Pairwise Spearman Correlations Between Context Conditions',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'spearman_heatmap.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_icc_by_assembly(assembly_icc_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot ICC values by assembly with error bars.

    Args:
        assembly_icc_df: DataFrame with assembly-specific ICC values
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(8, 6))
    x = range(len(assembly_icc_df))
    plt.errorbar(
        x,
        assembly_icc_df['icc'],
        yerr=[
            assembly_icc_df['icc'] - assembly_icc_df['ci_lower'],
            assembly_icc_df['ci_upper'] - assembly_icc_df['icc']
        ],
        marker='o',
        markersize=10,
        capsize=7,
        capthick=2,
        linewidth=2,
        linestyle='none'
    )
    plt.xticks(x, assembly_icc_df['assembly'], rotation=45, ha='right')
    plt.ylabel('ICC (2,1)', fontsize=12)
    plt.title('Score Reliability by Assembly', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.axhline(y=ICC_EXCELLENT, color='red', linestyle='--', alpha=0.5,
                label=f'ICC = {ICC_EXCELLENT} threshold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'icc_by_assembly.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_score_range_vs_length(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Scatter plot: score range vs. target utterance token count.

    Args:
        df: Results dataframe
        pivot_df: Pivoted scores
        output_dir: Directory to save plot
    """
    score_range = compute_score_range_per_utterance(pivot_df)
    token_counts = df.groupby('utterance_id')['target_token_count'].first()

    common_idx = score_range.index.intersection(token_counts.index)
    plot_df = pd.DataFrame({
        'token_count': token_counts.loc[common_idx],
        'score_range': score_range.loc[common_idx]
    })

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_df['token_count'], plot_df['score_range'], alpha=0.5, s=30)

    z = np.polyfit(plot_df['token_count'], plot_df['score_range'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df['token_count'].min(), plot_df['token_count'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2,
             label=f'Linear fit: y = {z[0]:.4f}x + {z[1]:.2f}')

    r, pval = pearsonr(plot_df['token_count'], plot_df['score_range'])
    plt.text(
        0.05, 0.95,
        f'Pearson r = {r:.3f}\np = {pval:.4f}',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    plt.xlabel('Target Utterance Token Count', fontsize=12)
    plt.ylabel('Score Range (max - min across conditions)', fontsize=12)
    plt.title('Context Sensitivity vs. Utterance Length', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'score_range_vs_length.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_actual_tokens_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Histogram of actual tokens used per budget condition.
    Shows variance introduced by "approximately N tokens" rule.

    Args:
        df: Results dataframe
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, budget in enumerate(BUDGET_ORDER):
        ax = axes[i]
        subset = df[df['budget_condition'] == budget]
        ax.hist(
            subset['actual_tokens_used'],
            bins=30,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        ax.axvline(
            subset['actual_tokens_used'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {subset["actual_tokens_used"].mean():.0f}'
        )
        ax.set_xlabel('Actual Tokens Used', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Budget: {budget}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'actual_tokens_distribution.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def identify_high_mad_utterances(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    threshold: float = DEFAULT_MAD_THRESHOLD,
    output_dir: Path = Path(".")
) -> pd.DataFrame:
    """
    Identify and save utterances with high Mean Absolute Deviation (context-sensitive).

    Args:
        df: Results dataframe
        pivot_df: Pivoted scores
        threshold: MAD threshold (default 1.0)
        output_dir: Directory to save results

    Returns:
        DataFrame of high-MAD utterances with their texts
    """
    mad = compute_mad_per_utterance(pivot_df)
    high_mad = mad[mad >= threshold].sort_values(ascending=False)

    if len(high_mad) == 0:
        print(f"  No utterances with MAD >= {threshold}")
        return pd.DataFrame()

    high_mad_data = []
    for utt_id in high_mad.index.tolist():
        utt_row = df[df['utterance_id'] == utt_id].iloc[0]
        scores = pivot_df.loc[utt_id].to_dict()
        high_mad_data.append({
            'utterance_id': utt_id,
            'assembly': utt_row['assembly'],
            'speaker': utt_row['speaker'],
            'text': utt_row['target_text'],
            'token_count': utt_row['target_token_count'],
            'mad': mad[utt_id],
            'score_range': pivot_df.loc[utt_id].max() - pivot_df.loc[utt_id].min(),
            **{f'score_{k}': v for k, v in scores.items()}
        })

    high_mad_df = pd.DataFrame(high_mad_data)

    output_path = output_dir / f'high_mad_utterances_threshold_{threshold}.csv'
    high_mad_df.to_csv(output_path, index=False)
    print(f"  Saved {len(high_mad_df)} high-MAD utterances to: {output_path}")

    return high_mad_df


def generate_summary_report(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    icc_val: float,
    ci_lower: float,
    ci_upper: float,
    friedman_stat: float,
    friedman_p: float,
    corr_matrix: pd.DataFrame,
    output_dir: Path = Path(".")
) -> None:
    """
    Generate a text summary report with key findings.

    Args:
        df: Results dataframe
        pivot_df: Pivoted scores
        icc_val: Overall ICC value
        ci_lower: ICC 95% CI lower bound
        ci_upper: ICC 95% CI upper bound
        friedman_stat: Friedman test statistic
        friedman_p: Friedman test p-value
        corr_matrix: Spearman correlation matrix
        output_dir: Directory to save report
    """
    report_lines = []

    report_lines.append("="*80)
    report_lines.append("CONTEXT SIZE CALIBRATION: SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("SAMPLE")
    report_lines.append("-"*80)
    report_lines.append(f"Total utterances sampled: {len(pivot_df)}")
    report_lines.append(f"Assemblies: {', '.join(df['assembly'].unique())}")
    n_by_assembly = df.groupby('assembly')['utterance_id'].nunique()
    for assembly, n in n_by_assembly.items():
        report_lines.append(f"  - {assembly}: {n} utterances")
    report_lines.append("")

    report_lines.append("SCORE RELIABILITY (ICC)")
    report_lines.append("-"*80)
    report_lines.append(f"ICC(2,1) across all context conditions: {icc_val:.3f}")
    report_lines.append(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    if icc_val >= ICC_EXCELLENT:
        interpretation = "Excellent reliability - scores are highly consistent across context sizes"
    elif icc_val >= ICC_GOOD:
        interpretation = "Good reliability - scores are reasonably consistent"
    elif icc_val >= ICC_MODERATE:
        interpretation = "Moderate reliability - some context sensitivity present"
    else:
        interpretation = "Poor reliability - scores vary substantially with context"

    report_lines.append(f"Interpretation: {interpretation}")
    report_lines.append("")

    report_lines.append("SYSTEMATIC BIAS TEST (Friedman)")
    report_lines.append("-"*80)
    report_lines.append(f"Friedman χ² = {friedman_stat:.2f}, p = {friedman_p:.4f}")

    if friedman_p < ALPHA:
        bias_interpretation = "SIGNIFICANT systematic effect - mean scores differ across conditions"
    else:
        bias_interpretation = "No significant systematic effect - mean scores stable across conditions"

    report_lines.append(f"Interpretation: {bias_interpretation}")
    report_lines.append("")

    report_lines.append("MEAN SCORES BY CONDITION")
    report_lines.append("-"*80)
    mean_scores = df.groupby('budget_condition')['score'].mean().sort_index()
    for condition, mean in mean_scores.items():
        report_lines.append(f"  {str(condition):15s}: {mean:.3f}")
    report_lines.append("")

    report_lines.append("PAIRWISE SPEARMAN CORRELATIONS")
    report_lines.append("-"*80)
    if '20_utterances' in corr_matrix.columns:
        baseline_corrs = corr_matrix['20_utterances'].drop('20_utterances').sort_values(ascending=False)
        report_lines.append("Correlations with 20-utterance baseline:")
        for condition, rho in baseline_corrs.items():
            report_lines.append(f"  {str(condition):15s}: ρ = {rho:.3f}")
        report_lines.append("")

    min_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).min().min()
    report_lines.append(f"Minimum pairwise correlation: {min_corr:.3f}")
    report_lines.append("")

    report_lines.append("CONTEXT SENSITIVITY BY UTTERANCE LENGTH")
    report_lines.append("-"*80)
    r, pval = analyze_context_sensitivity_by_length(df, pivot_df)
    report_lines.append(f"Correlation (length vs. score range): r = {r:.3f}, p = {pval:.4f}")

    if pval < ALPHA:
        if r < 0:
            length_interpretation = "Shorter utterances are MORE context-sensitive (significant)"
        else:
            length_interpretation = "Longer utterances are MORE context-sensitive (significant)"
    else:
        length_interpretation = "No significant relationship between length and context sensitivity"

    report_lines.append(f"Interpretation: {length_interpretation}")
    report_lines.append("")

    mad = compute_mad_per_utterance(pivot_df)
    high_mad_count = (mad >= DEFAULT_MAD_THRESHOLD).sum()
    report_lines.append("HIGHLY CONTEXT-SENSITIVE UTTERANCES")
    report_lines.append("-"*80)
    report_lines.append(
        f"Utterances with MAD >= {DEFAULT_MAD_THRESHOLD}: "
        f"{high_mad_count} ({high_mad_count/len(mad)*100:.1f}%)"
    )
    report_lines.append("")

    report_lines.append("RECOMMENDATION")
    report_lines.append("-"*80)

    recommendation = []
    if icc_val >= ICC_EXCELLENT and friedman_p >= ALPHA:
        recommendation.append(f"✓ High reliability (ICC >= {ICC_EXCELLENT})")
        recommendation.append(f"✓ No systematic bias (Friedman p >= {ALPHA})")

        median_20_tokens = df[df['budget_condition'] == '20_utterances']['actual_tokens_used'].median()
        closest_budget = min(TOKEN_BUDGETS, key=lambda x: abs(x - median_20_tokens))

        recommendation.append(f"✓ 20-utterance baseline has median {median_20_tokens:.0f} tokens")
        recommendation.append("")
        recommendation.append(f"RECOMMENDED CONTEXT BUDGET: {closest_budget} tokens")
        recommendation.append(f"  (Closest to existing 20-utterance approach)")

    elif icc_val >= ICC_EXCELLENT:
        recommendation.append(f"✓ High reliability (ICC >= {ICC_EXCELLENT})")
        recommendation.append("⚠ Systematic bias detected - investigate mean score trends")
        recommendation.append("")
        recommendation.append("RECOMMENDED: Choose budget based on mean score stability")

    else:
        recommendation.append("⚠ Moderate reliability - context matters")
        recommendation.append("")
        recommendation.append("RECOMMENDED: Use higher context budget (1500-2000 tokens)")
        recommendation.append("  OR report context as a potential confound in analysis")

    for line in recommendation:
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("="*80)

    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n  Saved summary report to: {report_path}")
    print("\n" + '\n'.join(report_lines))


def main() -> None:
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze context size calibration results"
    )
    parser.add_argument(
        '--results-file',
        required=True,
        help='Path to context_calibration_results.csv'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for plots and reports (defaults to same directory as results file)'
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    output_dir = Path(args.output_dir) if args.output_dir is not None else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CONTEXT SIZE CALIBRATION: ANALYSIS")
    print("="*80)
    print(f"Results file: {results_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    print("\n[1/8] Loading results...")
    df = load_results(results_path)

    print("\n[2/8] Reshaping data...")
    pivot_df = pivot_scores(df)
    print(f"  Pivoted shape: {pivot_df.shape} (utterances × conditions)")

    print("\n[3/8] Computing ICC...")
    icc_val, ci_lower, ci_upper = compute_icc(pivot_df)
    print(f"  ICC(2,1) = {icc_val:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")

    print("\n[4/8] Computing pairwise Spearman correlations...")
    corr_matrix = compute_pairwise_spearman(pivot_df)
    print("  Correlation matrix computed")

    print("\n[5/8] Running Friedman test for systematic bias...")
    friedman_stat, friedman_p = friedman_test(pivot_df)
    print(f"  Friedman χ² = {friedman_stat:.2f}, p = {friedman_p:.4f}")

    print("\n[6/8] Computing assembly-specific ICC...")
    assembly_icc_df = analyze_by_assembly(df, pivot_df)
    print(assembly_icc_df.to_string(index=False))

    print("\n[7/8] Identifying context-sensitive utterances...")
    high_mad_df = identify_high_mad_utterances(
        df, pivot_df, threshold=DEFAULT_MAD_THRESHOLD, output_dir=output_dir
    )

    print("\n[8/8] Generating visualizations...")
    plot_mean_scores_by_budget(df, output_dir)
    plot_spearman_heatmap(corr_matrix, output_dir)

    if len(assembly_icc_df) > 0:
        plot_icc_by_assembly(assembly_icc_df, output_dir)

    plot_score_range_vs_length(df, pivot_df, output_dir)
    plot_actual_tokens_distribution(df, output_dir)

    print("\nGenerating summary report...")
    generate_summary_report(
        df, pivot_df,
        icc_val, ci_lower, ci_upper,
        friedman_stat, friedman_p,
        corr_matrix,
        output_dir
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
