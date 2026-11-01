"""
Comprehensive Correlation Analysis Script

This script performs correlation analysis on assembly and expert speech data,
examining relationships between linguistic features and expertise measures.

Features:
- Loads and combines CSV files from assembly (8th amendment, gender equality,
  Scotland) and expert (constitution, future, gender) datasets
- Filters texts with configurable non-function word thresholds (5, 10, 20)
- Dynamically extracts numeric features, separating base vs finetuned model columns
- Applies feature-specific scaling based on each feature's computation method
- Computes Pearson correlation matrices with p-values
- Generates correlation heatmaps, p-value heatmaps, and temporal evolution plots
- Outputs results as PNG visualizations, JSON data, and text summaries

Usage:
    python corr.py

The script runs all combinations of:
- Content word threshold: 5, 10, 20
- Normalization scope: 'global' or 'by_topic'
- Include Scotland data: 'yes' or 'no'

Output structure:
    code/correlation_analysis/threshold_{N}/
        scotland{yes|no}_{norm}_featurespecific/
            assemblies/
            experts/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))
from shared.topic_maps import ASSEMBLY_TOPICS, EXPERT_TOPICS
from correlation_utils import (
    filter_syntactically_relevant, get_dynamic_features,
    format_feature_name, load_and_combine_data,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA: float = 0.05
PLOT_DPI: int = 300
MIN_VALID_PAIRS: int = 3
EPISTEMIC_BALANCE_NEUTRAL: float = 0.5
CORR_STRONG: float = 0.7
CORR_MODERATE: float = 0.4
EMBEDDING_KEYWORDS: List[str] = ['distance', 'similarity', 'expertise', 'overall']
MODEL_TYPES: List[str] = ['base', 'finetuned']
CONTENT_WORD_THRESHOLDS: List[int] = [5, 10, 20]
NORM_SCOPES: List[str] = ['global', 'by_topic']
SCOTLAND_OPTIONS: List[str] = ['yes', 'no']
TEMPORAL_PERIOD_COLORS: Dict[str, str] = {
    'Start': '#89CFF0', 'Middle': '#90EE90', 'End': '#FFB6C1',
}


def scale_features_for_correlation(df: pd.DataFrame, scope: str = 'global') -> pd.DataFrame:
    """
    Scale features appropriately for correlation analysis based on how each feature is computed.

    Feature-specific scaling rationale:
    - Group 1 (No scaling): Already [0,1] bounded with good distributions
    - Group 2 (MinMax): [0,1] bounded but may benefit from normalization for consistency
    - Group 3 (RobustScaler): Zero-inflated or unbounded, robust to outliers
    - Group 4 (StandardScaler): Approximately normal, benefits from z-score
    - Group 5 (Log + StandardScaler): Count data with power-law distribution
    - Group 6 (StandardScaler for embedding distances): Model-derived metrics

    Args:
        df: DataFrame with features
        scope: 'global' or 'by_topic' for scaling scope
    """
    df_scaled = df.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 1: NO SCALING NEEDED
    # Already [0,1] bounded with appropriate distributions
    # ═══════════════════════════════════════════════════════════════════════════
    # - jargon_density: log1p(count)/log1p(tokens) → [0,1], log handles skew
    # - lexical_sophistication: log1p(count)/log1p(tokens) → [0,1]
    # - syntactic_complexity: log1p(raw)/log1p(tokens) → [0,~1]
    # - epistemic_balance: hedge/(hedge+booster) → [0,1] proportion
    # - specificity_word_length: clip((mean-3)/5, 0, 1) → [0,1] explicitly
    # - mrc_familiarity: clip((mean-100)/600, 0, 1) → [0,1] explicitly
    # - mrc_concreteness: same as above
    # - mrc_imageability: same as above
    no_scale_cols = [
        'jargon_density', 'lexical_sophistication', 'syntactic_complexity',
        'epistemic_balance', 'specificity_word_length',
        'mrc_familiarity', 'mrc_concreteness', 'mrc_imageability'
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 2: MINMAX SCALING
    # [0,1] bounded but right-skewed from sqrt(ratio), benefits from rescaling
    # ═══════════════════════════════════════════════════════════════════════════
    # - hedge_density: sqrt(count/tokens) → right-skewed
    # - booster_density: sqrt(count/tokens) → right-skewed
    minmax_cols = ['hedge_density', 'booster_density']

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 3: ROBUST SCALING
    # Zero-inflated or potentially unbounded, need outlier-resistant scaling
    # Uses median and IQR instead of mean and std
    # ═══════════════════════════════════════════════════════════════════════════
    # - evidential_markers: sqrt(regex_matches/tokens) → many zeros, rare high values
    # - specificity_numbers: sqrt(num_count/tokens) → many texts have no numbers
    # - subjectivity_score: sqrt(accumulated_weights/tokens) → unbounded, varies widely
    robust_cols = ['evidential_markers', 'specificity_numbers', 'subjectivity_score']

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 4: STANDARD (Z-SCORE) SCALING
    # Approximately normal distribution, benefits from centering and unit variance
    # ═══════════════════════════════════════════════════════════════════════════
    # - reasoning_markers: sqrt(count/tokens) → more common than evidential,
    #   includes frequent words like "because", "therefore", "but"
    standard_cols = ['reasoning_markers']

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 5: LOG TRANSFORM + STANDARD SCALING
    # Raw counts with power-law distribution, need log to normalize
    # ═══════════════════════════════════════════════════════════════════════════
    # - length_tokens: raw count → highly right-skewed
    # - num_non_function_words: raw count → correlated with length, right-skewed
    log_standard_cols = ['length_tokens', 'num_non_function_words']

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP 6: STANDARD SCALING FOR MODEL-DERIVED METRICS
    # Embedding distances/similarities from centroid analysis
    # These are continuous and roughly normal, benefit from standardization
    # ═══════════════════════════════════════════════════════════════════════════
    # Numeric-only: 'expertise' also matches the free-text expertise_justification
    # column, which would crash StandardScaler
    embedding_cols = [col for col in df_scaled.columns
                      if any(x in col.lower() for x in EMBEDDING_KEYWORDS)
                      and pd.api.types.is_numeric_dtype(df_scaled[col])]

    if scope == 'by_topic' and 'topic' in df_scaled.columns:
        topics = df_scaled['topic'].unique()
    else:
        topics = [None]  # Global scaling

    def apply_scaling(columns, scaler_class, log_transform=False):
        """Apply scaling to columns, optionally by topic."""
        existing = [c for c in columns if c in df_scaled.columns]
        if not existing:
            return 0

        for topic in topics:
            if topic is not None:
                mask = df_scaled['topic'] == topic
            else:
                mask = slice(None)

            subset = df_scaled.loc[mask, existing].copy()
            subset = subset.fillna(0)

            if log_transform:
                subset = np.log1p(subset)

            if scaler_class is not None:
                scaler = scaler_class()
                scaled_values = scaler.fit_transform(subset)
                df_scaled.loc[mask, existing] = scaled_values
            elif log_transform:
                df_scaled.loc[mask, existing] = subset

        return len(existing)

    scaled_info = {}

    if 'epistemic_balance' in df_scaled.columns:
        df_scaled['epistemic_balance'] = df_scaled['epistemic_balance'].fillna(EPISTEMIC_BALANCE_NEUTRAL)
    scaled_info['no_scale'] = len([c for c in no_scale_cols if c in df_scaled.columns])

    scaled_info['minmax'] = apply_scaling(minmax_cols, MinMaxScaler)
    scaled_info['robust'] = apply_scaling(robust_cols, RobustScaler)
    scaled_info['standard'] = apply_scaling(standard_cols, StandardScaler)
    scaled_info['log_standard'] = apply_scaling(log_standard_cols, StandardScaler, log_transform=True)
    scaled_info['embedding'] = apply_scaling(embedding_cols, StandardScaler)

    scope_desc = f"by topic ({len(topics)} topics)" if topics[0] is not None else "globally"
    print(f"   ✓ Feature-specific scaling applied {scope_desc}:")
    print(f"      - No scaling: {scaled_info['no_scale']} features (already normalized)")
    print(f"      - MinMax: {scaled_info['minmax']} features (hedge/booster density)")
    print(f"      - Robust: {scaled_info['robust']} features (zero-inflated)")
    print(f"      - Standard: {scaled_info['standard']} features (reasoning markers)")
    print(f"      - Log+Standard: {scaled_info['log_standard']} features (count data)")
    print(f"      - Standard (embeddings): {scaled_info['embedding']} features")

    return df_scaled


class ComprehensiveCorrelationAnalyzer:
    def __init__(self, base_dir: Path = Path('data/assemblies/new_feats'),
                 norm_type: str = 'global',
                 include_scotland: str = 'yes',
                 content_word_threshold: int = 5,
                 output_root: Path = Path('code/correlation_analysis')):
        """
        Initialize analyzer.

        Args:
            base_dir: Path to data directory
            norm_type: 'global' or 'by_topic' - scope for feature scaling
            include_scotland: 'yes' or 'no' - whether to include Scotland data
            content_word_threshold: Minimum number of non-function words required (5, 10, or 20)
            output_root: Directory under which threshold_{N}/... result trees are written

        Note: scaler_type is no longer needed as each feature uses its optimal scaler
        based on how it's computed (see scale_features_for_correlation function).
        """
        self.base_dir = Path(base_dir)
        self.output_root = Path(output_root)
        self.assemblies_path = self.base_dir
        self.experts_path = self.base_dir / 'experts'
        self.norm_type = norm_type
        self.include_scotland = include_scotland
        self.content_word_threshold = content_word_threshold

        # COLUMNS TO IGNORE during dynamic extraction
        # These are either textual, metadata, or derived columns that shouldn't be correlated
        self.excluded_columns = [
            # Textual columns (explicit names)
            'text', 'speech',
            'speaker', 'name',
            'date',
            'justification',
            # Columns to exclude from analysis
            'Unnamed',  # pandas index artifacts
            'hedge_ratio',  # redundant with epistemic_balance
            'cluster_',  # clustering results, not features
            'id', 'index', 'row',  # identifiers
        ]

        if self.include_scotland == "yes":
            self.assembly_topics = ASSEMBLY_TOPICS
        else:
            self.assembly_topics = {k: v for k, v in ASSEMBLY_TOPICS.items() if k != "scotland"}

        self.expert_topics = EXPERT_TOPICS

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature-specific scaling based on how each feature is computed.

        This method uses the global scale_features_for_correlation function
        which applies appropriate scaling to each feature type:
        - No scaling for already-bounded features (MRC, epistemic_balance, etc.)
        - MinMax for hedge/booster density
        - RobustScaler for zero-inflated features (evidential, specificity_numbers)
        - StandardScaler for reasoning markers and embedding metrics
        - Log + StandardScaler for count data (length, non-function words)
        """
        scope = self.norm_type
        if scope == 'by_topic' and 'topic' not in df.columns:
            print("   ⚠️  'topic' column not found, falling back to global normalization")
            scope = 'global'
        return scale_features_for_correlation(df, scope=scope)

    def compute_correlations(self, df: pd.DataFrame, features: List[str],
                            method: str = 'pearson') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Compute correlation matrix with p-values."""
        available = [f for f in features if f in df.columns]

        if len(available) < 2:
            print(f"   ⚠️  Only {len(available)} features available, skipping")
            return None, None

        print(f"   ✓ Computing correlations for {len(available)} features")

        numeric_df = df[available]
        n = len(numeric_df.columns)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i, col1 in enumerate(numeric_df.columns):
            for j, col2 in enumerate(numeric_df.columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    mask = ~(numeric_df[col1].isna() | numeric_df[col2].isna())
                    valid_count = mask.sum()

                    if valid_count < MIN_VALID_PAIRS:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
                    else:
                        try:
                            if method == 'pearson':
                                corr, p_val = pearsonr(numeric_df[col1][mask],
                                                       numeric_df[col2][mask])
                            else:
                                corr, p_val = spearmanr(numeric_df[col1][mask],
                                                        numeric_df[col2][mask])
                            corr_matrix[i, j] = corr
                            p_matrix[i, j] = p_val
                        except Exception:
                            corr_matrix[i, j] = np.nan
                            p_matrix[i, j] = np.nan

        display_names = [format_feature_name(col) for col in numeric_df.columns]
        corr_df = pd.DataFrame(corr_matrix, index=display_names, columns=display_names)
        p_df = pd.DataFrame(p_matrix, index=display_names, columns=display_names)

        return corr_df, p_df

    def create_correlation_heatmaps(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                                   title: str, output_dir: Path) -> None:
        """Create and save separate correlation and p-value heatmaps."""
        if corr_df is None or p_df is None:
            return

        mask = np.triu(np.ones_like(corr_df, dtype=bool))

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={'label': 'Pearson r', 'shrink': 0.7},
                   linewidths=0.5, linecolor='lightgray',
                   annot_kws={'fontsize': 8})

        plt.title(f'{title}\nCorrelation Coefficients', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 12))
        sns.heatmap(p_df, mask=mask, annot=True, fmt='.3f',
                   cmap='RdYlGn_r', center=0.05, square=True,
                   cbar_kws={'label': 'p-value', 'shrink': 0.7},
                   linewidths=0.5, linecolor='lightgray',
                   annot_kws={'fontsize': 8},
                   vmin=0, vmax=0.1)

        plt.title(f'{title}\nP-values (Green = Significant)', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / 'p_values_heatmap.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved heatmaps to: {output_dir}")

    def save_correlation_summary(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                                 output_path: Path, title: str) -> None:
        """Save text summary of correlations."""
        if corr_df is None or p_df is None:
            return

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis includes {len(corr_df)} features\n")
            f.write(f"Content word threshold: >{self.content_word_threshold} non-function words\n")
            f.write(f"Features scaling: {self.norm_type} scope, feature-specific scalers\n")
            f.write("  - No scaling: jargon/lexical/syntactic density, MRC features, epistemic_balance\n")
            f.write("  - MinMax: hedge_density, booster_density\n")
            f.write("  - RobustScaler: evidential_markers, specificity_numbers, subjectivity_score\n")
            f.write("  - StandardScaler: reasoning_markers, embedding metrics\n")
            f.write("  - Log+StandardScaler: length_tokens, num_non_function_words\n\n")

            f.write(f"SIGNIFICANT CORRELATIONS (p < {ALPHA}):\n")
            f.write("-" * 80 + "\n\n")

            significant_pairs = []
            for i, row_name in enumerate(corr_df.index):
                for j, col_name in enumerate(corr_df.columns):
                    if i < j:
                        corr = corr_df.iloc[i, j]
                        p_val = p_df.iloc[i, j]
                        if not np.isnan(corr) and not np.isnan(p_val) and p_val < ALPHA:
                            significant_pairs.append({
                                'feature1': row_name,
                                'feature2': col_name,
                                'correlation': corr,
                                'p_value': p_val,
                                'strength': abs(corr)
                            })

            significant_pairs.sort(key=lambda x: x['strength'], reverse=True)
            for pair in significant_pairs:
                f.write(f"{pair['feature1']} <-> {pair['feature2']}\n")
                f.write(f"   r = {pair['correlation']:.3f}, p = {pair['p_value']:.4f}\n")
                f.write(f"   Interpretation: {'Strong' if pair['strength'] > CORR_STRONG else 'Moderate' if pair['strength'] > CORR_MODERATE else 'Weak'} ")
                f.write(f"{'positive' if pair['correlation'] > 0 else 'negative'} correlation\n\n")

            if not significant_pairs:
                f.write(f"No significant correlations found at p < {ALPHA}\n\n")
        print(f"   ✓ Saved summary: {output_path.name}")

    def save_correlation_json(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                             output_path: Path, metadata: Dict) -> None:
        """Save correlations to JSON."""
        if corr_df is None or p_df is None:
            return

        results = {
            'metadata': metadata,
            'correlations': {},
            'p_values': {},
            'significant_pairs': []
        }

        for i, row_name in enumerate(corr_df.index):
            results['correlations'][row_name] = {}
            results['p_values'][row_name] = {}
            for j, col_name in enumerate(corr_df.columns):
                corr = corr_df.iloc[i, j]
                p_val = p_df.iloc[i, j]
                results['correlations'][row_name][col_name] = float(corr) if not np.isnan(corr) else None
                results['p_values'][row_name][col_name] = float(p_val) if not np.isnan(p_val) else None

                if i < j and not np.isnan(corr) and not np.isnan(p_val) and p_val < ALPHA:
                    results['significant_pairs'].append({
                        'feature1': row_name,
                        'feature2': col_name,
                        'correlation': float(corr),
                        'p_value': float(p_val)
                    })

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✓ Saved JSON: {output_path.name}")

    def temporal_analysis(self, normalized_df: pd.DataFrame,
                         features: List[str], output_dir: Path,
                         dataset_name: str, model_type: str) -> None:
        """Analyze temporal patterns (start, middle, end) using normalized data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        topics = normalized_df['topic'].unique()

        for feature in features:
            temporal_data = []

            for topic in topics:
                df = normalized_df[normalized_df['topic'] == topic]

                if feature not in df.columns:
                    continue

                n = len(df)
                chunk_size = n // 3

                chunks = {
                    'Start': df.iloc[:chunk_size],
                    'Middle': df.iloc[chunk_size:2*chunk_size],
                    'End': df.iloc[2*chunk_size:]
                }

                for chunk_name, chunk_df in chunks.items():
                    values = chunk_df[feature].dropna()
                    for val in values:
                        temporal_data.append({
                            'topic': topic,
                            'period': chunk_name,
                            'value': val
                        })

            if not temporal_data:
                continue

            temporal_df = pd.DataFrame(temporal_data)

            fig, ax = plt.subplots(figsize=(14, 8))

            plot_topics = sorted(temporal_df['topic'].unique())
            periods = ['Start', 'Middle', 'End']
            colors = TEMPORAL_PERIOD_COLORS

            positions = []
            data_to_plot = []
            labels = []
            box_colors = []

            pos = 0
            for topic in plot_topics:
                for period in periods:
                    subset = temporal_df[(temporal_df['topic'] == topic) &
                                        (temporal_df['period'] == period)]
                    if len(subset) > 0:
                        data_to_plot.append(subset['value'].values)
                        positions.append(pos)
                        labels.append(f"{topic}\n{period}")
                        box_colors.append(colors[period])
                        pos += 1
                pos += 0.5

            parts = ax.violinplot(data_to_plot, positions=positions,
                                 showmeans=False, showmedians=True)

            for pc, color in zip(parts['bodies'], box_colors):
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1.5)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Scaled Value (feature-specific)', fontsize=13, fontweight='bold')
            ax.set_title(f'Temporal Evolution: {format_feature_name(feature)}\n'
                        f'{dataset_name.title()} - {model_type.title()}',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

            legend_elements = [Patch(facecolor=colors[p], label=p, alpha=0.7,
                                    edgecolor='black', linewidth=1.5)
                             for p in periods]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                     frameon=True, shadow=True)

            plt.tight_layout()

            safe_name = feature.replace(' ', '_').replace('/', '_').lower()
            filename = f'{safe_name}_temporal_{dataset_name}_{model_type}.png'
            plt.savefig(output_dir / filename, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()

        print(f"   ✓ Saved {len(features)} temporal plots to {output_dir}")

    def run_comprehensive_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CORRELATION ANALYSIS (DYNAMIC FEATURE EXTRACTION)")
        print(f"Content word threshold: {self.content_word_threshold}")
        print("=" * 80)

        output_base = self.output_root / f'threshold_{self.content_word_threshold}/scotland{self.include_scotland}_{self.norm_type}_featurespecific'
        output_base.mkdir(parents=True, exist_ok=True)

        print("\n📂 Loading Assemblies Data...")
        assemblies_data = load_and_combine_data(
            self.assemblies_path, self.assembly_topics, self.content_word_threshold
        )

        print("\n📂 Loading Experts Data...")
        experts_data = load_and_combine_data(
            self.experts_path, self.expert_topics, self.content_word_threshold
        )

        datasets = {
            'assemblies': assemblies_data,
            'experts': experts_data
        }

        for dataset_name, data_by_topic in datasets.items():
            if not data_by_topic:
                print(f"\n⚠️  No data for {dataset_name}, skipping...")
                continue

            combined_df = pd.concat(list(data_by_topic.values()), ignore_index=True)

            print(f"\n" + "=" * 80)
            print(f"ANALYZING: {dataset_name.upper()} ({len(combined_df)} total rows)")
            print("=" * 80)

            for model_type in MODEL_TYPES:
                print(f"\n🔬 {model_type.title()} Model Configuration")
                print("-" * 60)

                features = get_dynamic_features(combined_df, model_type, self.excluded_columns)
                print(f"   Found {len(features)} relevant features")

                df_normalized = self.normalize_features(combined_df)

                corr_df, p_df = self.compute_correlations(df_normalized, features)

                if corr_df is None:
                    print("   ⚠️  Skipping due to insufficient data")
                    continue

                output_dir = output_base / dataset_name / model_type
                output_dir.mkdir(parents=True, exist_ok=True)

                title = f"{dataset_name.title()} - {model_type.title()} Model"

                self.create_correlation_heatmaps(corr_df, p_df, title, output_dir)

                self.save_correlation_summary(
                    corr_df, p_df,
                    output_dir / 'correlation_summary.txt',
                    title
                )

                metadata = {
                    'dataset': dataset_name,
                    'model_type': model_type,
                    'n_samples': len(df_normalized),
                    'n_features': len(features),
                    'content_word_threshold': self.content_word_threshold,
                    'normalization_scope': self.norm_type,
                    'include_scotland': self.include_scotland,
                    'scaling_method': 'feature-specific',
                    'scaling_details': {
                        'no_scale': ['jargon_density', 'lexical_sophistication', 'syntactic_complexity',
                                     'epistemic_balance', 'specificity_word_length', 'mrc_*'],
                        'minmax': ['hedge_density', 'booster_density'],
                        'robust': ['evidential_markers', 'specificity_numbers', 'subjectivity_score'],
                        'standard': ['reasoning_markers', 'embedding_metrics'],
                        'log_standard': ['length_tokens', 'num_non_function_words']
                    }
                }
                self.save_correlation_json(
                    corr_df, p_df,
                    output_dir / 'correlations.json',
                    metadata
                )

                self.temporal_analysis(
                    df_normalized, features, output_dir / 'temporal',
                    dataset_name, model_type
                )

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_base}")


def main() -> None:
    """Run analysis with feature-specific scaling across multiple thresholds.

    Runs all combinations of:
    - Content word thresholds: 5, 10, 20 (minimum non-function words)
    - Normalization scope: 'global' or 'by_topic'
    - Include Scotland: 'yes' or 'no'

    Output structure:
        code/correlation_analysis/threshold_{N}/
            scotland{yes|no}_{norm}_featurespecific/
                assemblies/{base,finetuned}/
                experts/{base,finetuned}/

    Feature-specific scaling:
    - No scaling: Already bounded features (MRC, jargon/lexical density, etc.)
    - MinMax: hedge_density, booster_density (right-skewed [0,1])
    - RobustScaler: Zero-inflated features (evidential, specificity_numbers)
    - StandardScaler: Normally distributed (reasoning_markers, embeddings)
    - Log+StandardScaler: Count data (length_tokens, num_non_function_words)
    """
    parser = argparse.ArgumentParser(description="Comprehensive correlation analysis")
    parser.add_argument('--base-dir', type=Path, default=Path('data/assemblies/new_feats'),
                        help='Feature CSVs directory (experts read from its experts/ subdir)')
    parser.add_argument('--output-dir', type=Path, default=Path('code/correlation_analysis'),
                        help='Directory under which threshold_{N}/... result trees are written')
    args = parser.parse_args()

    for threshold in CONTENT_WORD_THRESHOLDS:
        print(f"\n\n{'#' * 80}")
        print(f"# CONTENT WORD THRESHOLD: {threshold}")
        print(f"{'#' * 80}")

        for norm_type in NORM_SCOPES:
            for scotland in SCOTLAND_OPTIONS:
                print(f"\n\n*** Running: threshold={threshold}, norm_scope={norm_type}, scotland={scotland} ***\n")
                analyzer = ComprehensiveCorrelationAnalyzer(
                    base_dir=args.base_dir,
                    norm_type=norm_type,
                    include_scotland=scotland,
                    content_word_threshold=threshold,
                    output_root=args.output_dir
                )
                analyzer.run_comprehensive_analysis()

    print("\n\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print("\nOutput structure:")
    print("  code/correlation_analysis/")
    for t in CONTENT_WORD_THRESHOLDS:
        print(f"    threshold_{t}/")
        print(f"      scotlandyes_global_featurespecific/")
        print(f"      scotlandyes_by_topic_featurespecific/")
        print(f"      scotlandno_global_featurespecific/")
        print(f"      scotlandno_by_topic_featurespecific/")


if __name__ == "__main__":
    main()
