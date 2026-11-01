"""Paper-era correlation analysis (frozen copy of code/correlation_analysis/corr.py as used for the
published figures). Kept here so reproduce/ keeps working after the refactor of code/. Do not edit."""
"""
Comprehensive Correlation Analysis Script

This script performs correlation analysis on assembly and expert speech data,
examining relationships between linguistic features and expertise measures.

Features:
- Loads and combines CSV files from assembly (8th amendment, gender equality, 
  Scotland) and expert (constitution, future, gender) datasets
- Filters texts with >5 non-function words for syntactic relevance
- Dynamically extracts numeric features, separating base vs finetuned model columns
- Normalizes features using Z-score or Min-Max scaling (global or by-topic)
- Computes Pearson correlation matrices with p-values
- Generates correlation heatmaps, p-value heatmaps, and temporal evolution plots
- Outputs results as PNG visualizations, JSON data, and text summaries

Usage:
    python new_corr.py

The script runs all combinations of:
- Normalization scope: 'global' or 'by_topic'
- Scaler type: 'zscore' or 'minmax'  
- Include Scotland data: 'yes' or 'no'

Output is saved to: code/correlation_analysis/results_scotland{yes|no}_{norm}_{scaler}/
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import sys

# Add path for filter function if needed
sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))

warnings.filterwarnings('ignore')

class ComprehensiveCorrelationAnalyzer:
    def __init__(self, base_dir: Path = Path('data/assemblies/processed_files'),
                 norm_type: str = 'global', scaler_type: str = 'zscore', 
                 include_scotland: str = 'yes'):
        """
        Initialize analyzer.
        """
        self.base_dir = Path(base_dir)
        self.assemblies_path = self.base_dir
        self.experts_path = self.base_dir / 'experts'
        self.norm_type = norm_type
        self.scaler_type = scaler_type
        self.include_scotland = include_scotland
        
        # COLUMNS TO IGNORE during dynamic extraction
        self.excluded_columns = [
            'text', 'speaker',
            'Unnamed', 
            'hedge_ratio',
            'cluster_',
            'date', 'justification', 
        ]
        
        # Topic mappings
        if self.include_scotland == "yes":
            self.assembly_topics = {
                "8th_amendment": "Constitutional Reform",
                "gendereq": "Gender Equality",
                "scotland": "Public Policy"
            }
        else:
            self.assembly_topics = {
                "8th_amendment": "Constitutional Reform",
                "gendereq": "Gender Equality"
            }
        
        self.expert_topics = {
            "constitution": "Constitutional Reform",
            "future": "Public Policy",
            "gender": "Gender Equality"
        }
        
    def filter_syntactically_relevant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to speeches with >5 non-function words."""
        if 'num_non_function_words' not in df.columns:
            print("   ⚠️  'num_non_function_words' not found, no filtering applied")
            return df
        
        original = len(df)
        df['num_non_function_words'] = pd.to_numeric(df['num_non_function_words'], errors='coerce')
        filtered = df[df['num_non_function_words'] > 5].copy()
        removed = original - len(filtered)
        
        print(f"   ✓ Filtered: {len(filtered)} rows kept ({removed} removed)")
        return filtered
    
    def get_dynamic_features(self, df: pd.DataFrame, model_type: str) -> List[str]:
        """Automatically extract relevant numeric features from the DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        features = []
        for col in numeric_cols:
            if any(ex in col for ex in self.excluded_columns):
                continue
            if col == 'id':
                continue
            
            is_base_col = 'base' in col.lower()
            is_fine_col = 'finetuned' in col.lower()
            
            if is_base_col and model_type != 'base':
                continue 
            if is_fine_col and model_type != 'finetuned':
                continue 
                
            features.append(col)
            
        seen = set()
        unique_features = [x for x in features if not (x in seen or seen.add(x))]
        
        expertise_cols = []
        distance_cols = []
        similarity_cols = []
        overall_cols = []
        rest_cols = []

        for f in unique_features:
            f_lower = f.lower()
            if 'expertise' in f_lower:
                expertise_cols.append(f)
            elif 'distance' in f_lower:
                distance_cols.append(f)
            elif 'similarity' in f_lower:
                similarity_cols.append(f)
            elif 'overall' in f_lower:
                overall_cols.append(f)
            else:
                rest_cols.append(f)
        
        return rest_cols + overall_cols + distance_cols + similarity_cols + expertise_cols

    def normalize_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Normalize features using instance's method and scope."""
        df_norm = df.copy()
        normalized_count = 0
        scope = self.norm_type
        scaler_type = self.scaler_type
        
        if scope == 'by_topic' and 'topic' not in df_norm.columns:
            print("   ⚠️  'topic' column not found, falling back to global normalization")
            scope = 'global'
        
        topics = df_norm['topic'].unique() if scope == 'by_topic' else [None]
        
        for col in features:
            if col not in df_norm.columns:
                continue
            
            for topic in topics:
                if scope == 'by_topic':
                    mask = df_norm['topic'] == topic
                    subset = df_norm.loc[mask, col]
                else:
                    mask = slice(None)
                    subset = df_norm[col]
                
                if scaler_type == 'zscore':
                    std_dev = subset.std()
                    if std_dev > 0:
                        df_norm.loc[mask, col] = (subset - subset.mean()) / std_dev
                    else:
                        df_norm.loc[mask, col] = 0.0
                elif scaler_type == 'minmax':
                    col_min = subset.min()
                    col_max = subset.max()
                    col_range = col_max - col_min
                    if col_range > 0:
                        df_norm.loc[mask, col] = (subset - col_min) / col_range
                    else:
                        df_norm.loc[mask, col] = 0.0
            
            normalized_count += 1
        
        scope_desc = f"by topic ({len(topics)} topics)" if scope == 'by_topic' else "globally"
        scaler_desc = "Z-score" if scaler_type == 'zscore' else "Min-Max"
        print(f"   ✓ Normalized {normalized_count} features {scope_desc} ({scaler_desc})")
        
        return df_norm
    
    def load_and_combine_data(self, base_path: Path, topic_folders: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load and combine CSV files by topic."""
        data_by_topic = {}
        
        for folder, topic_name in topic_folders.items():
            folder_path = base_path / folder
            if not folder_path.exists():
                if (base_path / f"{folder}_csv").exists():
                    folder_path = base_path / f"{folder}_csv"
                else:
                    print(f"   ⚠️  Folder not found: {folder_path}")
                    continue
            
            csv_files = list(folder_path.glob('*.csv'))
            if not csv_files:
                print(f"   ⚠️  No CSV files in {folder_path}")
                continue
            
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['source_file'] = csv_file.name
                    df['topic'] = topic_name
                    dfs.append(df)
                except Exception as e:
                    print(f"   ❌ Error reading {csv_file.name}: {e}")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                filtered = self.filter_syntactically_relevant(combined)
                
                if len(filtered) > 0:
                    data_by_topic[topic_name] = filtered
                    print(f"   ✓ {topic_name}: {len(csv_files)} files, {len(filtered)} rows")
        
        return data_by_topic
    
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
                    
                    if valid_count < 3:
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
        
        display_names = [self.format_feature_name(col) for col in numeric_df.columns]
        corr_df = pd.DataFrame(corr_matrix, index=display_names, columns=display_names)
        p_df = pd.DataFrame(p_matrix, index=display_names, columns=display_names)
        
        return corr_df, p_df
    
    def format_feature_name(self, name: str) -> str:
        """Format feature name for display."""
        name = name.replace('_norm', '').replace('_base', '').replace('_finetuned', '')
        if name.startswith('distance'):
            name = name.replace('distance', 'Distance to Centroid')
        elif name.startswith('similarity'):
            name = name.replace('similarity', 'Similarity to Centroid')
        return name.replace('_', ' ').title()
    
    def create_correlation_heatmaps(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                                   title: str, output_dir: Path):
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
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(output_dir / 'p_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved heatmaps to: {output_dir}")
    
    def save_correlation_summary(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                                 output_path: Path, title: str):
        """Save text summary of correlations."""
        if corr_df is None or p_df is None:
            return
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis includes {len(corr_df)} features\n")
            f.write("Only syntactically relevant texts (>5 non-function words)\n")
            f.write(f"Features normalized: {self.norm_type} ({self.scaler_type})\n\n")
            
            f.write("SIGNIFICANT CORRELATIONS (p < 0.05):\n")
            f.write("-" * 80 + "\n\n")
            
            significant_pairs = []
            for i, row_name in enumerate(corr_df.index):
                for j, col_name in enumerate(corr_df.columns):
                    if i < j:
                        corr = corr_df.iloc[i, j]
                        p_val = p_df.iloc[i, j]
                        if not np.isnan(corr) and not np.isnan(p_val) and p_val < 0.05:
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
                f.write(f"   Interpretation: {'Strong' if pair['strength'] > 0.7 else 'Moderate' if pair['strength'] > 0.4 else 'Weak'} ")
                f.write(f"{'positive' if pair['correlation'] > 0 else 'negative'} correlation\n\n")
            
            if not significant_pairs:
                f.write("No significant correlations found at p < 0.05\n\n")
        print(f"   ✓ Saved summary: {output_path.name}")
    
    def save_correlation_json(self, corr_df: pd.DataFrame, p_df: pd.DataFrame,
                             output_path: Path, metadata: Dict):
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
                
                if i < j and not np.isnan(corr) and not np.isnan(p_val) and p_val < 0.05:
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
                         dataset_name: str, model_type: str):
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
            colors = {'Start': '#89CFF0', 'Middle': '#90EE90', 'End': '#FFB6C1'}
            
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
            
            f_lower = feature.lower()
            is_discrete = any(x in f_lower for x in ['score', 'expertise', 'word', 'token'])
            
            if is_discrete:
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
            else:
                bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True,
                               widths=0.6, showfliers=False)
                
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(f'Normalized Value ({self.scaler_type.title()})', fontsize=13, fontweight='bold')
            ax.set_title(f'Temporal Evolution: {self.format_feature_name(feature)}\n'
                        f'{dataset_name.title()} - {model_type.title()}',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[p], label=p, alpha=0.7, 
                                    edgecolor='black', linewidth=1.5) 
                             for p in periods]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                     frameon=True, shadow=True)
            
            plt.tight_layout()
            
            safe_name = feature.replace(' ', '_').replace('/', '_').lower()
            filename = f'{safe_name}_temporal_{dataset_name}_{model_type}.png'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   ✓ Saved {len(features)} temporal plots to {output_dir}")
    
    def run_comprehensive_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CORRELATION ANALYSIS (DYNAMIC FEATURE EXTRACTION)")
        print("=" * 80)
        
        output_base = Path(f'code/correlation_analysis/results_scotland{self.include_scotland}_{self.norm_type}_{self.scaler_type}')
        output_base.mkdir(parents=True, exist_ok=True)
        
        print("\n📂 Loading Assemblies Data...")
        assemblies_data = self.load_and_combine_data(
            self.assemblies_path, self.assembly_topics
        )
        
        print("\n📂 Loading Experts Data...")
        experts_data = self.load_and_combine_data(
            self.experts_path, self.expert_topics
        )
        
        datasets = {
            'assemblies': assemblies_data,
            'experts': experts_data
        }
        
        model_types = ['base', 'finetuned']
        
        for dataset_name, data_by_topic in datasets.items():
            if not data_by_topic:
                print(f"\n⚠️  No data for {dataset_name}, skipping...")
                continue
            
            combined_df = pd.concat(list(data_by_topic.values()), ignore_index=True)
            
            print(f"\n" + "=" * 80)
            print(f"ANALYZING: {dataset_name.upper()} ({len(combined_df)} total rows)")
            print("=" * 80)
            
            for model_type in model_types:
                print(f"\n🔬 {model_type.title()} Model Configuration")
                print("-" * 60)
                
                features = self.get_dynamic_features(combined_df, model_type)
                print(f"   Found {len(features)} relevant features")
                
                df_normalized = self.normalize_features(combined_df, features)
                
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
                    'normalization': f'{self.norm_type} {self.scaler_type}'
                }
                self.save_correlation_json(
                    corr_df, p_df,
                    output_dir / 'correlations.json',
                    metadata
                )
                
                temporal_dir = output_dir / 'temporal'
                self.temporal_analysis(
                    df_normalized, features, temporal_dir,
                    dataset_name, model_type
                )
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_base}")


def main():
    for norm_type in ['global', 'by_topic']:
        for scaler_type in ['zscore', 'minmax']:
            for scotland in ['yes', 'no']:
                print(f"\n\n*** Running analysis with norm_type={norm_type}, scaler_type={scaler_type}, scotland={scotland} ***\n")
                analyzer = ComprehensiveCorrelationAnalyzer(
                    norm_type=norm_type,
                    scaler_type=scaler_type,
                    include_scotland=scotland
                )
                analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()