import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
import sys
import argparse

# Add paths for shared utilities and filter function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))
from shared.topic_maps import ASSEMBLY_TOPICS, EXPERT_TOPICS
from correlation_utils import (
    filter_syntactically_relevant, get_dynamic_features,
    format_feature_name, load_and_combine_data,
)

warnings.filterwarnings('ignore')

class DetailedTemporalAnalyzer:
    def __init__(self, base_dir: Path = Path('data/assemblies/processed_files')):
        """
        Initialize analyzer.
        """
        self.base_dir = Path(base_dir)
        self.assemblies_path = self.base_dir
        self.experts_path = self.base_dir / 'experts'
        
        # COLUMNS TO IGNORE during dynamic extraction
        self.excluded_columns = [
            'text', 'comment', 'statement', 'original_text', 
            'topic', 'source_file', 'Unnamed: 0', 
            'hedge_ratio', 
            'cluster_',    
            'year', 'date', 'id', 'speaker_id', 'speech_id', 'member_id'
        ]

        # Topic mappings
        self.assembly_topics = ASSEMBLY_TOPICS
        self.expert_topics = EXPERT_TOPICS
        
    def normalize_features_globally(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Apply Z-score normalization to ALL selected features."""
        df_norm = df.copy()
        for col in features:
            if col in df_norm.columns:
                std_dev = df_norm[col].std()
                if std_dev > 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].mean()) / std_dev
                else:
                    df_norm[col] = 0.0
        return df_norm
    
    def plot_detailed_temporal(self, df: pd.DataFrame, features: List[str], 
                             output_dir: Path, dataset_name: str, model_type: str):
        """
        Plot evolution of features over time for each file individually.
        Generates one plot per feature, with lines for each file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'source_file' not in df.columns or 'topic' not in df.columns:
            print("   ⚠️  Missing source_file or topic column")
            return

        # Get unique files and their topics for coloring
        # We group by source_file to ensure we process each file correctly
        unique_files = df['source_file'].unique()
        topics = df['topic'].unique()
        
        # Setup colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
        topic_color_map = dict(zip(topics, colors))
        
        print(f"   Generating plots for {len(features)} features...")
        
        for feature in features:
            plt.figure(figsize=(14, 8))
            
            # Plot each file as a line
            for fname in unique_files:
                file_data = df[df['source_file'] == fname]
                
                if feature not in file_data.columns or file_data.empty:
                    continue
                
                # Get topic for color
                topic = file_data['topic'].iloc[0]
                
                # Get values
                y_values = file_data[feature].values
                x_values = range(len(y_values))
                
                plt.plot(x_values, y_values, 
                        color=topic_color_map[topic], 
                        alpha=0.4, # Low alpha to handle many overlapping lines
                        linewidth=1.0)
            
            # Custom Legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=topic_color_map[t], lw=2, label=t)
                             for t in topics]
            
            plt.legend(handles=legend_elements, title="Topics", loc='upper right')
            
            plt.title(f'Detailed Temporal Evolution: {format_feature_name(feature)}\n'
                     f'{dataset_name.title()} - {model_type.title()}',
                     fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Speech Sequence Index', fontsize=12)
            plt.ylabel('Normalized Value (Z-Score)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            safe_name = feature.replace(' ', '_').replace('/', '_').lower()
            plt.savefig(output_dir / f"{safe_name}_detailed.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"   ✓ Saved detailed temporal plots to {output_dir}")

    def run_analysis(self, output_base: Path = Path('code/correlation_analysis/results')):
        print("\n" + "=" * 80)
        print("DETAILED TEMPORAL ANALYSIS (FILE-BY-FILE)")
        print("=" * 80)
        
        # Load data
        assemblies_data = load_and_combine_data(self.assemblies_path, self.assembly_topics)
        experts_data = load_and_combine_data(self.experts_path, self.expert_topics)
        
        datasets = {'assemblies': assemblies_data, 'experts': experts_data}
        model_types = ['base', 'finetuned']
        
        for dataset_name, data_by_topic in datasets.items():
            if not data_by_topic:
                continue
            
            combined_df = pd.concat(list(data_by_topic.values()), ignore_index=True)
            
            print(f"\nANALYZING: {dataset_name.upper()}")
            
            for model_type in model_types:
                print(f"  Model: {model_type}")
                
                # 1. Extract Features
                features = get_dynamic_features(combined_df, model_type, self.excluded_columns)
                
                # 2. Normalize
                df_normalized = self.normalize_features_globally(combined_df, features)
                
                # 3. Plot
                output_dir = output_base / dataset_name / model_type / 'detailed_temporal'
                self.plot_detailed_temporal(
                    df_normalized, features, output_dir,
                    dataset_name, model_type
                )

def main():
    parser = argparse.ArgumentParser(description="Detailed file-by-file temporal analysis")
    parser.add_argument('--base-dir', type=Path, default=Path('data/assemblies/processed_files'),
                        help='Processed files directory (experts read from its experts/ subdir)')
    parser.add_argument('--output-dir', type=Path, default=Path('code/correlation_analysis/results'),
                        help='Base directory for output plots')
    args = parser.parse_args()

    analyzer = DetailedTemporalAnalyzer(base_dir=args.base_dir)
    analyzer.run_analysis(output_base=args.output_dir)

if __name__ == "__main__":
    main()