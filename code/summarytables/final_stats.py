#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate concise corpus statistics and LaTeX tables for Citizens' Assemblies and Experts.
Creates separate reports with high-level summaries and feature statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys

# Add path for filter function
sys.path.append(str(Path(__file__).parent.parent / "handcrafted_features"))
try:
    from filter_function import is_substantive_speech_act
except ImportError:
    print("⚠️  Warning: Could not import filter_function, using fallback")
    def is_substantive_speech_act(num_words: int, threshold: int = 5) -> bool:
        return num_words >= threshold

# ============================================================================
# Configuration
# ============================================================================

ASSEMBLIES_DIR = Path("data/assemblies/processed_files")
EXPERTS_DIR = Path("data/assemblies/processed_files/experts")
OUTPUT_DIR = Path("corpus_statistics")

# Topic mappings
ASSEMBLY_TOPIC_MAP = {
    "8th_amendment": "Constitutional Reform",
    "gendereq": "Gender Equality",
    "scotland": "Public Policy"
}

EXPERT_TOPIC_MAP = {
    "constitution": "Constitutional Reform",
    "future": "Public Policy",
    "gender": "Gender Equality"
}

# Features to extract
FEATURES = [
    "jargon_density",
    "lexical_sophistication",
    "syntactic_complexity",
    "reasoning_markers",
    "evidential_markers",
    "length_tokens",
    "num_non_function_words",
    "expertise_llm_score",
    "expertise_justification",
    "hedge_density",
    "booster_density",
    "epistemic_balance",
    "overall_score",
    "specificity_length_norm",
    "specificity_mean_word_len_norm",
    "specificity_numbers_norm",
    "specificity_content_prop",
    "cluster_base",
    "distance_base",
    "similarity_base",
    "cluster_finetuned",
    "distance_finetuned",
    "similarity_finetuned",
    "new_jargon_density", "new_jargon_density_scaled", "new_jargon_count",
    "new_lexical_sophistication", "new_syntactic_complexity",
    "new_hedge_density", "new_booster_density", "new_epistemic_balance", 
    "new_reasoning_markers", "new_evidential_markers", "new_specificity_score",
    "new_subjectivity_score", "new_subjectivity_count",
    "mrc_familiarity", "mrc_concreteness", "mrc_imageability", 
]

CONTENT_THRESHOLD = 5

# ============================================================================
# Data Processing Functions
# ============================================================================

def load_and_filter_substantial_speeches(csv_path: Path, content_threshold: int = 5) -> pd.DataFrame:
    """Load CSV and filter for substantial speeches only."""
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return pd.DataFrame()
        
        # Check for required column
        if 'num_non_function_words' not in df.columns:
            print(f"    ⚠️  Missing 'num_non_function_words' in {csv_path.name}")
            return pd.DataFrame()
        
        # Filter for substantial speeches
        substantial_mask = df['num_non_function_words'].apply(
            lambda x: is_substantive_speech_act(int(x), content_threshold) if pd.notna(x) else False
        )
        
        return df[substantial_mask]
    
    except Exception as e:
        print(f"    ⚠️  Error loading {csv_path.name}: {e}")
        return pd.DataFrame()

def compute_topic_statistics(folder_path: Path, content_threshold: int = 5) -> Dict:
    """
    Compute statistics for all CSV files in a topic folder.
    
    Returns:
        Dictionary with:
        - total_speeches: total number of speeches
        - substantial_speeches: number of substantial speeches
        - avg_length: average length of substantial speeches (in words)
        - features: dict of {feature: {mean, std}} for each feature
    """
    csv_files = sorted(folder_path.glob("*.csv"))
    
    if not csv_files:
        return None
    
    all_substantial_dfs = []
    total_speeches = 0
    
    for csv_file in csv_files:
        try:
            # Load full data to count total speeches
            df_full = pd.read_csv(csv_file)
            total_speeches += len(df_full)
            
            # Load substantial speeches
            df_substantial = load_and_filter_substantial_speeches(csv_file, content_threshold)
            
            if not df_substantial.empty:
                all_substantial_dfs.append(df_substantial)
        
        except Exception as e:
            print(f"    ⚠️  Error processing {csv_file.name}: {e}")
            continue
    
    if not all_substantial_dfs:
        return {
            'total_speeches': total_speeches,
            'substantial_speeches': 0,
            'avg_length': 0.0,
            'features': {}
        }
    
    # Concatenate all substantial speeches
    df_all_substantial = pd.concat(all_substantial_dfs, ignore_index=True)
    
    # Compute high-level stats
    substantial_count = len(df_all_substantial)
    
    # Compute average length (try different text columns)
    text_col = None
    for col in ['text', 'comment', 'statement']:
        if col in df_all_substantial.columns:
            text_col = col
            break
    
    if text_col:
        word_counts = df_all_substantial[text_col].dropna().astype(str).apply(lambda x: len(x.split()))
        avg_length = float(word_counts.mean())
    else:
        # Fallback to num_non_function_words
        avg_length = float(df_all_substantial['num_non_function_words'].mean())
    
    # Compute feature statistics
    feature_stats = {}
    for feature in FEATURES:
        if feature in df_all_substantial.columns:
            values = pd.to_numeric(df_all_substantial[feature], errors='coerce').dropna()
            
            if len(values) > 0:
                feature_stats[feature] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'count': int(len(values))
                }
    
    return {
        'total_speeches': total_speeches,
        'substantial_speeches': substantial_count,
        'avg_length': avg_length,
        'features': feature_stats
    }

# ============================================================================
# Text Report Generation
# ============================================================================

def generate_text_report(results: Dict, output_path: Path, dataset_name: str) -> None:
    """Generate concise text report."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_name.upper()} - CORPUS STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Content word threshold: {CONTENT_THRESHOLD}\n")
        f.write("=" * 80 + "\n\n")
        
        for topic_name in sorted(results.keys()):
            stats = results[topic_name]
            
            if stats is None:
                f.write(f"{topic_name.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write("  No data available\n\n")
                continue
            
            f.write(f"{topic_name.upper()}\n")
            f.write("-" * 80 + "\n\n")
            
            # SECTION 1: High-level description
            f.write("1. HIGH-LEVEL SUMMARY\n")
            f.write("   " + "-" * 76 + "\n")
            f.write(f"   Total speeches:        {stats['total_speeches']:>8,}\n")
            f.write(f"   Substantial speeches:  {stats['substantial_speeches']:>8,} "
                   f"({stats['substantial_speeches']/stats['total_speeches']*100:.1f}%)\n")
            f.write(f"   Avg length (words):    {stats['avg_length']:>8.1f}\n")
            f.write("\n")
            
            # SECTION 2: Feature statistics
            f.write("2. FEATURE STATISTICS (SUBSTANTIAL SPEECHES ONLY)\n")
            f.write("   " + "-" * 76 + "\n")
            
            if stats['features']:
                f.write(f"   {'Feature':<35} {'Mean':>12} {'Std Dev':>12} {'N':>10}\n")
                f.write("   " + "-" * 76 + "\n")
                
                for feature in FEATURES:
                    if feature in stats['features']:
                        fstats = stats['features'][feature]
                        f.write(f"   {feature:<35} {fstats['mean']:>12.4f} {fstats['std']:>12.4f} {fstats['count']:>10,}\n")
            else:
                f.write("   No feature data available\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_latex_table_highlevel(results: Dict, output_path: Path, dataset_name: str) -> None:
    """Generate LaTeX table for high-level summary."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% High-level summary table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{dataset_name}: High-Level Corpus Statistics}}\n")
        f.write(f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}_highlevel}}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Topic & Total Speeches & Substantial Speeches & Avg Length (words) \\\\\n")
        f.write("\\midrule\n")
        
        for topic_name in sorted(results.keys()):
            stats = results[topic_name]
            
            if stats is None or stats['total_speeches'] == 0:
                continue
            
            # Escape underscores for LaTeX
            topic_latex = topic_name.replace('_', '\\_')
            
            f.write(f"{topic_latex} & {stats['total_speeches']:,} & "
                   f"{stats['substantial_speeches']:,} ({stats['substantial_speeches']/stats['total_speeches']*100:.1f}\\%) & "
                   f"{stats['avg_length']:.1f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_latex_table_features(results: Dict, output_path: Path, dataset_name: str) -> None:
    """Generate LaTeX table for feature statistics."""
    
    # Get all topics with data
    valid_topics = [t for t in sorted(results.keys()) if results[t] is not None and results[t]['features']]
    
    if not valid_topics:
        print(f"    ⚠️  No valid topics with features for {dataset_name}")
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% Feature statistics table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\footnotesize\n")  # Smaller font for readability
        f.write(f"\\caption{{{dataset_name}: Feature Statistics (Mean ± SD)}}\n")
        f.write(f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}_features}}\n")
        
        # Create dynamic column specification
        n_topics = len(valid_topics)
        col_spec = "l" + "c" * n_topics
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Feature & " + " & ".join(valid_topics) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Feature rows
        for feature in FEATURES:
            # Check if at least one topic has this feature
            has_feature = any(feature in results[t]['features'] for t in valid_topics if results[t])
            
            if not has_feature:
                continue
            
            # Escape underscores and format feature name
            feature_latex = feature.replace('_', '\\_')
            
            row = [feature_latex]
            
            for topic in valid_topics:
                stats = results[topic]
                if stats and feature in stats['features']:
                    fstats = stats['features'][feature]
                    row.append(f"{fstats['mean']:.3f} ± {fstats['std']:.3f}")
                else:
                    row.append("--")
            
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

# ============================================================================
# Main Processing
# ============================================================================

def process_dataset(base_dir: Path, topic_map: Dict, dataset_name: str) -> Dict:
    """Process all topics in a dataset."""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {dataset_name.upper()}")
    print("="*80)
    
    results = {}
    
    for folder_name, topic_name in topic_map.items():
        folder_path = base_dir / folder_name
        
        # Handle alternative folder names
        if not folder_path.exists():
            alt_names = [folder_name + "_csv", folder_name.replace("_csv", "")]
            for alt_name in alt_names:
                alt_path = base_dir / alt_name
                if alt_path.exists():
                    folder_path = alt_path
                    break
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            results[topic_name] = None
            continue
        
        print(f"\n📂 {topic_name}")
        csv_files = sorted(folder_path.glob("*.csv"))
        print(f"   Found {len(csv_files)} CSV file(s)")
        
        stats = compute_topic_statistics(folder_path, CONTENT_THRESHOLD)
        results[topic_name] = stats
        
        if stats:
            print(f"   ✓ Total: {stats['total_speeches']:,} speeches")
            print(f"   ✓ Substantial: {stats['substantial_speeches']:,} speeches")
            print(f"   ✓ Features: {len(stats['features'])} available")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate concise corpus statistics and LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports and tables"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Process Assemblies
    # ========================================================================
    
    if not ASSEMBLIES_DIR.exists():
        print(f"❌ Assemblies directory not found: {ASSEMBLIES_DIR}")
    else:
        assemblies_results = process_dataset(ASSEMBLIES_DIR, ASSEMBLY_TOPIC_MAP, "Citizens' Assemblies")
        
        # Generate text report
        print(f"\n📝 Generating assemblies text report...")
        assemblies_txt = args.output_dir / "assemblies_statistics.txt"
        generate_text_report(assemblies_results, assemblies_txt, "Citizens' Assemblies")
        print(f"   ✓ Saved to: {assemblies_txt}")
        
        # Generate LaTeX tables
        print(f"\n📊 Generating assemblies LaTeX tables...")
        
        assemblies_table_hl = args.output_dir / "assemblies_table_highlevel.tex"
        generate_latex_table_highlevel(assemblies_results, assemblies_table_hl, "Citizens' Assemblies")
        print(f"   ✓ High-level table: {assemblies_table_hl}")
        
        assemblies_table_feat = args.output_dir / "assemblies_table_features.tex"
        generate_latex_table_features(assemblies_results, assemblies_table_feat, "Citizens' Assemblies")
        print(f"   ✓ Features table: {assemblies_table_feat}")
    
    # ========================================================================
    # Process Experts
    # ========================================================================
    
    if not EXPERTS_DIR.exists():
        print(f"❌ Experts directory not found: {EXPERTS_DIR}")
    else:
        experts_results = process_dataset(EXPERTS_DIR, EXPERT_TOPIC_MAP, "Expert Speeches")
        
        # Generate text report
        print(f"\n📝 Generating experts text report...")
        experts_txt = args.output_dir / "experts_statistics.txt"
        generate_text_report(experts_results, experts_txt, "Expert Speeches")
        print(f"   ✓ Saved to: {experts_txt}")
        
        # Generate LaTeX tables
        print(f"\n📊 Generating experts LaTeX tables...")
        
        experts_table_hl = args.output_dir / "experts_table_highlevel.tex"
        generate_latex_table_highlevel(experts_results, experts_table_hl, "Expert Speeches")
        print(f"   ✓ High-level table: {experts_table_hl}")
        
        experts_table_feat = args.output_dir / "experts_table_features.tex"
        generate_latex_table_features(experts_results, experts_table_feat, "Expert Speeches")
        print(f"   ✓ Features table: {experts_table_feat}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("✅ ALL REPORTS AND TABLES GENERATED")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir.absolute()}")
    print("\nGenerated files:")
    print("  Text Reports:")
    print("    • assemblies_statistics.txt")
    print("    • experts_statistics.txt")
    print("  LaTeX Tables:")
    print("    • assemblies_table_highlevel.tex")
    print("    • assemblies_table_features.tex")
    print("    • experts_table_highlevel.tex")
    print("    • experts_table_features.tex")

if __name__ == "__main__":
    main()