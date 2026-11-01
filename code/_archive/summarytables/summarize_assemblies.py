#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive corpus statistics for both Citizens' Assemblies and Expert datasets.
Creates separate detailed reports with file-level and topic-level aggregations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "handcrafted_features"))
from shared.topic_maps import ASSEMBLY_TOPICS as ASSEMBLY_TOPIC_MAP, EXPERT_TOPICS as EXPERT_TOPIC_MAP
try:
    from filter_function import is_substantive_speech_act
except ImportError:
    print("⚠️  Warning: Could not import filter_function, using fallback")
    def is_substantive_speech_act(num_words: int, threshold: int = 5) -> bool:
        return num_words >= threshold


# Feature columns to analyze (if present)
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
]

def detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns in the dataframe.
    
    Returns:
        Dictionary with keys:
        - 'text': text columns
        - 'numeric': numeric feature columns
        - 'metadata': metadata columns
        - 'cluster': clustering columns
        - 'other': other columns
    """
    columns = {
        'text': [],
        'numeric': [],
        'metadata': [],
        'cluster': [],
        'other': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Text columns
        if col in ['text', 'comment', 'statement']:
            columns['text'].append(col)
        # Clustering/embedding columns
        elif any(x in col_lower for x in ['cluster', 'distance', 'similarity', 'entropy', 'confidence']):
            columns['cluster'].append(col)
        # Metadata columns
        elif col in ['speaker', 'date', 'session', 'topic', 'id', 'index']:
            columns['metadata'].append(col)
        # Numeric feature columns
        elif col in FEATURE_COLUMNS or (df[col].dtype in ['float64', 'int64', 'float32', 'int32'] 
                                        and col not in ['id', 'index']):
            columns['numeric'].append(col)
        else:
            columns['other'].append(col)
    
    return columns

def compute_text_statistics(texts: pd.Series) -> Dict:
    """Compute detailed text statistics."""
    # Remove NaN and convert to string
    valid_texts = texts.dropna().astype(str)
    
    if len(valid_texts) == 0:
        return {
            'count': 0,
            'mean_length_words': 0.0,
            'median_length_words': 0.0,
            'min_length_words': 0,
            'max_length_words': 0,
            'std_length_words': 0.0,
            'mean_length_chars': 0.0,
            'empty_count': 0,
            'very_short_count': 0,  # < 3 words
        }
    
    # Compute word lengths
    word_lengths = valid_texts.apply(lambda x: len(str(x).split()))
    char_lengths = valid_texts.apply(lambda x: len(str(x)))
    
    stats = {
        'count': int(len(valid_texts)),
        'mean_length_words': float(word_lengths.mean()),
        'median_length_words': float(word_lengths.median()),
        'min_length_words': int(word_lengths.min()),
        'max_length_words': int(word_lengths.max()),
        'std_length_words': float(word_lengths.std()),
        'mean_length_chars': float(char_lengths.mean()),
        'empty_count': int((word_lengths == 0).sum()),
        'very_short_count': int((word_lengths < 3).sum()),
        'percentile_25': float(word_lengths.quantile(0.25)),
        'percentile_75': float(word_lengths.quantile(0.75)),
    }
    
    return stats

def compute_substantive_statistics(df: pd.DataFrame, content_threshold: int = 5) -> Dict:
    """Compute statistics about substantive speeches."""
    if 'num_non_function_words' not in df.columns:
        return {
            'substantial_speeches': None,
            'substantial_percentage': None,
            'non_substantial_count': None
        }
    
    # Use the proper filter function
    substantive_mask = df['num_non_function_words'].apply(
        lambda x: is_substantive_speech_act(int(x), content_threshold) if pd.notna(x) else False
    )
    
    substantial_count = int(substantive_mask.sum())
    total_count = len(df)
    
    return {
        'substantial_speeches': substantial_count,
        'substantial_percentage': float(substantial_count / total_count * 100) if total_count > 0 else 0.0,
        'non_substantial_count': total_count - substantial_count,
        'content_word_stats': {
            'mean': float(df['num_non_function_words'].mean()) if 'num_non_function_words' in df.columns else None,
            'median': float(df['num_non_function_words'].median()) if 'num_non_function_words' in df.columns else None,
            'min': int(df['num_non_function_words'].min()) if 'num_non_function_words' in df.columns else None,
            'max': int(df['num_non_function_words'].max()) if 'num_non_function_words' in df.columns else None,
        }
    }

def compute_feature_statistics(df: pd.DataFrame, feature_col: str) -> Optional[Dict]:
    """Compute statistics for a single numeric feature."""
    if feature_col not in df.columns:
        return None
    
    values = pd.to_numeric(df[feature_col], errors='coerce').dropna()
    
    if len(values) == 0:
        return None
    
    return {
        'count': int(len(values)),
        'mean': float(values.mean()),
        'median': float(values.median()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max()),
        'missing': int(df[feature_col].isna().sum()),
        'missing_pct': float(df[feature_col].isna().sum() / len(df) * 100),
        'percentile_25': float(values.quantile(0.25)),
        'percentile_75': float(values.quantile(0.75)),
    }

def compute_file_statistics(csv_path: Path, content_threshold: int = 5) -> Dict:
    """
    Compute comprehensive statistics for a single CSV file.
    
    Returns:
        Dictionary with file-level statistics
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            'error': str(e),
            'file': csv_path.name
        }
    
    # Detect columns
    cols = detect_columns(df)
    
    # Basic file stats
    stats = {
        'file': csv_path.name,
        'file_path': str(csv_path),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_types': {
            'text': len(cols['text']),
            'numeric': len(cols['numeric']),
            'metadata': len(cols['metadata']),
            'cluster': len(cols['cluster']),
            'other': len(cols['other'])
        }
    }
    
    # Text column stats
    if cols['text']:
        text_col = cols['text'][0]  # Use first text column
        stats['text_column'] = text_col
        stats['text_stats'] = compute_text_statistics(df[text_col])
    
    # Substantial speeches statistics
    substantive_stats = compute_substantive_statistics(df, content_threshold)
    stats.update(substantive_stats)
    
    # Numeric column statistics
    stats['numeric_features'] = {}
    for col in cols['numeric']:
        feature_stats = compute_feature_statistics(df, col)
        if feature_stats is not None:
            stats['numeric_features'][col] = feature_stats
    
    # Clustering column statistics (if present)
    if cols['cluster']:
        stats['has_clustering'] = True
        stats['clustering_columns'] = cols['cluster']
        
        # Analyze cluster assignments if present
        for col in cols['cluster']:
            if 'cluster' in col.lower() and df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                cluster_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(cluster_vals) > 0:
                    stats['cluster_distribution'] = {
                        'unique_clusters': int(cluster_vals.nunique()),
                        'assignments': cluster_vals.value_counts().to_dict()
                    }
                    break
    else:
        stats['has_clustering'] = False
    
    # Overall score distribution (if present)
    if 'overall_score' in df.columns:
        scores = pd.to_numeric(df['overall_score'], errors='coerce').dropna()
        if len(scores) > 0:
            stats['score_distribution'] = {
                str(int(score)): int((scores == score).sum())
                for score in sorted(scores.unique())
            }
    
    # Metadata statistics
    if cols['metadata']:
        stats['metadata'] = {}
        for col in cols['metadata']:
            if col in df.columns:
                unique_vals = df[col].nunique()
                stats['metadata'][col] = {
                    'unique_count': int(unique_vals),
                    'missing_count': int(df[col].isna().sum())
                }
    
    return stats

def aggregate_topic_statistics(file_stats: List[Dict], content_threshold: int = 5) -> Dict:
    """
    Aggregate statistics across all files in a topic.
    
    Args:
        file_stats: List of file-level statistics dictionaries
        content_threshold: Threshold for substantive speeches
        
    Returns:
        Dictionary with aggregated topic statistics
    """
    # Filter out error files
    valid_stats = [s for s in file_stats if 'error' not in s]
    
    if not valid_stats:
        return {'error': 'No valid files'}
    
    agg = {
        'n_files': len(valid_stats),
        'total_rows': sum(s['total_rows'] for s in valid_stats),
        'avg_rows_per_file': float(np.mean([s['total_rows'] for s in valid_stats])),
    }
    
    # Aggregate substantial speeches
    if all('substantial_speeches' in s and s['substantial_speeches'] is not None for s in valid_stats):
        total_substantial = sum(s['substantial_speeches'] for s in valid_stats)
        total_non_substantial = sum(s['non_substantial_count'] for s in valid_stats)
        
        agg['substantial_speeches'] = total_substantial
        agg['non_substantial_speeches'] = total_non_substantial
        agg['substantial_percentage'] = float(total_substantial / agg['total_rows'] * 100) if agg['total_rows'] > 0 else 0.0
        
        # Content word statistics
        if all('content_word_stats' in s for s in valid_stats):
            all_means = [s['content_word_stats']['mean'] for s in valid_stats if s['content_word_stats']['mean'] is not None]
            if all_means:
                agg['content_word_stats'] = {
                    'avg_mean': float(np.mean(all_means)),
                    'overall_min': int(min(s['content_word_stats']['min'] for s in valid_stats if s['content_word_stats']['min'] is not None)),
                    'overall_max': int(max(s['content_word_stats']['max'] for s in valid_stats if s['content_word_stats']['max'] is not None)),
                }
    
    # Aggregate text stats
    if all('text_stats' in s for s in valid_stats):
        all_means = [s['text_stats']['mean_length_words'] for s in valid_stats]
        all_medians = [s['text_stats']['median_length_words'] for s in valid_stats]
        all_stds = [s['text_stats']['std_length_words'] for s in valid_stats]
        
        agg['text_stats'] = {
            'avg_mean_length': float(np.mean(all_means)),
            'avg_median_length': float(np.mean(all_medians)),
            'avg_std_length': float(np.mean(all_stds)),
            'overall_min': int(min(s['text_stats']['min_length_words'] for s in valid_stats)),
            'overall_max': int(max(s['text_stats']['max_length_words'] for s in valid_stats)),
            'total_empty': int(sum(s['text_stats']['empty_count'] for s in valid_stats)),
            'total_very_short': int(sum(s['text_stats']['very_short_count'] for s in valid_stats)),
        }
    
    # Aggregate numeric features
    all_features = set()
    for s in valid_stats:
        all_features.update(s.get('numeric_features', {}).keys())
    
    agg['numeric_features'] = {}
    for feature in sorted(all_features):
        # Collect values across all files
        feature_stats = [
            s['numeric_features'][feature]
            for s in valid_stats
            if feature in s.get('numeric_features', {})
        ]
        
        if feature_stats:
            # Weighted average by count
            total_count = sum(f['count'] for f in feature_stats)
            weighted_mean = sum(f['mean'] * f['count'] for f in feature_stats) / total_count if total_count > 0 else 0
            
            agg['numeric_features'][feature] = {
                'weighted_mean': float(weighted_mean),
                'avg_mean': float(np.mean([f['mean'] for f in feature_stats])),
                'avg_median': float(np.mean([f['median'] for f in feature_stats])),
                'overall_min': float(min(f['min'] for f in feature_stats)),
                'overall_max': float(max(f['max'] for f in feature_stats)),
                'avg_std': float(np.mean([f['std'] for f in feature_stats])),
                'total_missing': int(sum(f['missing'] for f in feature_stats)),
            }
    
    # Aggregate score distribution (if present)
    if any('score_distribution' in s for s in valid_stats):
        combined_scores = {}
        for s in valid_stats:
            if 'score_distribution' in s:
                for score, count in s['score_distribution'].items():
                    combined_scores[score] = combined_scores.get(score, 0) + count
        
        agg['score_distribution'] = combined_scores
    
    # Clustering statistics
    has_clustering = any(s.get('has_clustering', False) for s in valid_stats)
    agg['has_clustering'] = has_clustering
    
    if has_clustering:
        agg['files_with_clustering'] = sum(1 for s in valid_stats if s.get('has_clustering', False))
    
    return agg

def generate_report(
    topic_name: str,
    file_stats: List[Dict],
    topic_stats: Dict,
    output: List[str],
    include_file_details: bool = True
) -> None:
    """Generate formatted report section for a topic."""
    
    output.append("=" * 80)
    output.append(f"{topic_name.upper()}")
    output.append("=" * 80)
    output.append("")
    
    # Topic-level summary
    output.append("TOPIC SUMMARY:")
    output.append(f"  Files processed: {topic_stats.get('n_files', 0)}")
    output.append(f"  Total speeches: {topic_stats.get('total_rows', 0):,}")
    output.append(f"  Average speeches per file: {topic_stats.get('avg_rows_per_file', 0):.1f}")
    
    if 'substantial_speeches' in topic_stats:
        output.append(f"\n  Substantive speeches (≥5 content words):")
        output.append(f"    • Substantive: {topic_stats['substantial_speeches']:,} "
                     f"({topic_stats['substantial_percentage']:.1f}%)")
        output.append(f"    • Non-substantive: {topic_stats['non_substantial_speeches']:,} "
                     f"({100 - topic_stats['substantial_percentage']:.1f}%)")
        
        if 'content_word_stats' in topic_stats:
            cw = topic_stats['content_word_stats']
            output.append(f"    • Content words: {cw['avg_mean']:.1f} on average "
                         f"(range: {cw['overall_min']}-{cw['overall_max']})")
    
    if 'text_stats' in topic_stats:
        ts = topic_stats['text_stats']
        output.append(f"\n  Speech length (in words):")
        output.append(f"    • Mean: {ts['avg_mean_length']:.1f} ± {ts['avg_std_length']:.1f}")
        output.append(f"    • Median: {ts['avg_median_length']:.1f}")
        output.append(f"    • Range: {ts['overall_min']} - {ts['overall_max']}")
        
        if ts.get('total_empty', 0) > 0:
            output.append(f"    • Empty texts: {ts['total_empty']:,}")
        if ts.get('total_very_short', 0) > 0:
            output.append(f"    • Very short (<3 words): {ts['total_very_short']:,}")
    
    # Clustering info
    if topic_stats.get('has_clustering'):
        output.append(f"\n  Clustering: ✓ Present in {topic_stats.get('files_with_clustering', 0)} file(s)")
    
    # Score distribution
    if 'score_distribution' in topic_stats:
        output.append("\nOVERALL SCORE DISTRIBUTION:")
        for score in sorted(topic_stats['score_distribution'].keys(), key=lambda x: int(x)):
            count = topic_stats['score_distribution'][score]
            pct = count / topic_stats['total_rows'] * 100 if topic_stats['total_rows'] > 0 else 0
            bar = '█' * int(pct / 2)  # Visual bar
            output.append(f"  Score {score}: {count:>6,} ({pct:5.1f}%) {bar}")
    
    # Numeric features (topic-level)
    if topic_stats.get('numeric_features'):
        output.append("\nFEATURE STATISTICS (TOPIC-LEVEL):")
        
        # Prioritize key features
        key_features = ['overall_score', 'jargon_density', 'lexical_sophistication', 
                       'syntactic_complexity', 'num_non_function_words']
        
        # Show key features first
        shown_features = []
        for feature in key_features:
            if feature in topic_stats['numeric_features']:
                shown_features.append(feature)
        
        # Add remaining features
        for feature in sorted(topic_stats['numeric_features'].keys()):
            if feature not in shown_features:
                shown_features.append(feature)
        
        for feature in shown_features:
            fstats = topic_stats['numeric_features'][feature]
            output.append(f"\n  {feature}:")
            output.append(f"    Weighted mean: {fstats['weighted_mean']:.4f}")
            output.append(f"    Avg mean:      {fstats['avg_mean']:.4f}")
            output.append(f"    Avg median:    {fstats['avg_median']:.4f}")
            output.append(f"    Avg std:       {fstats['avg_std']:.4f}")
            output.append(f"    Range:         [{fstats['overall_min']:.4f}, {fstats['overall_max']:.4f}]")
            
            if fstats.get('total_missing', 0) > 0:
                output.append(f"    Missing:       {fstats['total_missing']:,}")
    
    # File-level details
    if include_file_details:
        output.append("\n" + "-" * 80)
        output.append("FILE-LEVEL DETAILS:")
        output.append("-" * 80)
        
        for fstats in file_stats:
            if 'error' in fstats:
                output.append(f"\n  ❌ {fstats['file']}: ERROR - {fstats['error']}")
                continue
            
            output.append(f"\n  📄 {fstats['file']}")
            output.append(f"     Total rows: {fstats['total_rows']:,}")
            
            if fstats.get('substantial_speeches') is not None:
                output.append(f"     Substantive: {fstats['substantial_speeches']:,} "
                             f"({fstats['substantial_percentage']:.1f}%)")
            
            if 'text_stats' in fstats:
                ts = fstats['text_stats']
                output.append(f"     Text length: {ts['mean_length_words']:.1f} ± {ts['std_length_words']:.1f} words")
                output.append(f"                  (median: {ts['median_length_words']:.0f}, "
                             f"range: {ts['min_length_words']}-{ts['max_length_words']})")
            
            # Clustering info
            if fstats.get('has_clustering'):
                output.append(f"     Clustering: ✓")
                if 'cluster_distribution' in fstats:
                    cd = fstats['cluster_distribution']
                    output.append(f"                 {cd['unique_clusters']} unique clusters")
            
            # Show top 3 features
            if fstats.get('numeric_features'):
                key_features = ['overall_score', 'jargon_density', 'lexical_sophistication']
                shown_features = {k: v for k, v in fstats['numeric_features'].items() 
                                if k in key_features}
                
                if shown_features:
                    output.append(f"     Key features:")
                    for feat, vals in shown_features.items():
                        output.append(f"       • {feat}: {vals['mean']:.3f} ± {vals['std']:.3f}")
    
    output.append("")

def process_assemblies(base_dir: Path, output_path: Path, content_threshold: int = 5) -> None:
    """Process Citizens' Assembly files and generate report."""
    
    print("\n" + "=" * 80)
    print("PROCESSING CITIZENS' ASSEMBLIES")
    print("=" * 80)
    
    output = []
    output.append("=" * 80)
    output.append("CITIZENS' ASSEMBLIES - CORPUS STATISTICS")
    output.append("=" * 80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Content word threshold: {content_threshold}")
    output.append("")
    
    all_results = {}
    
    # Process each assembly folder
    for folder_name, topic_name in ASSEMBLY_TOPIC_MAP.items():
        folder_path = base_dir / folder_name
        
        # Handle scotland_csv vs scotland
        if not folder_path.exists():
            alt_names = [folder_name + "_csv", folder_name.replace("_csv", "")]
            for alt_name in alt_names:
                alt_path = base_dir / alt_name
                if alt_path.exists():
                    folder_path = alt_path
                    break
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            continue
        
        print(f"\n📂 {topic_name}")
        
        # Find all CSV files
        csv_files = sorted(folder_path.glob("*.csv"))
        
        if not csv_files:
            print(f"   No CSV files found")
            continue
        
        print(f"   Processing {len(csv_files)} file(s)...")
        
        # Compute statistics for each file
        file_stats = []
        for csv_file in csv_files:
            print(f"      {csv_file.name}")
            stats = compute_file_statistics(csv_file, content_threshold)
            file_stats.append(stats)
        
        # Aggregate topic statistics
        topic_stats = aggregate_topic_statistics(file_stats, content_threshold)
        
        # Store results
        all_results[topic_name] = {
            'file_stats': file_stats,
            'topic_stats': topic_stats
        }
        
        # Generate report section
        generate_report(topic_name, file_stats, topic_stats, output, include_file_details=True)
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"\n✅ Assembly report saved to: {output_path}")
    
    # Save JSON version
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON data saved to: {json_path}")

def process_experts(base_dir: Path, output_path: Path, content_threshold: int = 5) -> None:
    """Process expert files and generate report."""
    
    print("\n" + "=" * 80)
    print("PROCESSING EXPERT SPEECHES")
    print("=" * 80)
    
    output = []
    output.append("=" * 80)
    output.append("EXPERT SPEECHES - CORPUS STATISTICS")
    output.append("=" * 80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Content word threshold: {content_threshold}")
    output.append("")
    
    all_results = {}
    
    # Process each expert topic folder
    for folder_name, topic_name in EXPERT_TOPIC_MAP.items():
        folder_path = base_dir / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            continue
        
        print(f"\n📂 {topic_name}")
        
        # Find all CSV files
        csv_files = sorted(folder_path.glob("*.csv"))
        
        if not csv_files:
            print(f"   No CSV files found")
            continue
        
        print(f"   Processing {len(csv_files)} file(s)...")
        
        # Compute statistics for each file
        file_stats = []
        for csv_file in csv_files:
            print(f"      {csv_file.name}")
            stats = compute_file_statistics(csv_file, content_threshold)
            file_stats.append(stats)
        
        # Aggregate topic statistics
        topic_stats = aggregate_topic_statistics(file_stats, content_threshold)
        
        # Store results
        all_results[topic_name] = {
            'file_stats': file_stats,
            'topic_stats': topic_stats
        }
        
        # Generate report section
        generate_report(topic_name, file_stats, topic_stats, output, include_file_details=True)
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"\n✅ Expert report saved to: {output_path}")
    
    # Save JSON version
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON data saved to: {json_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive corpus statistics for assemblies and experts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("../../data/assemblies/processed_files"),
        help="Base directory containing processed files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--content_threshold",
        type=int,
        default=5,
        help="Minimum number of content words for substantive speech (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.base_dir.exists():
        print(f"❌ Base directory not found: {args.base_dir}")
        return
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process assemblies
    assemblies_output = args.output_dir / "assemblies_statistics.txt"
    process_assemblies(args.base_dir, assemblies_output, args.content_threshold)
    
    # Process experts
    experts_dir = args.base_dir / "experts"
    if experts_dir.exists():
        experts_output = args.output_dir / "experts_statistics.txt"
        process_experts(experts_dir, experts_output, args.content_threshold)
    else:
        print(f"\n⚠️  Expert directory not found: {experts_dir}")
    
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS GENERATED")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")

if __name__ == "__main__":
    main()