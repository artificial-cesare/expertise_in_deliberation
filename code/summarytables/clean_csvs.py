#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean CSV files by keeping only specified columns.
Removes all other columns from both Citizens' Assemblies and Expert datasets.
"""

import pandas as pd
from pathlib import Path
from typing import List, Set
import argparse
import re
from datetime import datetime

# Columns to keep (exact names)
COLUMNS_TO_KEEP = [
    # Metadata
    "date",
    "speaker",
    "text",
    
    # Linguistic features
    "jargon_density",
    "lexical_sophistication",
    "syntactic_complexity",
    "hedge_ratio",
    "reasoning_markers",
    "evidential_markers",
    "hedge_density",
    "booster_density",
    "epistemic_balance",
    
    # Specificity features
    "specificity_length_norm",
    "specificity_mean_word_len_norm",
    "specificity_numbers_norm",
    "specificity_content_prop",
    
    # Length features
    "length_tokens",
    "num_non_function_words",
    
    # LLM features
    "expertise_llm_score",
    "expertise_justification",
    
    # Overall score
    "overall_score",
]

# Topic mappings
ASSEMBLY_TOPICS = ["constitution", "future", "gender"]
MODEL_TYPES = ["base", "finetuned"]
PCA_MODES = ["experts_only", "experts_and_assemblies"]

def generate_clustering_patterns() -> List[re.Pattern]:
    """
    Generate regex patterns for clustering columns.
    
    Format:
    - cluster_{topic}_{model_type}_{pca_mode}_k{k_val}
    - distance_{topic}_{model_type}_{pca_mode}_k{k_val}
    - similarity_{topic}_{model_type}_{pca_mode}_k{k_val}
    """
    patterns = []
    
    # Generate pattern for each prefix
    for prefix in ["cluster", "distance", "similarity"]:
        # Pattern matches: prefix_topic_modeltype_pcamode_kN
        # where topic, modeltype, pcamode are variable and N is a digit
        pattern = re.compile(
            rf"^{prefix}_({'|'.join(ASSEMBLY_TOPICS)})_"
            rf"({'|'.join(MODEL_TYPES)})_"
            rf"({'|'.join(PCA_MODES)})_"
            r"k\d+$"
        )
        patterns.append(pattern)
    
    return patterns

def identify_columns_to_keep(df: pd.DataFrame) -> Set[str]:
    """
    Identify which columns to keep in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Set of column names to keep
    """
    columns_to_keep = set()
    clustering_patterns = generate_clustering_patterns()
    
    for col in df.columns:
        # Check if column is in the fixed list
        if col in COLUMNS_TO_KEEP:
            columns_to_keep.add(col)
            continue
        
        # Check if column matches any clustering pattern
        for pattern in clustering_patterns:
            if pattern.match(col):
                columns_to_keep.add(col)
                break
    
    return columns_to_keep

def clean_csv_file(csv_path: Path, dry_run: bool = False) -> dict:
    """
    Clean a single CSV file by keeping only specified columns.
    
    Args:
        csv_path: Path to CSV file
        dry_run: If True, only report changes without modifying file
        
    Returns:
        Dictionary with cleaning statistics
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        original_columns = set(df.columns)
        original_count = len(original_columns)
        
        # Identify columns to keep
        columns_to_keep = identify_columns_to_keep(df)
        
        # Columns that exist and will be kept
        existing_to_keep = columns_to_keep.intersection(original_columns)
        
        # Columns that will be dropped
        columns_to_drop = original_columns - existing_to_keep
        
        stats = {
            'file': csv_path.name,
            'original_columns': original_count,
            'kept_columns': len(existing_to_keep),
            'dropped_columns': len(columns_to_drop),
            'kept_list': sorted(existing_to_keep),
            'dropped_list': sorted(columns_to_drop),
            'success': True,
            'error': None
        }
        
        # If not dry run, save cleaned file
        if not dry_run:
            # Keep only selected columns (in order they appear in original)
            columns_order = [col for col in df.columns if col in existing_to_keep]
            df_cleaned = df[columns_order]
            
            # Save (overwrite original)
            df_cleaned.to_csv(csv_path, index=False)
            stats['saved'] = True
        else:
            stats['saved'] = False
        
        return stats
        
    except Exception as e:
        return {
            'file': csv_path.name,
            'success': False,
            'error': str(e),
            'saved': False
        }

def process_directory(
    directory: Path,
    recursive: bool = False,
    dry_run: bool = False
) -> List[dict]:
    """
    Process all CSV files in a directory.
    
    Args:
        directory: Directory to process
        recursive: If True, process subdirectories recursively
        dry_run: If True, only report changes without modifying files
        
    Returns:
        List of statistics dictionaries
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    # Find CSV files
    if recursive:
        csv_files = sorted(directory.rglob("*.csv"))
    else:
        csv_files = sorted(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {directory}")
        return []
    
    print(f"\n{'📂' if recursive else '📄'} Processing {len(csv_files)} file(s) in {directory.name}")
    if dry_run:
        print("   [DRY RUN MODE - No files will be modified]")
    
    all_stats = []
    
    for csv_file in csv_files:
        print(f"   {csv_file.name}... ", end='', flush=True)
        stats = clean_csv_file(csv_file, dry_run)
        all_stats.append(stats)
        
        if stats['success']:
            print(f"✓ (kept {stats['kept_columns']}/{stats['original_columns']} columns)")
        else:
            print(f"❌ {stats['error']}")
    
    return all_stats

def generate_report(all_stats: List[dict], output_path: Path) -> None:
    """Generate a detailed cleaning report."""
    
    output = []
    output.append("=" * 80)
    output.append("CSV CLEANING REPORT")
    output.append("=" * 80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    
    # Summary statistics
    successful = [s for s in all_stats if s['success']]
    failed = [s for s in all_stats if not s['success']]
    
    output.append("SUMMARY:")
    output.append(f"  Total files processed: {len(all_stats)}")
    output.append(f"  Successful: {len(successful)}")
    output.append(f"  Failed: {len(failed)}")
    
    if successful:
        total_original = sum(s['original_columns'] for s in successful)
        total_kept = sum(s['kept_columns'] for s in successful)
        total_dropped = sum(s['dropped_columns'] for s in successful)
        
        output.append(f"\n  Total columns:")
        output.append(f"    • Original: {total_original:,}")
        output.append(f"    • Kept: {total_kept:,}")
        output.append(f"    • Dropped: {total_dropped:,}")
        output.append(f"    • Reduction: {total_dropped/total_original*100:.1f}%")
    
    # Columns kept (across all files)
    if successful:
        output.append("\n" + "=" * 80)
        output.append("COLUMNS KEPT (UNION ACROSS ALL FILES):")
        output.append("=" * 80)
        
        all_kept = set()
        for s in successful:
            all_kept.update(s['kept_list'])
        
        # Categorize columns
        metadata_cols = [c for c in all_kept if c in ['date', 'speaker', 'text']]
        feature_cols = [c for c in all_kept if c in COLUMNS_TO_KEEP and c not in metadata_cols]
        clustering_cols = [c for c in all_kept if c not in COLUMNS_TO_KEEP]
        
        if metadata_cols:
            output.append("\nMetadata columns:")
            for col in sorted(metadata_cols):
                output.append(f"  • {col}")
        
        if feature_cols:
            output.append("\nFeature columns:")
            for col in sorted(feature_cols):
                output.append(f"  • {col}")
        
        if clustering_cols:
            output.append("\nClustering columns:")
            for col in sorted(clustering_cols):
                output.append(f"  • {col}")
    
    # File-level details
    output.append("\n" + "=" * 80)
    output.append("FILE-LEVEL DETAILS:")
    output.append("=" * 80)
    
    for stats in all_stats:
        output.append(f"\n📄 {stats['file']}")
        
        if not stats['success']:
            output.append(f"   ❌ ERROR: {stats['error']}")
            continue
        
        output.append(f"   Original columns: {stats['original_columns']}")
        output.append(f"   Kept columns: {stats['kept_columns']}")
        output.append(f"   Dropped columns: {stats['dropped_columns']} ({stats['dropped_columns']/stats['original_columns']*100:.1f}%)")
        
        if stats['dropped_list']:
            output.append(f"\n   Dropped:")
            for col in stats['dropped_list']:
                output.append(f"     • {col}")
    
    # Failed files
    if failed:
        output.append("\n" + "=" * 80)
        output.append("FAILED FILES:")
        output.append("=" * 80)
        
        for stats in failed:
            output.append(f"\n❌ {stats['file']}")
            output.append(f"   Error: {stats['error']}")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"\n📊 Detailed report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Clean CSV files by keeping only specified columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes)
  python clean_csvs.py --dry_run
  
  # Clean assemblies only
  python clean_csvs.py --assemblies_only
  
  # Clean experts only
  python clean_csvs.py --experts_only
  
  # Clean everything
  python clean_csvs.py
  
Kept columns:
  - Metadata: date, speaker, text
  - Features: jargon_density, lexical_sophistication, etc.
  - Clustering: cluster_*, distance_*, similarity_* (matching pattern)
        """
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
        help="Output directory for report"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--assemblies_only",
        action="store_true",
        help="Only process assembly files"
    )
    parser.add_argument(
        "--experts_only",
        action="store_true",
        help="Only process expert files"
    )
    
    args = parser.parse_args()
    
    # Validate base directory
    if not args.base_dir.exists():
        print(f"❌ Base directory not found: {args.base_dir}")
        return
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CSV CLEANING SCRIPT")
    print("=" * 80)
    print(f"\nBase directory: {args.base_dir}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified")
    
    all_stats = []
    
    # Process assemblies
    if not args.experts_only:
        print("\n" + "=" * 80)
        print("PROCESSING CITIZENS' ASSEMBLIES")
        print("=" * 80)
        
        assembly_folders = ["8th_amendment", "scotland", "gendereq"]
        
        for folder_name in assembly_folders:
            folder_path = args.base_dir / folder_name
            
            # Handle alternative folder names
            if not folder_path.exists():
                alt_names = [folder_name + "_csv", folder_name.replace("_csv", "")]
                for alt_name in alt_names:
                    alt_path = args.base_dir / alt_name
                    if alt_path.exists():
                        folder_path = alt_path
                        break
            
            if folder_path.exists():
                stats = process_directory(folder_path, recursive=False, dry_run=args.dry_run)
                all_stats.extend(stats)
            else:
                print(f"\n⚠️  Folder not found: {folder_name}")
    
    # Process experts
    if not args.assemblies_only:
        print("\n" + "=" * 80)
        print("PROCESSING EXPERT SPEECHES")
        print("=" * 80)
        
        experts_dir = args.base_dir / "experts"
        
        if experts_dir.exists():
            # Process each expert topic
            expert_folders = ["constitution", "future", "gender"]
            
            for folder_name in expert_folders:
                folder_path = experts_dir / folder_name
                
                if folder_path.exists():
                    stats = process_directory(folder_path, recursive=False, dry_run=args.dry_run)
                    all_stats.extend(stats)
                else:
                    print(f"\n⚠️  Folder not found: {folder_name}")
        else:
            print(f"\n⚠️  Expert directory not found: {experts_dir}")
    
    # Generate report
    if all_stats:
        report_suffix = "_dry_run" if args.dry_run else ""
        report_path = args.output_dir / f"cleaning_report{report_suffix}.txt"
        generate_report(all_stats, report_path)
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ CLEANING COMPLETE")
        print("=" * 80)
        
        successful = sum(1 for s in all_stats if s['success'])
        failed = sum(1 for s in all_stats if not s['success'])
        
        print(f"\nFiles processed: {len(all_stats)}")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {failed}")
        
        if args.dry_run:
            print("\n⚠️  DRY RUN: No files were modified")
            print("   Run without --dry_run to apply changes")
        else:
            print(f"\n✅ Files cleaned and saved")
        
        print(f"\n📊 Report: {report_path}")
    else:
        print("\n⚠️  No files processed")

if __name__ == "__main__":
    main()