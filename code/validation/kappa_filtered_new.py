import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# ============================================================================
# ✅ ADD: Problematic speeches to exclude (reported by participants)
# ============================================================================
EXCLUDED_SPEECHES = {
    'experts/future/committees.parliament.uk_speeches.csv/row_69',
    'assemblies/scotland/8992int020_speeches.csv/row_67', 
    'experts/constitution/committees.parliament.uk_speeches.csv/row_0'
}

from io_utils import load_sampled_speeches, load_annotations, prepare_ratings_matrix

def filter_extreme_scores(ratings_matrix):
    """
    Filter speeches where ANY rater gave a score of 1 or 5.
    Returns filtered matrix and indices of extreme speeches.
    """
    # Convert to numeric
    numeric_matrix = ratings_matrix.apply(pd.to_numeric, errors='coerce')
    
    # Check if any score in each row is 1 or 5
    has_extreme = numeric_matrix.isin([1.0, 5.0]).any(axis=1)
    
    # Filter to only speeches with at least one extreme score
    filtered_matrix = ratings_matrix[has_extreme].copy()
    extreme_indices = ratings_matrix.index[has_extreme].tolist()
    
    return filtered_matrix, extreme_indices

def calculate_kappas(ratings_matrix, rater1, rater2):
    """Calculate unweighted and weighted kappa for two raters."""
    # Get valid pairs (both raters have scored)
    valid_mask = ratings_matrix[[rater1, rater2]].notna().all(axis=1)
    valid_ratings = ratings_matrix.loc[valid_mask, [rater1, rater2]]
    
    if len(valid_ratings) == 0:
        return None
    
    y1 = valid_ratings[rater1].astype(int)
    y2 = valid_ratings[rater2].astype(int)
    
    # Unweighted kappa
    kappa_unweighted = cohen_kappa_score(y1, y2)
    
    # Weighted kappa (quadratic)
    kappa_weighted = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return {
        'n_samples': int(len(valid_ratings)),
        'kappa_unweighted': float(kappa_unweighted),
        'kappa_weighted_quadratic': float(kappa_weighted),
        'speech_ids': valid_ratings.index.tolist()
    }

def main():
    # ========================================================================
    # ✅ ADD: Command-line argument parsing
    # ========================================================================
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate inter-rater agreement (Cohen\'s Kappa)')
    parser.add_argument(
        '--extreme-only',
        action='store_true',
        help='Only analyze speeches with extreme scores (1 or 5)'
    )
    parser.add_argument(
        '--json-path',
        type=str,
        default='data/validation/sampled_speeches.json',
        help='Path to sampled speeches JSON'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default='data/validation/annotations_csv',
        help='Directory containing annotation CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='code/validation/kappa_results.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Paths
    json_path = args.json_path
    csv_dir = args.csv_dir
    output_path = args.output
    
    # Load data
    print("Loading data...")
    speeches = load_sampled_speeches(json_path)
    annotations = load_annotations(csv_dir)
    
    # ========================================================================
    # ✅ DISPLAY: Show excluded speeches
    # ========================================================================
    print(f"\n🚫 Excluding {len(EXCLUDED_SPEECHES)} problematic speeches:")
    for speech_id in sorted(EXCLUDED_SPEECHES):
        print(f"   - {speech_id}")
    
    # Prepare full ratings matrix (with exclusions)
    print("\nPreparing ratings matrix (after exclusions)...")
    full_matrix = prepare_ratings_matrix(speeches, annotations)
    full_matrix = full_matrix[~full_matrix.index.isin(EXCLUDED_SPEECHES)]

    print(f"Total speeches (after exclusions): {len(full_matrix)}")
    
    # ========================================================================
    # ✅ CONDITIONAL: Filter to extreme scores if flag is set
    # ========================================================================
    if args.extreme_only:
        print("\n🔍 Filtering speeches with extreme scores (1 or 5)...")
        filtered_matrix, extreme_indices = filter_extreme_scores(full_matrix)
        print(f"Speeches with at least one extreme score: {len(filtered_matrix)}")
        print(f"Percentage: {len(filtered_matrix) / len(full_matrix) * 100:.1f}%")
        filter_description = 'Speeches where ANY rater (human or LLM) gave a score of 1 or 5'
    else:
        print("\n📊 Using all speeches (no extreme score filtering)...")
        filtered_matrix = full_matrix
        extreme_indices = full_matrix.index.tolist()
        filter_description = 'All speeches (no filtering)'
    
    # Get raters
    human_raters = [col for col in filtered_matrix.columns if col != 'llm']
    
    # Initialize results
    results = {
        'metadata': {
            'total_speeches': len(full_matrix),
            'analyzed_speeches': len(filtered_matrix),
            'analyzed_percentage': float(len(filtered_matrix) / len(full_matrix) * 100),
            'n_human_raters': len(human_raters),
            'filter_criterion': filter_description,
            'extreme_only': args.extreme_only,
            'excluded_speeches': sorted(list(EXCLUDED_SPEECHES)),
            'n_excluded': len(EXCLUDED_SPEECHES)
        },
        'speech_ids': extreme_indices
    }
    
    # Calculate LLM vs each human
    print("\nCalculating LLM vs human rater kappas...")
    results['llm_vs_humans'] = {}
    
    for human in human_raters:
        print(f"   LLM vs {human}...")
        kappa_result = calculate_kappas(filtered_matrix, 'llm', human)
        if kappa_result:
            results['llm_vs_humans'][f'llm_vs_{human}'] = kappa_result
    
    # Calculate inter-human
    print("\nCalculating inter-human kappas...")
    results['inter_human'] = {}
    
    for i, h1 in enumerate(human_raters):
        for h2 in human_raters[i+1:]:
            print(f"   {h1} vs {h2}...")
            kappa_result = calculate_kappas(filtered_matrix, h1, h2)
            if kappa_result:
                results['inter_human'][f'{h1}_vs_{h2}'] = kappa_result
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    # LLM vs humans summary
    llm_kappas_unw = [v['kappa_unweighted'] for v in results['llm_vs_humans'].values()]
    llm_kappas_w = [v['kappa_weighted_quadratic'] for v in results['llm_vs_humans'].values()]
    
    # Inter-human summary
    human_kappas_unw = [v['kappa_unweighted'] for v in results['inter_human'].values()]
    human_kappas_w = [v['kappa_weighted_quadratic'] for v in results['inter_human'].values()]
    
    results['summary'] = {
        'llm_vs_humans': {
            'avg_kappa_unweighted': float(np.mean(llm_kappas_unw)),
            'std_kappa_unweighted': float(np.std(llm_kappas_unw)),
            'min_kappa_unweighted': float(np.min(llm_kappas_unw)),
            'max_kappa_unweighted': float(np.max(llm_kappas_unw)),
            'avg_kappa_weighted': float(np.mean(llm_kappas_w)),
            'std_kappa_weighted': float(np.std(llm_kappas_w)),
        },
        'inter_human': {
            'avg_kappa_unweighted': float(np.mean(human_kappas_unw)),
            'std_kappa_unweighted': float(np.std(human_kappas_unw)),
            'min_kappa_unweighted': float(np.min(human_kappas_unw)),
            'max_kappa_unweighted': float(np.max(human_kappas_unw)),
            'avg_kappa_weighted': float(np.mean(human_kappas_w)),
            'std_kappa_weighted': float(np.std(human_kappas_w)),
        }
    }
    
    # Add comparison note
    if args.extreme_only:
        results['comparison_note'] = (
            f"These kappa scores are calculated ONLY on speeches where at least one rater "
            f"(human or LLM) assigned an extreme score (1 or 5). {len(EXCLUDED_SPEECHES)} "
            f"problematic speeches with high disagreement were excluded from the analysis."
        )
    else:
        results['comparison_note'] = (
            f"These kappa scores are calculated on ALL speeches (no extreme score filtering). "
            f"{len(EXCLUDED_SPEECHES)} problematic speeches with high disagreement were excluded from the analysis."
        )
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("KAPPA ANALYSIS - SUMMARY")
    print("="*70)
    
    print(f"\n📊 DATASET:")
    print(f"   Mode: {'EXTREME SCORES ONLY' if args.extreme_only else 'ALL SCORES'}")
    print(f"   Excluded speeches: {results['metadata']['n_excluded']}")
    print(f"   Total speeches (after exclusions): {results['metadata']['total_speeches']}")
    print(f"   Analyzed speeches: {results['metadata']['analyzed_speeches']}")
    print(f"   Percentage: {results['metadata']['analyzed_percentage']:.1f}%")
    
    print("\n🤖 LLM vs HUMAN RATERS:")
    print(f"   Average κ (unweighted): {results['summary']['llm_vs_humans']['avg_kappa_unweighted']:.3f} "
          f"(±{results['summary']['llm_vs_humans']['std_kappa_unweighted']:.3f})")
    print(f"   Average κ (weighted):   {results['summary']['llm_vs_humans']['avg_kappa_weighted']:.3f} "
          f"(±{results['summary']['llm_vs_humans']['std_kappa_weighted']:.3f})")
    print(f"   Range: [{results['summary']['llm_vs_humans']['min_kappa_unweighted']:.3f}, "
          f"{results['summary']['llm_vs_humans']['max_kappa_unweighted']:.3f}]")
    
    print("\n👥 INTER-HUMAN RELIABILITY:")
    print(f"   Average κ (unweighted): {results['summary']['inter_human']['avg_kappa_unweighted']:.3f} "
          f"(±{results['summary']['inter_human']['std_kappa_unweighted']:.3f})")
    print(f"   Average κ (weighted):   {results['summary']['inter_human']['avg_kappa_weighted']:.3f} "
          f"(±{results['summary']['inter_human']['std_kappa_weighted']:.3f})")
    print(f"   Range: [{results['summary']['inter_human']['min_kappa_unweighted']:.3f}, "
          f"{results['summary']['inter_human']['max_kappa_unweighted']:.3f}]")
    
    print("\n📋 INDIVIDUAL COMPARISONS:")
    print("\n   LLM vs Humans:")
    for pair, values in results['llm_vs_humans'].items():
        print(f"      {pair}: κ={values['kappa_unweighted']:.3f}, "
              f"κ_w={values['kappa_weighted_quadratic']:.3f} (n={values['n_samples']})")
    
    print("\n   Inter-Human:")
    for pair, values in results['inter_human'].items():
        print(f"      {pair}: κ={values['kappa_unweighted']:.3f}, "
              f"κ_w={values['kappa_weighted_quadratic']:.3f} (n={values['n_samples']})")
    
    print("\n" + "="*70)
    print(f"✅ Results saved to: {output_path}")
    print("="*70)

if __name__ == '__main__':
    main()