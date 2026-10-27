import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt

def load_sampled_speeches(json_path):
    """Load the sampled speeches from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['speeches']

def load_annotations(csv_dir):
    """Load all annotation CSV files from directory."""
    csv_dir = Path(csv_dir)
    annotations = []
    
    for csv_file in csv_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        annotations.append(df)
    
    return pd.concat(annotations, ignore_index=True)

def prepare_ratings_matrix(speeches, annotations):
    """Prepare a matrix of ratings for each speech."""
    speech_ids = [speech['speech_id'] for speech in speeches]
    raters = annotations['rater_id'].unique().tolist()
    raters.append('llm')
    
    ratings_matrix = pd.DataFrame(index=speech_ids, columns=raters, dtype=float)
    
    for _, row in annotations.iterrows():
        ratings_matrix.loc[row['item_id'], row['rater_id']] = float(row['score'])
    
    for speech in speeches:
        ratings_matrix.loc[speech['speech_id'], 'llm'] = float(speech['expertise_score'])
    
    return ratings_matrix

def calculate_raw_agreement(y1, y2):
    """Calculate percentage of identical labels."""
    return float(np.mean(y1 == y2) * 100)

def weighted_kappa(y1, y2, weights='quadratic'):
    """Calculate weighted Cohen's kappa for ordinal data."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2, weights=weights)

def compute_confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix with labels."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

def analyze_pairwise_agreement(ratings_matrix, rater1, rater2, labels):
    """Comprehensive pairwise analysis between two raters."""
    valid_mask = ratings_matrix[[rater1, rater2]].notna().all(axis=1)
    valid_ratings = ratings_matrix.loc[valid_mask, [rater1, rater2]]
    
    if len(valid_ratings) == 0:
        return None
    
    y1 = valid_ratings[rater1].astype(int)
    y2 = valid_ratings[rater2].astype(int)
    
    # 1. Raw agreement
    raw_agreement = calculate_raw_agreement(y1, y2)
    
    # 2. Unweighted kappa
    kappa_unweighted = cohen_kappa_score(y1, y2)
    
    # 3. Weighted kappa (quadratic)
    kappa_weighted = weighted_kappa(y1, y2, weights='quadratic')
    
    # 4. Confusion matrix
    cm = compute_confusion_matrix(y1, y2, labels)
    
    # 5. Per-category statistics
    disagreements = []
    for idx in valid_ratings.index:
        score1 = int(valid_ratings.loc[idx, rater1])
        score2 = int(valid_ratings.loc[idx, rater2])
        if score1 != score2:
            disagreements.append({
                'speech_id': idx,
                f'{rater1}_score': score1,
                f'{rater2}_score': score2,
                'difference': abs(score1 - score2)
            })
    
    return {
        'n_samples': int(len(valid_ratings)),
        'raw_agreement_pct': float(raw_agreement),
        'kappa_unweighted': float(kappa_unweighted),
        'kappa_weighted_quadratic': float(kappa_weighted),
        'confusion_matrix': cm.to_dict(),
        'disagreements': disagreements,
        'n_disagreements': len(disagreements),
        'mean_absolute_difference': float(np.mean(np.abs(y1 - y2)))
    }

def aggregate_human_ratings(ratings_matrix, method='mean'):
    """Aggregate human ratings (excluding LLM)."""
    human_raters = [col for col in ratings_matrix.columns if col != 'llm']
    human_ratings = ratings_matrix[human_raters].copy()
    
    # Convert to numeric, coercing errors to NaN
    human_ratings = human_ratings.apply(pd.to_numeric, errors='coerce')
    
    if method == 'mean':
        # Calculate mean and round, keeping NaN for rows with all NaN
        aggregated = human_ratings.mean(axis=1, skipna=True).round()
    elif method == 'median':
        aggregated = human_ratings.median(axis=1, skipna=True).round()
    elif method == 'mode':
        # Use scipy mode for each row
        def get_mode(row):
            valid_values = row.dropna()
            if len(valid_values) == 0:
                return np.nan
            result = mode(valid_values, keepdims=False)
            return result.mode
        
        aggregated = human_ratings.apply(get_mode, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated

def analyze_llm_vs_aggregated(ratings_matrix, aggregation_methods, labels):
    """Compare LLM against aggregated human ratings."""
    results = {}
    
    for method in aggregation_methods:
        human_agg = aggregate_human_ratings(ratings_matrix, method=method)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'llm': ratings_matrix['llm'],
            f'human_{method}': human_agg
        })
        
        # Remove rows with NaN
        comparison = comparison.dropna()
        
        if len(comparison) == 0:
            continue
        
        y_llm = comparison['llm'].astype(int)
        y_human = comparison[f'human_{method}'].astype(int)
        
        results[method] = analyze_pairwise_agreement(
            pd.DataFrame({'llm': y_llm, 'human': y_human}),
            'llm',
            'human',
            labels
        )
    
    return results

def identify_high_disagreement_speeches(ratings_matrix, speeches, top_n=10):
    """Identify speeches with highest disagreement across all raters."""
    disagreement_scores = []
    
    for speech_id in ratings_matrix.index:
        scores = ratings_matrix.loc[speech_id].dropna()
        
        if len(scores) > 1:
            # Convert to numeric
            scores = pd.to_numeric(scores, errors='coerce').dropna()
            
            if len(scores) <= 1:
                continue
                
            scores = scores.astype(int)
            
            # Calculate multiple disagreement metrics
            std = scores.std()
            range_val = scores.max() - scores.min()
            
            # Get speech text
            speech_text = next(
                (s['text'] for s in speeches if s['speech_id'] == speech_id),
                "N/A"
            )
            
            # Get all individual scores
            all_scores = {
                rater: int(ratings_matrix.loc[speech_id, rater])
                for rater in ratings_matrix.columns
                if pd.notna(ratings_matrix.loc[speech_id, rater])
            }
            
            disagreement_scores.append({
                'speech_id': speech_id,
                'std_dev': float(std),
                'range': int(range_val),
                'mean_score': float(scores.mean()),
                'n_raters': int(len(scores)),
                'individual_scores': all_scores,
                'text_preview': speech_text[:200] + "..." if len(speech_text) > 200 else speech_text
            })
    
    # Sort by standard deviation (primary) and range (secondary)
    disagreement_scores.sort(key=lambda x: (x['std_dev'], x['range']), reverse=True)
    
    return disagreement_scores[:top_n]

def generate_confusion_matrix_plots(pairwise_results, output_dir):
    """Generate heatmap plots for confusion matrices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair_name, results in pairwise_results.items():
        if results is None:
            continue
        
        cm_dict = results['confusion_matrix']
        cm_df = pd.DataFrame(cm_dict)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix: {pair_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        safe_name = pair_name.replace('/', '_').replace(' ', '_')
        plt.savefig(output_dir / f'confusion_matrix_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def diagnose_label_imbalance(ratings_matrix):
    """Analyze label distribution to diagnose imbalance."""
    all_ratings = []
    
    for col in ratings_matrix.columns:
        scores = pd.to_numeric(ratings_matrix[col], errors='coerce').dropna().astype(int)
        all_ratings.extend(scores.tolist())
    
    label_counts = pd.Series(all_ratings).value_counts().sort_index()
    
    return {
        'label_distribution': {int(k): int(v) for k, v in label_counts.items()},
        'most_common_label': int(label_counts.idxmax()),
        'least_common_label': int(label_counts.idxmin()),
        'imbalance_ratio': float(label_counts.max() / label_counts.min())
    }

def main():
    # Paths
    json_path = 'code/validation/sampled_speeches.json'
    csv_dir = 'code/validation/annotations_csv'
    output_path = 'code/validation/comprehensive_reliability_results.json'
    plots_dir = 'code/validation/confusion_matrices'
    
    # Load data
    print("Loading data...")
    speeches = load_sampled_speeches(json_path)
    annotations = load_annotations(csv_dir)
    
    # Prepare ratings matrix
    print("Preparing ratings matrix...")
    ratings_matrix = prepare_ratings_matrix(speeches, annotations)
    
    # Define labels (expertise scores 1-5)
    labels = [1, 2, 3, 4, 5]
    
    # Get human raters
    human_raters = [col for col in ratings_matrix.columns if col != 'llm']
    
    # Initialize results structure
    results = {
        'metadata': {
            'n_speeches': len(speeches),
            'n_human_raters': len(human_raters),
            'total_annotations': len(annotations)
        }
    }
    
    # 1. Label imbalance diagnosis
    print("\n1. Diagnosing label imbalance...")
    results['label_imbalance'] = diagnose_label_imbalance(ratings_matrix)
    
    # 2. LLM vs each human (with confusion matrices)
    print("\n2. Analyzing LLM vs each human rater...")
    results['llm_vs_individual_humans'] = {}
    
    for human in human_raters:
        print(f"   Analyzing LLM vs {human}...")
        analysis = analyze_pairwise_agreement(ratings_matrix, 'llm', human, labels)
        if analysis:
            results['llm_vs_individual_humans'][f'llm_vs_{human}'] = analysis
    
    # 3. Inter-human reliability
    print("\n3. Analyzing inter-human reliability...")
    results['inter_human_reliability'] = {}
    
    for i, h1 in enumerate(human_raters):
        for h2 in human_raters[i+1:]:
            print(f"   Analyzing {h1} vs {h2}...")
            analysis = analyze_pairwise_agreement(ratings_matrix, h1, h2, labels)
            if analysis:
                results['inter_human_reliability'][f'{h1}_vs_{h2}'] = analysis
    
    # 4. LLM vs aggregated human ratings
    print("\n4. Analyzing LLM vs aggregated human ratings...")
    aggregation_methods = ['mean', 'median', 'mode']
    results['llm_vs_aggregated_humans'] = analyze_llm_vs_aggregated(
        ratings_matrix, aggregation_methods, labels
    )
    
    # 5. High disagreement speeches
    print("\n5. Identifying speeches with highest disagreement...")
    results['high_disagreement_speeches'] = identify_high_disagreement_speeches(
        ratings_matrix, speeches, top_n=10
    )
    
    # 6. Summary statistics
    print("\n6. Computing summary statistics...")
    
    # Average metrics for LLM vs humans
    llm_human_kappas_unweighted = [
        v['kappa_unweighted'] 
        for v in results['llm_vs_individual_humans'].values()
    ]
    llm_human_kappas_weighted = [
        v['kappa_weighted_quadratic'] 
        for v in results['llm_vs_individual_humans'].values()
    ]
    llm_human_raw_agreement = [
        v['raw_agreement_pct'] 
        for v in results['llm_vs_individual_humans'].values()
    ]
    
    # Average metrics for inter-human
    inter_human_kappas_unweighted = [
        v['kappa_unweighted'] 
        for v in results['inter_human_reliability'].values()
    ]
    inter_human_kappas_weighted = [
        v['kappa_weighted_quadratic'] 
        for v in results['inter_human_reliability'].values()
    ]
    inter_human_raw_agreement = [
        v['raw_agreement_pct'] 
        for v in results['inter_human_reliability'].values()
    ]
    
    results['summary'] = {
        'llm_vs_humans': {
            'avg_kappa_unweighted': float(np.mean(llm_human_kappas_unweighted)),
            'avg_kappa_weighted': float(np.mean(llm_human_kappas_weighted)),
            'avg_raw_agreement_pct': float(np.mean(llm_human_raw_agreement)),
            'std_kappa_unweighted': float(np.std(llm_human_kappas_unweighted)),
        },
        'inter_human': {
            'avg_kappa_unweighted': float(np.mean(inter_human_kappas_unweighted)),
            'avg_kappa_weighted': float(np.mean(inter_human_kappas_weighted)),
            'avg_raw_agreement_pct': float(np.mean(inter_human_raw_agreement)),
            'std_kappa_unweighted': float(np.std(inter_human_kappas_unweighted)),
        }
    }
    
    # 7. Generate confusion matrix plots
    print("\n7. Generating confusion matrix visualizations...")
    all_pairwise = {**results['llm_vs_individual_humans'], **results['inter_human_reliability']}
    generate_confusion_matrix_plots(all_pairwise, plots_dir)
    
    # 8. Interpretation guide
    results['interpretation'] = {
        'kappa_interpretation': {
            '0.00-0.20': 'slight agreement',
            '0.21-0.40': 'fair agreement',
            '0.41-0.60': 'moderate agreement',
            '0.61-0.80': 'substantial agreement',
            '0.81-1.00': 'almost perfect agreement'
        },
        'weighted_vs_unweighted': (
            'Weighted kappa (quadratic) penalizes disagreements proportionally to their distance. '
            'If weighted κ >> unweighted κ, most disagreements are small (e.g., 3 vs 4). '
            'If weighted κ ≈ unweighted κ, disagreements are evenly distributed.'
        ),
        'raw_agreement_vs_kappa': (
            'Raw agreement can be high even with low kappa if labels are imbalanced. '
            'Kappa corrects for chance agreement. Large gap suggests label imbalance.'
        )
    }
    
    # Save results
    print(f"\n8. Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE RELIABILITY ANALYSIS - SUMMARY")
    print("="*70)
    
    print("\n📊 LABEL DISTRIBUTION:")
    for label, count in sorted(results['label_imbalance']['label_distribution'].items()):
        print(f"   Score {label}: {count} annotations")
    print(f"   Imbalance ratio: {results['label_imbalance']['imbalance_ratio']:.2f}:1")
    
    print("\n🤖 LLM vs HUMAN RATERS:")
    print(f"   Average κ (unweighted): {results['summary']['llm_vs_humans']['avg_kappa_unweighted']:.3f}")
    print(f"   Average κ (weighted):   {results['summary']['llm_vs_humans']['avg_kappa_weighted']:.3f}")
    print(f"   Average raw agreement:  {results['summary']['llm_vs_humans']['avg_raw_agreement_pct']:.1f}%")
    
    print("\n👥 INTER-HUMAN RELIABILITY:")
    print(f"   Average κ (unweighted): {results['summary']['inter_human']['avg_kappa_unweighted']:.3f}")
    print(f"   Average κ (weighted):   {results['summary']['inter_human']['avg_kappa_weighted']:.3f}")
    print(f"   Average raw agreement:  {results['summary']['inter_human']['avg_raw_agreement_pct']:.1f}%")
    
    print("\n📈 LLM vs AGGREGATED HUMANS:")
    for method, agg_results in results['llm_vs_aggregated_humans'].items():
        if agg_results:
            print(f"   {method.upper()}:")
            print(f"      κ (unweighted): {agg_results['kappa_unweighted']:.3f}")
            print(f"      κ (weighted):   {agg_results['kappa_weighted_quadratic']:.3f}")
            print(f"      Raw agreement:  {agg_results['raw_agreement_pct']:.1f}%")
    
    print("\n⚠️  TOP 5 HIGHEST DISAGREEMENT SPEECHES:")
    for i, speech in enumerate(results['high_disagreement_speeches'][:5], 1):
        print(f"\n   {i}. {speech['speech_id']}")
        print(f"      Std Dev: {speech['std_dev']:.2f} | Range: {speech['range']}")
        print(f"      Scores: {speech['individual_scores']}")
        print(f"      Preview: {speech['text_preview'][:100]}...")
    
    print("\n" + "="*70)
    print(f"✅ Complete results saved to: {output_path}")
    print(f"📊 Confusion matrices saved to: {plots_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()