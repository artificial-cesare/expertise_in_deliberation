"""
K-Means clustering with PCA dimensionality reduction and configuration search.

KEY FEATURES:
1. Global PCA: Fits on ALL expert speeches, transforms everything.
2. Best k selection: Global expert classification accuracy.
3. CSV updates: Distances calculated to closest CORRECT topic cluster.
4. Visualization: UMAP plots (Experts only & Experts + Assemblies).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
from scipy.special import softmax
from scipy.stats import entropy
import itertools
import umap  # Requires: pip install umap-learn

sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))
try:
    from filter_function import is_substantive_speech_act
except ImportError:
    # Fallback if file not found in context
    def is_substantive_speech_act(n, t): return n >= t

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    # Model settings
    use_finetuned: bool = False
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    finetuned_model: str = "/home/u933585/thesis/qwen3_expert_contrastive_final/best_model/"
    device: str = "cuda:0"
    batch_size: int = 16
    
    # Dimensionality reduction
    pca_n_components: int = 100
    
    # Clustering
    k_values: List[int] = field(default_factory=lambda: [2, 3, 4])
    kmeans_random_state: int = 42
    
    # Data paths
    base_data_dir: Path = Path("/home/u933585/thesis/data/assemblies/processed_files/")
    content_word_threshold: int = 5
    
    # Visualization
    max_assembly_plot_samples: int = 300  # Samples per topic for plotting
    random_seed: int = 42
    
    # Topics and assemblies
    topics: List[str] = field(default_factory=lambda: ["constitution", "future", "gender"])
    assembly_mappings: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "8th_amendment": ("constitution", "Constitutional Reform"),
        "scotland": ("future", "Public Policy"),
        "gendereq": ("gender", "Gender Equality")
    })
    # Colors for plotting
    topic_colors: Dict[str, str] = field(default_factory=lambda: {
        "constitution": '#ff7f0e', # Orange
        "future": '#2ca02c',      # Green
        "gender": "#d62728"       # Red
    })
    
    def __post_init__(self):
        self.expert_dir = self.base_data_dir / "experts"
        self.assembly_dir = self.base_data_dir
        
        self.model_path = self.finetuned_model if self.use_finetuned else self.base_model
        self.model_type = "finetuned" if self.use_finetuned else "base"
        self.output_dir = Path(f"{self.model_type}_kmeans_global_pca")
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.random_seed)

# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class TopicResult:
    """Results for one topic."""
    topic: str
    best_k: int
    kmeans_model: KMeans
    centroids_100d: np.ndarray
    expert_labels: np.ndarray
    expert_silhouette: float

@dataclass
class GlobalMetrics:
    """Global metrics across all topics."""
    expert_classification_accuracy: float
    per_topic_accuracy: Dict[str, float]
    avg_per_topic_silhouette: float

@dataclass
class CrossTopicResults:
    """Cross-topic assignment results for assemblies."""
    topic: str
    total_assemblies: int
    correct_topic_assignments: int
    accuracy: float

@dataclass
class ConfigurationResult:
    """Complete results for one configuration."""
    model_type: str
    k_combination: Dict[str, int]
    topic_results: Dict[str, TopicResult]
    global_metrics: GlobalMetrics
    cross_topic_results: Dict[str, CrossTopicResults]
    pca: PCA
    config: Config

# ============================================================================
# Helper Functions
# ============================================================================
def setup_device(device_str: str) -> str:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_str.split(":")[-1]))
        return device_str
    return "cpu"

def load_and_filter_csv(csv_path: Path, content_threshold: int) -> Tuple[List[str], List[int], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    text_col = next((col for col in ["text", "comment"] if col in df.columns), None)
    if not text_col or "num_non_function_words" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_path.name}")
    
    texts, valid_indices = [], []
    for idx, row in df.iterrows():
        if pd.notna(row.get(text_col)):
            text = str(row[text_col]).strip()
            if text and is_substantive_speech_act(int(row.get("num_non_function_words", 0)), content_threshold):
                texts.append(text)
                valid_indices.append(idx)
    return texts, valid_indices, df

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int, device: str) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, 
                       convert_to_numpy=True, show_progress_bar=True, device=device)

# ============================================================================
# Core Pipeline Functions
# ============================================================================
def fit_global_pca(expert_data: Dict[str, Dict], config: Config) -> PCA:
    """Fit one global PCA model on ALL expert speeches."""
    print(f"\n{'='*80}\nFITTING GLOBAL PCA\n{'='*80}")
    
    # Stack all expert embeddings from all topics
    all_expert_emb = np.vstack([data['embeddings'] for data in expert_data.values()])
    print(f"Fitting PCA on {len(all_expert_emb):,} expert embeddings (Global)")
    
    pca = PCA(n_components=config.pca_n_components, random_state=config.random_seed)
    pca.fit(all_expert_emb)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"✓ PCA fitted: {config.pca_n_components} components explain {explained_var:.1%} of variance")
    return pca

def find_best_k_combination_globally(
    expert_data: Dict[str, Dict],
    pca: PCA,
    k_values: List[int],
    topics: List[str],
    random_state: int
) -> Tuple[Dict[str, int], float, float, Dict[str, TopicResult]]:
    """Find best k combination based on global expert classification accuracy."""
    print(f"\n{'='*80}\nSEARCHING FOR BEST K COMBINATION\n{'='*80}")
    
    k_combinations = list(itertools.product(k_values, repeat=len(topics)))
    best_accuracy = -1
    best_avg_silhouette = -1
    best_k_combination = None
    best_topic_results = None
    
    # Pre-transform all expert data once
    expert_emb_100d = {t: pca.transform(d['embeddings']) for t, d in expert_data.items()}
    
    for k_combo in k_combinations:
        k_dict = {topic: k for topic, k in zip(topics, k_combo)}
        topic_results = {}
        
        # 1. Fit K-Means for each topic
        for topic, k in k_dict.items():
            if topic not in expert_emb_100d: continue
            
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(expert_emb_100d[topic])
            
            try:
                sil = silhouette_score(expert_emb_100d[topic], labels) if len(np.unique(labels)) > 1 else -1
            except: sil = -1
            
            topic_results[topic] = TopicResult(
                topic=topic, best_k=k, kmeans_model=kmeans,
                centroids_100d=kmeans.cluster_centers_.copy(),
                expert_labels=labels, expert_silhouette=sil
            )
            
        # 2. Compute Global Accuracy
        all_centroids = []
        centroid_topics = []
        for topic, result in topic_results.items():
            for _ in range(result.best_k):
                all_centroids.extend(result.centroids_100d)
                centroid_topics.extend([topic] * result.best_k)
        
        all_centroids_arr = np.vstack(all_centroids)
        total_correct, total_experts = 0, 0
        silhouettes = []
        
        for topic in topics:
            if topic not in expert_emb_100d: continue
            dists = euclidean_distances(expert_emb_100d[topic], all_centroids_arr)
            closest = dists.argmin(axis=1)
            assigned = [centroid_topics[i] for i in closest]
            
            total_correct += sum(1 for t in assigned if t == topic)
            total_experts += len(assigned)
            silhouettes.append(topic_results[topic].expert_silhouette)
            
        acc = total_correct / total_experts if total_experts > 0 else 0
        avg_sil = float(np.mean(silhouettes)) if silhouettes else 0
        
        if acc > best_accuracy or (acc == best_accuracy and avg_sil > best_avg_silhouette):
            best_accuracy = acc
            best_avg_silhouette = avg_sil
            best_k_combination = k_dict
            best_topic_results = topic_results
            
    print(f"✅ BEST: k={best_k_combination} | Acc: {best_accuracy:.1%} | Sil: {best_avg_silhouette:.4f}")
    return best_k_combination, best_accuracy, best_avg_silhouette, best_topic_results

def generate_plots(
    expert_data: Dict[str, Dict],
    assembly_data: Dict[str, Dict],
    topic_results: Dict[str, TopicResult],
    pca: PCA,
    config: Config
):
    """Generate UMAP plots for Experts only and Experts + Assemblies."""
    print(f"\n{'='*80}\nGENERATING UMAP PLOTS\n{'='*80}")
    
    # 1. Prepare Data for UMAP Fitting
    # We fit UMAP on Experts + Subset of Assemblies to define the space
    
    umap_training_data = []
    
    # Add all experts
    for topic, data in expert_data.items():
        umap_training_data.append(pca.transform(data['embeddings']))
        
    # Add subset of assemblies (at least 100, up to max_samples)
    assembly_plot_data = {} # Store for plotting later
    for folder, data in assembly_data.items():
        emb_100d = pca.transform(data['embeddings'])
        # Sample if too large
        if len(emb_100d) > config.max_assembly_plot_samples:
            indices = np.random.choice(len(emb_100d), config.max_assembly_plot_samples, replace=False)
            emb_100d = emb_100d[indices]
        
        umap_training_data.append(emb_100d)
        assembly_plot_data[folder] = {'emb_100d': emb_100d, 'topic': data['topic']}

    # Add Centroids
    all_centroids = []
    centroid_meta = [] # (topic, cluster_idx)
    for topic, res in topic_results.items():
        all_centroids.append(res.centroids_100d)
        for i in range(res.best_k):
            centroid_meta.append((topic, i))
    
    umap_training_data.append(np.vstack(all_centroids))
    
    # Stack everything for fit
    X_fit = np.vstack(umap_training_data)
    print(f"Fitting UMAP on {len(X_fit):,} points (Experts + Sampled Assemblies + Centroids)...")
    
    reducer = umap.UMAP(n_components=2, random_state=config.random_seed)
    X_2d = reducer.fit_transform(X_fit)
    
    # 2. Split back data
    cursor = 0
    expert_2d = {}
    for topic, data in expert_data.items():
        n = len(data['embeddings'])
        expert_2d[topic] = X_2d[cursor : cursor + n]
        cursor += n
        
    assembly_2d = {}
    for folder, data in assembly_plot_data.items():
        n = len(data['emb_100d'])
        assembly_2d[folder] = X_2d[cursor : cursor + n]
        cursor += n
        
    centroids_2d = X_2d[cursor:]
    
    # 3. Plotting Helper
    def plot_base(title, filename, include_assemblies=False):
        plt.figure(figsize=(12, 10))
        
        # Plot Experts
        alpha = 0.15 if include_assemblies else 0.6
        for topic, coords in expert_2d.items():
            color = config.topic_colors.get(topic, 'gray')
            plt.scatter(coords[:, 0], coords[:, 1], c=color, label=f"Expert: {topic.title()}", 
                       alpha=alpha, s=20, edgecolors='none')
            
        # Plot Assemblies (Optional)
        if include_assemblies:
            for folder, coords in assembly_2d.items():
                topic = assembly_plot_data[folder]['topic']
                color = config.topic_colors.get(topic, 'gray')
                plt.scatter(coords[:, 0], coords[:, 1], c=color, marker='^', 
                           label=f"Assembly: {topic.title()}", alpha=0.8, s=30, edgecolors='white', linewidth=0.5)

        # Plot Centroids
        for i, (topic, c_idx) in enumerate(centroid_meta):
            color = config.topic_colors.get(topic, 'black')
            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], c=color, marker='X', 
                       s=200, edgecolors='black', linewidth=2, zorder=10)
            plt.text(centroids_2d[i, 0]+0.1, centroids_2d[i, 1]+0.1, str(c_idx), 
                    fontsize=12, fontweight='bold')

        # Legend (Deduplicate)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(config.plots_dir / filename, dpi=300)
        plt.close()
        print(f"  Saved plot: {filename}")

    # 4. Generate Plots
    plot_base(f"Experts Only ({config.model_type})", "plot_experts_only.png", include_assemblies=False)
    plot_base(f"Experts + Assemblies ({config.model_type})", "plot_experts_assemblies.png", include_assemblies=True)

def save_assembly_assignments_forced_constraint(
    assembly_folder: str,
    raw_data: Dict,
    topic_results: Dict[str, TopicResult],
    pca: PCA,
    config: Config
) -> None:
    """
    Save cluster assignments forcing the CORRECT topic constraint.
    Calculates distance to the closest centroid belonging to the assembly's ground-truth topic.
    """
    correct_topic = raw_data['topic']
    
    # If we don't have a model for this topic, skip
    if correct_topic not in topic_results:
        return

    # Get centroids ONLY for the correct topic
    target_result = topic_results[correct_topic]
    target_centroids = target_result.centroids_100d
    
    # Transform embeddings
    embeddings_100d = pca.transform(raw_data['embeddings'])
    
    # Calculate distances to target centroids only
    distances = euclidean_distances(embeddings_100d, target_centroids)
    
    # Find closest cluster (index relative to this topic's k)
    closest_cluster_indices = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    # Calculate similarity (cosine) to the closest centroid
    # Note: cosine_similarity returns (n_samples, n_centroids)
    sims = cosine_similarity(embeddings_100d, target_centroids)
    max_sims = np.array([sims[i, closest_cluster_indices[i]] for i in range(len(embeddings_100d))])
    
    offset = 0
    for csv_file, texts, valid_indices, df in raw_data['files']:
        n_texts = len(texts)
        
        # Slice batch
        batch_labels = closest_cluster_indices[offset:offset+n_texts]
        batch_dists = min_distances[offset:offset+n_texts]
        batch_sims = max_sims[offset:offset+n_texts]
        offset += n_texts
        
        # Column names
        k_val = target_result.best_k
        suffix = f"{config.model_type}_k{k_val}"
        cluster_col = f"cluster_{correct_topic}_{suffix}"
        dist_col = f"distance_{correct_topic}_{suffix}"
        sim_col = f"similarity_{correct_topic}_{suffix}"
        
        # Initialize columns
        for col in [cluster_col, dist_col, sim_col]:
            if col not in df.columns: df[col] = np.nan
            
        # Update rows
        df.loc[valid_indices, cluster_col] = batch_labels
        df.loc[valid_indices, dist_col] = batch_dists
        df.loc[valid_indices, sim_col] = batch_sims
        
        df.to_csv(csv_file, index=False)

def analyze_cross_topic_assignments(
    assembly_data_raw: Dict[str, Dict],
    topic_results: Dict[str, TopicResult],
    pca: PCA
) -> Dict[str, CrossTopicResults]:
    """Analyze natural classification (without forcing constraint) for reporting."""
    # Build global centroid list
    all_centroids = []
    centroid_topics = []
    for topic, result in topic_results.items():
        for _ in range(result.best_k):
            all_centroids.extend(result.centroids_100d)
            centroid_topics.extend([topic] * result.best_k)
    all_centroids_arr = np.vstack(all_centroids)
    
    results = {}
    for folder, data in assembly_data_raw.items():
        emb_100d = pca.transform(data['embeddings'])
        dists = euclidean_distances(emb_100d, all_centroids_arr)
        closest = dists.argmin(axis=1)
        assigned = [centroid_topics[i] for i in closest]
        
        correct = sum(1 for t in assigned if t == data['topic'])
        acc = correct / len(assigned) if assigned else 0
        
        results[folder] = CrossTopicResults(
            topic=data['topic'], total_assemblies=len(assigned),
            correct_topic_assignments=correct, accuracy=acc
        )
    return results

def run_single_configuration(
    model: SentenceTransformer,
    expert_data: Dict[str, Dict],
    assembly_data_raw: Dict[str, Dict],
    config: Config
) -> ConfigurationResult:
    """Run complete pipeline for a single configuration."""
    print(f"\n{'='*80}\nRUNNING CONFIGURATION: {config.model_type.upper()}\n{'='*80}")
    
    # 1. Fit Global PCA on Experts
    pca = fit_global_pca(expert_data, config)
    
    # 2. Find best k combination
    k_combo, acc, sil, topic_results = find_best_k_combination_globally(
        expert_data, pca, config.k_values, config.topics, config.kmeans_random_state
    )
    
    # 3. Compute Global Metrics
    # Re-compute global accuracy for the best combo
    all_centroids = []
    centroid_topics = []
    for topic, res in topic_results.items():
        all_centroids.extend(res.centroids_100d)
        centroid_topics.extend([topic] * res.best_k)
    
    all_centroids_arr = np.vstack(all_centroids)
    total_correct, total_experts = 0, 0
    per_topic_acc = {}
    
    for topic, data in expert_data.items():
        emb = pca.transform(data['embeddings'])
        dists = euclidean_distances(emb, all_centroids_arr)
        closest = dists.argmin(axis=1)
        assigned = [centroid_topics[i] for i in closest]
        correct = sum(1 for t in assigned if t == topic)
        per_topic_acc[topic] = correct / len(assigned)
        total_correct += correct
        total_experts += len(assigned)
        
    global_metrics = GlobalMetrics(
        expert_classification_accuracy=total_correct/total_experts,
        per_topic_accuracy=per_topic_acc,
        avg_per_topic_silhouette=sil
    )
    
    # 4. Cross-topic analysis (Natural classification)
    cross_results = analyze_cross_topic_assignments(assembly_data_raw, topic_results, pca)
    
    # 5. Generate Plots
    generate_plots(expert_data, assembly_data_raw, topic_results, pca, config)
    
    return ConfigurationResult(
        model_type=config.model_type,
        k_combination=k_combo,
        topic_results=topic_results,
        global_metrics=global_metrics,
        cross_topic_results=cross_results,
        pca=pca,
        config=config
    )

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    base_config = Config()
    device = setup_device(base_config.device)
    
    # Load Data (Shared)
    print(f"\n{'='*80}\nLOADING DATA\n{'='*80}")
    
    # Models
    base_model = SentenceTransformer(base_config.base_model, device=device, trust_remote_code=True)
    finetuned_model = SentenceTransformer(base_config.finetuned_model, device=device, trust_remote_code=True)
    
    # Load Text
    expert_texts = {}
    for topic in base_config.topics:
        files = sorted((base_config.expert_dir / topic).glob("*.csv"))
        texts = []
        for f in files:
            t, _, _ = load_and_filter_csv(f, base_config.content_word_threshold)
            texts.extend(t)
        if len(texts) >= max(base_config.k_values): expert_texts[topic] = texts
            
    assembly_texts_raw = {}
    for folder, (topic, name) in base_config.assembly_mappings.items():
        files = sorted((base_config.assembly_dir / folder).glob("*.csv"))
        texts, file_data = [], []
        for f in files:
            t, idx, df = load_and_filter_csv(f, base_config.content_word_threshold)
            if t:
                texts.extend(t)
                file_data.append((f, t, idx, df))
        if texts:
            assembly_texts_raw[folder] = {'topic': topic, 'display_name': name, 'texts': texts, 'files': file_data}

    # Embed
    print("\nEmbedding Base...")
    exp_base = {t: {'embeddings': embed_texts(base_model, txts, 16, device)} for t, txts in expert_texts.items()}
    ass_base = {f: {**d, 'embeddings': embed_texts(base_model, d['texts'], 16, device)} for f, d in assembly_texts_raw.items()}
    
    print("\nEmbedding Finetuned...")
    exp_fine = {t: {'embeddings': embed_texts(finetuned_model, txts, 16, device)} for t, txts in expert_texts.items()}
    ass_fine = {f: {**d, 'embeddings': embed_texts(finetuned_model, d['texts'], 16, device)} for f, d in assembly_texts_raw.items()}

    # Run Configs
    all_results = {}
    
    # Config 1: Base
    cfg_base = Config(use_finetuned=False)
    cfg_base.__post_init__()
    all_results['base'] = run_single_configuration(base_model, exp_base, ass_base, cfg_base)
    
    # Config 2: Finetuned
    cfg_fine = Config(use_finetuned=True)
    cfg_fine.__post_init__()
    all_results['finetuned'] = run_single_configuration(finetuned_model, exp_fine, ass_fine, cfg_fine)
    
    # Update CSVs (Forced Constraint)
    print(f"\n{'='*80}\nUPDATING CSVs (FORCED TOPIC CONSTRAINT)\n{'='*80}")
    
    # Update Base
    print("Updating Base CSVs...")
    res = all_results['base']
    for folder, data in ass_base.items():
        save_assembly_assignments_forced_constraint(folder, data, res.topic_results, res.pca, res.config)
        
    # Update Finetuned
    print("Updating Finetuned CSVs...")
    res = all_results['finetuned']
    for folder, data in ass_fine.items():
        save_assembly_assignments_forced_constraint(folder, data, res.topic_results, res.pca, res.config)

    print("\n✅ Done! Check output folders for plots and CSVs.")

if __name__ == "__main__":
    main()