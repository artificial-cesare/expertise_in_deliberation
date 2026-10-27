"""
Fine-tuned model analysis pipeline:
1. Embed expert speech acts → compute centroids
2. Embed assembly speech acts → compute similarities to centroids
3. Fit UMAP on combined data
4. Create visualizations with outlier cropping
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add filter function
sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))
from filter_function import is_substantive_speech_act

try:
    from umap import UMAP
except ImportError:
    sys.exit("❌ Install UMAP: pip install umap-learn")

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    # Model settings
    use_finetuned: bool = True
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    finetuned_model: str = "/home/u933585/thesis/code/embeddings/SIMCSE/qwen3_expert_contrastive_v2/best_model/"
    device: str = "cuda:1"
    batch_size: int = 16
    
    # Data paths - unified base directory
    base_data_dir: Path = Path("/home/u933585/thesis/data/assemblies/processed_files/")
    
    # Topics and mappings
    topics: List[str] = None
    assembly_mappings: Dict[str, Tuple[str, str]] = None
    
    # Filtering and visualization
    content_word_threshold: int = 5
    crop_percentile: int = 90
    max_plot_samples: int = 200
    random_seed: int = 42
    
    # Colors and markers
    topic_colors: Dict[str, str] = None
    topic_markers: Dict[str, str] = None
    
    def __post_init__(self):
        self.topics = ["constitution", "future", "gender"]
        self.assembly_mappings = {
            "8th_amendment": ("constitution", "Constitutional Reform"),
            "scotland_csv": ("future", "Public Policy"),
            "gendereq": ("gender", "Gender Equality")
        }
        self.topic_colors = {
            "CONSTITUTION": "#ff7f0e",
            "FUTURE": "#2ca02c",
            "GENDER": "#d62728"
        }
        self.topic_markers = {
            "CONSTITUTION": "*",
            "FUTURE": "^",
            "GENDER": "s"
        }
        
        # Derived paths
        self.expert_dir = self.base_data_dir / "experts"
        self.assembly_dir = self.base_data_dir
        
        # Model properties
        self.model_path = self.finetuned_model if self.use_finetuned else self.base_model
        self.model_type = "finetuned" if self.use_finetuned else "base"
        self.output_dir = Path(f"{self.model_type}_analysis_output")
        self.centroid_dir = self.output_dir / "centroids"
        self.umap_path = self.output_dir / f"{self.model_type}_umap.pkl"
        
        # Setup
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.centroid_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.random_seed)

# ============================================================================
# Helper Functions
# ============================================================================
def setup_device(device_str: str) -> str:
    """Configure and return device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_str.split(":")[-1]))
        print(f"✓ Using {device_str}")
        return device_str
    print("⚠️  Using CPU")
    return "cpu"

def load_and_filter_csv(csv_path: Path, content_threshold: int, text_col_names: List[str] = None) -> Tuple[List[str], List[int], pd.DataFrame]:
    """Load CSV and return (texts, valid_indices, dataframe)"""
    if text_col_names is None:
        text_col_names = ["text", "comment"]
    
    df = pd.read_csv(csv_path)
    
    # Find text column
    text_col = next((col for col in text_col_names if col in df.columns), None)
    if not text_col:
        raise ValueError(f"No text column found in {csv_path.name}")
    
    if "num_non_function_words" not in df.columns:
        raise ValueError(f"Missing 'num_non_function_words' in {csv_path.name}")
    
    # Filter substantive speech acts
    texts, valid_indices = [], []
    for idx, row in df.iterrows():
        if pd.notna(row.get(text_col)):
            text = str(row[text_col]).strip()
            num_content = int(row.get("num_non_function_words", 0))
            if text and is_substantive_speech_act(num_content, content_threshold):
                texts.append(text)
                valid_indices.append(idx)
    
    return texts, valid_indices, df

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int, device: str, show_progress: bool = True) -> np.ndarray:
    """Encode texts and return normalized embeddings"""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        device=device
    )

def sample_by_topic(data_2d: np.ndarray, labels: List[str], topics: List[str], max_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Sample data points by topic for visualization"""
    sampled_data, sampled_labels = [], []
    
    for topic in topics:
        topic_upper = topic.upper()
        mask = np.array([topic_upper in l for l in labels])
        indices = np.where(mask)[0]
        
        if len(indices) > max_samples:
            indices = np.random.choice(indices, max_samples, replace=False)
        
        if len(indices) > 0:
            sampled_data.append(data_2d[indices])
            sampled_labels.extend([labels[i] for i in indices])
    
    return (np.vstack(sampled_data), sampled_labels) if sampled_data else (np.array([]).reshape(0, 2), [])

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    config = Config()
    device = setup_device(config.device)
    
    print("=" * 80)
    print(f"{config.model_type.upper()} MODEL ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Output: {config.output_dir}")
    print(f"Expert data: {config.expert_dir}")
    print(f"Assembly data: {config.assembly_dir}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = SentenceTransformer(config.model_path, device=device, trust_remote_code=True)
    model.max_seq_length = 1024
    
    # ========================================================================
    # STEP 1 & 2: Process Expert Data → Compute Centroids
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1-2: EXPERT DATA → CENTROIDS")
    print("=" * 80)
    
    expert_data = {}
    centroids = {}
    
    for topic in config.topics:
        print(f"\n📂 {topic.upper()}")
        topic_dir = config.expert_dir / topic
        
        if not topic_dir.exists():
            print(f"  ⚠️  Directory not found: {topic_dir}")
            continue
        
        all_texts, all_csv_info = [], []
        
        for csv_file in sorted(topic_dir.glob("*.csv")):
            try:
                texts, _, _ = load_and_filter_csv(csv_file, config.content_word_threshold)
                all_texts.extend(texts)
                all_csv_info.extend([csv_file.name] * len(texts))
                print(f"  ✓ {csv_file.name}: {len(texts)} substantive")
            except Exception as e:
                print(f"  ❌ {csv_file.name}: {e}")
        
        if not all_texts:
            print(f"  ⚠️  No substantive texts found for {topic}")
            continue
        
        print(f"  🔄 Embedding {len(all_texts)} texts...")
        embeddings = embed_texts(model, all_texts, config.batch_size, device)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid)
        
        expert_data[topic] = {
            'texts': all_texts,
            'embeddings': embeddings,
            'csv_info': all_csv_info
        }
        centroids[topic] = centroid
        
        # Save centroid
        np.save(config.centroid_dir / f"{config.model_type}_centroid_{topic}.npy", centroid)
        print(f"  ✓ Centroid saved")
    
    # ========================================================================
    # STEP 3 & 4: Process Assembly Data → Compute Similarities
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3-4: ASSEMBLY DATA → SIMILARITIES")
    print("=" * 80)
    
    assembly_data = {}
    similarity_col = f"similarity_to_centroid_{config.model_type}"
    distance_col = f"distance_to_centroid_{config.model_type}"
    
    for assembly_folder, (topic, display_name) in config.assembly_mappings.items():
        print(f"\n📂 {display_name}")
        folder_path = config.assembly_dir / assembly_folder
        
        if not folder_path.exists():
            print(f"  ⚠️  Directory not found: {folder_path}")
            continue
        
        if topic not in centroids:
            print(f"  ⚠️  No centroid for {topic}")
            continue
        
        centroid = centroids[topic].reshape(1, -1)
        csv_data_list = []
        
        for csv_file in sorted(folder_path.glob("*.csv")):
            try:
                texts, valid_indices, df = load_and_filter_csv(csv_file, config.content_word_threshold)
                
                if not texts:
                    print(f"  ⊘ {csv_file.name}: No substantive texts")
                    continue
                
                # Embed
                embeddings = embed_texts(model, texts, config.batch_size, device, show_progress=False)
                
                # Compute metrics
                similarities = cosine_similarity(embeddings, centroid).flatten()
                distances = euclidean_distances(embeddings, centroid).flatten()
                
                # Initialize columns with -1 (non-substantive marker)
                df[similarity_col] = -1.0
                df[distance_col] = -1.0
                
                # Fill substantive rows
                df.loc[valid_indices, similarity_col] = similarities
                df.loc[valid_indices, distance_col] = distances
                
                # Save
                df.to_csv(csv_file, index=False)
                
                csv_data_list.append({
                    'embeddings': embeddings,
                    'valid_indices': valid_indices
                })
                
                print(f"  ✓ {csv_file.name}: {len(texts)} substantive")
                
            except Exception as e:
                print(f"  ❌ {csv_file.name}: {e}")
        
        if csv_data_list:
            assembly_data[assembly_folder] = {
                'topic': topic,
                'display_name': display_name,
                'csv_data': csv_data_list
            }
        else:
            print(f"  ⚠️  No valid data for {display_name}")
    
    # ========================================================================
    # STEP 5: Fit UMAP
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: UMAP FITTING")
    print("=" * 80)
    
    all_embeddings, all_labels = [], []
    
    # Add expert embeddings
    for topic, data in expert_data.items():
        all_embeddings.append(data['embeddings'])
        all_labels.extend([f"EXPERT-{topic.upper()}"] * len(data['embeddings']))
    
    # Add assembly embeddings
    for assembly_folder, data in assembly_data.items():
        topic_upper = data['topic'].upper()
        emb_list = [csv_data['embeddings'] for csv_data in data['csv_data']]
        if emb_list:
            combined = np.vstack(emb_list)
            all_embeddings.append(combined)
            all_labels.extend([f"ASSEMBLY-{topic_upper}"] * len(combined))
    
    # Add centroids
    centroid_embeddings = list(centroids.values())
    centroid_labels = [f"CENTROID-{t.upper()}" for t in centroids.keys()]
    
    X_combined = np.vstack(all_embeddings + centroid_embeddings)
    labels_combined = all_labels + centroid_labels
    
    print(f"Total embeddings: {len(X_combined):,}")
    print(f"  - Expert: {sum(1 for l in all_labels if l.startswith('EXPERT')):,}")
    print(f"  - Assembly: {sum(1 for l in all_labels if l.startswith('ASSEMBLY')):,}")
    print(f"  - Centroids: {len(centroid_labels)}")
    
    # Fit UMAP
    print("🔄 Fitting UMAP...")
    umap_reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=config.random_seed,
        n_jobs=1,
        verbose=True
    )
    X_2d = umap_reducer.fit_transform(X_combined)
    
    # Save UMAP
    with open(config.umap_path, 'wb') as f:
        pickle.dump(umap_reducer, f)
    print(f"✓ UMAP saved: {config.umap_path}")
    
    # Split components
    n_expert = sum(1 for l in all_labels if l.startswith('EXPERT'))
    n_assembly = sum(1 for l in all_labels if l.startswith('ASSEMBLY'))
    n_total = n_expert + n_assembly
    
    expert_2d = X_2d[:n_expert]
    assembly_2d = X_2d[n_expert:n_total]
    centroids_2d = X_2d[n_total:]
    
    expert_labels_split = [l for l in all_labels if l.startswith('EXPERT')]
    assembly_labels_split = [l for l in all_labels if l.startswith('ASSEMBLY')]
    
    # ========================================================================
    # STEP 6: Crop Outliers
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 6: CROPPING OUTLIERS ({config.crop_percentile}th percentile)")
    print("=" * 80)
    
    center = np.mean(X_2d[:n_total], axis=0)
    distances = np.sqrt(np.sum((X_2d[:n_total] - center) ** 2, axis=1))
    threshold = np.percentile(distances, config.crop_percentile)
    
    expert_mask = distances[:n_expert] <= threshold
    assembly_mask = distances[n_expert:n_total] <= threshold
    
    expert_2d_cropped = expert_2d[expert_mask]
    expert_labels_cropped = [l for i, l in enumerate(expert_labels_split) if expert_mask[i]]
    
    assembly_2d_cropped = assembly_2d[assembly_mask]
    assembly_labels_cropped = [l for i, l in enumerate(assembly_labels_split) if assembly_mask[i]]
    
    print(f"Expert kept: {expert_mask.sum():,}/{len(expert_mask):,}")
    print(f"Assembly kept: {assembly_mask.sum():,}/{len(assembly_mask):,}")
    
    # Sample for plotting
    expert_2d_plot, expert_labels_plot = sample_by_topic(
        expert_2d_cropped, expert_labels_cropped, 
        [t.upper() for t in config.topics], config.max_plot_samples
    )
    assembly_2d_plot, assembly_labels_plot = sample_by_topic(
        assembly_2d_cropped, assembly_labels_cropped,
        [t.upper() for t in config.topics], config.max_plot_samples
    )
    
    # ========================================================================
    # STEP 7: Visualizations
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 7: VISUALIZATIONS")
    print("=" * 80)
    
    sns.set_style("whitegrid")
    
    def plot_scatter(ax, data_2d, labels, colors, alpha, size, edge, label_prefix):
        """Helper to plot scatter points"""
        for topic in [t.upper() for t in config.topics]:
            label_key = f"{label_prefix}-{topic}"
            mask = np.array([label_key in l for l in labels])
            if mask.any():
                ax.scatter(
                    data_2d[mask, 0], data_2d[mask, 1],
                    c=colors[topic],
                    label=f"{label_prefix.title()} {topic.title()}",
                    alpha=alpha, s=size,
                    edgecolors=edge if edge else 'none',
                    linewidths=0.5 if edge else 0
                )
    
    def plot_centroids(ax, centroids_2d, centroid_labels, colors, markers):
        """Helper to plot centroids"""
        for i, label in enumerate(centroid_labels):
            topic = label.split('-')[1]
            ax.scatter(
                centroids_2d[i, 0], centroids_2d[i, 1],
                c=colors[topic],
                marker=markers[topic],
                s=500, edgecolors='black', linewidths=2,
                label=f"{topic.title()} Centroid",
                zorder=10
            )
    
    # Visualization 1: Expert only
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_scatter(ax, expert_2d_plot, expert_labels_plot, config.topic_colors, 0.3, 30, None, "EXPERT")
    plot_centroids(ax, centroids_2d, centroid_labels, config.topic_colors, config.topic_markers)
    ax.set_xlabel("UMAP Dimension 1", fontsize=18)
    ax.set_ylabel("UMAP Dimension 2", fontsize=18)
    ax.set_title(
        f"{'Fine-tuned' if config.use_finetuned else 'Base'} Model: Expert Embeddings vs Centroids\n"
        f"Filtered (≥{config.content_word_threshold} content words) | Cropped at {config.crop_percentile}th percentile",
        fontsize=16, fontweight='bold', pad=20
    )
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    expert_viz = config.output_dir / f"{config.model_type}_expert_visualization.png"
    plt.savefig(expert_viz, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {expert_viz.name}")
    plt.close()
    
    # Visualization 2: Combined
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_scatter(ax, expert_2d_plot, expert_labels_plot, config.topic_colors, 0.15, 20, None, "EXPERT")
    plot_scatter(ax, assembly_2d_plot, assembly_labels_plot, config.topic_colors, 0.5, 50, 'black', "ASSEMBLY")
    plot_centroids(ax, centroids_2d, centroid_labels, config.topic_colors, config.topic_markers)
    ax.set_xlabel("UMAP Dimension 1", fontsize=18)
    ax.set_ylabel("UMAP Dimension 2", fontsize=18)
    ax.set_title(
        f"{'Fine-tuned' if config.use_finetuned else 'Base'} Model: Assembly vs Expert Embeddings\n"
        f"Filtered (≥{config.content_word_threshold} content words) | Cropped at {config.crop_percentile}th percentile",
        fontsize=16, fontweight='bold', pad=20
    )
    ax.legend(loc='best', fontsize=12, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_viz = config.output_dir / f"{config.model_type}_combined_visualization.png"
    plt.savefig(combined_viz, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {combined_viz.name}")
    plt.close()
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()