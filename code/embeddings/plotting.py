"""
Visualization Generator for Saved K-Means/PCA/UMAP Models.

CHANGES IMPLEMENTED:
1. Loads pre-trained PCA, UMAP, and Centroids from disk.
2. Text sizes increased (min 18).
3. Centroids: Black contours, numbers removed.
4. Combined Plot: Assemblies larger/opaque, Experts transparent.
5. Uniform Sampling: 100 Experts, 150 Assemblies per topic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import umap  # pip install umap-learn
import joblib 
import os

# Defaults are the GPU server layout; override with env vars to run elsewhere (see reproduce/umap_figures_v2.sh)
THESIS_ROOT = Path(os.getenv("THESIS_ROOT", "/home/u933585/thesis"))
DEVICE = os.getenv("DEVICE", "cuda:0")
PLOT_LIMITS = [x == "True" for x in os.getenv("PLOT_LIMITS", "True,False").split(",")]
import warnings

# Add path for custom modules
sys.path.append(str(Path(__file__).parent / ".." / ".." / "handcrafted_features"))

try:
    from filter_function import is_substantive_speech_act
except ImportError:
    warnings.warn("Could not import 'is_substantive_speech_act'. Using fallback logic.", UserWarning)
    def is_substantive_speech_act(n, t): return n >= t

# ============================================================================
# Configuration
# ============================================================================
class Config:
    def __init__(self, model_type="base", limit_samples=True):
        self.model_type = model_type  # 'base' or 'finetuned'
        self.limit_samples = limit_samples
        
        # Paths
        self.base_data_dir = THESIS_ROOT / "data/assemblies/processed_files"
        self.output_root = THESIS_ROOT / f"{self.model_type}_kmeans_global_pca"
        self.models_dir = self.output_root / "models"
        self.plots_dir = self.output_root / f"plots_visual_update_limit{self.limit_samples}"
        
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        if self.model_type == "finetuned":
            self.emb_model_path = str(THESIS_ROOT / "qwen3_expert_contrastive_final/best_model")
        else:
            self.emb_model_path = "Qwen/Qwen3-Embedding-0.6B"
            
        self.device = DEVICE
        self.batch_size = 16
        self.random_seed = 42
        self.content_word_threshold = 5
        
        # Sampling Limits for Visualization
        if self.limit_samples:
            self.samples_expert = 200
            self.samples_assembly = 250
        else:
            self.samples_expert = None
            self.samples_assembly = None
        
        # Topics
        self.topics = ["constitution", "future", "gender"]
        self.assembly_mappings = {
            "8th_amendment": ("constitution", "Constitutional Reform"),
            "scotland": ("future", "Public Policy"),
            "gendereq": ("gender", "Gender Equality")
        }
        
        # Colors
        self.topic_colors = {
            "constitution": "#2E86AB", # Deep Teal Blue
            "gender": "#A23B72",       # Rich Magenta
            "future": "#F18F01"        # Warm Orange
        }

        np.random.seed(self.random_seed)

# ============================================================================
# Helpers
# ============================================================================
def setup_device(device_str: str) -> str:
    if torch.cuda.is_available():
        return device_str
    return "cpu"

def load_and_filter_csv(csv_path: Path, content_threshold: int):
    df = pd.read_csv(csv_path)
    text_col = next((col for col in ["text", "comment"] if col in df.columns), None)
    texts = []
    if text_col and "num_non_function_words" in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row.get(text_col)):
                text = str(row[text_col]).strip()
                if text and is_substantive_speech_act(int(row.get("num_non_function_words", 0)), content_threshold):
                    texts.append(text)
    return texts

def embed_texts(model: SentenceTransformer, texts: list, batch_size: int, device: str) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, 
                       convert_to_numpy=True, show_progress_bar=True, device=device)

# ============================================================================
# Visualization Logic
# ============================================================================
def generate_plots(
    expert_data: dict,
    assembly_data: dict,
    config: Config
):
    print(f"\n--- Generating Plots for {config.model_type.upper()} ---")
    
    # 1. Load Artifacts
    print("Loading saved models (PCA, UMAP, Centroids)...")
    try:
        pca = joblib.load(config.models_dir / f"pca_model_{config.model_type}.pkl")
        reducer = joblib.load(config.models_dir / f"umap_model_{config.model_type}.pkl")
        centroids_100d = np.load(config.models_dir / f"centroids_100d_{config.model_type}.npy")
        centroid_labels = joblib.load(config.models_dir / f"centroid_labels_{config.model_type}.pkl")
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {config.models_dir}")
        print(e)
        return

    # 2. Transform Data (Embeddings -> PCA -> UMAP)
    # We must transform separately to maintain labels, then concat for dataframe
    
    plot_data = [] # List of dicts for DataFrame
    
    # Process Experts
    for topic, data in expert_data.items():
        if len(data['embeddings']) == 0: continue
        emb_100d = pca.transform(data['embeddings'])
        emb_2d = reducer.transform(emb_100d)
        
        for i in range(len(emb_2d)):
            plot_data.append({
                'x': emb_2d[i, 0],
                'y': emb_2d[i, 1],
                'type': 'expert',
                'topic': topic
            })
            
    # Process Assemblies
    for folder, data in assembly_data.items():
        if len(data['embeddings']) == 0: continue
        topic = data['topic']
        emb_100d = pca.transform(data['embeddings'])
        emb_2d = reducer.transform(emb_100d)
        
        for i in range(len(emb_2d)):
            plot_data.append({
                'x': emb_2d[i, 0],
                'y': emb_2d[i, 1],
                'type': 'assembly',
                'topic': topic
            })

    # Process Centroids
    centroids_2d = reducer.transform(centroids_100d)
    
    df_all = pd.DataFrame(plot_data)

    # 3. Plotting Function with New Style Requirements
    def create_plot(filename, include_assemblies=False):
        plt.figure(figsize=(18, 14)) # Large figure for readability
        ax = plt.gca()
        
        # --- SAMPLING LOGIC ---
        # If limits are set, sample. Otherwise, use all data.
        df_sampled = pd.DataFrame()
        
        # Sample/Select Experts
        for topic in config.topics:
            subset = df_all[(df_all['type'] == 'expert') & (df_all['topic'] == topic)]
            if not subset.empty:
                if config.samples_expert is not None:
                    n_samples = min(len(subset), config.samples_expert)
                    df_sampled = pd.concat([df_sampled, subset.sample(n=n_samples, random_state=config.random_seed)])
                else:
                    # No limit - use all samples
                    df_sampled = pd.concat([df_sampled, subset])
        
        # Sample/Select Assemblies (if included)
        if include_assemblies:
            for topic in config.topics:
                subset = df_all[(df_all['type'] == 'assembly') & (df_all['topic'] == topic)]
                if not subset.empty:
                    if config.samples_assembly is not None:
                        n_samples = min(len(subset), config.samples_assembly)
                        df_sampled = pd.concat([df_sampled, subset.sample(n=n_samples, random_state=config.random_seed)])
                    else:
                        # No limit - use all samples
                        df_sampled = pd.concat([df_sampled, subset])

        # --- DRAWING ---
        
        # Requirement 3: If combined, experts are transparent. If alone, opaque.
        exp_alpha = 0.3 if include_assemblies else 0.7
        exp_size = 120
        ass_alpha = 0.6 if config.limit_samples else 0.3
        ass_size = 180 if include_assemblies else 100

        
        for topic, group in df_sampled[df_sampled['type'] == 'expert'].groupby('topic'):
            color = config.topic_colors.get(topic, 'gray')
            plt.scatter(
                group.x, group.y, 
                c=color, 
                label=f"Expert: {topic.title()}",
                alpha=exp_alpha,
                s=exp_size, 
                marker='o', 
                edgecolors='none' # Remove edges for cleaner transparency look
            )

        # 2. Draw Assemblies (if requested)
        if include_assemblies:
            # Requirement 3: Assemblies bigger and opaque
            for topic, group in df_sampled[df_sampled['type'] == 'assembly'].groupby('topic'):
                color = config.topic_colors.get(topic, 'gray')
                plt.scatter(
                    group.x, group.y, 
                    c=color, 
                    marker='^',
                    label=f"Assembly: {topic.title()}", 
                    alpha=ass_alpha,      # Very Opaque
                    s=ass_size,          # Bigger
                    edgecolors='none', 
                    linewidth=1.0
                )

        # 3. Draw Centroids
        # Requirement 2: Black contours, remove numbers
        for i, (topic, c_idx) in enumerate(centroid_labels):
            color = config.topic_colors.get(topic, 'black')
            plt.scatter(
                centroids_2d[i, 0], centroids_2d[i, 1], 
                c=color, 
                marker='o',
                s=500,              # Large size
                edgecolors='black', # Requirement 2: Black Contour
                linewidth=3.0,      # Thick line
                zorder=20, 
                label=f"Centroid ({topic.title()})" if f"Centroid ({topic.title()})" not in [l for h, l in zip(*ax.get_legend_handles_labels())] else ""
            )
            # Note: plt.text (numbers) removed as per Requirement 2

        # --- STYLING ---
        # Requirement 1: Make all text bigger (min 18)
        
        # De-duplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Legend position: bottom right for finetuned, upper right for base
        legend_loc = 'lower right' if config.model_type == 'finetuned' else 'upper right'
        
        leg = plt.legend(
            by_label.values(), by_label.keys(), 
            loc=legend_loc, 
            frameon=True, 
            framealpha=0.95, 
            fontsize=20,          # Big font
            title="Data Source", 
            title_fontsize=22     # Big Title
        )
        
        # Increase dot size in legend for visibility
        for handle in leg.legend_handles:
            handle.set_sizes([150]) 
            handle.set_alpha(1) # Legend icons should be fully visible

        title_str = f"Experts + Assemblies ({config.model_type.upper()})" if include_assemblies else f"Experts Only ({config.model_type.upper()})"
        plt.title(title_str, fontsize=28, fontweight='bold', pad=25)
        plt.xlabel("UMAP Dimension 1", fontsize=22)
        plt.ylabel("UMAP Dimension 2", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        save_path = config.plots_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # Generate the two versions
    create_plot("viz_experts_only.png", include_assemblies=False)
    create_plot("viz_experts_assemblies.png", include_assemblies=True)


# ============================================================================
# Main Execution
# ============================================================================
def main():
    device = setup_device(DEVICE)
    
    # We need to load data and embed it to have points to plot
    # We do this once per model type (Base vs Finetuned)
    
    for m_type in ['base', 'finetuned']:
        for limit_samples in PLOT_LIMITS:
            print(f"Preparing data for model type: {m_type}, limit_samples: {limit_samples}")
            config = Config(model_type=m_type, limit_samples=limit_samples)
            print(f"\n{'='*60}\nPROCESSING: {m_type.upper()} (LIMIT={limit_samples})\n{'='*60}")
            
            # 1. Load Data
            print("Loading text data...")
            expert_texts = {}
            expert_dir = config.base_data_dir / "experts"
            for topic in config.topics:
                files = sorted((expert_dir / topic).glob("*.csv"))
                texts = []
                for f in files:
                    texts.extend(load_and_filter_csv(f, config.content_word_threshold))
                expert_texts[topic] = texts

            assembly_texts = {}
            for folder, (topic, name) in config.assembly_mappings.items():
                files = sorted((config.base_data_dir / folder).glob("*.csv"))
                texts = []
                for f in files:
                    texts.extend(load_and_filter_csv(f, config.content_word_threshold))
                if texts:
                    assembly_texts[folder] = {'topic': topic, 'texts': texts}

            # 2. Embed Data (Required to project onto saved UMAP)
            print(f"Loading {m_type} embedding model...")
            model = SentenceTransformer(config.emb_model_path, device=device, trust_remote_code=True)
            
            print("Generating embeddings...")
            exp_data = {t: {'embeddings': embed_texts(model, txts, config.batch_size, device)} 
                    for t, txts in expert_texts.items()}
            
            ass_data = {f: {**d, 'embeddings': embed_texts(model, d['texts'], config.batch_size, device)} 
                    for f, d in assembly_texts.items()}

            # 3. Generate Visualizations using saved PCA/UMAP
            generate_plots(exp_data, ass_data, config)
            
            # Cleanup to free GPU memory for next iteration
            del model
            torch.cuda.empty_cache()

    print("\n✅ Visualization update complete.")

if __name__ == "__main__":
    main()