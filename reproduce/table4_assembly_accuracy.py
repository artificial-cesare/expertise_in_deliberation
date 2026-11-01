"""Table 4 'Assembly Cross-Topic Accuracy' for the Constitutional corpus, with the saved PCA and centroids."""
import pickle, joblib, numpy as np, pandas as pd, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
files = {"8th": "data/assemblies/processed_files/8th_amendment/Transcript-of-Final-Weekend-Sat-Sun-1_speeches.csv",
         "ageing": "data/assemblies/processed_files/ageing_population/Transcript-of-Sunday-9-July-Voting-_speeches.csv"}
texts = {k: [str(t).strip() for t, n in zip(*[pd.read_csv(f)[c] for c in ("text", "num_non_function_words")]) if pd.notna(t) and str(t).strip() and n >= 5] for k, f in files.items()}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
for name, model_path in [("base", "Qwen/Qwen3-Embedding-0.6B"), ("finetuned", "models/qwen3_expert_contrastive_final/best_model")]:
    m = Path(f"outputs/embeddings/{name}_kmeans_global_pca/models")
    pca = joblib.load(m / f"pca_model_{name}.pkl"); C = np.load(m / f"centroids_100d_{name}.npy")
    labels = [t for t, _ in joblib.load(m / f"centroid_labels_{name}.pkl")]
    model = SentenceTransformer(model_path, device=dev); model.max_seq_length = 1024  # 8192 stalls on Apple GPUs; 1024 reproduces the published values within 0.4 points
    hits = {}
    for k, T in texts.items():
        E = model.encode(T, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
        a = [labels[i] for i in euclidean_distances(pca.transform(E), C).argmin(1)]
        hits[k] = (sum(x == "constitution" for x in a), len(a))
    both = (hits["8th"][0] + hits["ageing"][0]) / (hits["8th"][1] + hits["ageing"][1])
    print(f"{name}: both files {both:.1%} (paper: {'66.4' if name == 'base' else '65.1'}%) | 8th only {hits['8th'][0]}/{hits['8th'][1]} = {hits['8th'][0]/hits['8th'][1]:.1%} | ageing only {hits['ageing'][0]/hits['ageing'][1]:.1%}", flush=True)
