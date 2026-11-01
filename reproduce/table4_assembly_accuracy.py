"""Table 4, 'Assembly Cross-Topic Accuracy', recomputed with the saved PCA and k-means centroids.

Embeds every substantial utterance (>= 5 content words) of each assembly, projects it with the saved
expert-only PCA, assigns it to the nearest centroid (Euclidean, 100-d) and counts how often that centroid
belongs to the assembly's topic. Same logic as analyze_cross_topic_assignments() in code/embeddings/k_means_exp.py.

Models keep their own max_seq_length (as k_means_exp.py does): 512 for the fine-tuned model, the
tokenizer default for the base model. Long utterances need a GPU with enough memory for the base model.

Run from the repo root, e.g. on the GPU server:
  python reproduce/table4_assembly_accuracy.py --device cuda:0
  python reproduce/table4_assembly_accuracy.py --models finetuned --device mps   # laptop, fine-tuned only
"""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances

ASSEMBLIES = {"8th_amendment": "constitution", "gendereq": "gender", "scotland": "future",
              "ageing_population": None}  # ageing: not part of the paper, reported for comparison only
WEIGHT_ORDER = ["8th_amendment", "gendereq", "scotland"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-dir", type=Path, default=Path("data/assemblies/processed_files"))
ap.add_argument("--kmeans-dir", type=Path, default=Path("outputs/embeddings"),
                help="folder holding {base,finetuned}_kmeans_global_pca/models/")
ap.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
ap.add_argument("--finetuned-model", default="models/qwen3_expert_contrastive_final/best_model")
ap.add_argument("--models", nargs="+", default=["base", "finetuned"], choices=["base", "finetuned"])
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--max-seq-length", type=int, default=None, help="override the model's own limit (not for reproduction)")
ap.add_argument("--out", type=Path, default=Path("outputs/paper_v2/table4_assembly_accuracy.json"))
args = ap.parse_args()

texts = {}
for folder in ASSEMBLIES:
    t = []
    for f in sorted((args.data_dir / folder).glob("*_speeches.csv")):
        d = pd.read_csv(f)
        t += [str(x).strip() for x, n in zip(d["text"], d["num_non_function_words"]) if pd.notna(x) and str(x).strip() and n >= 5]
    if t: texts[folder] = t
print({k: len(v) for k, v in texts.items()}, "| device:", args.device, flush=True)

results = {}
for name in args.models:
    m = args.kmeans_dir / f"{name}_kmeans_global_pca" / "models"
    pca = joblib.load(m / f"pca_model_{name}.pkl")
    C = np.load(m / f"centroids_100d_{name}.npy")
    labels = [t for t, _ in joblib.load(m / f"centroid_labels_{name}.pkl")]
    model = SentenceTransformer(args.base_model if name == "base" else args.finetuned_model, device=args.device)
    if args.max_seq_length: model.max_seq_length = args.max_seq_length
    print(f"\n{name}: max_seq_length={model.max_seq_length}, {len(labels)} centroids", flush=True)
    results[name] = {}
    for folder, T in texts.items():
        t0 = time.time()
        # longest first so a memory problem shows up immediately
        order = np.argsort([-len(x) for x in T])
        E = model.encode([T[i] for i in order], batch_size=args.batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        E = E[np.argsort(order)]
        assigned = [labels[i] for i in euclidean_distances(pca.transform(E), C).argmin(1)]
        topic = ASSEMBLIES[folder] or "constitution"
        hits = sum(a == topic for a in assigned)
        results[name][folder] = {"correct": hits, "n": len(assigned), "accuracy": hits / len(assigned)}
        print(f"  {folder:18s} {hits}/{len(assigned)} = {hits / len(assigned):.1%}  ({time.time() - t0:.0f}s)", flush=True)
    w = sum(results[name][f]["correct"] for f in WEIGHT_ORDER if f in results[name])
    n = sum(results[name][f]["n"] for f in WEIGHT_ORDER if f in results[name])
    results[name]["weighted_average"] = w / n
    print(f"  weighted average (paper corpus): {w / n:.1%}", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(results, indent=1))
print("\nsaved", args.out)
