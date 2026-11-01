"""Do the embedding results depend on the PCA reduction? (2026-11-01)

1. Re-embeds expert (UK hearings) and assembly utterances (>= 5 content words) with the base and fine-tuned
   Qwen3 models and checks that the stored distance/similarity columns are the PCA-space values described in the
   paper (100-d PCA fitted on experts, nearest centroid of the assembly's topic).
2. Compares that measure with three alternatives:
   - UMAP: 10-d UMAP (cosine metric) fitted on expert embeddings, assemblies transformed; k-means per topic with
     the same k as the PCA run; distance to the nearest centroid of the assembly's topic.
   - UMAP kNN: mean distance to the 15 nearest expert utterances of the assembly's topic in the UMAP space
     (UMAP preserves local neighbourhoods better than global distances).
   - Raw kNN: mean cosine similarity to the 15 nearest expert utterances of the topic in the full 1024-d space
     (no reduction at all).
   For each: assembly cross-topic accuracy, correlation with the LLM score (raw and at equal length and assembly),
   and a known-groups check (expert witnesses vs citizens in the constitutional-reform session, at equal length).

Run from the repo root (CL env):  python reproduce/embedding_space_check.py [--device mps]
Embeddings are cached in outputs/paper_v2/embedding_check/.
"""
import argparse, json, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, residual
from scipy.stats import pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--models", nargs="+", default=["base", "finetuned"])
args = ap.parse_args()
OUT = Path("outputs/paper_v2/embedding_check"); OUT.mkdir(parents=True, exist_ok=True)
TOPIC = {"Constitutional Reform": "constitution", "Gender Equality": "gender", "Public Policy": "future"}
MODEL = {"base": "Qwen/Qwen3-Embedding-0.6B", "finetuned": "models/qwen3_expert_contrastive_final/best_model"}
K = {"base": {"constitution": 3, "future": 3, "gender": 3}, "finetuned": {"constitution": 3, "future": 4, "gender": 3}}

D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True)
S["topic"] = S.assembly.map(TOPIC)
ex = []
for topic in ("constitution", "future", "gender"):
    for f in sorted(Path(f"data/assemblies/processed_files/experts/{topic}").glob("*_speeches.csv")):
        d = pd.read_csv(f)
        d = d[(d.num_non_function_words >= 5) & d.text.notna()]
        ex += [(topic, str(t).strip()) for t in d.text if str(t).strip()]
E_topic = np.array([t for t, _ in ex]); E_text = [x for _, x in ex]
print("experts:", pd.Series(E_topic).value_counts().to_dict(), "| assembly:", len(S), flush=True)

def embed(name, which, texts):
    p = OUT / f"{name}_{which}.npy"
    if p.exists(): return np.load(p)
    from sentence_transformers import SentenceTransformer
    dev = args.device if name == "finetuned" else "cpu"  # base on CPU, as in table4 (long inputs)
    m = SentenceTransformer(MODEL[name], device=dev)
    if name == "finetuned": m.max_seq_length = 512
    t0 = time.time(); order = np.argsort([-len(x) for x in texts])
    X = m.encode([texts[i] for i in order], batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    X = X[np.argsort(order)]; np.save(p, X); print(f"  embedded {name}/{which}: {len(texts)} in {time.time()-t0:.0f}s", flush=True)
    return X

def stats(score, label):
    """Correlations with the LLM score, known-groups gap. Higher score = closer to expert speech."""
    s = S.assign(m=score)
    r, p = pearsonr(s.m, s.expertise_llm_score)
    cov = [s.loglen] + [s.assembly == a for a in sorted(s.assembly.unique())[1:]]
    pr, pp = pearsonr(residual(s.m.values, cov), residual(s.expertise_llm_score.values, cov))
    c = s[(s.assembly == "Constitutional Reform") & s.role.isin(["expert", "participant"])]
    rr = residual(c.m.values, [c.loglen]); sd = c.m.std()
    gap = (rr[(c.role == "expert").values].mean() - rr[(c.role == "participant").values].mean()) / sd
    within = {a: round(pearsonr(g.m, g.expertise_llm_score)[0], 3) for a, g in s.groupby("assembly")}
    return {"measure": label, "r_llm": round(r, 3), "p": float(p), "partial_r_llm": round(pr, 3), "partial_p": float(pp),
            "within_assembly_r": within, "expert_vs_citizen_gap_sd_equal_length": round(float(gap), 2)}

res = {}
for name in args.models:
    print(f"\n== {name}", flush=True)
    Xe, Xa = embed(name, "experts", E_text), embed(name, "assembly", list(S.text.astype(str)))
    R = {"measures": []}
    # ---- 1. PCA (paper) ------------------------------------------------------------------------
    mdir = Path(f"outputs/embeddings/{name}_kmeans_global_pca/models")
    pca = joblib.load(mdir / f"pca_model_{name}.pkl"); C = np.load(mdir / f"centroids_100d_{name}.npy")
    labels = np.array([t for t, _ in joblib.load(mdir / f"centroid_labels_{name}.pkl")])
    Za = pca.transform(Xa); dist = np.full(len(S), np.nan); sim = np.full(len(S), np.nan)
    for t in TOPIC.values():
        i = (S.topic == t).values; Ct = C[labels == t]
        dd = euclidean_distances(Za[i], Ct); j = dd.argmin(1)
        dist[i] = dd.min(1); sim[i] = cosine_similarity(Za[i], Ct)[np.arange(i.sum()), j]
    R["check_stored"] = {"distance_r": round(pearsonr(dist, S[f"distance_{name}"])[0], 4),
                         "distance_max_abs_diff": round(float(np.nanmax(np.abs(dist - S[f"distance_{name}"]))), 4),
                         "similarity_r": round(pearsonr(sim, S[f"similarity_{name}"])[0], 4)}
    acc = np.mean(labels[euclidean_distances(Za, C).argmin(1)] == S.topic.values)
    R["measures"].append({**stats(S[f"similarity_{name}"].values, "PCA-100 centroid similarity (paper)"), "assembly_accuracy": round(acc, 3)})
    R["measures"].append(stats(-S[f"distance_{name}"].values, "PCA-100 centroid distance (paper, negated)"))
    # ---- 2. UMAP centroids and kNN ---------------------------------------------------------------
    from umap import UMAP
    um = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit(Xe)
    Ue, Ua = um.embedding_, um.transform(Xa)
    cents, clab = [], []
    for t, k in K[name].items():
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Ue[E_topic == t]); cents.append(km.cluster_centers_); clab += [t] * k
    Cu = np.vstack(cents); clab = np.array(clab)
    exp_acc = np.mean(clab[euclidean_distances(Ue, Cu).argmin(1)] == E_topic)
    acc_u = np.mean(clab[euclidean_distances(Ua, Cu).argmin(1)] == S.topic.values)
    du = np.array([euclidean_distances(Ua[i:i+1], Cu[clab == t]).min() for i, t in enumerate(S.topic)])
    R["measures"].append({**stats(-du, "UMAP-10 centroid distance (negated)"), "assembly_accuracy": round(acc_u, 3),
                          "expert_accuracy": round(exp_acc, 3)})
    knn_u = np.zeros(len(S)); knn_r = np.zeros(len(S))
    for t in TOPIC.values():
        i = (S.topic == t).values
        knn_u[i] = -np.sort(euclidean_distances(Ua[i], Ue[E_topic == t]), 1)[:, :15].mean(1)
        knn_r[i] = np.sort(Xa[i] @ Xe[E_topic == t].T, 1)[:, -15:].mean(1)
    R["measures"].append(stats(knn_u, "UMAP-10 kNN (15) distance (negated)"))
    R["measures"].append(stats(knn_r, "Full-space kNN (15) cosine similarity"))
    # nearest-neighbour topic accuracy in full space, for comparison with the centroid accuracies
    nn = np.argsort(-(Xa @ Xe.T), 1)[:, :15]
    acc_r = np.mean([pd.Series(E_topic[row]).mode()[0] == t for row, t in zip(nn, S.topic)])
    R["measures"][-1]["assembly_accuracy"] = round(float(acc_r), 3)
    res[name] = R
    print(json.dumps(R["check_stored"]))
    for m in R["measures"]:
        print(f"  {m['measure']:45s} r={m['r_llm']:+.3f} partial={m['partial_r_llm']:+.3f} "
              f"gap(SD)={m['expert_vs_citizen_gap_sd_equal_length']:+.2f} acc={m.get('assembly_accuracy','')} within={m['within_assembly_r']}", flush=True)

(OUT / "embedding_space_check.json").write_text(json.dumps(res, indent=1))
print("saved", OUT / "embedding_space_check.json")
