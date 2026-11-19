"""Expert centroids, distances and similarities from the witnesses' turns only (2026-11-12).

The expert reference is restricted in two ways (see reproduce/hearing_roles.py):
  - data/assemblies/processed_files/experts/gender/committees.parliament.uk_speeches.csv was processed from an HTML
    file byte-identical to committees.parliament.uk5.html, so that hearing entered the expert corpus twice (104 of
    1,667 expert utterances); the copy is dropped;
  - the hearings also hold the turns of the chairs and of the committee members who question the witnesses; only the
    witnesses' turns are kept (822 of the 1,563 utterances of the deduplicated hearings).
The previous versions (all turns of the deduplicated hearings; all turns with the duplicate) are kept in
outputs/paper_v2/_all_hearing_turns_2026-11-12/ and _with_duplicate_2026-11-12/. This script repeats the pipeline of
code/embeddings/k_means_exp.py on the witnesses' turns:

  1. global PCA (100 components, random_state 42) fitted on all expert embeddings;
  2. k-means per topic (n_init 10, random_state 42) for every k in {2, 3, 4}^3; the combination with the highest
     expert classification accuracy (nearest centroid, Euclidean, 100-d) wins, ties broken by mean silhouette;
  3. for every substantial assembly utterance: Euclidean distance to the nearest centroid of its assembly's topic,
     and cosine similarity to that same centroid (the stored distance_*/similarity_* columns);
  4. Table 4 metrics: expert accuracy (global, per topic), mean silhouette, assembly cross-topic accuracy.
The ageing-population voting session is embedded and written to embeddings_v2.csv (against the constitutional hearings)
for traceability, but since 2026-11-19 it is not part of the paper's corpus: the assembly accuracies and their weighted
average leave it out and report it on a separate line.

With all turns and the duplicate included, this reproduces the stored PCA and centroids to 1e-6 (checked below).
Embeddings are the cached ones of reproduce/embedding_space_check.py (same models and settings).
Outputs: outputs/paper_v2/centroids_v2/{base,finetuned}/ and outputs/paper_v2/embeddings_v2.csv, which
paper_v2_stats.load() merges over the stored columns.

Run from the repo root (CL env):  python reproduce/centroids_v2.py
"""
import itertools, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load
from hearing_roles import expert_corpus

TOPICS = ["constitution", "future", "gender"]
TOPIC_OF = {"Constitutional Reform": "constitution", "Gender Equality": "gender", "Public Policy": "future"}
CACHE = Path("outputs/paper_v2/embedding_check")
OUT = Path("outputs/paper_v2/centroids_v2")

E = expert_corpus()
keep = ((E.role == "witness") & ~E.duplicate).values  # witnesses' turns, duplicate hearing dropped

D = load(include_ageing_embeddings=True, embeddings="stored", include_ageing=True)  # ageing rows kept in embeddings_v2.csv
S = D[D.num_non_function_words >= 5].reset_index(drop=True)
AGE = (S.folder == "ageing_population").values
S_topic = S.assembly.map(TOPIC_OF).values


def fit(Xe, topics):
    pca = PCA(n_components=100, random_state=42).fit(Xe)
    Z = pca.transform(Xe)
    best = None
    for combo in itertools.product([2, 3, 4], repeat=3):
        cents, labs, sils = [], [], []
        for t, k in zip(TOPICS, combo):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z[topics == t])
            cents.append(km.cluster_centers_); labs += [t] * k
            sils.append(silhouette_score(Z[topics == t], km.labels_))
        C, labs = np.vstack(cents), np.array(labs)
        assigned = labs[euclidean_distances(Z, C).argmin(1)]
        acc, sil = float(np.mean(assigned == topics)), float(np.mean(sils))
        if best is None or acc > best["acc"] or (acc == best["acc"] and sil > best["sil"]):
            best = {"k": dict(zip(TOPICS, combo)), "acc": acc, "sil": sil, "C": C, "labels": labs,
                    "per_topic": {t: float(np.mean(assigned[topics == t] == t)) for t in TOPICS}}
    return pca, best


rows = {"file": S.file, "row": S.row}
metrics = {}
for name in ("base", "finetuned"):
    Xe_all = np.load(CACHE / f"{name}_experts.npy")
    assert len(Xe_all) == len(E), "expert cache does not match the expert files"
    Xa = np.zeros((len(S), Xe_all.shape[1]), dtype=Xe_all.dtype)
    Xa[~AGE] = np.load(CACHE / f"{name}_assembly.npy")
    if AGE.any(): Xa[AGE] = np.load(CACHE / f"{name}_assembly_ageing.npy")  # ageing session, only with load(include_ageing=True)

    # check: with the duplicate, the stored models come back
    stored = Path(f"outputs/embeddings/{name}_kmeans_global_pca/models")
    pca_s, C_s = joblib.load(stored / f"pca_model_{name}.pkl"), np.load(stored / f"centroids_100d_{name}.npy")
    pca_chk, best_chk = fit(Xe_all, E.topic.values)
    dev = float(euclidean_distances(pca_chk.inverse_transform(best_chk["C"]), pca_s.inverse_transform(C_s)).min(1).max())
    print(f"{name}: all turns with duplicate k={best_chk['k']} acc={best_chk['acc']:.3f}; max centroid deviation from stored = {dev:.1e}")

    pca, best = fit(Xe_all[keep], E.topic.values[keep])
    Za = pca.transform(Xa)
    dist, sim = np.full(len(S), np.nan), np.full(len(S), np.nan)
    for t in TOPICS:
        i = S_topic == t
        Ct = best["C"][best["labels"] == t]
        dd = euclidean_distances(Za[i], Ct); j = dd.argmin(1)
        dist[i] = dd.min(1); sim[i] = cosine_similarity(Za[i], Ct)[np.arange(i.sum()), j]
    rows[f"distance_{name}"], rows[f"similarity_{name}"] = dist, sim
    assigned = best["labels"][euclidean_distances(Za, best["C"]).argmin(1)]
    hit = assigned == S_topic
    by = {}
    for key, mask in {"Constitutional Reform": (S.folder == "8th_amendment").values,  # Eighth Amendment session only
                      "Gender Equality": (S.assembly == "Gender Equality").values,
                      "Public Policy": (S.assembly == "Public Policy").values,
                      "ageing session (not in the paper)": AGE}.items():
        by[key.strip()] = {"correct": int(hit[mask].sum()), "n": int(mask.sum()), "accuracy": round(float(hit[mask].mean()), 4)}
    metrics[name] = {"k": best["k"], "n_experts": int(keep.sum()), "expert_accuracy": round(best["acc"], 4),
                     "per_topic_expert_accuracy": {t: round(v, 4) for t, v in best["per_topic"].items()},
                     "mean_silhouette": round(best["sil"], 4), "assembly_accuracy": by,
                     "assembly_weighted_average": round(float(hit[~AGE].mean()), 4),  # without the ageing session
                     "check_all_turns_with_duplicate": {"k": best_chk["k"], "expert_accuracy": round(best_chk["acc"], 4),
                                              "max_centroid_deviation_from_stored": dev}}
    od = OUT / name; od.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, od / "pca.pkl"); np.save(od / "centroids_100d.npy", best["C"]); joblib.dump(list(best["labels"]), od / "centroid_labels.pkl")
    old_d, old_s = S[f"distance_{name}"].values, S[f"similarity_{name}"].values
    print(f"{name}: witnesses only ({keep.sum()} utterances) k={best['k']} acc={best['acc']:.3f} sil={best['sil']:.4f} "
          f"assembly acc (no ageing)={hit[~AGE].mean():.3f} | r(new, old) distance={np.corrcoef(dist, old_d)[0,1]:.3f} similarity={np.corrcoef(sim, old_s)[0,1]:.3f}")

pd.DataFrame(rows).to_csv("outputs/paper_v2/embeddings_v2.csv", index=False)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
