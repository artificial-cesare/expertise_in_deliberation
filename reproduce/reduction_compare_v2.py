"""Linear (PCA) vs non-linear (UMAP) reduction: what does the reduction lose? (2026-11-12)

Spaces, all built from the same cached Qwen3 embeddings (expert reference: the witnesses' turns of the
hearings, without the duplicated gender hearing; reproduce/hearing_roles.py):
  full-1024 (no reduction), PCA-100 (main analysis), PCA-10, UMAP-100, UMAP-10.
PCA as in centroids_v2.py (fitted on experts); UMAP fitted on experts (cosine, 15 neighbours, min_dist 0, seed 42),
assembly utterances transformed. In every space: k-means per topic with the k of the main analysis, then for each
assembly utterance the Euclidean distance and cosine similarity to the nearest centroid of its topic.

Reported per space and model:
  - variance kept (PCA only): share of expert variance explained; out-of-sample share of assembly variance
  - trustworthiness (15 neighbours) of the assembly utterances against the full space
  - Spearman agreement of distance/similarity with the full-space values
  - partial r of similarity with the LLM score (log length + assembly); expert-citizen gap (Eighth Amendment session,
    equal log length, in SD) with its OLS p-value
  - cross-validated R^2 (ridge, folds grouped by transcript) predicting the LLM score residualised on log length and
    assembly from the coordinates of the space: does the space still hold expertise-related signal at all?
Run from the repo root (CL env):  python reproduce/reduction_compare_v2.py
"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, t as tdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.manifold import trustworthiness
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import GroupKFold, cross_val_predict
from umap import UMAP
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, residual
from hearing_roles import expert_corpus

CACHE, CV2 = Path("outputs/paper_v2/embedding_check"), Path("outputs/paper_v2/centroids_v2")
TOPIC_OF = {"Constitutional Reform": "constitution", "Gender Equality": "gender", "Public Policy": "future"}
K = {m: v["k"] for m, v in json.loads((CV2 / "metrics.json").read_text()).items()}

E = expert_corpus()  # cache order; reference = witnesses' turns, duplicate hearing dropped
keep = ((E.role == "witness") & ~E.duplicate).values; E_topic = E.topic.values[keep]
D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True)
AGE = (S.folder == "ageing_population").values; S_topic = S.assembly.map(TOPIC_OF).values
cov = [S.loglen] + [S.assembly == a for a in sorted(S.assembly.unique())[1:]]
y_adj = residual(S.expertise_llm_score.values, cov)
kg = S[(S.folder == "8th_amendment") & S.role.isin(["expert", "participant"])].index.values
groups = S.file.values


def measures(Ze, Za, k):
    cents, labs = [], []
    for t, kk in k.items():
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(Ze[E_topic == t]); cents.append(km.cluster_centers_); labs += [t] * kk
    C, labs = np.vstack(cents), np.array(labs)
    dist, sim = np.zeros(len(Za)), np.zeros(len(Za))
    for t in k:
        i = S_topic == t; Ct = C[labs == t]
        dd = euclidean_distances(Za[i], Ct); j = dd.argmin(1)
        dist[i] = dd.min(1); sim[i] = cosine_similarity(Za[i], Ct)[np.arange(i.sum()), j]
    acc = float(np.mean(labs[euclidean_distances(Za, C).argmin(1)] == S_topic))
    return dist, sim, acc


def validity(m):
    pr, pp = pearsonr(residual(m, cov), y_adj)
    c = S.loc[kg]; mv = m[kg]; e = (c.role == "expert").values.astype(float)
    X = np.column_stack([np.ones(len(c)), c.loglen.values, e]); b, *_ = np.linalg.lstsq(X, mv, rcond=None); r = mv - X @ b
    se = np.sqrt(r @ r / (len(c) - 3) * np.linalg.inv(X.T @ X)[2, 2])
    return round(float(pr), 4), float(pp), round(float(b[2] / mv.std()), 2), float(2 * tdist.sf(abs(b[2] / se), len(c) - 3))


res = []
for name in ("base", "finetuned"):
    Xe = np.load(CACHE / f"{name}_experts.npy")[keep]
    Xa = np.zeros((len(S), Xe.shape[1]), dtype=Xe.dtype)
    Xa[~AGE] = np.load(CACHE / f"{name}_assembly.npy")
    if AGE.any(): Xa[AGE] = np.load(CACHE / f"{name}_assembly_ageing.npy")  # ageing session, only with load(include_ageing=True)
    spaces = {"Full (1,024-d)": (Xe, Xa, None)}
    for nc in (100, 10):
        p = joblib.load(CV2 / name / "pca.pkl") if nc == 100 else PCA(n_components=nc, random_state=42).fit(Xe)
        Za = p.transform(Xa); rec = p.inverse_transform(Za)
        oos = 1 - ((Xa - rec) ** 2).sum() / ((Xa - Xa.mean(0)) ** 2).sum()
        spaces[f"PCA-{nc}"] = (p.transform(Xe), Za, {"expert_var": round(float(p.explained_variance_ratio_.sum()), 4), "assembly_var": round(float(oos), 4)})
    for nc in (100, 10):
        u = UMAP(n_components=nc, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit(Xe)
        spaces[f"UMAP-{nc}"] = (u.embedding_, u.transform(Xa), None)
    full_d, full_s, _ = measures(Xe, Xa, K[name])
    for sp, (Ze, Za, var) in spaces.items():
        d, s, acc = (full_d, full_s, None) if sp.startswith("Full") else measures(Ze, Za, K[name])
        if acc is None: acc = measures(Xe, Xa, K[name])[2]
        pr_s, pp_s, gap_s, gp_s = validity(s); pr_d, pp_d, gap_d, gp_d = validity(-d)
        r2 = 1 - np.sum((y_adj - cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Za, y_adj, groups=groups,
                                                         cv=GroupKFold(5))) ** 2) / np.sum((y_adj - y_adj.mean()) ** 2)
        trust = 1.0 if sp.startswith("Full") else float(trustworthiness(Xa, Za, n_neighbors=15))
        row = {"model": name, "space": sp, "expert_var": (var or {}).get("expert_var"), "assembly_var": (var or {}).get("assembly_var"),
               "trustworthiness": round(trust, 4), "rho_dist_full": round(float(spearmanr(d, full_d)[0]), 3),
               "rho_sim_full": round(float(spearmanr(s, full_s)[0]), 3), "accuracy": round(acc, 3),
               "partial_r_sim": pr_s, "p_sim": pp_s, "gap_sim_sd": gap_s, "gap_sim_p": round(gp_s, 3),
               "partial_r_negdist": pr_d, "p_negdist": pp_d, "gap_negdist_sd": gap_d, "gap_negdist_p": round(gp_d, 3),
               "cv_r2_adjusted_llm": round(float(r2), 4)}
        res.append(row); print({k: v for k, v in row.items() if not k.startswith("p_")}, flush=True)
out = pd.DataFrame(res); out.to_csv("outputs/paper_v2/reduction_compare_v2.csv", index=False)
print(out.drop(columns=["p_sim", "p_negdist"]).to_string(index=False))
