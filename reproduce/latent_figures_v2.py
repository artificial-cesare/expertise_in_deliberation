"""Figures 1-2 (revised, 2026-11-12): UMAP views of the expert and assembly embeddings.

Same procedure as code/embeddings/k_means_exp.generate_plots and the same drawing code (plotting.generate_plots),
with the revised inputs: PCA and centroids from reproduce/centroids_v2.py (witnesses' turns only, no duplicated
gender hearing), the ageing-population voting session left out (since 2026-11-19), and the cached embeddings of
embedding_space_check.py.
A 2-d UMAP (random_state 42) is fitted on all expert (witness) utterances, up to 300 sampled utterances per assembly
transcript folder and the centroids, all in the 100-d PCA space; the old pickled UMAP was fitted in the old PCA space
and cannot be reused, so the layout differs from the published figures.

Run from the repo root (CL env):  python reproduce/latent_figures_v2.py
"""
import os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from umap import UMAP

ROOT = Path("outputs/paper_v2/latent_v2")
os.environ["THESIS_ROOT"] = str(ROOT.resolve()); os.environ["PLOT_LIMITS"] = "True"
sys.path[:0] = [str(Path("code/embeddings").resolve()), str(Path("code/handcrafted_features").resolve()),
                str(Path(__file__).resolve().parent)]
import plotting  # noqa: E402  (reads THESIS_ROOT at import)
from paper_v2_stats import load  # noqa: E402
from hearing_roles import expert_corpus  # noqa: E402

CACHE, CV2 = Path("outputs/paper_v2/embedding_check"), Path("outputs/paper_v2/centroids_v2")
FOLDER_TOPIC = {"8th_amendment": "constitution", "ageing_population": "constitution", "gendereq": "gender", "scotland": "future"}

E = expert_corpus()  # cache order; reference = witnesses' turns, duplicate hearing dropped
keep = ((E.role == "witness") & ~E.duplicate).values; E_topic = E.topic.values[keep]
D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True); AGE = (S.folder == "ageing_population").values
FOLDER_TOPIC = {f: t for f, t in FOLDER_TOPIC.items() if (S.folder == f).any()}  # the ageing folder only with load(include_ageing=True)

for name in ("base", "finetuned"):
    Xe = np.load(CACHE / f"{name}_experts.npy")[keep]
    Xa = np.zeros((len(S), Xe.shape[1]), dtype=Xe.dtype)
    Xa[~AGE] = np.load(CACHE / f"{name}_assembly.npy")
    if AGE.any(): Xa[AGE] = np.load(CACHE / f"{name}_assembly_ageing.npy")  # ageing session, only with load(include_ageing=True)
    pca = joblib.load(CV2 / name / "pca.pkl"); C = np.load(CV2 / name / "centroids_100d.npy")
    labels = list(joblib.load(CV2 / name / "centroid_labels.pkl"))
    rng = np.random.RandomState(42)
    fit = [pca.transform(Xe)]
    for folder in FOLDER_TOPIC:
        Z = pca.transform(Xa[(S.folder == folder).values])
        fit.append(Z[rng.choice(len(Z), 300, replace=False)] if len(Z) > 300 else Z)
    fit.append(C)
    reducer = UMAP(n_components=2, random_state=42).fit(np.vstack(fit))
    mdir = ROOT / f"{name}_kmeans_global_pca" / "models"; mdir.mkdir(parents=True, exist_ok=True)
    meta, count = [], {}
    for t in labels:
        meta.append((t, count.get(t, 0))); count[t] = count.get(t, 0) + 1
    joblib.dump(pca, mdir / f"pca_model_{name}.pkl"); joblib.dump(reducer, mdir / f"umap_model_{name}.pkl")
    np.save(mdir / f"centroids_100d_{name}.npy", C); joblib.dump(meta, mdir / f"centroid_labels_{name}.pkl")
    expert_data = {t: {"embeddings": Xe[E_topic == t]} for t in ("constitution", "future", "gender")}
    assembly_data = {f: {"topic": t, "embeddings": Xa[(S.folder == f).values]} for f, t in FOLDER_TOPIC.items()}
    plotting.generate_plots(expert_data, assembly_data, plotting.Config(model_type=name, limit_samples=True))
print("plots in", ROOT)
