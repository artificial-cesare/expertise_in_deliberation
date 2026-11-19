"""Embedding measures on topic-subtracted embeddings: style relative to the speaker's own discussion (2026-11-12).

Distance to expert centroids mostly tracks topic. Here the topic and setting of each discussion are subtracted
before any comparison, in the full 1,024-d space:
  - hearings: for each hearing h, m_h = speaker-balanced mean embedding of all its substantial turns (witnesses,
    committee members, chair); every witness turn becomes x' = e - m_h. The mean of x' over a hearing's witnesses is
    the direction in which witnesses depart from their own hearing;
  - assemblies: for each transcript s (session or table discussion), m_s = speaker-balanced mean of all its
    substantial turns; every utterance becomes x' = e - m_s.
Speaker-balanced: the mean of per-speaker means, so that a speaker with many turns does not define the reference.
Topic-specific as before: assembly utterances are compared only with the witnesses of their assembly's topic.

Then, as in centroids_v2.py, the subtracted embeddings are reduced: PCA (100 components) fitted on the witnesses' x',
k-means per topic with k chosen from {2, 3, 4}^3 by expert topic accuracy (ties: silhouette), and for every assembly
utterance the Euclidean distance and cosine similarity to the nearest centroid of its topic. The projection does not
re-centre (Z = x' V^T, V = PCA components): the origin stays at each discussion's own mean, which is what makes the
cosine a statement about the direction of departure from one's own discussion. Euclidean distances are the same with
or without re-centring. A full-space variant compares x' with the single witness direction of the topic
(mean x' of its witnesses), without reduction.

Checks:
  1. topic removed: expert and assembly cross-topic accuracy of the nearest centroid (chance = 1/3);
  2. validity: partial r with the LLM score (log length + assembly); expert witnesses vs citizens in the Eighth
     Amendment session at equal log length (SD of the measure, OLS p); facilitators vs participants in the
     public-policy discussions at equal log length;
  3. topic specificity: cosines between the three witness directions; permutation test (topic labels shuffled over
     the 14 hearings) of their between-topic dispersion;
  4. alignment: cosine between the in-session expert direction (Eighth Amendment experts minus citizens, both
     session-centred) and each topic's witness direction; null from shuffling expert/citizen labels.
The unsubtracted measures (witness-only centroids, embeddings_v2.csv) are evaluated alongside.
Run from the repo root (CL env):  python reproduce/style_contrast_v2.py
Outputs: outputs/paper_v2/style_contrast_v2.json, style_contrast_v2.csv (per-utterance values).
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, t as tdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load, residual
from hearing_roles import expert_corpus

CACHE, OUT = Path("outputs/paper_v2/embedding_check"), Path("outputs/paper_v2")
TOPICS = ["constitution", "future", "gender"]
TOPIC_OF = {"Constitutional Reform": "constitution", "Gender Equality": "gender", "Public Policy": "future"}
rng = np.random.default_rng(42)

E = expert_corpus(); E = E[~E.duplicate].reset_index(drop=False)  # 'index' = position in the 1,667-row cache
W = (E.role == "witness").values; E_topic = E.topic.values
D = load(); S = D[D.num_non_function_words >= 5].reset_index(drop=True)
AGE = (S.folder == "ageing_population").values; S_topic = S.assembly.map(TOPIC_OF).values
cov = [S.loglen] + [S.assembly == a for a in sorted(S.assembly.unique())[1:]]
y_adj = residual(S.expertise_llm_score.values, cov)
KG = S[(S.folder == "8th_amendment") & S.role.isin(["expert", "participant"])].index.values
PP = S[(S.assembly == "Public Policy") & S.role_group.isin(["facilitation", "participant"])].index.values


def centre(X, unit, speaker):
    """x - speaker-balanced mean of its unit (hearing or transcript)."""
    Xc = np.empty_like(X)
    for u in np.unique(unit):
        i = unit == u
        spk = pd.Series(np.arange(i.sum())).groupby(speaker[i]).apply(list)
        m = np.mean([X[i][idx].mean(0) for idx in spk], axis=0)
        Xc[i] = X[i] - m
    return Xc


def gap(m, idx, flag):
    """Group difference at equal log length: OLS of m on [1, loglen, flag]; returns (SD units, two-sided p)."""
    c = S.loc[idx]; mv = m[idx]; f = flag(c).astype(float)
    X = np.column_stack([np.ones(len(c)), c.loglen.values, f]); b, *_ = np.linalg.lstsq(X, mv, rcond=None); r = mv - X @ b
    se = np.sqrt(r @ r / (len(c) - 3) * np.linalg.inv(X.T @ X)[2, 2])
    return round(float(b[2] / mv.std()), 2), round(float(2 * tdist.sf(abs(b[2] / se), len(c) - 3)), 4)


def validity(m, label, **extra):
    pr, pp = pearsonr(residual(m, cov), y_adj)
    g, gp = gap(m, KG, lambda c: (c.role == "expert").values)
    f, fp = gap(m, PP, lambda c: (c.role_group == "facilitation").values)
    within = {a: round(float(pearsonr(residual(m[i], [S.loglen[i]]), residual(S.expertise_llm_score.values[i], [S.loglen[i]]))[0]), 3)
              for a in TOPIC_OF for i in [(S.assembly == a).values]}
    return {"measure": label, "partial_r_llm": round(float(pr), 3), "partial_p": float(pp), "within_assembly_partial_r": within,
            "expert_vs_citizen_gap_sd": g, "expert_vs_citizen_p": gp, "facilitator_vs_participant_gap_sd": f,
            "facilitator_vs_participant_p": fp, **extra}


def fit_centroids(Ze, topics):
    best = None
    for combo in itertools.product([2, 3, 4], repeat=3):
        cents, labs, sils = [], [], []
        for t, k in zip(TOPICS, combo):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Ze[topics == t])
            cents.append(km.cluster_centers_); labs += [t] * k; sils.append(silhouette_score(Ze[topics == t], km.labels_))
        C, labs = np.vstack(cents), np.array(labs)
        acc, sil = float(np.mean(labs[euclidean_distances(Ze, C).argmin(1)] == topics)), float(np.mean(sils))
        if best is None or acc > best["acc"] or (acc == best["acc"] and sil > best["sil"]):
            best = {"k": dict(zip(TOPICS, combo)), "acc": acc, "sil": sil, "C": C, "labels": labs}
    return best


def nearest(Za, C, labs):
    dist, sim = np.full(len(Za), np.nan), np.full(len(Za), np.nan)
    for t in TOPICS:
        i = S_topic == t; Ct = C[labs == t]
        dd = euclidean_distances(Za[i], Ct); j = dd.argmin(1)
        dist[i] = dd.min(1); sim[i] = cosine_similarity(Za[i], Ct)[np.arange(i.sum()), j]
    acc = float(np.mean(labs[euclidean_distances(Za, C).argmin(1)] == S_topic))
    return dist, sim, acc


def dispersion(dirs_by_hearing, topic_of_hearing):
    d̄ = dirs_by_hearing.mean(0)
    return sum(((dirs_by_hearing[topic_of_hearing == t].mean(0) - d̄) ** 2).sum() * (topic_of_hearing == t).sum() for t in TOPICS)


res, per_utt = {}, {"file": S.file, "row": S.row}
for name in ("base", "finetuned"):
    Xall = np.load(CACHE / f"{name}_experts.npy"); Xe = Xall[E["index"].values]
    Xa = np.zeros((len(S), Xall.shape[1]), dtype=Xall.dtype)
    Xa[~AGE] = np.load(CACHE / f"{name}_assembly.npy")
    if AGE.any(): Xa[AGE] = np.load(CACHE / f"{name}_assembly_ageing.npy")  # ageing session, only with load(include_ageing=True)
    Xe_c = centre(Xe, E.hearing.values, E.speaker.values)[W]          # witnesses relative to their hearing
    Xa_c = centre(Xa, S.file.values, S.speaker.astype(str).values)   # utterances relative to their transcript
    Wt = E_topic[W]; R = {"measures": []}

    # ---- reduced: PCA-100 on witnesses' x', no re-centring; k-means per topic ------------------------------------
    V = PCA(n_components=100, random_state=42).fit(Xe_c).components_
    Ze, Za = Xe_c @ V.T, Xa_c @ V.T
    best = fit_centroids(Ze, Wt)
    dist, sim, acc = nearest(Za, best["C"], best["labels"])
    R["pca_subtracted"] = {"k": best["k"], "expert_accuracy": round(best["acc"], 3), "silhouette": round(best["sil"], 4),
                           "assembly_accuracy": round(acc, 3), "dist_sim_r": round(float(pearsonr(dist, sim)[0]), 3)}
    R["measures"].append(validity(sim, "Subtracted, PCA-100 centroid similarity", assembly_accuracy=round(acc, 3)))
    R["measures"].append(validity(-dist, "Subtracted, PCA-100 centroid distance (negated)"))
    per_utt[f"style_distance_{name}"], per_utt[f"style_similarity_{name}"] = dist, sim

    # ---- full space: single witness direction per topic -------------------------------------------------------
    dirs = {t: Xe_c[Wt == t].mean(0) for t in TOPICS}
    Cd = np.vstack([dirs[t] for t in TOPICS]); labs = np.array(TOPICS)
    d_full, s_full, acc_full = nearest(Xa_c, Cd, labs)
    exp_acc_full = float(np.mean(labs[euclidean_distances(Xe_c, Cd).argmin(1)] == Wt))
    R["measures"].append(validity(s_full, "Subtracted, full-space cosine to topic witness direction",
                                  assembly_accuracy=round(acc_full, 3), expert_accuracy=round(exp_acc_full, 3)))
    R["measures"].append(validity(-d_full, "Subtracted, full-space distance to topic witness direction (negated)"))

    # ---- unsubtracted reference (witness-only centroids, as in the paper) ---------------------------------------
    R["measures"].append(validity(S[f"similarity_{name}"].values, "Unsubtracted, PCA-100 centroid similarity (paper)"))
    R["measures"].append(validity(-S[f"distance_{name}"].values, "Unsubtracted, PCA-100 centroid distance (paper, negated)"))

    # ---- 3. topic specificity of the witness directions ---------------------------------------------------------
    H = E[W].hearing.values; hs = np.unique(H)
    dh = np.vstack([Xe_c[H == h].mean(0) for h in hs]); th = np.array([h.split("/")[0] for h in hs])
    obs = dispersion(dh, th); null = np.array([dispersion(dh, rng.permutation(th)) for _ in range(10000)])
    R["topic_directions"] = {"cosines": {f"{a}-{b}": round(float(cosine_similarity(dirs[a][None], dirs[b][None])[0, 0]), 3)
                                         for a, b in itertools.combinations(TOPICS, 2)},
                             "hearing_direction_cosines_within_topic": round(float(np.mean([cosine_similarity(dh[i][None], dh[j][None])[0, 0]
                                 for i, j in itertools.combinations(range(len(hs)), 2) if th[i] == th[j]])), 3),
                             "hearing_direction_cosines_between_topics": round(float(np.mean([cosine_similarity(dh[i][None], dh[j][None])[0, 0]
                                 for i, j in itertools.combinations(range(len(hs)), 2) if th[i] != th[j]])), 3),
                             "dispersion_permutation_p": float((null >= obs).mean()), "n_hearings": len(hs)}

    # ---- 4. alignment of the in-session expert direction with the witness directions ----------------------------
    c = S.loc[KG]; ex = (c.role == "expert").values; Xk = Xa_c[KG]
    d_in = Xk[ex].mean(0) - Xk[~ex].mean(0)
    cos_ = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    perm = [Xk[p].mean(0) - Xk[~p].mean(0) for p in (rng.permutation(ex) for _ in range(5000))]
    R["in_session_alignment"] = {}
    for t in TOPICS:
        nul = np.array([cos_(d, dirs[t]) for d in perm]); o = cos_(d_in, dirs[t])
        R["in_session_alignment"][t] = {"cosine": round(o, 3), "null_mean": round(float(nul.mean()), 3),
                                        "null_sd": round(float(nul.std()), 3), "p_one_sided": float((nul >= o).mean())}
    res[name] = R
    print(f"\n== {name}: {json.dumps(R['pca_subtracted'])}")
    for m in R["measures"]:
        print(f"  {m['measure']:72s} partial r={m['partial_r_llm']:+.3f} gap={m['expert_vs_citizen_gap_sd']:+.2f} (p={m['expert_vs_citizen_p']:.3f}) "
              f"fac-part={m['facilitator_vs_participant_gap_sd']:+.2f} (p={m['facilitator_vs_participant_p']:.3f}) acc={m.get('assembly_accuracy', '')} within={m['within_assembly_partial_r']}")
    print("  topic directions:", R["topic_directions"])
    print("  in-session alignment:", R["in_session_alignment"])

(OUT / "style_contrast_v2.json").write_text(json.dumps(res, indent=1))
pd.DataFrame(per_utt).to_csv(OUT / "style_contrast_v2.csv", index=False)
print("saved", OUT / "style_contrast_v2.json")
