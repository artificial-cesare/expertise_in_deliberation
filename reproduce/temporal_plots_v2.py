"""Appendix C (revised): temporal plots that match the Kruskal-Wallis tests in Table 7.

Windows: each transcript is cut into thirds by row order (start / middle / end) and the thirds are
pooled per assembly, exactly as in the tests (the old plots cut the concatenated assembly instead).
Substantial utterances (>= 5 content words), corrected features, raw (unscaled) values.
Each panel: window mean with 95% CI (t), test result from outputs/paper_v2/stats.json.
Run from the repo root after paper_v2_stats.py:  python reproduce/temporal_plots_v2.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load

FEATURES = {"expertise_llm_score": "LLM expertise score (1–5)", "overall_score": "ArgQA overall score",
            "lexical_sophistication": "Lexical sophistication", "similarity_finetuned": "Similarity to expert centroid (fine-tuned)",
            "similarity_base": "Similarity to expert centroid (base)", "length_tokens": "Utterance length (tokens)",
            "specificity_length_norm": "Normalised length", "specificity_content_prop": "Content-word proportion",
            "specificity_numbers_norm": "Numerical specificity"}
ASSEMBLIES = ["Constitutional Reform", "Gender Equality", "Public Policy"]
WINDOWS = ["start", "middle", "end"]
INK, MUTED, MARK, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

D = load(); S = D[D.num_non_function_words >= 5]
T = pd.DataFrame(json.load(open("outputs/paper_v2/stats.json"))["temporal_ge5"]).set_index(["assembly", "feature"])
out = Path("outputs/paper_v2/temporal"); out.mkdir(parents=True, exist_ok=True)


def fmt_p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


for feat, label in FEATURES.items():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, a in zip(axes, ASSEMBLIES):
        g = S[S.assembly == a]
        m, lo, hi = [], [], []
        for w in WINDOWS:
            x = g.loc[g.window == w, feat].dropna()
            se = x.std(ddof=1) / np.sqrt(len(x)); h = se * stats.t.ppf(.975, len(x) - 1)
            m.append(x.mean()); lo.append(x.mean() - h); hi.append(x.mean() + h)
        xs = np.arange(3)
        ax.plot(xs, m, color=MARK, lw=2, zorder=2)
        ax.errorbar(xs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o", ms=7, color=MARK,
                    ecolor=MARK, elinewidth=2, capsize=0, mec="white", mew=2, zorder=3)
        ax.set_xticks(xs, ["Start", "Middle", "End"]); ax.set_xlim(-.4, 2.4)
        ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
        stat, sig = "", False
        if (a, feat) in T.index:
            r = T.loc[(a, feat)]; sig = r.q < .05
            q = "q < .001" if r.q < .001 else f"q = {r.q:.3f}".replace("0.", ".")
            stat = f"{fmt_p(r.p)}, {q}, ε² = {r.eps2:.3f}, n = {int(r.n)}".replace("ε² = 0.", "ε² = .")
        ax.set_title(a, fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=18)
        ax.text(0, 1.02, stat, transform=ax.transAxes, fontsize=8.5, color=INK if sig else MUTED,
                fontweight="bold" if sig else "normal", va="bottom")
    axes[0].set_ylabel("Mean (raw scale)")
    fig.suptitle(f"{label}: mean and 95% CI by third of each transcript", x=.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / f"{feat}_temporal.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", out / f"{feat}_temporal.png")
