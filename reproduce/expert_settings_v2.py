"""Experts in the assembly vs witnesses in the UK parliamentary hearings (paper table "expert_settings", 2026-11-12).

Groups (substantial utterances, >= 5 content words):
  - assembly experts: the Expert Advisory Group / medical experts of the Eighth Amendment session;
  - assembly citizens: participants of the same session;
  - parliamentary witnesses and committee members: reproduce/hearing_roles.py, duplicated gender hearing dropped,
    chairs left out.
Per group: speakers (distinct labels in the assembly, where generic labels such as CITIZEN prevent a count of
persons; distinct persons in the hearings), turns, mean LLM score and ArgQA, median tokens, disfluencies per 100 tokens. Gap: OLS
of the score on group dummies and log length (reference: parliamentary witnesses), HC1 standard errors. Also: does
the disfluency rate lower the LLM score at equal length? (assembly experts and citizens of the Eighth Amendment session;
until 2026-11-19 also the ballot-design speakers of the ageing session, which is no longer in the corpus.)
Run from the repo root (CL env):  python reproduce/expert_settings_v2.py   -> outputs/paper_v2/expert_settings_v2.json
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load
from hearing_roles import expert_corpus

DISF = re.compile(r"\b(um+|uh+|erm|er|you know|i mean|sort of|kind of|sorry|okay|ok)\b|--|\.\.\.", re.I)
REP = re.compile(r"\b(\w+)\s+\1\b", re.I)
TITLE = re.compile(r"^(TABLE \d+ - |DR\.? |PROFESSOR |MRS?\.? |MS\.? |SIR |LORD |BARONESS |VISCOUNT )+|( MP)$| OF .*$")


def disfluency(t):
    t = str(t); return 100 * (len(DISF.findall(t)) + len(REP.findall(t))) / max(len(t.split()), 1)


def persons(labels):
    """Distinct persons behind speaker labels: same surname = same person unless the first initials differ
    ("MR WILLIAMS" + "HYWEL WILLIAMS" = 1; "HYWEL WILLIAMS" + "CRAIG WILLIAMS" = 2)."""
    first = {}
    for lab in set(labels):
        x = re.sub(r"\bMC\s+", "MC", TITLE.sub("", lab.upper().strip()).strip()).split()
        first.setdefault(x[-1], set()).update(n[0] for n in x[:-1][:1])  # initial: spelling variants of a first name match
    return sum(max(1, len(v)) for v in first.values())


D = load(); A = D[(D.folder == "8th_amendment") & (D.num_non_function_words >= 5)]
A = A[A.role.isin(["expert", "participant"])].assign(
    group=lambda d: np.where(d.role == "expert", "Assembly experts", "Assembly citizens"))
E = expert_corpus(); E = E[~E.duplicate & E.role.isin(["witness", "member"])]
E = E.assign(group=np.where(E.role == "witness", "Parliamentary witnesses", "Committee members"))
cols = ["group", "speaker", "text", "expertise_llm_score", "overall_score", "length_tokens"]
X = pd.concat([A[cols], E[cols]], ignore_index=True)
X["loglen"] = np.log1p(X.length_tokens); X["disfl"] = X.text.map(disfluency)

ORDER = ["Assembly experts", "Assembly citizens", "Parliamentary witnesses", "Committee members"]
OUT = {"groups": {}}
fits = {y: smf.ols(f"{y} ~ C(group, Treatment('Parliamentary witnesses')) + loglen", X).fit(cov_type="HC1")
        for y in ("expertise_llm_score", "overall_score")}
for g in ORDER:
    s = X[X.group == g]; key = f"C(group, Treatment('Parliamentary witnesses'))[T.{g}]"
    OUT["groups"][g] = {"speakers": persons(s.speaker) if g.startswith(("Parl", "Comm")) else s.speaker.nunique(), "n": len(s), "llm": round(s.expertise_llm_score.mean(), 2),
                        "argqa": round(s.overall_score.mean(), 2), "median_tokens": float(s.length_tokens.median()),
                        "disfl_per100": round(s.disfl.mean(), 2),
                        **({f"gap_{y}": round(float(m.params[key]), 2) for y, m in fits.items()} if g != ORDER[2] else {}),
                        **({f"gap_{y}_p": float(m.pvalues[key]) for y, m in fits.items()} if g != ORDER[2] else {})}
e, w = X[X.group == ORDER[0]].expertise_llm_score, X[X.group == ORDER[2]].expertise_llm_score
OUT["assembly_experts_vs_witnesses_mwu_p"] = float(mannwhitneyu(e, w).pvalue)
irish = X[X.group.str.startswith("Assembly")].copy()
irish["loglen"] = np.log1p(irish.length_tokens); irish["disfl"] = irish.text.map(disfluency)
m = smf.ols("expertise_llm_score ~ loglen + disfl", irish).fit(cov_type="HC1")
OUT["irish_disfluency_effect"] = {"n": len(irish), "coef_per_100_tokens": round(float(m.params.disfl), 3), "p": float(m.pvalues.disfl)}
Path("outputs/paper_v2/expert_settings_v2.json").write_text(json.dumps(OUT, indent=1))
print(pd.DataFrame(OUT["groups"]).T.to_string())
print({k: v for k, v in OUT.items() if k != "groups"})
