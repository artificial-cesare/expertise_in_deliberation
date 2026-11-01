"""Markdown tables of every number the revised paper needs, from outputs/paper_v2/*. Prints to stdout."""
import json
import pandas as pd
S = json.load(open("outputs/paper_v2/stats.json"))
A = ["Constitutional Reform", "Gender Equality", "Public Policy"]
NAMES = {"jargon_density": "Jargon density", "lexical_sophistication": "Lexical sophistication (corrected)",
         "syntactic_complexity": "Syntactic complexity", "reasoning_markers": "Reasoning markers",
         "evidential_markers": "Evidential markers (v2)", "length_tokens": "Length (tokens)",
         "num_non_function_words": "Non-function words", "hedge_density": "Hedge density", "booster_density": "Booster density",
         "epistemic_balance": "Epistemic balance", "specificity_length_norm": "Normalised length",
         "specificity_mean_word_len_norm": "Mean word length", "specificity_numbers_norm": "Numerical specificity",
         "specificity_content_prop": "Content-word proportion", "overall_score": "ArgQA overall score",
         "distance_base": "Distance (base)", "similarity_base": "Similarity (base)",
         "distance_finetuned": "Distance (fine-tuned)", "similarity_finetuned": "Similarity (fine-tuned)",
         "expertise_llm_score": "LLM expertise score"}
P = lambda p: "< .001" if p < .001 else f"{p:.3f}".lstrip("0")
R = lambda r: f"{r:+.2f}".replace("+0.", ".").replace("-0.", "−.")
md = lambda df: df.to_markdown(index=False, disable_numparse=True)

print("### Table 1: corpus\n")
t = S["table1"]
print(md(pd.DataFrame([{"Assembly": a, "Total utterances": f"{t[a]['total']:,}", "Substantial (≥ 5 content words)": f"{t[a]['substantial']:,} ({t[a]['pct']}%)",
                         "Mean length, substantial (words)": t[a]["avg_words"]} for a in A])))
print(f"\nTotal: {t['total']['total']:,} utterances, {t['total']['substantial']:,} substantial.\n")
print("### Tables 3 and 6: mean ± SD, substantial utterances\n")
print(md(pd.DataFrame([{"Feature": f"`{f}`", **{a: S["table3_6"][a][f] for a in A}} for f in NAMES])))
c, pl, pa = S["corr_ge5"], S["partial_ge5"], S["partial_argqa_ge5"]
print(f"\n### Correlations with the LLM score (n = {c['expertise_llm_score']['overall_score']['n']:,}; epistemic balance n = {c['expertise_llm_score']['epistemic_balance']['n']:,})\n")
rows = []
for f, v in c["expertise_llm_score"].items():
    q = pl.get(f, {})
    rows.append({"Feature": NAMES[f], "r": R(v["r"]), "p": P(v["p"]),
                 "ArgQA r": R(c["overall_score"][f]["r"]) if f in c["overall_score"] else "",
                 "Partial r (length)": R(q["len"]["r"]) if q else "—", "Partial r (length + assembly)": R(q["len_asm"]["r"]) if q else "—",
                 "p (partial)": P(q["len_asm"]["p"]) if q else ""})
print(md(pd.DataFrame(rows)))
print(f"\nLog length vs LLM r = {R(c['loglen']['expertise_llm_score']['r'])}, vs ArgQA r = {R(c['loglen']['overall_score']['r'])}. "
      f"Variance in LLM score explained (R²): assembly {S['r2_ge5']['assembly']}, log length {S['r2_ge5']['loglen']}, both {S['r2_ge5']['both']}.")
print("ArgQA partial r (length + assembly): " + ", ".join(f"{NAMES[k]} {R(v['r'])}" for k, v in pa.items()))
print(f"Distance vs similarity: base r = {R(c['dist_sim_base']['r'])}, fine-tuned r = {R(c['dist_sim_finetuned']['r'])}. "
      f"Base similarity vs mean word length r = {R(c['similarity_base']['specificity_mean_word_len_norm']['r'])}, vs ArgQA r = {R(c['similarity_base']['overall_score']['r'])}; "
      f"fine-tuned similarity vs ArgQA r = {R(c['similarity_finetuned']['overall_score']['r'])} (all p < .001).\n")
print("### Within-assembly correlations with the LLM score\n")
w = S["within_ge5"]
print(md(pd.DataFrame([{"Feature": NAMES[f], **{f"{a} (n = {w[a]['overall_score']['n']})": (R(w[a][f]["r"]) if w[a].get(f) else "—") for a in A}}
                       for f in ["overall_score", "specificity_length_norm", "length_tokens", "specificity_mean_word_len_norm", "jargon_density",
                                 "lexical_sophistication", "syntactic_complexity", "specificity_numbers_norm", "epistemic_balance", "similarity_base", "similarity_finetuned"]])))
print("\n### LLM score distribution (share of substantial utterances)\n")
print(md(pd.DataFrame([{"Assembly": a, **{f"{int(float(k))}": f"{v:.0%}" for k, v in S["llm_dist"][a].items()}} for a in A])))
print("\n### Speaker roles\n")
r = pd.DataFrame(S["roles"]); r = r[r.role_group != "unknown"]
print(md(r.rename(columns={"assembly": "Assembly", "role_group": "Role", "n": "n", "llm_mean": "LLM mean", "argqa_mean": "ArgQA mean", "median_tokens": "Median tokens"})))
k = S["known_groups"]
print(f"\nKnown-groups check (Constitutional): experts {k['expert_mean']} (n = {k['expert_n']}) vs citizens {k['participant_mean']} (n = {k['participant_n']}), "
      f"Mann-Whitney p = {k['mwu_p']:.1e}, rank-biserial = {abs(k['rank_biserial']):.2f}; gap at equal log length = +{k['length_adjusted_gap']} points.\n")
for lab in ["published", "revised"]:
    q = S[f"kappa_{lab}"]
    print(f"Kappa ({lab}, n = {q['n']}): LLM vs each annotator {q['llm_vs_human']} → mean {q['llm_mean']} ± {q['llm_sd']}; "
          f"annotator pairs {q['human_pairs']} → mean {q['human_mean']} ± {q['human_sd']}; unweighted: LLM {q['unweighted_llm']}, humans {q['unweighted_human']}.")
print("\n### Table 7: temporal tests (≥ 5 content words, per-transcript thirds, BH-FDR over all 60 tests)\n")
T = pd.DataFrame(S["temporal_ge5"]); T = T[(T.q < .05) | T.feature.isin(["expertise_llm_score", "overall_score", "lexical_sophistication"])]
T = T[~T.feature.isin(["distance_finetuned", "num_non_function_words"])]
print(md(pd.DataFrame([{"Assembly": x.assembly, "Feature": NAMES[x.feature], "Means (start / middle / end)": f"{x.mean_start:.3g} / {x.mean_middle:.3g} / {x.mean_end:.3g}",
                         "p": P(x.p), "q": P(x.q), "ε²": f"{x.eps2:.3f}".lstrip("0"), "Trend": x.trend if x.q < .05 else "n.s."} for x in T.itertuples()])))
