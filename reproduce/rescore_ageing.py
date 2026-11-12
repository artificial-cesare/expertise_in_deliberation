"""Rescore the ageing-population voting session with its own topic description (2026-11-12).

The transcript was scored on 2025-10-31 from the 8th_amendment folder, so its prompt named the Eighth Amendment as
the topic. This rescoring uses the same prompt (Appendix A, llm_utils.SYSTEM_PROMPTS["paper"]), model, temperature,
20-utterance context (llm_expert_extract_new.get_context_window) and threshold (>= 5 content words); only the topic
string changes (shared.topic_maps: "ageing_population"). The data CSV is not modified: results go to
outputs/paper_v2/llm_v2.csv, which paper_v2_stats.load() merges over the stored scores.

Run from the repo root (CL env, OPENAI_API_KEY set):  python reproduce/rescore_ageing.py
"""
import csv, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
sys.path[:0] = ["code/LLM_expertise", "code/handcrafted_features", "code"]
from llm_expert_extract_new import get_context_window, get_text_from_row
from llm_utils import score_expertise_with_llm
from shared.topic_maps import extract_topic

F = Path("data/assemblies/processed_files/ageing_population/Transcript-of-Sunday-9-July-Voting-_speeches.csv")
rows = list(csv.DictReader(open(F, encoding="utf-8")))
topic = extract_topic(str(F))
assert "ageing" in topic, topic
key = os.environ["OPENAI_API_KEY"]
todo = [i for i, r in enumerate(rows) if get_text_from_row(r) and float(r.get("num_non_function_words") or 0) >= 5]
print(f"topic: {topic!r} | {len(todo)} of {len(rows)} rows to score", flush=True)


def score(i):
    for _ in range(4):
        s, j = score_expertise_with_llm(get_text_from_row(rows[i]), get_context_window(rows, i), topic, key, "gpt-4o-mini", prompt="paper")
        if s is not None: return i, s, j
    return i, None, j


with ThreadPoolExecutor(8) as ex:
    res = {i: (s, j) for i, s, j in ex.map(score, todo)}
out = pd.DataFrame({"file": F.name, "row": range(len(rows)),
                    "expertise_llm_score": [res.get(i, (None, None))[0] for i in range(len(rows))],
                    "expertise_justification": [res.get(i, (None, None))[1] for i in range(len(rows))],
                    "topic": topic, "prompt": "paper"})
out.to_csv("outputs/paper_v2/llm_v2.csv", index=False)
old = pd.read_csv(F).expertise_llm_score
m = out.expertise_llm_score.notna()
print(f"failed: {sum(v[0] is None for v in res.values())} | mean old topic {old[m].mean():.3f} -> new topic {out.expertise_llm_score[m].mean():.3f} "
      f"| identical {(old[m] == out.expertise_llm_score[m]).mean():.2f} | n = {m.sum()}")
