"""Size of the LLM judge's context: the 20-utterance window and the full transcripts (2026-11-19).

Numbers for the paper's context-calibration paragraph and Limitations. Tokens are counted with the tokenizer of
gpt-4o-mini (tiktoken o200k_base), as in code/LLM_expertise/context_size_calibration.py.
  - per transcript file: turns and tokens of all turns (the full history a judge or annotator would read);
  - per scored utterance (>= 5 content words): tokens of the 20 preceding turns of any length, which is the
    context of llm_expert_extract_new.get_context_window(window_size=20); median, IQR, P10/P90, share above
    500 and 1,000 tokens, by assembly.
Run from the repo root (CL env):  python reproduce/context_window_v2.py   -> outputs/paper_v2/context_window_v2.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, tiktoken
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_v2_stats import load

OUT = Path("outputs/paper_v2/context_window_v2.json")
enc = tiktoken.get_encoding("o200k_base")
D = load()
D["tok"] = [len(enc.encode(str(t))) for t in D.text]

files, wins = [], []
for f, g in D.groupby("file", sort=False):
    g = g.sort_values("row")
    files.append({"file": f, "folder": g.folder.iat[0], "assembly": g.assembly.iat[0], "turns": len(g), "tokens": int(g.tok.sum())})
    cs = np.concatenate([[0], np.cumsum(g.tok.values)])
    for i, ok in enumerate((g.num_non_function_words >= 5).values):
        if ok: wins.append({"assembly": g.assembly.iat[0], "folder": g.folder.iat[0], "ctx": int(cs[i] - cs[max(0, i - 20)])})
F, W = pd.DataFrame(files), pd.DataFrame(wins)


def summ(x):
    return {"n": len(x), "median": float(x.median()), "iqr": x.quantile([.25, .75]).tolist(), "p10_p90": x.quantile([.1, .9]).tolist(),
            "share_gt500": round((x > 500).mean(), 3), "share_gt1000": round((x > 1000).mean(), 3), "max": int(x.max())}


S = {"transcripts": {"n": len(F), "median_turns": float(F.turns.median()), "median_tokens": float(F.tokens.median()),
                     "max_tokens": int(F.tokens.max()), "by_folder": F.groupby("folder")[["turns", "tokens"]].median().to_dict("index")},
     "window_all": summ(W.ctx), "window_by_assembly": {a: summ(g.ctx) for a, g in W.groupby("assembly")},
     "window_le500_by_assembly": W[W.ctx <= 500].assembly.value_counts().to_dict(),
     "eighth_amendment_files": F[F.folder == "8th_amendment"][["file", "turns", "tokens"]].to_dict("records")}
OUT.write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))
print("->", OUT)
