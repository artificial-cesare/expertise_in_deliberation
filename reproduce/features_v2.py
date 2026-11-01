"""Corrected and extended features for the revised paper. Leaves the processed CSVs untouched.

lexical_sophistication_v2 fixes a mismatch in new_feats_extract.py: the 5th/95th Zipf percentiles were
taken over content words, but each utterance's mean Zipf was taken over all alphabetic tokens, function
words included, so most utterances clipped to 0. v2 averages over content words, the same set the
percentiles come from. Everything else is as in the original: spaCy en_core_web_sm, wordfreq Zipf (en),
percentiles per transcript file, scaled (p95 - mean) / max(p95 - p5, 1) and clipped to [0, 1],
NaN when an utterance has fewer than 5 content words.

evidential_markers_v2 uses reproduce/evidential_v2.py (broader patterns, by evidence type).

Writes outputs/paper_v2/features_v2.csv (one row per utterance: assembly folder, file, row).
Run from the repo root:  python reproduce/features_v2.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, spacy
from wordfreq import zipf_frequency
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidential_v2 import evidential_counts, evidential_v2

NLP = spacy.load("en_core_web_sm")
ROOT = Path("data/assemblies/processed_files")


def content_zipfs(doc):
    zs = [zipf_frequency(t.text.lower(), "en") for t in doc if t.is_alpha and not t.is_stop]
    return [z for z in zs if z > 0]


rows = []
for folder in ["8th_amendment", "gendereq", "scotland"]:
    for f in sorted((ROOT / folder).glob("*_speeches.csv")):
        df = pd.read_csv(f)
        texts = df["text"].fillna("").astype(str).tolist()
        docs = list(NLP.pipe(texts, batch_size=64))
        # percentiles exactly as compute_corpus_statistics: lowercased text, content words, per file
        pool = [z for d in NLP.pipe([t.lower() for t in texts if t.strip()], batch_size=64) for z in content_zipfs(d)]
        p95, p5 = (np.percentile(pool, 95), np.percentile(pool, 5)) if pool else (6.5, 3.0)
        if p95 <= p5: p95, p5 = 6.5, 3.0
        for i, (t, d) in enumerate(zip(texts, docs)):
            n_alpha = sum(t_.is_alpha for t_ in d)
            n_content = sum(t_.is_alpha and not t_.is_stop for t_ in d)
            rec = {"folder": folder, "file": f.name, "row": i}
            if n_content < 5:
                rec.update(lexical_sophistication_v2=np.nan, evidential_markers_v2=np.nan)
            else:
                zs = content_zipfs(d)
                avg = np.mean(zs) if zs else p95
                rec["lexical_sophistication_v2"] = float(np.clip((p95 - avg) / max(p95 - p5, 1.0), 0, 1))
                rec["evidential_markers_v2"] = evidential_v2(t, n_alpha)
                rec.update({f"evid_{k}": v for k, v in evidential_counts(t).items()})
            rows.append(rec)
        print(f"{folder}/{f.name}: {len(df)} rows, zipf p5={p5:.2f} p95={p95:.2f}", flush=True)
out = Path("outputs/paper_v2"); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "features_v2.csv", index=False)
print("saved", out / "features_v2.csv")
