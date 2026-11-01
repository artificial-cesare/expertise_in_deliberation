#!/usr/bin/env bash
# Rebuild every paper result that needs only the frozen CSVs (no GPU, no API key).
# Run from the repo root:  bash reproduce/run_all.sh   (set PY=/path/to/python to pick an env)
set -euo pipefail
PY="${PY:-python}"
export MPLBACKEND=Agg

shasum -a 256 -c --quiet data/CHECKSUMS.sha256   # the data is the frozen version
"$PY" tests/test_reproduce.py

# Tables 1, 3, 6
"$PY" code/summarytables/final_stats.py --output_dir outputs/corpus_statistics
# Table 5
mkdir -p outputs/validation
"$PY" code/validation/kappa_filtered_new.py --json-path data/validation/sampled_speeches.json \
    --csv-dir data/validation/annotations_csv --output outputs/validation/kappa_results.json
# Figs 3-4, Appendix B and C: as published, then with Scotland's LLM scores
"$PY" reproduce/correlations.py --scotland-llm drop
"$PY" reproduce/correlations.py --scotland-llm keep
# Section 5.4, Table 7
"$PY" reproduce/temporal_kw.py

shasum -a 256 -c --quiet data/CHECKSUMS.sha256   # nothing above rewrote the data
echo "Done. Results are in outputs/."
