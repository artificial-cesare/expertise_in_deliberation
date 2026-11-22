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
# Figs 3-4, Appendix B and C: as published (Overleaf, > 5 content words, no Scotland LLM scores),
# then >= 5 content words with Scotland's LLM scores
"$PY" reproduce/correlations.py --scotland-llm drop --legacy-gt5
"$PY" reproduce/correlations.py --scotland-llm keep
# Section 5.4, Table 7: as published (no filter), then >= 5 content words
"$PY" reproduce/temporal_kw.py --min-words 0
"$PY" reproduce/temporal_kw.py

# ---- Revised paper (v2): every number, table and figure, into outputs/paper_v2/ ----
# Needs, besides the CSVs, three inputs in outputs/ that are neither tracked nor checksummed:
#   - paper_v2/embedding_check/*.npy: Qwen3 embeddings (embedding_space_check.py writes them, GPU or MPS);
#   - paper_v2/calibration_appendixA/ge{5,10,20}/: the three calibration runs (OpenAI API);
#   - embeddings/{base,finetuned}_kmeans_global_pca/models/: the stored PCA and centroids (GPU server),
#     which centroids_v2.py checks it reproduces.
# rescore_ageing.py (API) is not rerun: since 2026-11-19 the ageing session is left out of the corpus.
for p in outputs/paper_v2/embedding_check/{base,finetuned}_{experts,assembly}.npy \
         outputs/paper_v2/calibration_appendixA/ge{5,10,20}/context_calibration_results.csv \
         outputs/embeddings/{base,finetuned}_kmeans_global_pca/models; do
    [ -e "$p" ] || { echo "Missing $p: see the comment above the v2 section of run_all.sh." >&2; exit 1; }
done
"$PY" reproduce/features_v2.py              # corrected lexical sophistication, broader evidential markers
"$PY" reproduce/centroids_v2.py             # witness centroids, embedding columns, Table 4
"$PY" reproduce/paper_v2_stats.py           # Tables 1, 3, 5, 6, 7 and the correlations (stats.json)
"$PY" reproduce/figures_v2.py               # Figs 3-4, Appendix B
"$PY" reproduce/temporal_plots_v2.py        # Appendix C (reads stats.json)
"$PY" reproduce/latent_figures_v2.py        # Figs 1-2 (UMAP)
"$PY" reproduce/embedding_space_check.py    # alternative embedding measures (reads centroids_v2)
"$PY" reproduce/reduction_compare_v2.py     # PCA vs UMAP
"$PY" reproduce/style_contrast_v2.py        # topic-subtracted embeddings
"$PY" reproduce/length_corrected_v2.py      # length-corrected scores
"$PY" reproduce/robustness_v2.py            # roles and content-word threshold
"$PY" reproduce/role_correlations_v2.py     # correlations by speaker role
"$PY" reproduce/expert_settings_v2.py       # assembly experts vs parliamentary witnesses
"$PY" reproduce/expert_length_v2.py         # length in expert speech
"$PY" reproduce/expert_gap_length_v2.py     # expert gap at equal length (exploratory)
"$PY" reproduce/irish_only_v2.py            # the two Irish assemblies alone
"$PY" reproduce/context_window_v2.py        # size of the judge's context
"$PY" reproduce/calibration_v2.py           # context calibration
"$PY" reproduce/rewrite_numbers.py > outputs/paper_v2/numbers.md

shasum -a 256 -c --quiet data/CHECKSUMS.sha256   # nothing above rewrote the data
echo "Done. Results are in outputs/."
