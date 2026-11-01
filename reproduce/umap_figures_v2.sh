#!/usr/bin/env bash
# Figures 1-2 (revised): UMAP plots of expert and assembly utterances, without the ageing-population transcript.
# Reuses code/embeddings/plotting.py with the saved PCA, UMAP and centroids. The UMAP model was pickled under
# Python 3.10, so this needs an environment matching qwen3_requirements.txt (e.g. conda env "umap310").
# Run from the repo root:  PY=/opt/miniconda3/envs/umap310/bin/python bash reproduce/umap_figures_v2.sh
set -euo pipefail
PY="${PY:-python}"
ROOT=outputs/paper_v2/umap_root            # mirrors the server's ~/thesis layout; plots land here
mkdir -p "$ROOT/data/assemblies" "$ROOT/base_kmeans_global_pca" "$ROOT/finetuned_kmeans_global_pca"
ln -sfn "$PWD/data/assemblies/processed_files" "$ROOT/data/assemblies/processed_files"
ln -sfn "$PWD/models/qwen3_expert_contrastive_final" "$ROOT/qwen3_expert_contrastive_final"
for m in base finetuned; do ln -sfn "$PWD/outputs/embeddings/${m}_kmeans_global_pca/models" "$ROOT/${m}_kmeans_global_pca/models"; done
THESIS_ROOT="$PWD/$ROOT" DEVICE="${DEVICE:-cpu}" PLOT_LIMITS=True "$PY" code/embeddings/plotting.py
ls "$ROOT"/*_kmeans_global_pca/plots_visual_update_limitTrue/
