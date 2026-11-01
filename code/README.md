# Thesis Analysis Code

Per-script reference for the pipeline. The root `README.md` has the directory
map, what is tracked, setup, and how to reproduce the paper.

Run everything from the **repo root**. Scripts self-manage `sys.path`; default
paths are relative to the repo root.

## Pipeline

```
raw transcripts (data/assemblies/raw/)
        │
[1] preprocessing/new_preprocess.py ............ OpenAI cleaning + regex parsing
        │      → data/assemblies/new_processed_files/ (committed set: processed_files/)
        ▼
[2] handcrafted_features/feats_extract_final.py . linguistic features (spaCy/MRC/MPQA/domain terms)
        │      → data/assemblies/new_feats/  (paper features: new_feats_extract.py)
        ▼
[3] LLM_expertise/llm_expert_extract_new.py ..... LLM expertise scores (1–5) ⚠ edits CSVs IN PLACE
        │
        ├──▶ [4] correlation_analysis/  (corr.py, stats_time_final.py, detailed_temporal_analysis.py)
        ├──▶ [5] regression_analysis/regression_script.py
        ├──▶ [6] validation/           (sampling → kappa / comprehensive reliability)
        └──▶ [7] summarytables/final_stats.py
```

Suffixes like `_new` / `_final` are historical — each script below is the
canonical (and only) version of its stage.

## Scripts

**`preprocessing/new_preprocess.py`** — cleans raw PDF/RTF/HTML transcripts
into `date,speaker,text` CSVs via gpt-4o-mini (**needs `OPENAI_API_KEY`**).
Writes to `new_processed_files/`, never overwriting the committed set.
Supports `--dry-run`, `--limit N`, `--folders`.

**`handcrafted_features/`**
- `new_feats_extract.py` — the extractor behind the feature columns in
  `processed_files/` and the paper (per-file normalisation).
- `feats_extract_final.py` — newer extractor (`new_*` and `mrc_*` columns); extracts all linguistic features
  (`--processed_dir`, `--output_dir`, `--domain_terms`, ...).
- `dom_terms_final_exp.py` — builds the domain-jargon lexicon from
  `data/vocab/`. ⚠ default `--output` overwrites the checked-in
  `domain_terms_clean_exp.json`; pass an explicit path when experimenting.
- `mrc_lookup.py`, `filter_function.py` — imported modules.

**`LLM_expertise/`**
- `llm_expert_extract_new.py` — LLM expertise scoring (**OpenAI**).
  ⚠ **rewrites input CSVs in place** (`--base-dir`); use a copy unless
  updating the dataset. `--force-rescore` to redo scored rows.
- `context_size_calibration.py` (**OpenAI**) → `analyze_calibration.py`;
  `verify_setup.py` pre-flight check; `llm_utils.py` shared module.
  Docs: `QUICKSTART.md`, `README_CALIBRATION.md`, `IMPLEMENTATION_NOTES.md`.

**`correlation_analysis/`**
- `corr.py` — correlation heatmaps/JSONs across threshold × normalization ×
  Scotland combinations (`--base-dir`, `--output-dir`). Final exports live in
  top-level `results/`.
- `stats_time_final.py` — time-window stats + Kruskal-Wallis + LaTeX
  (`--input`, `--output`).
- `detailed_temporal_analysis.py` — per-file temporal plots (`--base-dir`,
  `--output-dir`).
- `correlation_utils.py` — imported module.

**`regression_analysis/regression_script.py`** — regression of features on
`expertise_llm_score` (`--dataset`, `--model_type` required).

**`validation/`**
- `extract_speeches_context.py` — samples speeches for annotation
  (`--data-dir`, `--output`).
- `kappa_filtered_new.py` — Cohen's κ, `--extreme-only` variant.
- `comprehensive_reliability.py` — full reliability suite (`--json-path`,
  `--csv-dir`, `--output`, `--plots-dir`).
- `io_utils.py` — imported module. Inputs (`sampled_speeches.json`,
  `annotations_csv/`) live in `data/validation/` (not in git: they quote
  Scotland transcripts); the result JSONs are committed here, the confusion
  matrices go to `outputs/validation/`.

**`summarytables/`**
- `final_stats.py` — corpus statistics + LaTeX tables (`--output_dir`,
  default `outputs/corpus_statistics/`).
- `clean_csvs.py` — ⚠ column pruner that **rewrites CSVs in place**; always
  `--dry_run` first.

**`shared/topic_maps.py`** — canonical topic mappings (imported by 9 scripts).
**`export_requirements.py`** — secret-filtered `pip freeze` of the running env.

**`embeddings/`** (GPU) — `final_finetune.py` (contrastive fine-tuning of
Qwen3-Embedding-0.6B), `k_means_exp.py` (k selection, Table 4), `final_centroids.py`
(distance/similarity columns), `plotting.py` (UMAP figures), `run_analysis_simple.sh`.
They still hold the GPU server's absolute paths.

**`ArgQualityAdapters/code/`** — `inference_simple.py` and `run_batch_inference.sh`
write the ArgQA `overall_score` column (in place).

Requirements files (repo root): `llm_exp_requirements.txt` (OpenAI/feature pipeline),
`qwen3_requirements.txt` (Qwen/GPU stack), and
`LLM_expertise/requirements_calibration.txt` (calibration only).

## Gotchas

1. `llm_expert_extract_new.py` and `clean_csvs.py` mutate their input CSVs.
2. The CSVs in `data/assemblies/processed_files/` carry the feature/score
   columns behind the paper, from `new_feats_extract.py`; `feats_extract_final.py`
   computes different values.
3. `corr.py`, `stats_time_final.py` and `regression_script.py` default to
   `data/assemblies/new_feats/`; pass an explicit features directory to use
   `processed_files/`. The paper's figures come from `reproduce/paper_corr.py`.
4. `regression_script.py` imports `scale_features_for_correlation` from
   `corr.py`.

`_archive/` holds superseded scripts and orphaned outputs (see its README),
including the first preprocessing script, `preprocess_llm.py`.
