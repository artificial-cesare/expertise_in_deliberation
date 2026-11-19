# Expertise in Deliberation

Code for *How wise are citizens' assemblies? An automated measure of manifested expertise*.
It measures the expertise that participants express in citizens'-assembly transcripts in two ways,
(1) embedding proximity to UK Parliament expert testimony and (2) GPT-4o-mini scoring on a 1–5 scale,
and compares both with 14 handcrafted linguistic features and the ArgQA argument-quality score.

Assemblies: Irish Citizens' Assembly on the Eighth Amendment ("Constitutional Reform"), Irish
Citizens' Assembly on Gender Equality, and Citizens' Assembly of Scotland ("Public Policy").

## Layout

The repo root is the project root: run every script from here.

```
expertise_in_deliberation/
├── code/                      # pipeline (tracked; see code/README.md for every script)
│   ├── preprocessing/         # raw PDF/RTF/HTML -> date,speaker,text CSVs (OpenAI)
│   ├── handcrafted_features/  # 14 linguistic features, jargon lexicon
│   ├── LLM_expertise/         # GPT-4o-mini expertise scores + context-size calibration
│   ├── ArgQualityAdapters/    # ArgQA overall score (adapted from Falk & Lapesa 2023)
│   ├── embeddings/            # Qwen3 fine-tuning, k-means centroids, UMAP plots (GPU)
│   ├── correlation_analysis/  # correlations, temporal analysis
│   ├── regression_analysis/
│   ├── validation/            # human annotation sampling and Cohen's kappa
│   ├── summarytables/         # corpus statistics and LaTeX tables
│   ├── shared/                # topic mappings used across scripts
│   └── _archive/              # superseded scripts, kept for traceability
├── reproduce/                 # one-command rebuild of the paper's CPU-only results (tracked)
├── tests/                     # toy checks for reproduce/ (tracked)
├── records/                   # small text records of runs that need a GPU or the API (tracked)
├── data/                      # local only, except the public sources listed below
├── models/                    # fine-tuned Qwen3 model, best epoch only (local only, ~1 GB)
├── outputs/                   # everything the scripts generate (local only)
├── llm_exp_requirements.txt   # CPU / OpenAI environment
└── qwen3_requirements.txt     # GPU / embedding environment
```

### What is in git and what is not

| Path | In git | Notes |
|---|---|---|
| `code/`, `reproduce/`, `tests/`, `records/` | yes | |
| `data/vocab/` | yes | UK Parliament hearings used to build the jargon lexicon |
| `data/assemblies/raw/{8th_amendment,gendereq,experts}/` | yes | public Irish transcripts (PDF) and expert hearings (HTML) |
| `data/CHECKSUMS.sha256` | yes | hashes of the processed CSVs |
| `data/assemblies/raw/scotland/` | **never** | UK Data Archive licence (SN 8992) |
| `data/assemblies/processed_files/`, `new_feats/` | no | CSVs with all features and scores; Scotland files are derived from licensed data |
| `data/validation/` | no | annotation sample and ratings; quotes Scotland transcripts |
| `data/experts/`, `data/mrc-psycholinguistics-master/`, `data/subjectivity_clues_hltemnlp05/` | no | expert CSVs and third-party lexicons |
| `models/`, `outputs/` | no | |

Local data layout the scripts expect:

```
data/assemblies/processed_files/{8th_amendment,gendereq,scotland}/*_speeches.csv
data/assemblies/processed_files/experts/{constitution,future,gender}/*_speeches.csv
data/assemblies/raw/scotland/8992int0NN.rtf
data/validation/{sampled_speeches.json,annotations_csv/,ratings_matrix.csv}
data/mrc-psycholinguistics-master/mrc2.db
data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff
models/qwen3_expert_contrastive_final/best_model/
```

Check that the processed CSVs are the version behind the paper with
`shasum -a 256 -c data/CHECKSUMS.sha256`.

## Data sources

- Irish assemblies: <https://citizensassembly.ie> (Meetings pages).
- Citizens' Assembly of Scotland: UK Data Archive, <http://doi.org/10.5255/UKDA-SN-8992-1>, access on request.
- Expert testimony: <https://committees.parliament.uk/publications/oral-evidence/>.
- MRC Psycholinguistic Database and the MPQA subjectivity lexicon (Wilson et al., 2005), for the newer feature set only.

## Setup

```bash
conda create -n CL python=3.13 && conda activate CL
pip install -r llm_exp_requirements.txt
python -m spacy download en_core_web_sm
export OPENAI_API_KEY=...            # only for preprocessing and LLM scoring
```

The embedding scripts need a GPU and `qwen3_requirements.txt`. They still carry the absolute
paths of the GPU server (`/home/u933585/thesis/...`); edit the paths at the top of each script,
or set the environment variables in `final_finetune.py`, before running them.

## Reproducing the paper

Everything that needs only the processed CSVs (no GPU, no API key) rebuilds with one command:

```bash
bash reproduce/run_all.sh        # PY=/path/to/python picks the environment
```

| Paper result | Script | Rebuilt from the CSVs? |
|---|---|---|
| Tables 1, 3, 6 (corpus, features, scores) | `code/summarytables/final_stats.py` | yes, identical |
| Table 4 (k-means accuracy) | `code/embeddings/k_means_exp.py` | needs the GPU; recorded in `records/kmeans_*.txt` |
| Table 5 (kappa) | `code/validation/kappa_filtered_new.py` | yes, identical |
| Figs 1–2 (UMAP) | `code/embeddings/plotting.py` | needs the GPU and `models/` |
| Figs 3–4, Appendix B (correlations) | `reproduce/correlations.py` | yes, pixel-identical with `--scotland-llm drop --legacy-gt5` |
| Appendix C (temporal plots) | `reproduce/correlations.py` | yes, up to file read order |
| Results "Time Difference Analysis", Table 7 | `reproduce/temporal_kw.py` | yes, all values, with `--min-words 0` |
| Context calibration (Methods) | `code/LLM_expertise/context_size_calibration.py` | needs the API; the surviving run is in `records/context_calibration_summary.txt` |

`reproduce/paper_corr.py` is a frozen copy of the correlation code the figures were made with.
`code/correlation_analysis/corr.py` is the later, refactored version and scales features differently.
The features in the CSVs were computed by `code/handcrafted_features/new_feats_extract.py`;
`feats_extract_final.py` is the newer extractor and writes the `new_*` and `mrc_*` columns instead.

### Points to know before reusing the numbers

- **The published correlation figures leave out Scotland's LLM scores.** All 17 values in the
  LLM-score row of Fig 3 match a run where Scotland's scores are missing, so those cells use the two
  Irish assemblies only (n = 537), while all other cells use all three (n = 3,183). Scotland's CSVs
  do contain LLM scores now (scored for all three assemblies on 2025-10-31, `outputs/logs/`), so the
  figure was most likely made from an earlier copy of the CSVs. `--scotland-llm drop` reproduces the figures;
  `keep` uses all three (for example, LLM score vs. normalised length goes from r = .51 to .32;
  vs. ArgQA it stays at .67–.68).
- **Content-word threshold: ≥ 5 everywhere** (3,607 substantial utterances): tables, correlations,
  temporal tests, embeddings, ArgQA and LLM scoring, and the context calibration. The published
  (Overleaf) correlation figures used > 5 (3,183) and the published temporal tests no filter;
  `correlations.py --legacy-gt5` and `temporal_kw.py --min-words 0` rebuild those.
- **The LLM scores use a fixed 20-utterance context window** (`llm_expert_extract_new.py`), not a
  token budget. `llm_utils.SYSTEM_PROMPTS["paper"]` is the Appendix A prompt that produced the scores
  and is the default; `"lived_experience"` is a later revision. The first calibration run
  (`outputs/paper_v2/calibration/`) used the later prompt; `outputs/paper_v2/calibration_appendixA/`
  repeats it with the Appendix A prompt at ≥ 5, 10 and 20 content words.
- **Duplicated expert hearing.** `experts/gender/committees.parliament.uk_speeches.csv` was processed from an HTML
  file identical to `committees.parliament.uk5.html`. `reproduce/centroids_v2.py` refits the PCA and k-means without it
  (it reproduces the stored models to 1e-6 when the copy is kept) and writes `outputs/paper_v2/embeddings_v2.csv`;
  the fine-tuned model was trained with the copy included.
- **Expert reference = the witnesses' turns.** The hearing transcripts also hold the turns of the committee members who
  question the witnesses and of the chair. `reproduce/hearing_roles.py` labels every hearing speaker (witnesses from
  the list at the head of each official report) and `centroids_v2.py`, `embedding_space_check.py`,
  `latent_figures_v2.py` and `reduction_compare_v2.py` use only the witnesses' turns (822 of the 1,563 deduplicated
  utterances); the fine-tuned model was trained on all turns. Results with all turns are kept in
  `outputs/paper_v2/_all_hearing_turns_2026-11-12/`. `reproduce/expert_settings_v2.py` compares assembly experts,
  citizens, witnesses and committee members (LLM score, ArgQA, disfluencies).
- **Length in expert speech** (`reproduce/expert_length_v2.py`, paper table `expert_length`): LLM score by length
  quarter for the experts and citizens of the Eighth Amendment session, correlations with log length by half,
  the low-scoring expert turns and embedding distance by quarter; then citizens of all assemblies against assembly
  experts and parliamentary witnesses (common token bins, slopes, interaction). Writes `outputs/paper_v2/expert_length_v2.json`.
  `reproduce/expert_gap_length_v2.py` (exploratory) tests whether experts speak longer and score higher at equal length
  (Eighth Amendment session, hearings, pooled): OLS and ordered logit with a length spline, length matching, a stratified
  permutation test and per-speaker residuals. Writes `outputs/paper_v2/expert_gap_length_v2.json`.
- **Context size** (`reproduce/context_window_v2.py`): tokens (gpt-4o-mini tokenizer) in the judge's 20-utterance window
  and in the full transcripts, for the paper's context-calibration paragraph and Limitations.
  Writes `outputs/paper_v2/context_window_v2.json`.
- **Topic-subtracted embeddings** (`reproduce/style_contrast_v2.py`, exploratory): each utterance minus the
  speaker-balanced mean of its own hearing or transcript, in the full space, then PCA-100 and k-means on the witnesses
  as in the main analysis; with topic-removal, validity and topic-specificity checks.
- **Ageing session rescored.** It was first scored with the Eighth Amendment topic string;
  `reproduce/rescore_ageing.py` rescored it with its own topic into `outputs/paper_v2/llm_v2.csv`.
  `paper_v2_stats.load()` merges both files over the stored CSV columns (originals kept as `*_v1`); the data CSVs are unchanged.
- **Normalised features are scaled per transcript file**, not per corpus
  (`compute_corpus_statistics` runs once per CSV in `new_feats_extract.py`).
- **The embedding columns** (`distance_*`, `similarity_*`) were written on the GPU server by a
  variant of `code/embeddings/final_centroids.py`; its fine-tuned model path
  (`SIMCSE/qwen3_expert_contrastive_v2`) differs from the one `k_means_exp.py` and `plotting.py`
  use (`qwen3_expert_contrastive_final`, the model kept in `models/`).
- `llm_expert_extract_new.py`, `clean_csvs.py`, the ArgQA batch script and the embedding scripts
  rewrite CSVs in place. Run them on a copy.

## License

MIT, see `LICENSE`. `code/ArgQualityAdapters/` is adapted from
<https://github.com/Blubberli/ArgQualityAdapters>.
