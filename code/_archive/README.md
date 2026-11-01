# Archive

Files moved here during the 2026-07-17 repository review. Nothing in this
directory is used by the current pipeline; everything is kept for provenance
only. Do not import from or run anything in here.

## summarytables/

- **`summarize_assemblies.py`** — superseded by `summarytables/final_stats.py`.
  Both scripts wrote `assemblies_statistics.txt` and `experts_statistics.txt`
  to the same output directory and clobbered each other's outputs.
  `final_stats.py` (concise stats + LaTeX tables) is the canonical version.
- **`assemblies_summary.txt`** — orphaned output; no current script produces a
  file with this name (it predates the summarize/final_stats split).

## validation/

- **`inter_rater_reliability_results.json`** — output of
  `calculate_inter_rater_reliability.py`, which was deleted in commit
  `5297f02` ("Delete superseded validation scripts"). The canonical
  reliability results are `validation/comprehensive_reliability_results.json`
  and `validation/kappa_results.json`.
- **`extreme_scores_kappa_results.json`** — orphaned output; the current
  `kappa_filtered_new.py` writes to `kappa_results.json` by default and no
  script produces this filename anymore.

## preprocessing/

- **`preprocess_llm.py`** — first two-pass OpenAI preprocessing script, superseded by
  `preprocessing/new_preprocess.py`. Kept because it is the version described in the
  paper's Preprocessing section.
