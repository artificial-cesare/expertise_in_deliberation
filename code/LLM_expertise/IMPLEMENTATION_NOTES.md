# Implementation Notes: Pseudocode → Python Mapping

This document maps your original pseudocode specification to the actual Python implementation.

## Architecture Overview

```
Pseudocode Section → Python File(s) → Function(s)
```

---

## Section 0: Constants and Configuration

**Pseudocode:**
```python
CONTEXT_BUDGETS = [0, 500, 1000, 1500, 2000]
SUBSET_SIZE = 200
RANDOM_SEED = 42
MODEL = "gpt-4o-mini"
ASSEMBLIES = ["8th_amendment", "gendereq", "scotland"]
```

**Implementation:**
📁 `context_size_calibration.py` (lines 25-31)

✓ Exactly as specified

---

## Section 1: Load and Prepare Data

**Pseudocode:**
```python
for each assembly in ASSEMBLIES:
    load CSV (columns: speaker, text, date)
    add column: row_index
    add column: token_count
    add column: content_word_count
    add column: is_substantial
```

**Implementation:**
📁 `context_size_calibration.py`

- **Function:** `load_assembly_data()` (lines 113-201)
- **Helper:** `count_tokens()` (lines 43-51)
- **Helper:** `count_content_words()` (lines 54-79)

**Key details:**
- Uses `tiktoken` for token counting (exact tokenization as model sees it)
- Reads `num_non_function_words` from CSV if available (from your existing pipeline)
- Fallback to `count_content_words()` if not present
- Returns list of `Utterance` dataclass objects (type-safe)

---

## Section 2: Stratified Sampling

**Pseudocode:**
```python
pool = all substantial utterances across all assemblies
for each assembly:
    n_from_this = proportional_to_corpus_size
    split assembly into thirds by row_index
    sample from each third
    oversample long utterances (>300 tokens)
```

**Implementation:**
📁 `context_size_calibration.py`

- **Function:** `stratified_sample()` (lines 204-285)

**Stratification strategy:**
1. **By assembly:** Proportional allocation based on substantial utterance count
2. **By temporal position:** Split each assembly into early/mid/late thirds
3. **By length:** 30% from long utterances (>300 tokens) if available

**Differences from pseudocode:**
- Uses `random.sample()` instead of manual shuffling (cleaner)
- Handles edge case where third has fewer utterances than requested
- Prints diagnostic summary

---

## Section 3: Build Context Windows

**Pseudocode:**
```python
function build_context(target_row_index, assembly_df, token_budget):
    if token_budget == 0:
        return "", 0
    walk backwards from target
    first utterance: always include
    subsequent: check budget, don't truncate
    reverse to chronological order
    return context_string, tokens_used
```

**Implementation:**
📁 `context_size_calibration.py`

- **Function:** `build_context()` (lines 288-352)
- **Baseline function:** `build_context_20_utterances()` (lines 355-389)

**Key details:**
- Walks backwards through utterances sorted by `row_index`
- Never truncates mid-utterance (respects token boundaries)
- Always includes at least one preceding utterance (even if over budget)
- Returns `(context_string, actual_tokens_used, n_utterances)`
- Context formatted as: `[row_idx] SPEAKER: text`

**Implementation note:**
- Uses pre-organized `utterances_by_file` dict for O(1) lookup instead of repeated filtering

---

## Section 4: Run LLM Scoring

**Pseudocode:**
```python
for each utterance in sampled_200:
    for budget in CONTEXT_BUDGETS + ["20_utterances"]:
        context = build_context(...)
        score, justification = call_llm_expertise(...)
        results.append({...})
        sleep(rate_limit_delay)
```

**Implementation:**
📁 `context_size_calibration.py`

- **Main function:** `run_calibration_experiment()` (lines 478-637)
- **LLM scoring:** `score_expertise_with_llm()` (lines 392-475)

**Key details:**
- **Randomization:** Shuffles utterance order AND condition order per utterance (prevents API caching)
- **Your existing prompt:** Copied verbatim from `llm_expert_extract_new.py` (no changes!)
- **Error handling:** Returns `None` for failed API calls, saves error message
- **Progress tracking:** Prints status per utterance and condition

**Results CSV columns:**
```
utterance_id, assembly, file_name, row_index, speaker,
target_text, target_token_count, target_content_words,
budget_condition, actual_tokens_used, n_utterances_in_context,
score, justification
```

---

## Section 5: Analysis — Score Stability

**Pseudocode:**
```python
# Reshape to utterances × conditions
pivot = pivot_table(...)

# ICC
icc = compute_ICC(pivot, model="two-way random, single measures")

# Pairwise Spearman
for each pair (c1, c2):
    spearman_rho = spearman_correlation(...)

# Mean Absolute Deviation
pivot["mad"] = mean_absolute_deviation_across_columns(...)
```

**Implementation:**
📁 `analyze_calibration.py`

- **Pivot:** `pivot_scores()` (lines 34-48)
- **ICC:** `compute_icc()` (lines 51-84) — uses `pingouin.intraclass_corr()`
- **Spearman:** `compute_pairwise_spearman()` (lines 87-118)
- **MAD:** `compute_mad_per_utterance()` (lines 133-146)

**ICC implementation:**
- Uses **ICC(2,1)** = two-way random, single measures
- "Two-way" = both utterances AND conditions are random samples
- "Single" = not averaged across raters
- Returns value + 95% CI

**Spearman matrix:**
- Computes all pairwise correlations (N×N matrix)
- Drops NaN pairs before correlation
- Returns as DataFrame for easy indexing

---

## Section 5b-5d: Systematic Bias and Variance

**Pseudocode:**
```python
# Friedman test (non-parametric repeated measures)
friedman_stat, p = friedman_test(...)

# Mean score by condition
for each condition:
    mean_score = mean(pivot[condition])

# Variance by condition
for each condition:
    score_variance = variance(pivot[condition])
```

**Implementation:**
📁 `analyze_calibration.py`

- **Friedman:** `friedman_test()` (lines 149-170)
- **Summary stats:** Computed in `generate_summary_report()` (lines 379-527)

**Friedman test:**
- Uses `scipy.stats.friedmanchisquare()`
- Null hypothesis: Median scores equal across conditions
- More appropriate than ANOVA for ordinal Likert data

---

## Section 6: Where Does Context Matter Most?

**Pseudocode:**
```python
# By utterance length
correlation = pearson(target_token_count, score_range)

# By position in transcript
for utterances where row_index < 20:
    compare scores at available_context vs budget

# By assembly
for each assembly:
    compute ICC within assembly
```

**Implementation:**
📁 `analyze_calibration.py`

- **Length analysis:** `analyze_context_sensitivity_by_length()` (lines 173-196)
- **By assembly:** `analyze_by_assembly()` (lines 199-235)

**Length analysis:**
- Computes `score_range` = max - min across conditions
- Pearson correlation: `score_range ~ token_count`
- Negative correlation → shorter utterances more context-sensitive

**Assembly-specific ICC:**
- Subsets pivot table by assembly
- Recomputes ICC for each subset
- Useful for detecting corpus-specific effects

---

## Section 7: Decision Rule

**Pseudocode:**
```python
# CRITERION 1: ICC >= 0.80
# CRITERION 2: No systematic bias (Friedman p >= .05)
# CRITERION 3: Spearman with 20-utt baseline >= 0.90
# CRITERION 4: Practical (~1000 tokens for humans)
```

**Implementation:**
📁 `analyze_calibration.py`

- **Function:** `generate_summary_report()` (lines 379-527)
- **Section:** "RECOMMENDATION" (lines 493-520)

**Decision logic:**
```python
if icc_val >= 0.80 and friedman_p >= 0.05:
    # High reliability + no bias
    # Choose budget closest to 20-utt median
elif icc_val >= 0.80:
    # High reliability but bias present
    # Choose based on mean score stability
else:
    # Moderate/low reliability
    # Recommend higher budget or report as confound
```

---

## Section 8: Outputs to Generate

**Pseudocode:**
```python
# Save:
# 1. results CSV
# 2. ICC value and CI
# 3. Pairwise Spearman matrix
# 4. Plot: mean score ± 95% CI by budget
# 5. Plot: ICC by assembly
# 6. Plot: score_range vs token_count
# 7. Plot: distribution of actual tokens used
# 8. List of high-MAD utterances
```

**Implementation:**
📁 `analyze_calibration.py`

| Output | Function | File |
|--------|----------|------|
| Results CSV | `run_calibration_experiment()` | `context_calibration_results.csv` |
| Summary report | `generate_summary_report()` | `summary_report.txt` |
| Mean scores plot | `plot_mean_scores_by_budget()` | `mean_score_by_budget.png` |
| Spearman heatmap | `plot_spearman_heatmap()` | `spearman_heatmap.png` |
| ICC by assembly | `plot_icc_by_assembly()` | `icc_by_assembly.png` |
| Length scatter | `plot_score_range_vs_length()` | `score_range_vs_length.png` |
| Token distribution | `plot_actual_tokens_distribution()` | `actual_tokens_distribution.png` |
| High-MAD utterances | `identify_high_mad_utterances()` | `high_mad_utterances_threshold_1.0.csv` |

**All plots:**
- 300 DPI publication quality
- Seaborn "whitegrid" style
- Saved as PNG with tight bounding boxes

---

## Deviations from Pseudocode

### Minor Enhancements

1. **Type safety:** Used `@dataclass` for `Utterance` (clearer than dicts)

2. **Error handling:** LLM scoring can return `None` (failed API call), tracked separately

3. **Diagnostic output:** Added progress bars and intermediate statistics during execution

4. **Modular design:** Split into two scripts (`calibration` + `analyze`) instead of monolithic

5. **Reproducibility:** All random operations seeded, conditions randomized per-utterance

### Simplifications

1. **No position-based analysis:** Didn't implement special handling for `row_index < 20` (can add if needed)

2. **Fixed MAD threshold:** Hard-coded at 1.0 (could make configurable)

3. **Single ICC type:** Only ICC(2,1), not ICC(1,1) or ICC(3,1)

### Extensions

1. **Shell wrapper:** `run_calibration.sh` runs both steps in sequence

2. **Comprehensive README:** Full usage guide with examples

3. **Dependencies file:** `requirements_calibration.txt`

---

## Testing Checklist

Before running on full dataset, verify:

- [ ] OpenAI API key is set: `echo $OPENAI_API_KEY`
- [ ] Dependencies installed: `pip install -r requirements_calibration.txt`
- [ ] Data directory exists: `ls data/assemblies/processed_files/8th_amendment`
- [ ] CSV files have required columns: `speaker, text, num_non_function_words`

**Test run (small sample):**

Edit `SUBSET_SIZE = 10` in `context_size_calibration.py`, then:

```bash
python context_size_calibration.py --delay 0.2
python analyze_calibration.py --results-file ./results/context_calibration/context_calibration_results.csv
```

Should complete in ~2 minutes and cost ~$0.05.

---

## Performance Notes

**Expected metrics (full run):**

- **API calls:** 200 utterances × 6 conditions = 1,200 calls
- **Input tokens:** ~500 avg/call × 1,200 = 600K tokens
- **Output tokens:** ~50 avg/call × 1,200 = 60K tokens
- **Cost (gpt-4o-mini):**
  - Input: 600K × $0.15/1M = $0.09
  - Output: 60K × $0.60/1M = $0.04
  - **Total: ~$0.13 USD**
- **Runtime:** 15-20 minutes (with 0.5s delay)

**Optimization options:**

1. Reduce delay (`--delay 0.2`) if no rate limits → 8-10 minutes
2. Run in parallel (split by assembly) → 5-7 minutes (requires code changes)
3. Use batched requests (OpenAI Batch API) → 50% cost reduction (async, slower)

---

## Citation & Acknowledgments

This implementation follows psychometric best practices:

- **ICC(2,1):** Shrout & Fleiss (1979) "Intraclass correlations"
- **Stratified sampling:** Ensures representativeness (Cochran 1977)
- **Friedman test:** Non-parametric alternative to repeated measures ANOVA (appropriate for Likert scales)
- **Spearman ρ:** Rank-based correlation (robust to outliers, suitable for ordinal data)

If you use this in a paper, mention:

> Context size was calibrated using a stratified sample (N=200) across token budgets (0, 500, 1000, 1500, 2000 tokens). Inter-rater reliability was assessed using intraclass correlation coefficient ICC(2,1), and systematic bias was tested using the Friedman test.
