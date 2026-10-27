# Context Size Calibration for Expertise Scoring

This directory contains tools for testing how sensitive LLM expertise scores are to the amount of preceding context in deliberative assembly transcripts.

## Overview

The calibration experiment addresses a key methodological question: **How much context should we provide to the LLM when scoring expertise?**

Your current approach uses a fixed 20-utterance window, which varies in token count depending on utterance length. This experiment tests:

1. **Score stability**: Do scores remain consistent across different context sizes?
2. **Systematic bias**: Does more/less context systematically increase/decrease scores?
3. **Context sensitivity**: Which utterances are most affected by context size?

## Files

```
context_size_calibration.py  - Main experiment script (run LLM scoring)
analyze_calibration.py       - Analysis script (compute statistics & plots)
requirements_calibration.txt - Python dependencies
README_CALIBRATION.md        - This file
```

## Installation

Install required dependencies:

```bash
pip install -r requirements_calibration.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Step 1: Run the Calibration Experiment

This will:
- Sample 200 utterances stratified by assembly and temporal position
- Score each utterance with 6 different context conditions:
  - 0 tokens (no context)
  - 500 tokens
  - 1000 tokens
  - 1500 tokens
  - 2000 tokens
  - 20 utterances (your current baseline)
- Save all scores to CSV

```bash
python context_size_calibration.py \
    --base-dir ./data/assemblies/processed_files \
    --output-dir ./results/context_calibration \
    --model gpt-4o-mini \
    --delay 0.5
```

**Arguments:**
- `--base-dir`: Directory containing processed CSV files (default: `./data/assemblies/processed_files`)
- `--output-dir`: Where to save results (default: `./results/context_calibration`)
- `--model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--delay`: Delay between API calls in seconds (default: 0.5)

**Expected runtime:** ~15-20 minutes
**Expected cost:** <$1 USD with gpt-4o-mini

**Output:**
- `context_calibration_results.csv`: All scores (200 utterances × 6 conditions = 1200 rows)

### Step 2: Analyze Results

This will compute statistical measures and generate visualizations:

```bash
python analyze_calibration.py \
    --results-file ./results/context_calibration/context_calibration_results.csv \
    --output-dir ./results/context_calibration
```

**Arguments:**
- `--results-file`: Path to the results CSV from Step 1 (required)
- `--output-dir`: Where to save plots and reports (default: same as results file)

**Outputs:**

**Statistical Results:**
- `summary_report.txt`: Full text summary with recommendations

**Visualizations:**
- `mean_score_by_budget.png`: Mean scores ± 95% CI across conditions (tests for systematic bias)
- `spearman_heatmap.png`: Pairwise rank correlations between conditions
- `icc_by_assembly.png`: Reliability scores for each assembly
- `score_range_vs_length.png`: Context sensitivity by utterance length
- `actual_tokens_distribution.png`: Distribution of actual tokens used per budget

**High-MAD Utterances:**
- `high_mad_utterances_threshold_1.0.csv`: Utterances with high context sensitivity (MAD ≥ 1.0)

## Interpreting Results

### 1. Overall Reliability: ICC (Intraclass Correlation Coefficient)

The ICC(2,1) measures how consistent scores are across context conditions.

**Interpretation:**
- **ICC ≥ 0.80**: Excellent reliability → context size doesn't matter much
- **0.70 ≤ ICC < 0.80**: Good reliability → minor context effects
- **0.60 ≤ ICC < 0.70**: Moderate reliability → context matters
- **ICC < 0.60**: Poor reliability → scores highly context-dependent

### 2. Systematic Bias: Friedman Test

Tests if mean scores differ significantly across conditions.

**Interpretation:**
- **p ≥ 0.05**: No systematic bias → safe to use any budget
- **p < 0.05**: Systematic bias present → check `mean_score_by_budget.png` for trends

### 3. Backward Compatibility: Spearman Correlations

Compares each token budget to your current 20-utterance baseline.

**Interpretation:**
- **ρ ≥ 0.90**: Very high agreement → safe to switch
- **0.80 ≤ ρ < 0.90**: Good agreement → mostly compatible
- **ρ < 0.80**: Low agreement → switching would change results substantially

### 4. Context Sensitivity by Length

Correlation between utterance length and score variability.

**Interpretation:**
- **r < 0, p < 0.05**: Shorter utterances are more context-sensitive (expected)
- **r ≈ 0**: Length doesn't affect context sensitivity
- **r > 0, p < 0.05**: Longer utterances are more context-sensitive (unexpected)

## Decision Criteria (from pseudocode)

Choose your context budget based on:

1. **ICC ≥ 0.80** across conditions
   - If true: Context doesn't matter much, any budget works
   - If false: Use higher budget (1500-2000) or report as potential confound

2. **No systematic bias** (Friedman p ≥ 0.05)
   - If true: Mean scores stable, safe to proceed
   - If false: Investigate trends and discuss in methods section

3. **High correlation with 20-utterance baseline** (ρ ≥ 0.90)
   - Ensures smooth transition from your current approach

4. **Practical feasibility**
   - ~1000 tokens ≈ 1-2 pages of text
   - Should align with what human annotators can reasonably read

**Recommended approach:**
If multiple budgets satisfy all criteria, choose the one closest to your 20-utterance baseline's median token count. This minimizes disruption to your existing results.

## Example Workflow

```bash
# 1. Run experiment (20 minutes, <$1)
python context_size_calibration.py

# 2. Analyze results
python analyze_calibration.py \
    --results-file ./results/context_calibration/context_calibration_results.csv

# 3. Review summary report
cat ./results/context_calibration/summary_report.txt

# 4. Examine plots
open ./results/context_calibration/*.png

# 5. Inspect high-MAD utterances (optional)
head ./results/context_calibration/high_mad_utterances_threshold_1.0.csv
```

## What to Report in Your Paper

Based on your results, include a methods subsection like:

> **Context window calibration.** To determine the appropriate amount of context for expertise scoring, we conducted a calibration study on a stratified sample of 200 utterances. Each utterance was scored under six context conditions (0, 500, 1000, 1500, 2000 tokens, and a 20-utterance baseline). Scores showed [high/moderate] stability across conditions (ICC(2,1) = X.XX, 95% CI [X.XX, X.XX]), with [no significant / a small] systematic effect of context size on mean scores (Friedman χ² = X.XX, p = X.XX). Based on this analysis, we fixed the context window at [N] tokens for all subsequent scoring, which corresponds to approximately [M] preceding utterances and aligns with [rationale].

Include `spearman_heatmap.png` and `mean_score_by_budget.png` in supplementary materials.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'pingouin'`
- **Fix:** `pip install pingouin`

**Issue:** `OpenAI API rate limit errors`
- **Fix:** Increase `--delay` parameter (e.g., `--delay 1.0`)

**Issue:** Very low ICC (<0.5)
- **Interpretation:** Scores are highly context-dependent
- **Action:**
  1. Check `high_mad_utterances_threshold_1.0.csv` for qualitative patterns
  2. Consider using higher context budgets (1500-2000)
  3. Report context as a potential confound in your analysis

**Issue:** Systematic bias detected (Friedman p < 0.05)
- **Interpretation:** Context size affects mean scores
- **Action:**
  1. Check `mean_score_by_budget.png` for trend direction
  2. If monotonic (more context → higher scores), discuss in limitations
  3. If non-monotonic, may indicate optimal context budget

## Advanced Usage

### Modify Context Budgets

Edit `CONTEXT_BUDGETS` in `context_size_calibration.py`:

```python
CONTEXT_BUDGETS = [0, 250, 500, 750, 1000, 1500, 2000]  # Finer granularity
```

### Change Sample Size

Edit `SUBSET_SIZE`:

```python
SUBSET_SIZE = 300  # Larger sample (takes longer, costs more)
```

### Adjust Stratification

Modify `stratified_sample()` function to change:
- Proportional vs. equal allocation across assemblies
- Oversampling of long utterances
- Temporal stratification (currently early/mid/late thirds)

### Test Different Models

```bash
# Test with more capable model (more expensive)
python context_size_calibration.py --model gpt-4o

# Test with cheaper model
python context_size_calibration.py --model gpt-3.5-turbo
```

## Contact & Citation

If you use this calibration approach, consider citing the methodology:

- Stratified sampling ensures representativeness across assemblies and temporal positions
- Token-based budgets (vs. utterance-based) control for length variation
- ICC(2,1) as primary reliability metric follows psychometric best practices
- Friedman test for systematic bias (non-parametric, appropriate for ordinal Likert scales)

## Changelog

- **v1.0** (2026-02-15): Initial implementation
  - Stratified sampling by assembly and position
  - 6 context conditions (0, 500, 1000, 1500, 2000 tokens + 20-utterance baseline)
  - ICC, Spearman, Friedman tests
  - 5 visualization types
  - Automated decision criteria
