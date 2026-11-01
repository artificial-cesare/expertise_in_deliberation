# Quick Start Guide: Context Size Calibration

**Goal:** Determine the optimal context window size for expertise scoring.

**Time:** 15-20 minutes | **Cost:** <$1 USD

---

## Step 0: Verify Setup (5 minutes)

```bash
cd code/LLM_expertise

# Install dependencies
pip install -r requirements_calibration.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Verify everything is ready
python verify_setup.py
```

If all checks pass ✓, proceed to Step 1.

If any checks fail ✗, fix the issues indicated (the script will tell you what to do).

---

## Step 1: Run Complete Workflow (15 minutes)

**Option A: Automated (recommended)**

```bash
./run_calibration.sh
```

This runs both experiment and analysis automatically.

**Option B: Manual (step-by-step)**

```bash
# Run experiment
python context_size_calibration.py

# Run analysis
python analyze_calibration.py \
    --results-file ./results/context_calibration/context_calibration_results.csv
```

---

## Step 2: Review Results (5 minutes)

```bash
cd ../../results/context_calibration/

# Read summary report (start here!)
cat summary_report.txt

# View plots
open *.png
```

**Key file:** `summary_report.txt`

Look for:
1. **ICC value** (should be ≥ 0.80 for good reliability)
2. **Friedman test p-value** (should be ≥ 0.05 for no systematic bias)
3. **Recommendation section** (tells you what context budget to use)

---

## Step 3: Decide on Context Budget

Based on the recommendation in `summary_report.txt`:

**If ICC ≥ 0.80 and Friedman p ≥ 0.05:**
→ Context size doesn't matter much. Use the recommended budget (likely closest to your current 20-utterance approach).

**If ICC < 0.80:**
→ Scores are context-sensitive. Consider:
- Using a higher budget (1500-2000 tokens)
- Reporting context as a potential confound
- Examining `high_mad_utterances_threshold_1.0.csv` for qualitative patterns

---

## What You'll Get

### Files (in `results/context_calibration/`)

**Data:**
- `context_calibration_results.csv` — All 1,200 scores (200 utterances × 6 conditions)
- `high_mad_utterances_threshold_1.0.csv` — Context-sensitive utterances for qualitative review

**Report:**
- `summary_report.txt` — Statistical analysis + recommendation (read this first!)

**Plots:**
- `mean_score_by_budget.png` — Tests for systematic bias
- `spearman_heatmap.png` — Correlation matrix (how similar are conditions?)
- `icc_by_assembly.png` — Reliability by assembly
- `score_range_vs_length.png` — Do short utterances vary more with context?
- `actual_tokens_distribution.png` — Variance in actual token counts per budget

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'pingouin'"**
```bash
pip install pingouin
```

**"Error: OPENAI_API_KEY environment variable not set"**
```bash
export OPENAI_API_KEY="your-key-here"
```

**"No CSV files found!"**
- Check that `data/assemblies/processed_files/` exists
- Verify assemblies have CSV files: `ls data/assemblies/processed_files/8th_amendment/`

**API rate limit errors**
```bash
# Increase delay between calls
python context_size_calibration.py --delay 1.0
```

---

## For More Details

- **Full documentation:** [README_CALIBRATION.md](README_CALIBRATION.md)
- **Implementation notes:** [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- **Code comments:** See inline documentation in `.py` files

---

## Expected Output Summary

**Console output (abbreviated):**
```
==============================================================================
CONTEXT SIZE CALIBRATION EXPERIMENT
==============================================================================
Model: gpt-4o-mini
Context budgets: [0, 500, 1000, 1500, 2000]
Sample size: 200
...

[1/5] Loading assembly data...
[2/5] Performing stratified sampling...
[3/5] Organizing utterances by file...
[4/5] Running LLM scoring...
  Scoring utterance 1/200...
  ...
[5/5] Saving results...

==============================================================================
ANALYSIS
==============================================================================
[1/8] Loading results...
[2/8] Reshaping data...
[3/8] Computing ICC...
  ICC(2,1) = 0.XXX [95% CI: 0.XXX, 0.XXX]
[4/8] Computing pairwise Spearman correlations...
[5/8] Running Friedman test...
  Friedman χ² = X.XX, p = 0.XXX
[6/8] Computing assembly-specific ICC...
[7/8] Identifying context-sensitive utterances...
[8/8] Generating visualizations...

RECOMMENDATION
------------------------------------------------------------------------------
✓ High reliability (ICC >= 0.80)
✓ No systematic bias (Friedman p >= 0.05)

RECOMMENDED CONTEXT BUDGET: XXXX tokens
```

---

## What to Report in Your Thesis

After running the calibration, add this to your Methods section:

> **Context window calibration.** To determine the appropriate amount of context for expertise scoring, we conducted a calibration study on a stratified sample of 200 utterances (proportionally sampled across assemblies and temporal positions). Each utterance was scored under six context conditions: 0, 500, 1000, 1500, 2000 tokens, and a 20-utterance baseline. Scores showed [excellent/good/moderate] stability across conditions (ICC(2,1) = X.XX, 95% CI [X.XX, X.XX]), with [no significant / a small but significant] effect of context size on mean scores (Friedman χ² = X.XX, p = X.XX). Based on this analysis, we fixed the context window at N tokens for all subsequent scoring, which corresponds to approximately M preceding utterances.

Include `spearman_heatmap.png` and `mean_score_by_budget.png` in supplementary materials.

---

## Questions?

See [README_CALIBRATION.md](README_CALIBRATION.md) for:
- Detailed usage instructions
- Interpretation guidelines
- Advanced configuration options
- Decision criteria explanations
