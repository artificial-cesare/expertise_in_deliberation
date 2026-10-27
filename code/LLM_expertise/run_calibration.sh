#!/bin/bash
#
# Complete Context Size Calibration Workflow
# Runs both experiment and analysis in sequence
#

set -e  # Exit on error

# Configuration
BASE_DIR="../../data/assemblies/processed_files"
OUTPUT_DIR="../../results/context_calibration"
MODEL="gpt-4o-mini"
DELAY=0.5

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "=============================================================================="
echo "CONTEXT SIZE CALIBRATION: COMPLETE WORKFLOW"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL"
echo "  Rate limit delay: ${DELAY}s"
echo ""
echo "This will:"
echo "  1. Sample 200 utterances (stratified)"
echo "  2. Score with 6 context conditions (0, 500, 1000, 1500, 2000, 20-utterances)"
echo "  3. Compute ICC, Spearman correlations, Friedman test"
echo "  4. Generate 5 plots + summary report"
echo ""
echo "Estimated time: 15-20 minutes"
echo "Estimated cost: <\$1 USD (gpt-4o-mini)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: Run experiment
echo ""
echo "=============================================================================="
echo "STEP 1/2: Running Calibration Experiment"
echo "=============================================================================="
echo ""

python context_size_calibration.py \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --delay "$DELAY"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Experiment failed. Check output above for details."
    exit 1
fi

# Step 2: Analyze results
echo ""
echo "=============================================================================="
echo "STEP 2/2: Analyzing Results"
echo "=============================================================================="
echo ""

python analyze_calibration.py \
    --results-file "$OUTPUT_DIR/context_calibration_results.csv" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Analysis failed. Check output above for details."
    exit 1
fi

# Summary
echo ""
echo "=============================================================================="
echo "WORKFLOW COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - summary_report.txt (read this first!)"
echo "  - mean_score_by_budget.png"
echo "  - spearman_heatmap.png"
echo "  - icc_by_assembly.png"
echo "  - score_range_vs_length.png"
echo "  - actual_tokens_distribution.png"
echo "  - high_mad_utterances_threshold_1.0.csv"
echo ""
echo "Next steps:"
echo "  1. Review summary_report.txt for recommendations"
echo "  2. Examine plots to understand score stability"
echo "  3. Decide on context budget for your main analysis"
echo ""
