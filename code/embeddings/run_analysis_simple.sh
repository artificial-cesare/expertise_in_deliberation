#!/bin/bash
# filepath: run_analysis_simple.sh

# Configuration
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "ANALYSIS PIPELINE"
echo "========================================"

# Phase 1: Base model
echo ""
echo "Running base model analysis..."
export USE_FINETUNED=0
python code/embeddings/final_centroids.py > "${LOG_DIR}/base_${TIMESTAMP}.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Base model complete"
else
    echo "✗ Base model failed - check ${LOG_DIR}/base_${TIMESTAMP}.log"
    exit 1
fi

# Phase 2: Fine-tuned model
echo ""
echo "Running fine-tuned model analysis..."
export USE_FINETUNED=1
python code/embeddings/final_centroids.py > "${LOG_DIR}/finetuned_${TIMESTAMP}.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Fine-tuned model complete"
else
    echo "✗ Fine-tuned model failed - check ${LOG_DIR}/finetuned_${TIMESTAMP}.log"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ PIPELINE COMPLETE"
echo "========================================"
echo "Outputs:"
echo "  - base_kmeans_global_pca/"
echo "  - finetuned_kmeans_global_pca/"
echo "Logs:"
echo "  - ${LOG_DIR}/base_${TIMESTAMP}.log"
echo "  - ${LOG_DIR}/finetuned_${TIMESTAMP}.log"