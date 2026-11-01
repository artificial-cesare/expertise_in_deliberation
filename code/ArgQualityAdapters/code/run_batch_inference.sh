#!/usr/bin/env bash
set -euo pipefail

########## CONFIG ##########
# Path to the repo's /code folder (where inference_simple.py lives)
CODE_DIR="code/ArgQualityAdapters/code"

# Where your input CSVs live (parent directory)
ASSEMBLIES_DIR="data/assemblies/processed_files"

# Inference batch size (lower if you see OOM)
BATCH_SIZE=32

# Quality dimension to evaluate
DIMENSION="overall"

######## END CONFIG ########

# ---- sanity checks ----
if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi
if [[ ! -d "$ASSEMBLIES_DIR" ]]; then
  echo "ERROR: ASSEMBLIES_DIR not found: $ASSEMBLIES_DIR" >&2
  exit 1
fi

# Ensure we're in the right conda env (optional but helpful)
if command -v conda >/dev/null 2>&1; then
  ENV_NAME="$(basename "$CONDA_PREFIX" 2>/dev/null || true)"
  if [[ "${ENV_NAME:-}" != "argqa" ]]; then
    echo "NOTE: You are not in the 'argqa' env (current: ${ENV_NAME:-none})."
    echo "      Activate it first:  conda activate argqa"
    exit 1
  fi
fi

# Function: detect column name (text or comment)
detect_column() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
try:
    df = pd.read_csv(p, nrows=0)
    if "text" in df.columns:
        print("text")
    elif "comment" in df.columns:
        print("comment")
    else:
        print("__ERR__")
        sys.exit(1)
except Exception as e:
    print(f"__ERR__ {e}", file=sys.stderr)
    sys.exit(2)
PY
}

echo "========================================"
echo "GPU/CPU Detection"
echo "========================================"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("✓ No GPU available, will use CPU")
PY

echo ""
echo "========================================"
echo "Scanning directories"
echo "========================================"

# Find CSV files in both assemblies subdirectories and experts subdirectories
mapfile -t assembly_files < <(find "$ASSEMBLIES_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.csv' ! -path "*/experts/*" | sort)
mapfile -t expert_files < <(find "$ASSEMBLIES_DIR/experts" -mindepth 2 -maxdepth 2 -type f -name '*.csv' | sort)

# Combine arrays
files=("${assembly_files[@]}" "${expert_files[@]}")

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No CSV files found in $ASSEMBLIES_DIR subdirectories or experts folders"
  exit 0
fi

echo "Found ${#assembly_files[@]} assembly CSV files"
echo "Found ${#expert_files[@]} expert CSV files"
echo "Total: ${#files[@]} CSV files to process"
echo ""

# Run inference
cd "$CODE_DIR"

processed=0
skipped=0

for csv in "${files[@]}"; do
  base="$(basename "$csv")"
  stem="${base%.csv}"
  
  # Get relative path for display
  rel_path="${csv#$ASSEMBLIES_DIR/}"
  
  echo "----------------------------------------------------------------"
  echo "[$((processed + skipped + 1))/${#files[@]}] Processing: $rel_path"
  
  # Detect column name
  col_name=$(detect_column "$csv")
  if [[ "$col_name" == "__ERR__" ]]; then
    echo "  ⚠️  Skipping: no 'text' or 'comment' column found" >&2
    ((skipped++))
    continue
  fi
  
  echo "  ✓ Detected column: '$col_name'"
  echo "  → Dimension: $DIMENSION | Batch: $BATCH_SIZE"
  echo "  → Updating file in-place: $csv"
  
  # Run the repo's simple inference script (outputs to same file)
  if python inference_simple.py "$csv" "$col_name" "$BATCH_SIZE" "$DIMENSION" "$csv"; then
    echo "  ✓ Successfully updated"
    ((processed++))
  else
    echo "  ✗ Failed to process" >&2
    ((skipped++))
  fi
  echo ""
done

echo "========================================"
echo "Summary"
echo "========================================"
echo "Total files: ${#files[@]}"
echo "Processed:   $processed"
echo "Skipped:     $skipped"
echo ""
echo "All files updated in-place in: $ASSEMBLIES_DIR"