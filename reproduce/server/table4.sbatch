#!/bin/bash
# Table 4 assembly accuracy on a GPU node. Submit from the repo root:  sbatch reproduce/server/table4.sbatch
#SBATCH --job-name=table4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/paper_v2/table4_%j.log
## If the job stays PENDING or is rejected, add the GPU partition from `sinfo`, e.g.:
##SBATCH --partition=<gpu-partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
eval "$(conda shell.bash hook)"
conda activate qwen3_embed
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # model is in ~/.cache; compute nodes may lack internet
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch, sentence_transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'st', sentence_transformers.__version__, 'sklearn', sklearn.__version__)"
# both models: the fine-tuned run checks that the GPU reproduces the laptop values exactly
python reproduce/table4_assembly_accuracy.py --models base finetuned --device cuda:0 --batch-size 8 \
  --out outputs/paper_v2/table4_server.json
