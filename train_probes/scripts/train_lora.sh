#!/usr/bin/env bash
# train_lora.sh
#
# Train a P(IK) probe using a LoRA adapter as the unlearned model, then
# evaluate the trained probe on GPQA Diamond using the base model.
#
# The LoRA adapter is loaded on top of the base model before training.
# Set MERGE_LORA=true to merge adapter weights into the base model before
# inference (reduces memory overhead at the cost of flexibility).
#
# Usage:
#   bash scripts/train_lora.sh
#
# Edit the variables below before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pik_dropout

# Prevent user site-packages (~/.local) from shadowing conda env packages
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Path to the LoRA adapter directory (produced by e.g. SFT / unlearning fine-tune)
LORA_ADAPTER="/path/to/lora-adapter-dir"

# Path to the base model the adapter was trained on top of
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Directory where the probe checkpoint and results will be written
WORKDIR="./results/lora_experiment"

# Evaluation dataset (gpqa | mmlu-pro | test-ARC)
DATASET="gpqa"

# Probe architecture (linear | single-mlp)
PROBE_TYPE="linear"

# Residual dropout rate injected during training inference (0 to disable)
DROPOUT="0.1"

# Batch sizes — reduce if you run out of GPU memory
TRAIN_BATCH=6
EVAL_BATCH=4

# Random seed
SEED=42

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

mkdir -p "$WORKDIR"

python main.py \
    --unlearned_model_path "$LORA_ADAPTER" \
    --base_model_path      "$BASE_MODEL" \
    --using_lora \
    --dataset              "$DATASET" \
    --workdir              "$WORKDIR" \
    --probe_type           "$PROBE_TYPE" \
    --dropout              "$DROPOUT" \
    --train_batch          "$TRAIN_BATCH" \
    --eval_batch           "$EVAL_BATCH" \
    --seed                 "$SEED"

echo "Done. Results written to: $WORKDIR"
