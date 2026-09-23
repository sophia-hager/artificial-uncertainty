#!/usr/bin/env bash
# train_and_eval.sh
#
# Full pipeline: train a P(IK) probe on the unlearned model's calibration set,
# then evaluate the trained probe on GPQA Diamond using the base model.
#
# Usage:
#   bash scripts/train_and_eval.sh
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

# Path to the unlearned model (HuggingFace ID or local directory)
UNLEARNED_MODEL="/path/to/unlearned-model"

# Path to the base (non-unlearned) model used for evaluation
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Directory where the probe checkpoint and results will be written
WORKDIR="./results/train_and_eval"

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
    --unlearned_model_path "$UNLEARNED_MODEL" \
    --base_model_path      "$BASE_MODEL" \
    --dataset              "$DATASET" \
    --workdir              "$WORKDIR" \
    --probe_type           "$PROBE_TYPE" \
    --dropout              "$DROPOUT" \
    --train_batch          "$TRAIN_BATCH" \
    --eval_batch           "$EVAL_BATCH" \
    --seed                 "$SEED"

echo "Done. Results written to: $WORKDIR"
