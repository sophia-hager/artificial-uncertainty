#!/usr/bin/env bash
# eval_only.sh
#
# Evaluate a pre-trained P(IK) probe on the GPQA Diamond test split.
# Skips training entirely — requires an existing checkpoint in WORKDIR.
#
# Usage:
#   bash scripts/eval_only.sh
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

# Path to the base (non-unlearned) model used for evaluation
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Directory containing an existing checkpoint.pth from a previous training run
WORKDIR="./results/train_and_eval"

# Evaluation dataset (gpqa | mmlu-pro | test-ARC)
DATASET="gpqa"

# Probe architecture — must match the checkpoint (linear | single-mlp)
PROBE_TYPE="linear"

# Batch size for evaluation inference
EVAL_BATCH=4

# Use the held-out test split (--test) or the validation split (remove flag)
USE_TEST=true

# Random seed
SEED=42

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

TEST_FLAG=""
if [ "$USE_TEST" = true ]; then
    TEST_FLAG="--test"
fi

python main.py \
    --base_model_path "$BASE_MODEL" \
    --dataset         "$DATASET" \
    --workdir         "$WORKDIR" \
    --probe_type      "$PROBE_TYPE" \
    --eval_batch      "$EVAL_BATCH" \
    --seed            "$SEED" \
    --skip_train \
    $TEST_FLAG

echo "Done. Results written to: $WORKDIR"
