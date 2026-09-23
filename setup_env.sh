#!/usr/bin/env bash
# setup_env.sh
#
# Creates a clean conda environment and installs all dependencies for the
# P(IK) with Dropout project.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Requirements:
#   - conda (Miniconda or Anaconda) must be installed and on PATH
#   - An NVIDIA GPU with a CUDA 12.1-compatible driver for GPU support
#     (adjust CUDA_VERSION below if your driver supports a different version)
#
# To check your installed CUDA driver version:
#   nvidia-smi

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_NAME="pik_dropout"
PYTHON_VERSION="3.11"

# CUDA version to use for the PyTorch wheel.
# Options: cpu | cu118 | cu121 | cu124
# Run `nvidia-smi` to see the CUDA version supported by your driver.
CUDA_VERSION="cu121"

# ---------------------------------------------------------------------------
# Create environment
# ---------------------------------------------------------------------------

echo "==> Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Activate within the script (works when conda is initialised for bash)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Prevent user site-packages (~/.local) from interfering with the env.
# This ensures pip installs go into the conda env, not ~/.local.
export PYTHONNOUSERSITE=1

echo "==> Installing PyTorch (CUDA=$CUDA_VERSION)"
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

echo "==> Installing project dependencies"
pip install -r "$(dirname "$0")/../requirements.txt"

echo ""
echo "==> Environment setup complete."
echo "    Activate with:  conda activate $ENV_NAME"
echo "    Verify GPU:     python -c \"import torch; print(torch.cuda.is_available())\""
