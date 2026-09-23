# P(IK) with Dropout

A framework for probing whether a language model "knows" an answer (P(IK)) by
training a lightweight probe on its hidden states.  The probe can be trained on
an **unlearned** model to detect which questions the model still answers
correctly, and evaluated on a base model.  Optionally, **residual dropout** is
injected into the LLM during training inference to simulate uncertainty on easy
examples.

---

## Repository structure

```
.
├── README.md
├── requirements.txt         # Python package dependencies
├── setup_env.sh             # Create and configure the conda environment
├── data/
│   ├── cal.jsonl            # Calibration training set
│   ├── train.jsonl          # Training examples
│   └── validation.jsonl     # Validation examples
└── train_probes/
    ├── main.py              # Entry point: probe training + evaluation
    ├── experiment.py        # ExperimentRunner: orchestrates caching, training, eval
    ├── baselines.py         # HiddenStateRegression probe + cache utilities
    ├── hidden_dropout.py    # Context managers for residual dropout / embedding noise
    ├── metrics.py           # Accuracy, AUROC, ECE, Brier score, and ACE metrics
    ├── data_utils.py        # Dataset formatting and chat-template helpers
    ├── utils.py             # Dataset loading, helpers, and BatchInference utility
    └── scripts/
        ├── train_and_eval.sh  # Full pipeline: train probe + evaluate
        ├── eval_only.sh       # Evaluate a pre-trained probe (skip training)
        └── train_lora.sh      # Train using a LoRA adapter as the unlearned model
```

---

## Setup

### Requirements

- Python ≥ 3.10
- An NVIDIA GPU with CUDA support (the model loading and inference code assumes a CUDA device)
- conda (Miniconda or Anaconda) — or a plain `venv` if you prefer

### Option A — Automated setup (conda)

The `setup_env.sh` script creates a fresh conda environment, installs
PyTorch for the right CUDA version, and then installs all remaining packages
from `requirements.txt`:

```bash
# Check which CUDA version your driver supports
nvidia-smi

# Edit CUDA_VERSION at the top of the script if needed (default: cu121)
bash setup_env.sh

# Activate the environment
conda activate pik_dropout
```

To verify the GPU is visible after activation:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Option B — Manual setup (pip / venv)

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all remaining dependencies
pip install -r requirements.txt
```

### Key packages

| Package | Role |
|---------|------|
| `torch` | Inference, probe training, hidden-state extraction |
| `transformers` | Model and tokenizer loading, generation |
| `datasets` | Dataset loading and preprocessing (HuggingFace) |
| `peft` | Loading LoRA adapters |
| `torcheval` | Streaming AUROC / accuracy metrics |
| `torchmetrics` | Binary calibration error (ECE) |
| `scikit-learn` | Brier score, AUROC, binned calibration |
| `absl-py` | Flag parsing for `main.py` |
| `tqdm` | Progress bars |

### Data cache directory

`ExperimentRunner` reads and writes gzip-compressed hidden-state caches to a
configurable directory (default: `"./cached_datasets/"`).  Set `--data_cache`
or update the default in `experiment.py` before running.

### Calibration data

The `calibration_varied` training set is loaded from `../data/cal.jsonl` (relative
to `utils.py`).  The file should be a JSONL where each row contains:
`{"Question": ..., "options": [...], "answer": ...}`.



---

## Quickstart

> Assumes you have an unlearned model checkpoint and a base model available
> (locally or via HuggingFace Hub), and that `../data/cal.jsonl` is populated.

The three scripts in `scripts/` cover the most common workflows end-to-end.
Edit the variables at the top of each script before running.

### 1 — Train probe + evaluate (full pipeline)

```bash
bash scripts/train_and_eval.sh
```

Trains a linear P(IK) probe on the unlearned model's calibration set (with
residual dropout), then evaluates the trained probe on GPQA Diamond using the
base model.  Writes results to `$WORKDIR`.

### 2 — Evaluate a pre-trained probe on the test split

```bash
bash scripts/eval_only.sh
```

Skips training entirely and scores the GPQA Diamond **test** split against an
existing checkpoint in `$WORKDIR`.

### 3 — Train using a LoRA adapter as the unlearned model

```bash
bash scripts/train_lora.sh
```

Same as (1) but loads the unlearned model as a LoRA adapter merged on top of
the base model.

---



All experiments are driven through `main.py` using [abseil-py](https://abseil.io/docs/python/guides/flags) flags.

### Basic probe training + evaluation

Train a linear probe on the unlearned model's calibration set, then evaluate on
GPQA diamond:

```bash
python main.py \
  --unlearned_model_path /path/to/unlearned-model \
  --base_model_path      meta-llama/Llama-3.1-8B-Instruct \
  --dataset              gpqa \
  --workdir              ./results/my_experiment \
  --probe_type           linear \
  --dropout              0.1 \
  --train_batch          6 \
  --eval_batch           4
```

Results are written to `./results/my_experiment/`:
- `checkpoint.pth` — probe weights
- `config.txt` — run configuration
- `predictions_gpqa.pkl` — `(predictions, scores, examples)` tuple
- `metrics_gpqa.json` — calibration metrics (accuracy, AUROC, Brier, ACE, ECE)

---

### Skip training (evaluate only)

If a checkpoint already exists in `--workdir`, skip the training phase and go
straight to evaluation:

```bash
python main.py \
  --base_model_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset         gpqa \
  --workdir         ./results/my_experiment \
  --skip_train
```

---

### Test split evaluation

Evaluate on the GPQA diamond test set instead of the validation set:

```bash
python main.py \
  --base_model_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset         gpqa \
  --workdir         ./results/my_experiment \
  --skip_train \
  --test
```

Metrics are saved as `TEST_gpqa.json`.

---

### Using a LoRA adapter as the unlearned model

```bash
python main.py \
  --unlearned_model_path /path/to/lora-adapter-dir \
  --base_model_path      meta-llama/Llama-3.1-8B-Instruct \
  --using_lora \
  --dataset   gpqa \
  --workdir   ./results/lora_experiment
```

---

### Targeting a specific training-set accuracy

Filter the training cache so that the probe sees examples at a target accuracy
(e.g. 0.6, meaning 60 % of training examples are answered correctly):

```bash
python main.py \
  --unlearned_model_path /path/to/unlearned-model \
  --base_model_path      meta-llama/Llama-3.1-8B-Instruct \
  --dataset              gpqa \
  --workdir              ./results/acc_experiment \
  --train_accuracy       0.6
```

---

### Aligning training sets across models

When comparing multiple models, use `--reference_cache` to restrict training
to the same questions used for a reference model:

```bash
python main.py \
  --unlearned_model_path /path/to/unlearned-model \
  --base_model_path      meta-llama/Llama-3.1-8B-Instruct \
  --dataset              gpqa \
  --workdir              ./results/aligned_experiment \
  --reference_cache      /path/to/reference_model_TRAIN.gz
```

---


## Key flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--base_model_path` | — | Base (non-unlearned) model used for evaluation |
| `--unlearned_model_path` | — | Unlearned model used during training |
| `--dataset` | `gpqa` | Evaluation dataset |
| `--cal_set` | `calibration` | Training/calibration dataset |
| `--workdir` | — | Output directory |
| `--skip_train` | `False` | Skip training and evaluate only |
| `--probe_type` | `linear` | `linear` or `single-mlp` |
| `--dropout` | `0.1` | Residual dropout rate during training inference |
| `--prompt_only` | `False` | Use prompt hidden state instead of generated token |
| `--using_lora` | `False` | Treat `--unlearned_model_path` as a LoRA adapter dir |
| `--train_accuracy` | `None` | Target accuracy for the training cache filter |
| `--reference_cache` | `None` | Cache path for cross-model question alignment |
| `--test` | `False` | Evaluate on the test split |
| `--limit` | `None` | Maximum number of examples |
| `--seed` | `42` | Random seed |
| `--train_batch` | `6` | Training inference batch size |
| `--eval_batch` | `4` | Evaluation inference batch size |
| `--n_eval_examples` | `None` | Cap on cached evaluation examples |

---

## Dropout uncertainty (`hidden_dropout.py`)

The `dropout_uncertainty` context manager injects residual dropout into
selected decoder sub-layers during inference, perturbing the residual stream
without changing model weights.

```python
from hidden_dropout import dropout_uncertainty

with dropout_uncertainty(model, dropout_rate=0.1, target="mlp"):
    outputs = model.generate(...)
```

**Targets:** `"mlp"` | `"attention"` | `"both"` | `"post_layer"`

**Layer selection:**

```python
# Apply only to the top 8 decoder layers
with dropout_uncertainty(model, dropout_rate=0.1, target="mlp", layers="top_8"):
    ...

# Apply to layers 10–20 (exclusive end)
with dropout_uncertainty(model, dropout_rate=0.1, target="mlp", layers=(10, 20)):
    ...
```

---

## Supported datasets

| Name | Description |
|------|-------------|
| `gpqa` | GPQA non-diamond validation split (50 examples) |
| `gpqa` + `--test` | GPQA Diamond test split |
| `mmlu-pro` | MMLU-SemiPro STEM categories (test or train split) |
| `test-ARC` | ARC-Easy test split (requires `--test`) |
| `calibration` | Calibration training set loaded from `../data/cal.jsonl` |
