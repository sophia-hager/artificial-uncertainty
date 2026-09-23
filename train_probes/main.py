"""main.py – Entry point for probe training and evaluation.

Trains a hidden-state regression probe (P(IK)) on a calibration set using
an unlearned model, then evaluates the trained probe on a held-out dataset
using the base model.

Two-phase design
----------------
1. **Training** (skipped when ``--skip_train`` is set):
   - Loads the *unlearned* model.
   - Runs generation on the calibration set, optionally with residual dropout.
   - Trains a linear (or single-MLP) probe to predict per-example correctness
     from the last-token hidden state.
   - Saves the probe checkpoint and hidden-state cache to ``--workdir``.

2. **Evaluation**:
   - Loads the *base* (non-unlearned) model.
   - Loads the probe checkpoint from ``--workdir``.
   - Runs generation on ``--dataset`` and computes calibration metrics.
   - Saves metrics JSON and a predictions pickle to ``--workdir``.
"""

import os
from typing import Sequence

from absl import app, flags
from transformers import set_seed

import baselines
import experiment
from utils import load_dataset


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

flags.DEFINE_string("base_model_path", None,
                    "HuggingFace model ID or path to the base (non-unlearned) model used for evaluation.")
flags.DEFINE_string("unlearned_model_path", None,
                    "HuggingFace model ID or path to the unlearned model used for training.")
flags.DEFINE_string("dataset", "gpqa",
                    "Dataset to run evaluation on.")
flags.DEFINE_string("workdir", None,
                    "Directory to save checkpoints and results to.")
flags.DEFINE_bool("skip_train", False,
                  "Skip the training phase and go straight to evaluation.")
flags.DEFINE_string("cal_set", "calibration",
                    "Name of the calibration dataset used for training.")
flags.DEFINE_string("limit", None,
                    "Maximum number of examples to use for training/evaluation.")
flags.DEFINE_integer("eval_batch", 4,
                     "Batch size used during evaluation inference.")
flags.DEFINE_integer("train_batch", 6,
                     "Batch size used during training inference.")
flags.DEFINE_integer("seed", 42,
                     "Random seed.")
flags.DEFINE_bool("difficult", False,
                  "When True, use the harder evaluation split for GPQA.")
flags.DEFINE_bool("prompt_only", False,
                  "If True, train the probe on the last token of the prompt rather than the generated output.")
flags.DEFINE_float("dropout", 0.1,
                   "Residual dropout rate injected into the model during training inference. "
                   "Set to 0 to disable.")
flags.DEFINE_string("probe_type", "linear",
                    "Probe architecture: 'linear' or 'single-mlp'.")
flags.DEFINE_bool("using_lora", False,
                  "If True, --unlearned_model_path is a LoRA adapter directory and will be "
                  "merged with --base_model_path for inference.")
flags.DEFINE_integer("n_eval_examples", None,
                     "Cap on the number of examples evaluated from the cache. "
                     "If None (default), all cached examples are used.")
flags.DEFINE_bool("test", False,
                  "If True, evaluate on the test split rather than the validation split.")
flags.DEFINE_float("train_accuracy", None,
                   "Target accuracy for the training/calibration set. "
                   "If a pre-filtered cache already exists it is reused; otherwise the base "
                   "cache is filtered on the fly and saved for future runs.")
flags.DEFINE_string("reference_cache", None,
                    "Path to a cache file from another model. When set, the training cache is "
                    "restricted to the same questions found in this reference cache, ensuring "
                    "consistent question sets across models.")

FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Task instruction
# ---------------------------------------------------------------------------

EMNLP_TASK_INSTRUCTION = (
    "Answer the following question. Enclose concise reasoning in <reasoning> </reasoning> "
    "tags and your FINAL answer in <answer> </answer> tags without any of your work, "
    "like this: \"If each of Lisa's 7 chickens "
    "lays 6 eggs, how many eggs does Lisa have?\n"
    "A) 24\n"
    "B) 35\n"
    "C) 42\n"
    "D) 50\n"
    "<reasoning> This can be solved with multiplication. The answer is 7*6, or 42."
    "</reasoning> <answer> C) 42 </answer>.\"\n"
    "Your answer should not include words.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise ValueError(f'Unexpected CLI arguments: {argv!r}')

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not FLAGS.skip_train:
        datasets = {
            "train": load_dataset(FLAGS.cal_set, test=False, limit=FLAGS.limit)
        }
        model = baselines.HiddenStateRegression(
            model_name_or_path=FLAGS.unlearned_model_path,
            base_model_name_or_path=FLAGS.base_model_path,
            architecture=FLAGS.probe_type,
            train_batch_size=FLAGS.train_batch,
            eval_batch_size=FLAGS.eval_batch,
            is_lora_adapter=FLAGS.using_lora,
            replace=False,
            after_question_only=FLAGS.prompt_only,
            dropout=FLAGS.dropout,
            seed=FLAGS.seed,
        )
        runner = experiment.ExperimentRunner(
            model, datasets, data_description=datasets,
            skip_train=False,
            task_type="multiple_choice",
            task_instruction=EMNLP_TASK_INSTRUCTION,
            workdir=FLAGS.workdir,
            cal_set_name=FLAGS.cal_set,
            train_accuracy=FLAGS.train_accuracy,
            reference_cache=FLAGS.reference_cache,
            dropout=FLAGS.dropout,
        )
        cfg_path = os.path.join(runner.workdir, "config.txt")
        print(f"Writing config to: {cfg_path}")
        with open(cfg_path, 'w') as fp:
            fp.write(str({f"Model: {FLAGS.unlearned_model_path}, prompt: {EMNLP_TASK_INSTRUCTION}"}))
        runner.run()

    # ------------------------------------------------------------------
    # Phase 2: Evaluation
    # ------------------------------------------------------------------
    datasets = {
        "eval": load_dataset(FLAGS.dataset, test=FLAGS.test,
                             limit=FLAGS.limit, difficult=FLAGS.difficult)
    }
    model = baselines.HiddenStateRegression(
        model_name_or_path=FLAGS.base_model_path,
        architecture=FLAGS.probe_type,
        train_batch_size=FLAGS.train_batch,
        pretrained_model_path=os.path.join(FLAGS.workdir, "checkpoint.pth"),
        eval_batch_size=FLAGS.eval_batch,
        after_question_only=FLAGS.prompt_only,
        dropout=0.0,
        seed=FLAGS.seed,
    )
    runner = experiment.ExperimentRunner(
        model, datasets, data_description=datasets,
        skip_train=True,
        dataset_name=FLAGS.dataset,
        task_type="multiple_choice",
        task_instruction=EMNLP_TASK_INSTRUCTION,
        workdir=FLAGS.workdir,
        test=FLAGS.test,
        dropout=0.0,
        n_eval_examples=FLAGS.n_eval_examples,
    )
    runner.run()


if __name__ == '__main__':
    set_seed(42)
    app.run(main)
