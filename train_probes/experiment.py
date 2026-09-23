import os
import json
import hashlib
import dataclasses
import pickle
import torch

from pathlib import Path
from typing import Any, Optional
from absl import logging
from torch.utils.data import Dataset, random_split

import baselines
from baselines import filter_cache_by_accuracy, get_questions_from_cache
import metrics
import data_utils
from data_utils import prepare_dataset, is_correct_multiple_choice


@dataclasses.dataclass
class ExperimentRunner:
    """Orchestrates training and evaluation of a ``HiddenStateRegression`` probe.

    Handles dataset splitting, hidden-state caching (read and write), probe
    training, and metric computation.  Results are written to *workdir*.

    Parameters
    ----------
    model : HiddenStateRegression
        The probe model (and underlying LLM) to train/evaluate.
    datasets : dict
        A dict with optional keys ``'train'`` and ``'eval'``, each a
        HuggingFace Dataset.
    data_description : dict
        Metadata dict logged alongside results.
    task_type : str
        ``"multiple_choice"`` — the only supported task type.
    task_instruction : str
        System prompt / task instruction prepended to every example.
    workdir : str
        Directory where checkpoints and results are written.
    data_cache : str
        Directory that stores gzip-compressed hidden-state caches.
    cal_set_name : str
        Name of the calibration (training) dataset, used to name cache files.
    train_accuracy : float or None
        Target accuracy for the training set.  If set, the training cache is
        filtered to achieve approximately this accuracy before training.
    reference_cache : str or None
        Path to a cache file from another model.  When set, the training cache
        is restricted to the same questions found in this reference cache,
        ensuring consistent question sets across models.
    """
    model: Any
    datasets: dict
    data_description: dict
    task_type: str
    task_instruction: str
    heldout_ratio: float = 0.1
    seed: int = 42
    data_cache: str = "./cached_datasets/"
    workdir: str = "/tmp/experiment"
    n_eval_examples: Optional[int] = None
    skip_train: bool = False
    dataset_name: str = ""
    test: bool = False
    dropout: float = 0.0
    cal_set_name: str = ""
    train_accuracy: Optional[float] = None
    reference_cache: Optional[str] = None

    def __post_init__(self):
        # Ensure working directory exists
        work_dir = Path(self.workdir)
        work_dir.mkdir(parents=True, exist_ok=True)

        m = self.model.model_name_or_path.split("/")[-1]
        m_train = f"{m}_dropout={self.dropout}"

        # Evaluation cache paths
        if self.test:
            self.cache_path = Path(self.data_cache + self.dataset_name + "_" + m + "_TEST.gz")
        else:
            self.cache_path = Path(self.data_cache + self.dataset_name + "_" + m + ".gz")
        self.temp_path = Path(self.data_cache + self.dataset_name + "_" + m + "_TEST-TEMP.gz")

        # Training (calibration) cache paths
        cal_name = self.cal_set_name or "TRAIN"
        self.train_cache_base_path = Path(
            self.data_cache + cal_name + "_" + m_train + "_TRAIN.gz"
        )

        # Determine effective training cache path (may include accuracy/reference tags)
        if self.train_accuracy is not None and self.reference_cache:
            acc_tag = f"{self.train_accuracy:.4f}".rstrip('0').rstrip('.')
            ref_tag = hashlib.md5(self.reference_cache.encode()).hexdigest()[:8]
            self.train_cache_path = Path(
                self.data_cache + cal_name + "_" + m_train
                + f"_TRAIN_acc{acc_tag}_ref-{ref_tag}.gz"
            )
        elif self.train_accuracy is not None:
            acc_tag = f"{self.train_accuracy:.4f}".rstrip('0').rstrip('.')
            self.train_cache_path = Path(
                self.data_cache + cal_name + "_" + m_train + f"_TRAIN_acc{acc_tag}.gz"
            )
        elif self.reference_cache:
            ref_tag = hashlib.md5(self.reference_cache.encode()).hexdigest()[:8]
            self.train_cache_path = Path(
                self.data_cache + cal_name + "_" + m_train + f"_TRAIN_ref-{ref_tag}.gz"
            )
        else:
            self.train_cache_path = self.train_cache_base_path

        # Answer-correctness function
        if self.task_type != "multiple_choice":
            raise ValueError(f"Unknown task_type: {self.task_type!r}. Only 'multiple_choice' is supported.")
        self.prediction_check_fn = data_utils.is_correct_multiple_choice

        # Conversation format varies by model family
        if "gemma-3" in self.model.model_name_or_path:
            self.conversation_format = "user/assistant"
        elif "Ministral" in self.model.model_name_or_path:
            self.conversation_format = "task_as_system/user/assistant"
        else:
            self.conversation_format = "system/user"

        logging.info(f"Conversation format: {self.conversation_format}")

        # Prepare training split
        if self.model.requires_supervision() or 'train' in self.datasets:
            if "checkpoint.pth" in os.listdir(self.workdir) and self.skip_train:
                self.train_split = None
                self.train_dataset = None
                self.valid_dataset = None
            elif 'train' not in self.datasets:
                raise ValueError("Need training data for models that require supervision.")
            else:
                train_data = self.datasets['train']
                splits = train_data.train_test_split(
                    shuffle=True, test_size=self.heldout_ratio, seed=self.seed)
                self.train_split = splits['train']
                self.valid_split = splits['test']
                self.train_dataset = prepare_dataset(
                    self.train_split, self.model.tokenizer,
                    task_instruction=self.task_instruction,
                    conversation_format=self.conversation_format)
                self.valid_dataset = prepare_dataset(
                    self.valid_split, self.model.tokenizer,
                    task_instruction=self.task_instruction,
                    conversation_format=self.conversation_format)

        # Prepare evaluation split
        if 'eval' in self.datasets:
            if not self.cache_path.exists():
                self.eval_split = self.datasets['eval']
                self.eval_dataset = prepare_dataset(
                    self.eval_split, self.model.tokenizer,
                    task_instruction=self.task_instruction,
                    conversation_format=self.conversation_format)
                if self.n_eval_examples:
                    subset_indices = list(range(self.n_eval_examples))
                    print(f"Using a subset of {self.n_eval_examples} evaluation examples")
                    self.eval_dataset = torch.utils.data.Subset(
                        self.eval_dataset, subset_indices)

    def run(self):
        """Execute the experiment: train the probe (if needed) then evaluate."""
        if self.model.requires_supervision() and self.train_dataset is not None:
            # ------------------------------------------------------------------
            # Training: prefer cached hidden states; otherwise run live inference
            # ------------------------------------------------------------------
            use_train_cache = False

            # Build reference-question filter if --reference_cache was given
            reference_questions = None
            if self.reference_cache:
                ref_path = Path(self.reference_cache)
                if ref_path.exists():
                    print(f"Loading reference cache for cross-model alignment: {ref_path}")
                    reference_questions = get_questions_from_cache(str(ref_path))
                    print(f"  → {len(reference_questions)} unique questions in reference cache")
                else:
                    print(f"WARNING: reference_cache path not found, skipping alignment: {ref_path}")

            # Decide which training cache to use / build
            if self.train_cache_path.exists():
                print(f"Found existing training cache: {self.train_cache_path}")
                use_train_cache = True
            elif self.train_accuracy is not None and self.train_cache_base_path.exists():
                print(
                    f"Filtered train cache not found. "
                    f"Building from base cache: {self.train_cache_base_path}"
                )
                achieved = filter_cache_by_accuracy(
                    src_cache_path=str(self.train_cache_base_path),
                    dst_cache_path=str(self.train_cache_path),
                    target_accuracy=self.train_accuracy,
                    prediction_check_fn=self.prediction_check_fn,
                    reference_questions=reference_questions,
                )
                print(f"Filtered training cache saved to: {self.train_cache_path} (accuracy={achieved:.4f})")
                use_train_cache = True
            elif self.reference_cache and self.train_cache_base_path.exists():
                print(f"Filtering base train cache by reference questions → {self.train_cache_path}")
                achieved = filter_cache_by_accuracy(
                    src_cache_path=str(self.train_cache_base_path),
                    dst_cache_path=str(self.train_cache_path),
                    target_accuracy=None,
                    prediction_check_fn=self.prediction_check_fn,
                    reference_questions=reference_questions,
                )
                print(f"Reference-filtered cache saved to: {self.train_cache_path}")
                use_train_cache = True

            if use_train_cache:
                print(f"Training probe from cache: {self.train_cache_path}")
                self.model.fit_from_cache(
                    train_cache=str(self.train_cache_path),
                    prediction_check_fn=self.prediction_check_fn,
                )
            else:
                print(f"No training cache found; running live inference and caching to: {self.train_cache_base_path}")
                self.train_cache_base_path.parent.mkdir(parents=True, exist_ok=True)
                open(str(self.train_cache_base_path), 'w').close()
                self.model.fit(
                    train_dataset=self.train_dataset,
                    valid_dataset=self.valid_dataset,
                    prediction_check_fn=self.prediction_check_fn,
                    cache=str(self.train_cache_base_path),
                )
                if self.train_accuracy is not None:
                    print(f"Building accuracy-filtered cache (target={self.train_accuracy}) → {self.train_cache_path}")
                    filter_cache_by_accuracy(
                        src_cache_path=str(self.train_cache_base_path),
                        dst_cache_path=str(self.train_cache_path),
                        target_accuracy=self.train_accuracy,
                        prediction_check_fn=self.prediction_check_fn,
                        reference_questions=reference_questions,
                    )

            torch.cuda.empty_cache()

            ckpt_path = os.path.join(self.workdir, "checkpoint.pth")
            print(f"Saving probe checkpoint to: {ckpt_path}")
            self.model.save(ckpt_path)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if 'eval' in self.datasets and self.datasets['eval']:
            if not self.cache_path.exists():
                print("Obtaining predictions with scores on evaluation dataset")
                predictions, scores, examples = self.model.predict_with_confidences(
                    self.eval_dataset, cache=self.temp_path)
                self.temp_path.rename(str(self.cache_path))
            else:
                print("Obtaining scores on cached predictions")
                predictions, scores, examples = self.model.predict_with_confidences_cached(
                    cache=self.cache_path,
                    max_examples=self.n_eval_examples,
                )

            print(f"Writing predictions, scores, and examples to {self.workdir}")
            predictions_path = os.path.join(self.workdir, f"predictions_{self.dataset_name}.pkl")
            with open(predictions_path, 'wb') as fp:
                pickle.dump((predictions, scores, examples), fp)

            print("Computing metrics")
            results = metrics.evaluate(predictions, scores, examples, self.prediction_check_fn)
            results['llm'] = self.model.model_name_or_path

            if self.test:
                metrics_path = os.path.join(self.workdir, f"TEST_{self.dataset_name}.json")
            else:
                metrics_path = os.path.join(self.workdir, f"metrics_{self.dataset_name}.json")
            print(f"Saving metrics to: {metrics_path}")
            with open(metrics_path, 'w') as fp:
                json.dump(results, fp)
