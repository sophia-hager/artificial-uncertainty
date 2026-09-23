import contextlib
import os
import re
import dataclasses
import gzip
import json
import pickle
import functools

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from abc import ABC
from typing import Any
from collections.abc import Callable
from collections import Counter

from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torcheval.metrics import BinaryAccuracy, BinaryAUROC
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    EosTokenCriteria,
    set_seed,
)
from peft import PeftModel
from absl import logging

from metrics import AccuracyMetric, compute_metrics, get_batch_correctness_labels
from data_utils import dict_of_lists_to_list_of_dicts, remove_leading_repeated_word, show_diff
from hidden_dropout import dropout_uncertainty


STOP_SEQUENCES = ["[INST]", "None", "User:"]


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def stream_compressed_data(filename):
    """Yield one JSON-decoded record at a time from a gzip-compressed JSONL file."""
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def get_questions_from_cache(cache_path):
    """Return the set of question texts stored in a cache file.

    Each batch record's ``batch`` dict is expected to contain a ``'Question'``
    key (or fall back to ``'formatted_chat'``).
    """
    questions = set()
    for record in stream_compressed_data(cache_path):
        batch = record.get("batch", {})
        if "Question" in batch:
            qs = batch["Question"]
            if isinstance(qs, list):
                questions.update(qs)
            else:
                questions.add(qs)
        elif "formatted_chat" in batch:
            chats = batch["formatted_chat"]
            if isinstance(chats, list):
                questions.update(chats)
            else:
                questions.add(chats)
    return questions


def _compute_record_accuracy(record, prediction_check_fn):
    """Return (n_correct, n_total) for a single cache record."""
    batch = record.get("batch", {})
    predictions = record.get("predictions", [])
    n_correct = 0
    n_total = len(predictions)
    for i, pred in enumerate(predictions):
        example = {k: (v[i] if isinstance(v, list) else v) for k, v in batch.items()}
        try:
            if prediction_check_fn(pred, example["answer"]):
                n_correct += 1
        except Exception:
            pass
    return n_correct, n_total


def filter_cache_by_accuracy(
        src_cache_path,
        dst_cache_path,
        target_accuracy,
        prediction_check_fn,
        reference_questions=None,
):
    """Read *src_cache_path*, optionally restrict to *reference_questions*,
    then greedily select records to reach *target_accuracy*, and write the
    result to *dst_cache_path*.

    If *target_accuracy* is ``None``, all matched examples are kept (i.e. only
    the ``reference_questions`` filter is applied, with no accuracy sub-sampling).

    Strategy (when target_accuracy is set)
    ----------------------------------------
    We want the final subset accuracy to be as close to ``target_accuracy`` as
    possible.  We do this in two passes:

    1. Load all records (filtered by ``reference_questions`` if given) and
       compute per-example correctness.
    2. Sort examples so that correct ones come first, then pick a prefix whose
       accuracy is closest to the target.

    Returns the achieved accuracy.
    """
    # --- Pass 1: collect all examples -----------------------------------------
    all_examples = []
    for record in stream_compressed_data(src_cache_path):
        batch = record.get("batch", {})
        predictions = record.get("predictions", [])

        if reference_questions is not None:
            qs = batch.get("Question", batch.get("formatted_chat", []))
            if isinstance(qs, list):
                if not all(q in reference_questions for q in qs):
                    continue
            else:
                if qs not in reference_questions:
                    continue

        n = len(predictions)
        for i in range(n):
            example = {k: (v[i] if isinstance(v, list) else v) for k, v in batch.items()}
            try:
                correct = prediction_check_fn(predictions[i], str(example.get("answer", "")))
            except Exception:
                correct = False
            first_state = record.get("first_state", [])
            last_state = record.get("last_state", [])
            all_examples.append({
                "first_state": first_state[i] if isinstance(first_state, list) and len(first_state) > i else first_state,
                "last_state": last_state[i] if isinstance(last_state, list) and len(last_state) > i else last_state,
                "prediction": predictions[i],
                "example": example,
                "correct": correct,
            })

    if not all_examples:
        raise ValueError(f"No examples found in cache (after optional filtering): {src_cache_path}")

    total = len(all_examples)
    n_correct_total = sum(1 for e in all_examples if e["correct"])
    base_accuracy = n_correct_total / total
    print(f"[filter_cache] Base accuracy: {base_accuracy:.4f} over {total} examples")

    # --- Pass 2: select a subset whose accuracy ≈ target ----------------------
    if target_accuracy is None:
        selected = all_examples
        achieved = base_accuracy
        print(f"[filter_cache] No accuracy target; keeping all {len(selected)} matched examples (accuracy={achieved:.4f})")
    else:
        correct_examples = [e for e in all_examples if e["correct"]]
        wrong_examples = [e for e in all_examples if not e["correct"]]

        # n_wrong = n_correct * (1 - target) / target
        n_c = len(correct_examples)
        n_w_needed = int(round(n_c * (1.0 - target_accuracy) / target_accuracy))
        n_w_needed = min(n_w_needed, len(wrong_examples))

        selected = correct_examples + wrong_examples[:n_w_needed]
        achieved = len(correct_examples) / len(selected) if selected else 0.0
        print(f"[filter_cache] Selected {len(selected)} examples, achieved accuracy: {achieved:.4f} (target: {target_accuracy:.4f})")

    # --- Write filtered cache -------------------------------------------------
    with gzip.open(dst_cache_path, 'wt', encoding='utf-8') as f:
        for ex in selected:
            record = {
                "first_state": [ex["first_state"]],
                "last_state": [ex["last_state"]],
                "predictions": [ex["prediction"]],
                "batch": {k: [v] for k, v in ex["example"].items()},
            }
            f.write(json.dumps(record) + '\n')

    print(f"[filter_cache] Wrote filtered cache to: {dst_cache_path}")
    return achieved


# ---------------------------------------------------------------------------
# HiddenStateRegression
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HiddenStateRegression:
    """Linear (or single-MLP) probe trained on LLM hidden states.

    When ``architecture = "linear"``, this corresponds to the linear P(IK)
    method described in:

        https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/p_ik.py

    The probe is trained to predict whether the LLM's answer is correct from
    the hidden state extracted at the last generated token.  During training,
    optional stochastic residual dropout (``dropout > 0``) is injected into
    the base model via the ``dropout_uncertainty`` context manager.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or path to the model used for inference.
    dropout : float
        Residual dropout rate injected during training inference (0 disables).
    architecture : str
        Probe architecture: ``"linear"`` or ``"single-mlp"``.
    after_question_only : bool
        If True, use the hidden state at the last token of the *prompt* rather
        than the last generated token.
    base_model_name_or_path : str or None
        Required only when ``is_lora_adapter=True``; points to the base model.
    is_lora_adapter : bool
        If True, ``model_name_or_path`` is a LoRA adapter directory that will
        be attached to ``base_model_name_or_path``.
    pretrained_model_path : str or None
        Path to a previously saved checkpoint.  If set, the probe weights are
        loaded from this file (skipping training).
    """
    model_name_or_path: str
    dropout: float = 0.1
    seed: int = 42
    train_batch_size: int = 50
    eval_batch_size: int = 50
    learning_rate: float = 5e-4
    ckpt_path: str = ""
    architecture: str = "linear"
    max_new_tokens: int = 1024
    temperature: float = 1.
    torch_dtype: Any = torch.bfloat16
    num_train_examples: int = 1000
    replace: bool = False
    num_epochs: int = 3
    eval_every_n: int = 100
    eval_subset_size: int = 50
    disable_tqdm: bool = False
    pretrained_model_path: str = None
    after_question_only: bool = False
    base_model_name_or_path: str = None
    is_lora_adapter: bool = False
    merge_lora_for_inference: bool = False

    def __post_init__(self):
        if self.is_lora_adapter:
            if not self.base_model_name_or_path:
                raise ValueError("base_model_name_or_path must be set when is_lora_adapter=True")

            adapter_dir = self.model_name_or_path
            base_id = self.base_model_name_or_path

            print(f"Loading base LLM: {base_id}")
            config = AutoConfig.from_pretrained(base_id)
            try:
                hidden_size = config.hidden_size
            except AttributeError:
                hidden_size = config.text_config.hidden_size

            base = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
            base.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(base_id)

            print(f"Loading LoRA adapters from: {adapter_dir}")
            lora_model = PeftModel.from_pretrained(base, adapter_dir)

            if self.merge_lora_for_inference:
                print("Merging LoRA into base weights for inference…")
                self.model = lora_model.merge_and_unload()
            else:
                self.model = lora_model
        else:
            print(f"Loading LLM: {self.model_name_or_path}")
            torch.manual_seed(self.seed)
            config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                hidden_dropout_prob=self.dropout,
                attention_probs_dropout_prob=self.dropout,
                dropout=self.dropout,
            )
            try:
                hidden_size = config.hidden_size
            except AttributeError:
                hidden_size = config.text_config.hidden_size

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                config=config,
                device_map="auto",
                attn_implementation="eager",
            )
            target_device = next(self.model.parameters()).device
            self.device = target_device

            if self.dropout > 0:
                self.model.train()
            else:
                self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # Build probe head
        if self.architecture == "linear":
            self.score_model = nn.Sequential(nn.Linear(hidden_size, 1))
            self.score_model = self.score_model.to(self.torch_dtype).to(device=target_device)
            if self.pretrained_model_path is not None:
                print("Loading previous checkpoint")
                self.score_model.load_state_dict(
                    torch.load(self.pretrained_model_path)["model_state_dict"])
        elif self.architecture == "single-mlp":
            self.score_model = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
            self.score_model = self.score_model.to(self.torch_dtype).to(device=target_device)
            if self.pretrained_model_path is not None:
                print("Loading previous checkpoint")
                self.score_model.load_state_dict(
                    torch.load(self.pretrained_model_path)["model_state_dict"])
        else:
            raise ValueError(f"Unknown architecture: {self.architecture!r}")

        # Tokenizer / padding configuration per model family
        model_name = self.model_name_or_path.lower()
        if 'gemma' in model_name:
            self.return_token_type_ids = True
        else:
            self.return_token_type_ids = False

        if 'llama-3.2' in model_name or 'llama-3.1' in model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.eos_token_id
        elif 'olmo-2' in model_name or 'phi' in model_name or 'gemma' in model_name:
            self.pad_token = self.tokenizer.pad_token
            self.pad_token_id = self.tokenizer.pad_token_id
        elif 'ministral' in model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token = self.tokenizer.pad_token
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            raise NotImplementedError(
                f"Pad token not configured for model: {self.model_name_or_path}"
            )

        self.stopping_criteria = [EosTokenCriteria(self.tokenizer.eos_token_id)]

    def requires_supervision(self):
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, dataset, prediction_check_fn, criterion, shuffle=False):
        """Evaluate the probe on a subset of *dataset* and print metrics."""
        set_seed(self.seed)
        torch.manual_seed(self.seed)
        num_data = len(dataset)
        indices = list(range(num_data))
        if shuffle:
            np.random.shuffle(indices)
        subset_indices = indices[:self.eval_subset_size]
        sampler = SubsetRandomSampler(subset_indices)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.eval_batch_size)
        self.score_model.eval()
        total_eval_loss = 0
        valid_llm_accuracy = AccuracyMetric()
        valid_calibration_metrics = {
            "calibration_accuracy": BinaryAccuracy(),
            "calibration_auroc": BinaryAUROC(),
        }
        inner_loop = tqdm(data_loader, position=0, leave=True, disable=self.disable_tqdm)
        for batch in inner_loop:
            with torch.inference_mode():
                inputs = self.tokenizer(
                    batch['formatted_chat'],
                    return_tensors="pt",
                    padding=True,
                    padding_side='left',
                    return_token_type_ids=self.return_token_type_ids,
                    add_special_tokens=False,
                )
                inputs = inputs.to(self.device)
                outputs, scores, first, last = self.forward(**inputs)
            scores = torch.squeeze(scores)
            prompt_length = inputs['input_ids'].shape[1]
            generated = outputs.sequences[:, prompt_length:]
            predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            labels = get_batch_correctness_labels(predictions, batch, prediction_check_fn)
            loss = criterion(scores, labels.to(scores.device))
            total_eval_loss += loss.item()
            valid_llm_accuracy.update(labels.cpu().detach())
            for metric in valid_calibration_metrics.values():
                metric.update(scores.cpu().detach(), labels.cpu().detach())
            acc = valid_llm_accuracy.compute().item()

        avg_eval_loss = total_eval_loss / len(data_loader)
        print(f"Validation Loss: {avg_eval_loss:.2f}, LLM accuracy: {acc:.2f}")
        for k, v in valid_calibration_metrics.items():
            print(f"\t{k} {v.compute().item():.2f}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_dataset, valid_dataset, prediction_check_fn, cache):
        """Train the probe from scratch using live model inference.

        Hidden states and predictions are written to *cache* (a gzip-compressed
        JSONL file) so that subsequent runs can use ``fit_from_cache`` instead.

        Parameters
        ----------
        train_dataset : Dataset
            Formatted training dataset (must have a ``'formatted_chat'`` column).
        valid_dataset : Dataset
            Formatted validation dataset (currently unused in the training loop).
        prediction_check_fn : callable
            ``(prediction: str, reference: str) -> bool``
        cache : str or None
            Path to write the hidden-state cache.  Truncated/created if it
            doesn't exist.  Skipped if ``None``.
        """
        set_seed(self.seed)
        torch.manual_seed(self.seed)
        llm_accuracy_metric = AccuracyMetric()
        calibration_metrics = {
            "calibration_accuracy": BinaryAccuracy(),
            "calibration_auroc": BinaryAUROC(),
        }
        optimizer = AdamW(self.score_model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        for _epoch in range(0, self.num_epochs):
            sampler = RandomSampler(train_dataset, replacement=False,
                                    num_samples=len(train_dataset))
            train_dataloader = DataLoader(train_dataset, sampler=sampler,
                                          batch_size=self.train_batch_size)
            self.score_model.train()
            loop = tqdm(train_dataloader, leave=True, disable=self.disable_tqdm)

            for i, batch in enumerate(loop):
                optimizer.zero_grad()
                inputs = self.tokenizer(
                    batch['formatted_chat'],
                    return_tensors="pt",
                    return_token_type_ids=self.return_token_type_ids,
                    padding=True,
                    padding_side='left',
                    add_special_tokens=False,
                )
                inputs = inputs.to(self.device)
                outputs, scores, first_hidden, last_hidden = self.forward(**inputs)
                scores = torch.squeeze(scores)
                prompt_length = inputs['input_ids'].shape[1]
                generated = outputs.sequences[:, prompt_length:]
                predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

                labels = get_batch_correctness_labels(predictions, batch, prediction_check_fn)
                llm_accuracy_metric.update(labels)
                loss = criterion(scores, labels.to(scores.device))
                loss.backward()
                optimizer.step()
                for metric in calibration_metrics.values():
                    metric.update(scores.cpu().detach(), labels.cpu().detach())
                loop.set_description(f"Step {i}")
                loop.set_postfix(
                    loss=loss.item(),
                    llm_accuracy=llm_accuracy_metric.compute().item(),
                    **compute_metrics(calibration_metrics),
                )

                # Cache hidden states for future reuse
                if cache is not None:
                    try:
                        with gzip.open(cache, 'at', encoding='utf-8') as f:
                            try:
                                f.write(json.dumps({
                                    "first_state": first_hidden,
                                    "last_state": last_hidden,
                                    "predictions": predictions,
                                    "batch": batch,
                                }) + '\n')
                            except Exception:
                                b = dict(batch)
                                b.pop("llama_correct", None)
                                f.write(json.dumps({
                                    "first_state": first_hidden,
                                    "last_state": last_hidden,
                                    "predictions": predictions,
                                    "batch": b,
                                }) + '\n')
                    except Exception as e:
                        print(f"WARNING: failed to write training cache record: {e}")

        print("ACCURACY:", llm_accuracy_metric.compute())

    def fit_from_cache(self, train_cache, prediction_check_fn):
        """Train the probe from pre-cached hidden states.

        Reads batches from the compressed cache written by ``fit`` or
        ``predict_with_confidences``, computes correctness labels from stored
        predictions, and updates the probe.

        Parameters
        ----------
        train_cache : str
            Path to the gzip-compressed JSONL cache file.
        prediction_check_fn : callable
            ``(prediction: str, reference: str) -> bool``
        """
        import random as _random

        set_seed(self.seed)
        torch.manual_seed(self.seed)
        llm_accuracy_metric = AccuracyMetric()
        calibration_metrics = {
            "calibration_accuracy": BinaryAccuracy(),
            "calibration_auroc": BinaryAUROC(),
        }
        optimizer = AdamW(self.score_model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        self.score_model.train()
        _records = list(stream_compressed_data(train_cache))
        _rng = _random.Random(self.seed)
        _rng.shuffle(_records)
        loop = tqdm(_records, leave=True, disable=self.disable_tqdm, desc="Training from cache")

        for i, record in enumerate(loop):
            try:
                hidden = record["first_state"] if self.after_question_only else record["last_state"]
                scores = self.cached_forward(hidden)
                scores = scores.squeeze(-1)

                predictions = record["predictions"]
                batch = record["batch"]
                labels = get_batch_correctness_labels(predictions, batch, prediction_check_fn)

                optimizer.zero_grad()
                loss = criterion(scores, labels.to(scores.device))
                loss.backward()
                optimizer.step()

                llm_accuracy_metric.update(labels.cpu().detach())
                for metric in calibration_metrics.values():
                    metric.update(scores.cpu().detach(), labels.cpu().detach())

                loop.set_postfix(
                    loss=loss.item(),
                    llm_accuracy=llm_accuracy_metric.compute().item(),
                    **compute_metrics(calibration_metrics),
                )
            except Exception as e:
                print(f"WARNING: skipping cache record {i}: {e}")

        print("ACCURACY (from cache):", llm_accuracy_metric.compute())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                token_type_ids=None):
        """Run one batched generation step and return outputs, probe scores, and hidden states.

        Applies ``dropout_uncertainty`` when ``self.dropout > 0``.

        Returns
        -------
        outputs : GenerateOutput
        scores : torch.Tensor, shape (batch_size, 1)
        first : list
            Last-layer hidden state at the first generated token, per example.
        last : list
            Last-layer hidden state at the EOS token, per example.
        """
        if token_type_ids is None:
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                      'token_type_ids': token_type_ids}

        set_seed(self.seed)
        torch.manual_seed(self.seed)
        ctx = (
            dropout_uncertainty(self.model, dropout_rate=self.dropout, target="mlp")
            if self.dropout > 0
            else contextlib.nullcontext()
        )
        with ctx:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pad_token_id,
            )

        batch_size = len(outputs.sequences)
        first = []
        last = []
        hidden_states = []

        for i in range(batch_size):
            sequence = outputs.sequences[i]
            input_ids_i = inputs['input_ids'][i]
            prompt_length = input_ids_i.shape[0]
            generation = sequence[prompt_length:]
            n_generated = len(generation)
            take_last = False

            try:
                indices = torch.nonzero(generation == self.tokenizer.eos_token_id)
                end_pos = indices[0]
            except Exception:
                try:
                    eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    indices = torch.nonzero(generation == eot_id)
                    end_pos = indices[0]
                except Exception:
                    end_pos = n_generated
                    take_last = True

            n_hidden = len(outputs.hidden_states)
            assert n_generated == n_hidden
            hidden = outputs.hidden_states

            if len(hidden) == 1:
                logging.warning('Taking first and only generation for hidden state!')
                last.append(hidden[0].cpu())
            elif self.after_question_only:
                pass  # handled below
            elif ((n_generated - 1) >= len(hidden)) or take_last:
                logging.error('Taking last state because n_generated is too large')

            if ((n_generated - 1) >= len(hidden)) or take_last:
                last.append(hidden[-1][-1][i, -1, :].cpu().float().numpy().tolist())
            else:
                last.append(hidden[end_pos - 1][-1][i, -1, :].cpu().float().numpy().tolist())

            first.append(hidden[0][-1][i, -1, :].cpu().float().numpy().tolist())

            if self.after_question_only:
                last_input = hidden[0]
            elif ((n_generated - 1) >= len(hidden)) or take_last:
                last_input = hidden[-1]
            else:
                last_input = hidden[end_pos - 1]

            last_layer = last_input[-1]
            last_token_embedding = last_layer[i, -1, :]
            hidden_states.append(last_token_embedding)

        assert len(hidden_states) == batch_size
        hidden_states = torch.stack(hidden_states)
        scores = self.score_model(hidden_states)
        return outputs, scores, first, last

    def cached_forward(self, hidden_states):
        """Run the probe head on pre-computed hidden states loaded from cache."""
        hidden_states_tensor = (
            torch.tensor(hidden_states)
            .bfloat16()
            .to(next(self.score_model.parameters()).device)
        )
        return self.score_model(hidden_states_tensor)

    def predict_with_confidences(self, dataset, cache):
        """Run inference on *dataset* and return predictions, scores, and examples.

        Also writes a hidden-state cache to *cache* for future reuse.

        Returns
        -------
        all_predictions : list of str
        all_scores : list of torch.Tensor
        all_examples : list of dict
        """
        set_seed(self.seed)
        torch.manual_seed(self.seed)
        data_loader = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False)
        self.score_model.eval()
        inner_loop = tqdm(data_loader, position=0, leave=True, disable=self.disable_tqdm)
        all_predictions = []
        all_scores = []
        all_examples = []

        open(cache, "w").close()

        for batch in inner_loop:
            with torch.inference_mode():
                inputs = self.tokenizer(
                    batch['formatted_chat'],
                    return_token_type_ids=self.return_token_type_ids,
                    return_tensors="pt",
                    padding=True,
                    padding_side='left',
                    add_special_tokens=False,
                )
                inputs = inputs.to(self.device)
                outputs, scores, first_hidden, last_hidden = self.forward(**inputs)

            scores = torch.squeeze(scores).cpu().float()
            prompt_length = inputs['input_ids'].shape[1]
            generated = outputs.sequences[:, prompt_length:]
            predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

            with gzip.open(cache, 'at', encoding='utf-8') as f:
                try:
                    f.write(json.dumps({
                        "first_state": first_hidden,
                        "last_state": last_hidden,
                        "predictions": predictions,
                        "batch": batch,
                    }) + '\n')
                except Exception:
                    b = dict(batch)
                    b.pop("llama_correct", None)
                    f.write(json.dumps({
                        "first_state": first_hidden,
                        "last_state": last_hidden,
                        "predictions": predictions,
                        "batch": b,
                    }) + '\n')

            list_batch = dict_of_lists_to_list_of_dicts(batch)
            for i in range(outputs.sequences.shape[0]):
                all_predictions.append(predictions[i])
                all_scores.append(scores[i])
                all_examples.append(list_batch[i])

        return all_predictions, all_scores, all_examples

    def predict_with_confidences_cached(self, cache, max_examples: int | None = None):
        """Score cached examples without running the base LLM.

        Reads hidden states from *cache*, runs the probe head, and returns
        predictions, scores, and example metadata.

        Parameters
        ----------
        cache : str or Path
            Path to a gzip-compressed JSONL cache file.
        max_examples : int or None
            If set, stop after returning this many examples.
        """
        set_seed(self.seed)
        torch.manual_seed(self.seed)
        self.score_model.eval()
        all_predictions = []
        all_scores = []
        all_examples = []

        for record in tqdm(stream_compressed_data(cache)):
            if max_examples is not None and len(all_predictions) >= max_examples:
                break
            try:
                hidden = record["first_state"] if self.after_question_only else record["last_state"]
                scores = self.cached_forward(hidden)
                scores = scores.squeeze(-1).cpu().float()

                list_batch = dict_of_lists_to_list_of_dicts(record["batch"])
                for i in range(scores.shape[0]):
                    all_predictions.append(record["predictions"][i])
                    all_scores.append(scores[i])
                    all_examples.append(list_batch[i])
                    if max_examples is not None and len(all_predictions) >= max_examples:
                        break
            except Exception:
                pass

        return all_predictions, all_scores, all_examples

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the probe checkpoint to *path*."""
        model_args = dataclasses.asdict(self)
        model_args['torch_dtype'] = str(model_args['torch_dtype'])
        checkpoint = {
            'model_state_dict': self.score_model.state_dict(),
            'model_args': model_args,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, ckpt_path):
        """Load a ``HiddenStateRegression`` instance from a checkpoint file."""
        checkpoint = torch.load(ckpt_path)
        if 'model_args' not in checkpoint:
            raise ValueError(
                "Checkpoint is not compatible with this version of the code. "
                "Please re-train the model."
            )
        model_args = checkpoint['model_args']
        dtype_map = {
            'torch.bfloat16': torch.bfloat16,
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
        }
        if model_args['torch_dtype'] not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype: {model_args['torch_dtype']}")
        model_args['torch_dtype'] = dtype_map[model_args['torch_dtype']]
        model = cls(**model_args)
        model.score_model.load_state_dict(checkpoint['model_state_dict'])
        return model
