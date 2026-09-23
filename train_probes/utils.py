import dataclasses
import json
import os
import random
from typing import Any

import torch
from datasets import load_dataset as _load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_dataset(dataset, test, limit=None, difficult=False):
    """Load and return a HuggingFace Dataset for the given dataset name.

    Parameters
    ----------
    dataset : str
        Name of the dataset split to load. Supported values: ``'gpqa'``,
        ``'mmlu-pro'``, ``'test-ARC'``, ``'calibration'``.
    test : bool
        When ``True``, load the held-out test split rather than the
        validation split.
    limit : int or None
        If set, randomly subsample to at most *limit* examples.
    difficult : bool
        For ``'gpqa'``: when ``True``, use a smaller easy-question set to
        make the mixed dataset more challenging overall.

    Returns
    -------
    datasets.Dataset
    """
    random.seed(42)
    print(f"Loading dataset: {dataset}")

    if dataset == 'mmlu-pro':
        d = _load_dataset('answerdotai/MMLU-SemiPro')
        stem_categories = [
            "math", "health", "chemistry", "physics",
            "engineering", "biology", "computer science",
        ]
        if test:
            pred_dataset = d["test"].shuffle(seed=42)
            pred_dataset = pred_dataset.filter(
                lambda ex: ex["category"] in stem_categories)
        else:
            pred_dataset = d["train"].shuffle(seed=42)
            pred_dataset = pred_dataset.filter(
                lambda ex: ex["category"] in stem_categories).select(range(1000))
        pred_dataset = pred_dataset.remove_columns(
            ["question_id", "answer", "cot_content", "category", "src"])
        pred_dataset = pred_dataset.rename_column("question", "Question")
        pred_dataset = pred_dataset.rename_column("answer_index", "answer")

    elif dataset == "test-ARC":
        if test:
            keys = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
                    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            easy = _load_dataset('allenai/ai2_arc', 'ARC-Easy')["test"]
            c = [{"Question": i["question"],
                  "options": i["choices"]["text"],
                  "answer": keys[i["answerKey"]]}
                 for i in easy]
            pred_dataset = Dataset.from_list(c)
        else:
            raise ValueError("test-ARC only supports test=True")

    elif dataset == 'gpqa':
        pred_dataset, diamond_dataset = _load_gpqa_split()
        if test:
            pred_dataset = diamond_dataset["train"]
        else:
            pred_dataset = _gpqa_non_diamond(pred_dataset, diamond_dataset).select(range(50))
            num = 10 if difficult else 25
            # ARC examples deliberately omitted in current configuration
            _ = _load_dataset('allenai/ai2_arc', 'ARC-Challenge')["validation"].select(range(num))
            _ = _load_dataset('allenai/ai2_arc', 'ARC-Easy')["validation"].select(range(num))
            return pred_dataset


    elif dataset == 'calibration':
        pred_dataset = _load_easy_calibration_set()

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if limit is not None:
        pred_dataset = pred_dataset.shuffle(seed=42).select(
            range(min(int(limit), len(pred_dataset))))
    return pred_dataset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_gpqa_split():
    """Return (main, diamond) GPQA splits, both without the Canary String column."""
    pred_dataset = _load_dataset('jeggers/gpqa_formatted', 'main')
    diamond_dataset = _load_dataset('jeggers/gpqa_formatted', 'diamond')
    pred_dataset = pred_dataset.remove_columns(["Canary String"])
    return pred_dataset, diamond_dataset


def _gpqa_non_diamond(pred_dataset, diamond_dataset):
    """Return the subset of pred_dataset whose questions are not in diamond_dataset."""
    diamond_questions = {d["Question"] for d in diamond_dataset["train"]}
    keep = [
        i for i, ex in enumerate(pred_dataset["train"])
        if ex["Question"] not in diamond_questions
    ]
    return pred_dataset["train"].select(keep)


def _load_easy_calibration_set():
    """Load the default easy calibration set from ``../data/cal.jsonl``.

    Expected format: a JSONL file with columns ``Question``, ``options``,
    ``answer``.  The path is resolved relative to this file's location so the
    repository can be used from any working directory.
    """
    cal_path = os.path.join(os.path.dirname(__file__), "..", "data", "cal.jsonl")
    return _load_dataset("json", data_files=cal_path)["train"]


def get_probs_labels(dictionary, i):
    """Extract probability and label lists from a predictions dictionary.

    Parameters
    ----------
    dictionary : list of dict
        Each element should have keys ``'counts'``, ``'label'``, ``'probability'``,
        and ``'answer'``.
    i : str
        ``'sample'`` to expand per-sample counts; any other string for single
        probability entries.
    """
    labels = []
    probs = []
    for item in dictionary:
        if i == "sample":
            for count in item["counts"]:
                if count == item["label"]:
                    labels.append(1)
                else:
                    labels.append(0)
                probs.append(item["counts"][count] / 20)
        else:
            probs.append(item["probability"])
            if item["answer"] == item["label"]:
                labels.append(1)
            else:
                labels.append(0)
    return labels, probs


# ---------------------------------------------------------------------------
# Batched inference utility
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BatchInference:
    """Batched LLM inference for generating predictions over a dataset.

    Loads a model, formats the dataset with ``prepare_dataset``, runs
    generation in batches, and writes results as JSONL to ``output_path``.

    Parameters
    ----------
    dataset : datasets.Dataset
        A HuggingFace Dataset with at least a ``'Question'`` and ``'options'``
        column (formatted by ``prepare_dataset`` internally).
    model_name_or_path : str
        HuggingFace model ID or local path.
    task_instruction : str
        System prompt used to format each example.
    num_samples : int
        Number of independent samples to generate per example.
    output_path : str
        JSONL file where predictions are written.
    batch_size : int
        Number of examples per inference batch.
    """
    dataset: Any
    model_name_or_path: str
    task_instruction: str
    num_samples: int = 1
    output_path: str = "predictions.jsonl"
    max_new_tokens: int = 1024
    num_shards: int = 1
    shard_index: int = 0
    compile_model: bool = False
    show_progress: bool = False
    do_sample: bool = True
    top_k: int = 1
    top_p: float = 0.95
    temperature: float = 1.
    repetition_penalty: float = 1.
    torch_dtype: Any = torch.bfloat16
    batch_size: int = 64

    def __post_init__(self):
        if not torch.cuda.is_available():
            raise ValueError("No GPU available.")
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")

        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.model.eval()

        if self.compile_model:
            print("Compiling model.")
            self.model = torch.compile(self.model)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Using EOS token as pad token: {tokenizer.pad_token}")
        elif tokenizer.pad_token is None:
            raise ValueError(
                "The tokenizer does not have a pad token and no EOS token was found. "
                "Please use a different tokenizer or manually pad your inputs."
            )
        self.tokenizer = tokenizer

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=self.num_samples,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        from data_utils import prepare_dataset, dict_of_lists_to_list_of_dicts
        self._dict_of_lists_to_list_of_dicts = dict_of_lists_to_list_of_dicts

        self.dataset = prepare_dataset(
            self.dataset, self.tokenizer, task_instruction=self.task_instruction)

        if self.num_shards > 1:
            self.dataset = self.dataset.shard(
                num_shards=self.num_shards, index=self.shard_index)

        self.data_loader = DataLoader(
            self.dataset, shuffle=False, batch_size=self.batch_size, collate_fn=None)

    def run(self):
        """Run inference and write predictions to ``self.output_path``."""
        if self.num_samples > 1:
            raise NotImplementedError("num_samples > 1 is not yet supported.")

        disable = not self.show_progress
        with open(self.output_path, 'w') as fp:
            for batch in tqdm(self.data_loader, disable=disable):
                inputs = self.tokenizer(
                    batch['formatted_chat'],
                    return_tensors="pt",
                    padding=True,
                    padding_side='left',
                )
                inputs = inputs.to(self.model.device)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, generation_config=self.generation_config)
                outputs = self.tokenizer.batch_decode(
                    outputs[:, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                )
                examples = self._dict_of_lists_to_list_of_dicts(batch)
                for i, output in enumerate(outputs):
                    example = examples[i]
                    example['generated_answer'] = output
                    fp.write(json.dumps(example) + '\n')
