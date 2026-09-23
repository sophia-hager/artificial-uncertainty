import torch
from torcheval.metrics.metric import Metric
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryAUROC
from torchmetrics.classification import BinaryCalibrationError
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import KBinsDiscretizer
import torch.nn as nn
import numpy as np


class AccuracyMetric(Metric[torch.Tensor]):
    """Streaming accuracy metric compatible with the torcheval Metric API."""

    def __init__(self, device=None) -> None:
        super().__init__(device=device)
        self._add_state("outcomes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, new_outcomes):
        self.outcomes = torch.cat((self.outcomes, new_outcomes))
        return self

    @torch.inference_mode()
    def compute(self):
        total = torch.sum(self.outcomes)
        return total / self.outcomes.shape[0]

    @torch.inference_mode()
    def merge_state(self, metrics):
        outcomes = [self.outcomes]
        for metric in metrics:
            outcomes.append(metric.outcomes)
        self.outcomes = torch.cat(outcomes)
        return self


def compute_metrics(metrics):
    """Compute and return a dict of scalar values from a metrics dict."""
    computed_metrics = dict()
    for key in sorted(metrics.keys()):
        metric = metrics[key]
        computed_metrics[key] = metric.compute().item()
    return computed_metrics


def get_correctness_label(prediction, reference, prediction_check_fn):
    """Return a correctness label (bool) for a single prediction."""
    return prediction_check_fn(prediction, reference)


def get_batch_correctness_labels(predictions, batch, prediction_check_fn):
    """Return a float tensor of per-example correctness labels for a batch."""
    labels = []
    for i, prediction in enumerate(predictions):
        reference = batch['answer'][i]
        if type(reference) == int:
            reference = 'abcde'[reference]
        label = get_correctness_label(prediction, reference, prediction_check_fn)
        labels.append(label)
    return torch.tensor(labels).float()


def compute_binned_metrics(scores, labels, num_bins=5, encode="ordinal", strategy="uniform"):
    """Compute AUROC and top-bin accuracy using a binned approximation.

    Note: only ``num_bins=5`` is currently supported.
    """
    if num_bins != 5:
        raise NotImplementedError
    binner = KBinsDiscretizer(num_bins, encode=encode, strategy=strategy)
    scores = np.array(scores).reshape(-1, 1)
    bins = binner.fit_transform(scores)
    probs = [(0.2 * b[0]) - 0.1 for b in bins]
    arr_labels = np.array(labels)
    top = [l for b, l in zip(bins, labels) if b > 2]
    return {
        f'calibration_auroc_nbins={num_bins}': roc_auc_score(arr_labels, probs),
        f'high_accuracy_nbins={num_bins}': np.array(top).mean(),
        f'high_accuracy_total={num_bins}': len(top),
    }


def evaluate(predictions, scores, examples, prediction_check_fn):
    """Compute calibration metrics over a set of predictions and scores.

    Parameters
    ----------
    predictions : list of str
        Model-generated text responses.
    scores : list of torch.Tensor
        Raw logit scores produced by the score model (before sigmoid).
    examples : list of dict
        Per-example metadata dicts containing at least an ``'answer'`` key.
    prediction_check_fn : callable
        Function ``(prediction: str, reference: str) -> bool`` that returns
        True if the prediction is correct.

    Returns
    -------
    dict
        Metric values including ``'llm_accuracy'``, ``'Brier'``, ``'var'``,
        ``'calibration_accuracy'``, ``'calibration_auroc'``, and
        ``'binary_calibration_error'``.
    """
    llm_accuracy = AccuracyMetric()
    calibration_metrics = {
        "calibration_accuracy": BinaryAccuracy(),
        "calibration_auroc": BinaryAUROC(),
        "binary_calibration_error": BinaryCalibrationError(n_bins=15, norm='l1'),
    }
    criterion = nn.BCEWithLogitsLoss()
    n_examples = 0
    total_eval_loss = 0.
    labels = []
    norm_scores = []

    for example, prediction, score in zip(examples, predictions, scores):
        try:
            raw_answer = example['answer']
            if isinstance(raw_answer, (list, tuple)):
                reference = raw_answer[0]
                if isinstance(reference, int):
                    reference = 'abcde'[reference]
            elif isinstance(raw_answer, int):
                reference = 'abcde'[raw_answer]
            else:
                reference = raw_answer

            label = get_correctness_label(prediction, reference, prediction_check_fn)
            labels.append(label)
            label = torch.tensor(label).float()
            n_examples += 1
            score = torch.tensor(score).float()
            loss = criterion(score, label)
            total_eval_loss += loss.item()
            score = torch.sigmoid(torch.unsqueeze(score, 0)).clamp(min=1e-7, max=1 - 1e-7)
            norm_scores.append(score)
            label = torch.unsqueeze(label, 0)
            llm_accuracy.update(label)
            for metric in calibration_metrics.values():
                metric.update(score, label)
        except Exception:
            pass

    results = {
        'llm_accuracy': llm_accuracy.compute().item(),
        'Brier': brier_score_loss(labels, [l.item() for l in norm_scores]),
        'var': np.var([l.item() for l in norm_scores]),
    }
    for k, v in calibration_metrics.items():
        results[k] = v.compute().item()
    return results
