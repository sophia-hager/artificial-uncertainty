"""hidden_dropout.py

Artificial uncertainty via dropout and noise injection.

Provides context managers that inject perturbations into transformer
decoder layers during inference, creating degraded / uncertain model
trajectories on easy data without any weight updates.

Two perturbation mechanisms:

1. **Residual dropout** (``dropout_uncertainty``): forward hooks on
   attention/MLP sub-layers apply dropout to their outputs before they
   are added to the residual stream. Supports layer-selective dropout
   via the ``layers`` parameter.

2. **Embedding noise** (``embedding_noise``): a forward hook on the
   input embedding layer adds scaled Gaussian noise, producing
   perturbations that propagate naturally through the full model.
   Compatible with any attention implementation (no eager mode needed).

Supported dropout targets (controlled by the ``target`` argument):
  - ``"both"``      — hooks on attention AND MLP sub-layers (default)
  - ``"attention"``  — hooks on attention sub-layers only
  - ``"mlp"``        — hooks on MLP sub-layers only
  - ``"post_layer"`` — hooks on the full decoder layer output

Supported layer specifications (controlled by the ``layers`` argument):
  - ``None``        — all decoder layers (default, backward compatible)
  - ``"top_N"``     — only the last N decoder layers
  - ``"bottom_N"``  — only the first N decoder layers
  - ``(start, end)``— only decoder layers with index in [start, end)
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DropoutTarget = Literal["both", "attention", "mlp", "post_layer"]
VALID_TARGETS: set[str] = {"both", "attention", "mlp", "post_layer"}

# Type alias for the layers specification
LayerSpec = None | str | tuple[int, int]


def parse_layer_spec(spec: str | None) -> LayerSpec:
    """Parse a layer specification string into a canonical form.

    Accepts:
      - None or "" -> None (all layers)
      - "top_N"    -> "top_N"   [handled by _filter_by_layer_index]
      - "bottom_N" -> "bottom_N"
      - "S-E"      -> (S, E)    [integer range, end exclusive]

    Returns:
        None, a string like "top_8", or a (start, end) tuple.
    """
    if spec is None or spec == "":
        return None
    spec = spec.strip()
    m = re.match(r"^(top|bottom)_(\d+)$", spec, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    m = re.match(r"^(\d+)-(\d+)$", spec)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(
        f"Invalid layer spec {spec!r}. "
        f"Expected 'top_N', 'bottom_N', or 'start-end'."
    )


def _find_decoder_layers(model: nn.Module) -> list[tuple[int, str, nn.Module]]:
    """Find all decoder layers, returning (index, name, module) triples.

    The index is the position among all DecoderLayer modules found,
    ordered by their appearance in ``model.named_modules()``.
    """
    layers = []
    idx = 0
    for name, module in model.named_modules():
        if "DecoderLayer" in type(module).__name__:
            layers.append((idx, name, module))
            idx += 1
    return layers


def _resolve_layer_indices(
    total_layers: int,
    layers: LayerSpec,
) -> set[int]:
    """Resolve a layer spec into a set of integer indices."""
    if layers is None:
        return set(range(total_layers))

    if isinstance(layers, tuple):
        start, end = layers
        if start < 0 or end > total_layers or start >= end:
            raise ValueError(
                f"Layer range ({start}, {end}) is invalid for a model "
                f"with {total_layers} decoder layers."
            )
        return set(range(start, end))

    m = re.match(r"^(top|bottom)_(\d+)$", layers)
    if not m:
        raise ValueError(f"Cannot resolve layer spec {layers!r}")

    direction = m.group(1)
    n = int(m.group(2))
    if n > total_layers:
        raise ValueError(
            f"Requested {direction}_{n} but model has only "
            f"{total_layers} decoder layers."
        )

    if direction == "top":
        return set(range(total_layers - n, total_layers))
    else:
        return set(range(n))


def _find_modules(
    model: nn.Module,
    target: DropoutTarget,
    layers: LayerSpec = None,
) -> list[tuple[str, nn.Module]]:
    """Find modules to attach dropout hooks to.

    For ``"attention"``, ``"mlp"``, and ``"both"``: searches for direct
    children of DecoderLayer modules whose class names contain
    'Attention' or 'MLP'.

    For ``"post_layer"``: returns the DecoderLayer modules themselves.

    Args:
        model: The transformer model.
        target: Which sub-layers to select within each decoder layer.
        layers: Which decoder layers to include. See module docstring.
    """
    decoder_layers = _find_decoder_layers(model)
    total = len(decoder_layers)

    if total == 0:
        return []

    allowed = _resolve_layer_indices(total, layers)

    results = []
    for idx, dec_name, dec_module in decoder_layers:
        if idx not in allowed:
            continue

        if target == "post_layer":
            results.append((dec_name, dec_module))
            continue

        for child_name, child_module in dec_module.named_children():
            child_cls = type(child_module).__name__
            is_attn = "Attention" in child_cls
            is_mlp = "MLP" in child_cls

            if target == "both" and (is_attn or is_mlp):
                results.append((f"{dec_name}.{child_name}", child_module))
            elif target == "attention" and is_attn:
                results.append((f"{dec_name}.{child_name}", child_module))
            elif target == "mlp" and is_mlp:
                results.append((f"{dec_name}.{child_name}", child_module))

    return results


@contextlib.contextmanager
def dropout_uncertainty(
    model: nn.Module,
    dropout_rate: float = 0.2,
    target: DropoutTarget = "both",
    layers: LayerSpec = None,
) -> Iterator[nn.Module]:
    """Context manager: inject residual dropout to induce uncertainty.

    Registers forward hooks on selected sub-layers inside the model's
    decoder layers.  Each hook applies dropout to the sub-layer's output
    tensor(s), perturbing the residual stream that the logit lens reads.

    Args:
        model: A HuggingFace causal LM.
        dropout_rate: Dropout probability to inject.
        target: Which sub-layers to attach hooks to.
            ``"both"`` — attention and MLP (default).
            ``"attention"`` — attention only.
            ``"mlp"`` — MLP only.
            ``"post_layer"`` — full decoder layer output.
        layers: Which decoder layers to include.
            ``None`` — all layers (default).
            ``"top_N"`` — last N layers only.
            ``"bottom_N"`` — first N layers only.
            ``(start, end)`` — layers in [start, end) only.

    Yields:
        The same model with dropout hooks active.
    """
    if target not in VALID_TARGETS:
        raise ValueError(
            f"Invalid dropout target {target!r}. "
            f"Must be one of {sorted(VALID_TARGETS)}"
        )

    modules = _find_modules(model, target, layers=layers)

    if len(modules) == 0:
        logger.warning(
            f"No modules found for target={target!r}, layers={layers!r}. "
            "Dropout injection will have no effect."
        )

    hooks = []

    def _make_hook(p: float):
        """Create a hook that applies dropout to the first output tensor."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                dropped = F.dropout(output[0], p=p, training=True)
                return (dropped,) + output[1:]
            else:
                return F.dropout(output, p=p, training=True)
        return hook_fn

    for name, mod in modules:
        h = mod.register_forward_hook(_make_hook(dropout_rate))
        hooks.append(h)

    layers_desc = f", layers={layers}" if layers is not None else ""
    logger.info(
        f"Injected residual dropout (p={dropout_rate}, target={target}"
        f"{layers_desc}) via hooks on {len(modules)} modules"
    )

    try:
        yield model
    finally:
        for h in hooks:
            h.remove()
        logger.info(
            f"Removed {len(hooks)} dropout hooks, restored original model"
        )


@contextlib.contextmanager
def embedding_noise(
    model: nn.Module,
    noise_scale: float = 1.0,
) -> Iterator[nn.Module]:
    """Context manager: inject Gaussian noise at the embedding layer.

    Registers a forward hook on the model's input embedding module that
    adds ``noise_scale * std(output) * randn_like(output)`` to the
    embedding output. The noise propagates naturally through all
    transformer layers without requiring eager attention mode.

    Args:
        model: A HuggingFace causal LM.
        noise_scale: Multiplier for the noise magnitude. At scale=1.0,
            noise standard deviation equals the embedding output's
            standard deviation (per-forward-pass).

    Yields:
        The same model with the embedding noise hook active.
    """
    embed = model.get_input_embeddings()
    if embed is None:
        raise ValueError(
            "Cannot find input embeddings via model.get_input_embeddings(). "
            "Ensure the model is a HuggingFace causal LM."
        )

    def _noise_hook(module, input, output):
        std = output.std()
        noise = torch.randn_like(output) * std * noise_scale
        return output + noise

    hook = embed.register_forward_hook(_noise_hook)
    logger.info(
        f"Injected embedding noise (scale={noise_scale}) on "
        f"{type(embed).__name__}"
    )

    try:
        yield model
    finally:
        hook.remove()
        logger.info("Removed embedding noise hook, restored original model")
