"""Wrap preloaded model in lm-eval's HFLM. Lazy lm_eval import, no device relocation.

Spec 04 owns dtype/device_map/generation_config. This never reloads, casts,
or moves the model (no device=); profiled and evaluated objects remain identical.
"""

from __future__ import annotations

from typing import Any


def build_hflm(
    model: Any, tokenizer: Any, *, batch_size: int, max_length: int | None = None
):
    """Return an lm_eval HFLM wrapping the preloaded model. Imports lm_eval lazily."""
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
