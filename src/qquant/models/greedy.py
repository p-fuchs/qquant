"""Greedy-decoding enforcement, kept torch-free so it is unit-testable with a stub.

Qwen ships ``generation_config`` with ``do_sample=true``; ``force_greedy`` clamps
the model to deterministic greedy decoding at load time. Spec 05 mirrors this in
``gen_kwargs``.
"""

from __future__ import annotations

from typing import Any


def force_greedy(model: Any) -> None:
    """Force greedy decoding in place. Idempotent.

    Sets ``do_sample=False``, ``num_beams=1`` and clears ``temperature/top_p/top_k``
    on ``model.generation_config``; also clears those sampling fields on
    ``model.config`` when present, so a stray sampling default cannot leak into
    generation.
    """
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        gen.do_sample = False
        gen.num_beams = 1
        gen.temperature = None
        gen.top_p = None
        gen.top_k = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, None)
