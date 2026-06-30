"""Spec 12 extensions — a gated sibling of the v1 core.

This package promotes the five spec-only entries of
``docs/specs/12-extensions-catalog.md`` (Mistral, JudgeBench, W8A8/SmoothQuant,
quantized KV cache, QLoRA recovery) into runnable code WITHOUT touching the v1 SSOT.

It defines its own id space (``EXT_VARIANT_IDS`` / ``EXT_TASK_IDS``) and
``ExtVariant`` / ``ExtTask`` dataclasses (subclasses of the v1 ``Variant`` / ``Task``),
so the v1 matrix / cell-writer / resume / aggregation machinery is reused verbatim while
``qquant.registry.VARIANT_IDS`` / ``TASK_IDS`` and the v1 registries stay unchanged.

Everything here is GATED: ext variants ship ``enabled: false`` and the only entry point
is ``qquant-ext`` writing under a separate ``results-ext/`` root. Import-time torch-free
(torch/transformers/llmcompressor are imported lazily inside method bodies).
"""

from __future__ import annotations
