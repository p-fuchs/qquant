"""EXT-4 quantized-KV-cache helpers (efficiency sub-study). Import-time torch-free.

The runtime path reuses everything: the paired ids (``bf16`` ↔ ``bf16-kvq4``,
``bnb-nf4`` ↔ ``bnb-nf4-kvq4``) are loaded by ``qquant.ext.loaders.ext_load_variant``
(which installs the quantized cache on ``generation_config``) and profiled by the v1
efficiency profiler unchanged. The only ext-specific logic is *pairing* a kvq variant
with its fp16-KV baseline so the report can quote the long-context memory/throughput
headroom and the (lossy-cache) quality delta.
"""

from __future__ import annotations

from qquant.ext.registry import ExtVariant

# kvq id -> the v1 base variant id it should be compared against (its fp16-KV baseline).
KVQ_BASELINE: dict[str, str] = {
    "bf16-kvq4": "bf16",
    "bnb-nf4-kvq4": "bnb-nf4",
}


def is_kvq_variant(variant: ExtVariant) -> bool:
    """True when the variant carries a quantized-KV-cache knob."""
    return variant.kv_cache is not None


def baseline_id_for(variant_id: str) -> str | None:
    """The fp16-KV baseline (a v1 variant id) a kvq variant is measured against."""
    return KVQ_BASELINE.get(variant_id)


def kvq_pairs(variants: dict[str, ExtVariant]) -> list[tuple[str, str]]:
    """``(kvq_id, baseline_id)`` pairs for every enabled kvq variant present."""
    pairs: list[tuple[str, str]] = []
    for vid, v in variants.items():
        if v.enabled and is_kvq_variant(v):
            baseline = baseline_id_for(vid)
            if baseline is not None:
                pairs.append((vid, baseline))
    return pairs
