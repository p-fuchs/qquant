"""EXT-4 KV-cache pairing helpers (torch-free)."""

from __future__ import annotations

from qquant.ext.kvcache import baseline_id_for, is_kvq_variant, kvq_pairs
from qquant.ext.registry import ExtVariant


def _kvq(vid, enabled=True):
    return ExtVariant(
        id=vid,
        quant_method="bf16",
        source="baseline",
        enabled=enabled,
        model_id="x",
        revision=None,
        kv_cache={"backend": "quanto", "nbits": 4},
    )


def test_baseline_mapping():
    assert baseline_id_for("bf16-kvq4") == "bf16"
    assert baseline_id_for("bnb-nf4-kvq4") == "bnb-nf4"
    assert baseline_id_for("nope") is None


def test_is_kvq_variant():
    assert is_kvq_variant(_kvq("bf16-kvq4"))
    plain = ExtVariant(
        id="w8a8-selfquant",
        quant_method="compressed-tensors",
        source="selfquant",
        enabled=True,
        model_id="x",
        revision=None,
    )
    assert not is_kvq_variant(plain)


def test_kvq_pairs_only_enabled():
    variants = {
        "bf16-kvq4": _kvq("bf16-kvq4", enabled=True),
        "bnb-nf4-kvq4": _kvq("bnb-nf4-kvq4", enabled=False),
    }
    assert kvq_pairs(variants) == [("bf16-kvq4", "bf16")]
