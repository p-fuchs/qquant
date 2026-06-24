from __future__ import annotations

import pytest

from qquant.models.loaders import ModelSource, resolve_model_source
from qquant.registry import load_variants


def test_resolve_official_source():
    variants = load_variants()
    src = resolve_model_source(variants["gptq-official"])
    assert isinstance(src, ModelSource)
    assert src.path_or_repo == "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    assert src.revision == "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"
    assert src.is_local is False


def test_resolve_baseline_source():
    variants = load_variants()
    src = resolve_model_source(variants["bf16"])
    assert src.path_or_repo == "Qwen/Qwen2.5-7B-Instruct"
    assert src.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert src.is_local is False


def test_resolve_selfquant_source_existing(tmp_path):
    variants = load_variants()
    ckpt = tmp_path / "self-quant" / "gptq-selfquant"
    ckpt.mkdir(parents=True)
    src = resolve_model_source(variants["gptq-selfquant"], checkpoints_root=tmp_path)
    assert src.path_or_repo == str(ckpt)
    assert src.revision is None
    assert src.is_local is True


def test_resolve_selfquant_missing_raises(tmp_path):
    variants = load_variants()
    with pytest.raises(FileNotFoundError):
        resolve_model_source(variants["awq-selfquant"], checkpoints_root=tmp_path)


def test_builders_cover_quant_methods_exactly():
    from qquant.models.loaders import BUILDERS
    from qquant.registry import QUANT_METHODS

    assert set(BUILDERS.keys()) == set(QUANT_METHODS)


def test_every_registered_variant_maps_to_a_builder():
    from qquant.models.loaders import BUILDERS

    variants = load_variants()
    for v in variants.values():
        assert v.quant_method in BUILDERS, f"{v.id}: no builder for {v.quant_method!r}"
