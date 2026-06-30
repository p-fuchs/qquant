"""EXT loader: KV-cache gen-kwargs, dispatch, adapter/KV wiring — all torch-free."""

from __future__ import annotations

import pytest

import qquant.ext.loaders as loaders
from qquant.ext.loaders import (
    build_kv_cache_generation_kwargs,
    ext_load_variant,
    resolve_local_path,
)
from qquant.ext.registry import ExtVariant


def test_kv_cache_kwargs_none():
    assert build_kv_cache_generation_kwargs(None) == {}


def test_kv_cache_kwargs_quantized():
    kw = build_kv_cache_generation_kwargs({"backend": "quanto", "nbits": 4})
    assert kw["cache_implementation"] == "quantized"
    assert kw["cache_config"] == {"backend": "quanto", "nbits": 4}


def test_resolve_local_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_local_path("local:checkpoints/qlora/none", checkpoints_root=tmp_path)


class _GenCfg:
    def __init__(self):
        self.do_sample = True


class _FakeModel:
    def __init__(self):
        self.generation_config = _GenCfg()
        self.config = None
        self.hf_device_map = {"": 0}


def _install_fake_builder(monkeypatch, model):
    def fake_builder(variant, source, opts):
        return model, object(), {"dtype": "bfloat16", "transformers_version": "test"}

    monkeypatch.setitem(loaders.BUILDERS, "bf16", fake_builder)


def test_ext_load_applies_kv_cache(monkeypatch):
    model = _FakeModel()
    _install_fake_builder(monkeypatch, model)
    variant = ExtVariant(
        id="bf16-kvq4",
        quant_method="bf16",
        source="baseline",
        enabled=True,
        model_id="some/repo",
        revision="abc",
        kv_cache={"backend": "quanto", "nbits": 4},
    )
    loaded = ext_load_variant("bf16-kvq4", variants={"bf16-kvq4": variant})
    # force_greedy ran (do_sample cleared) AND the quantized cache was installed.
    assert model.generation_config.do_sample is False
    assert model.generation_config.cache_implementation == "quantized"
    assert loaded.metadata["kv_cache"]["cache_config"]["nbits"] == 4


def test_ext_load_disabled_rejected():
    variant = ExtVariant(
        id="bf16-kvq4",
        quant_method="bf16",
        source="baseline",
        enabled=False,
        model_id="some/repo",
        revision="abc",
    )
    with pytest.raises(ValueError, match="disabled"):
        ext_load_variant("bf16-kvq4", variants={"bf16-kvq4": variant})


def test_ext_load_unknown_id():
    with pytest.raises(KeyError):
        ext_load_variant("nope", variants={})


def test_ext_load_refuses_cpu_offload(monkeypatch):
    model = _FakeModel()
    model.hf_device_map = {"": "cpu"}
    _install_fake_builder(monkeypatch, model)
    variant = ExtVariant(
        id="bf16-kvq4",
        quant_method="bf16",
        source="baseline",
        enabled=True,
        model_id="some/repo",
        revision="abc",
    )
    with pytest.raises(RuntimeError, match="offloaded"):
        ext_load_variant("bf16-kvq4", variants={"bf16-kvq4": variant})
