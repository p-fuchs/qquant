from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

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


class _FakeParam:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeModel:
    """Stand-in for a loaded transformers model — no torch needed."""

    def __init__(self, device_map):
        self.hf_device_map = device_map
        self.generation_config = SimpleNamespace(
            do_sample=True, num_beams=3, temperature=0.7, top_p=0.9, top_k=40
        )
        self.config = SimpleNamespace()

    def parameters(self):
        yield _FakeParam("torch.bfloat16")


def _fake_builder(model):
    def builder(variant, source, opts):
        return model, object(), {"dtype": "bfloat16", "transformers_version": "5.10.1"}

    return builder


def test_load_variant_dispatches_builds_and_assembles_metadata(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    lv = loaders.load_variant("bf16")
    assert lv.model is model
    assert lv.variant.id == "bf16"
    md = lv.metadata
    assert md["variant_id"] == "bf16"
    assert md["quant_method"] == "bf16"
    assert md["model_source"] == "Qwen/Qwen2.5-7B-Instruct"
    assert md["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert md["is_local"] is False
    assert md["dtype"] == "bfloat16"
    assert md["param_dtype"] == "torch.bfloat16"
    assert md["attn_implementation"] == "sdpa"
    assert md["device_map"] == {"": 0}
    assert isinstance(md["load_seconds"], float)
    assert md["transformers_version"] == "5.10.1"
    # greedy forced post-build:
    assert lv.model.generation_config.do_sample is False
    assert lv.model.generation_config.num_beams == 1


def test_load_variant_unknown_id_raises_keyerror():
    from qquant.models import loaders

    with pytest.raises(KeyError):
        loaders.load_variant("nope-not-a-variant")


def test_load_variant_disabled_raises_valueerror():
    from qquant.models import loaders
    from qquant.registry import Variant

    fake = {
        "bf16": Variant(
            id="bf16",
            quant_method="bf16",
            source="baseline",
            enabled=False,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            notes="",
        )
    }
    with pytest.raises(ValueError):
        loaders.load_variant("bf16", variants=fake)


def test_load_variant_rejects_cpu_offload(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0, "model.layers.20": "cpu"})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    with pytest.raises(RuntimeError):
        loaders.load_variant("bf16")


def test_loaded_variant_context_manager_unloads(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    with loaders.load_variant("bf16") as lv:
        assert lv.model is model
    assert lv.model is None  # unload() ran on context exit


def test_smoke_test_load_never_raises(monkeypatch):
    from qquant.models import loaders

    def boom(*args, **kwargs):
        raise RuntimeError("kapow")

    monkeypatch.setattr(loaders, "load_variant", boom)
    ok, msg = loaders.smoke_test_load("awq-official")
    assert ok is False
    assert "kapow" in msg


def test_importing_qquant_models_is_torch_free():
    """Importing the package (not running a builder) must not pull torch in.

    Run in a fresh interpreter so another test's torch import can't mask a regression.
    """
    code = (
        "import importlib, sys;"
        "importlib.import_module('qquant.models');"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, f"qquant.models imported torch: {result.stderr}"
