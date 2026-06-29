from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from qquant.eval.hflm import build_hflm


def _install_fake_lm_eval(monkeypatch):
    calls = {}

    class FakeHFLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    pkg = ModuleType("lm_eval")
    models = ModuleType("lm_eval.models")
    hf = ModuleType("lm_eval.models.huggingface")
    hf.HFLM = FakeHFLM
    monkeypatch.setitem(sys.modules, "lm_eval", pkg)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf)
    return calls


def test_build_hflm_passes_preloaded_model_no_device(monkeypatch):
    calls = _install_fake_lm_eval(monkeypatch)
    model = SimpleNamespace(name="m")
    tok = SimpleNamespace(name="t")
    build_hflm(model, tok, batch_size=4, max_length=4096)
    assert calls["pretrained"] is model
    assert calls["tokenizer"] is tok
    assert calls["batch_size"] == 4
    assert calls["max_length"] == 4096
    assert "device" not in calls  # never relocate a quantized model
