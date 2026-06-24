from __future__ import annotations

from types import SimpleNamespace

from qquant.models.greedy import force_greedy


def _stub_model():
    gen = SimpleNamespace(
        do_sample=True, num_beams=4, temperature=0.7, top_p=0.8, top_k=20
    )
    cfg = SimpleNamespace(temperature=0.7, top_p=0.8, top_k=20)
    return SimpleNamespace(generation_config=gen, config=cfg)


def test_force_greedy_sets_greedy_fields():
    m = _stub_model()
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
    assert m.generation_config.temperature is None
    assert m.generation_config.top_p is None
    assert m.generation_config.top_k is None
    assert m.config.temperature is None
    assert m.config.top_p is None
    assert m.config.top_k is None


def test_force_greedy_idempotent():
    m = _stub_model()
    force_greedy(m)
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
    assert m.generation_config.temperature is None


def test_force_greedy_tolerates_missing_config():
    gen = SimpleNamespace(
        do_sample=True, num_beams=2, temperature=1.0, top_p=1.0, top_k=50
    )
    m = SimpleNamespace(generation_config=gen, config=None)
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
