from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip module if torch absent (laptop)

from qquant.models import load_variant, smoke_test_load  # noqa: E402
from qquant.registry import enabled_variants, load_variants  # noqa: E402

pytestmark = pytest.mark.gpu


def _core_enabled_ids():
    """Enabled official-core variants.

    Self-quant checkpoints don't exist until Spec 07.
    """
    variants = load_variants()
    return [v.id for v in enabled_variants(variants) if v.source != "selfquant"]


@pytest.mark.parametrize("variant_id", _core_enabled_ids())
def test_load_variant_no_offload_and_generates(variant_id):
    with load_variant(variant_id) as lv:
        values = {str(d) for d in lv.metadata["device_map"].values()}
        assert not (values & {"cpu", "disk"}), f"offload in {lv.metadata['device_map']}"
        device = next(lv.model.parameters()).device
        enc = lv.tokenizer("The capital of France is", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = lv.model.generate(**enc, max_new_tokens=1, do_sample=False)
        assert out.shape[-1] == enc["input_ids"].shape[-1] + 1


@pytest.mark.parametrize("variant_id", _core_enabled_ids())
def test_greedy_generate_is_deterministic(variant_id):
    with load_variant(variant_id) as lv:
        device = next(lv.model.parameters()).device
        enc = lv.tokenizer("List three primary colors:", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        def gen():
            return lv.model.generate(**enc, max_new_tokens=8, do_sample=False).tolist()

        assert gen() == gen()


def test_smoke_test_load_awq_official_reports():
    ok, msg = smoke_test_load("awq-official")
    assert isinstance(ok, bool)
    assert isinstance(msg, str) and msg


def test_unload_allows_sequential_load_without_oom():
    ids = _core_enabled_ids()
    if len(ids) < 2:
        pytest.skip("need >= 2 enabled core variants for a sequential-load check")
    first = load_variant(ids[0])
    first.unload()
    second = load_variant(ids[1])
    second.unload()
