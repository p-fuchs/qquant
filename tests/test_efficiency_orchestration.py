from __future__ import annotations

from qquant.efficiency.profiler import ProfileConfig, profile_variant
from qquant.efficiency.schema import validate_efficiency
from qquant.registry import load_variants


class _FakeLoaded:
    def __init__(self, variant):
        self.variant = variant
        self.model = object()
        self.tokenizer = object()
        self.metadata = {"dtype": "bfloat16", "model_revision": variant.revision}
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def disk(self, variant):
        self.calls.append("disk")
        return {
            "weights_bytes": 15231000000,
            "source": "hf-cache",
            "weights_gib": 14.19,
        }

    def runtime_facts(self, loaded, cfg):
        self.calls.append("runtime")
        return {
            "torch_version": "2.12.1+cu126",
            "transformers_version": "5.10.1",
            "cuda_version": "12.6",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "dtype": "bfloat16",
            "expandable_segments": True,
            "timestamp": "2026-06-29T00:00:00Z",
            "hostname": "test",
        }

    def load_and_measure(self, load_model, variant, cfg):
        self.calls.append("load")
        loaded = load_model(variant.id, device_map={"": 0})
        return loaded, {
            "weights_resident_bytes": 15000000000,
            "load_peak_bytes": 15500000000,
        }

    def throughput_and_genpeak(self, loaded, cfg):
        self.calls.append("throughput")
        tp = {
            b: {
                "prompt_tokens": n,
                "gen_tokens": cfg.decode_tokens,
                "prefill_tok_s": 900.0,
                "decode_tok_s": 60.0,
                "e2e_latency_s": 4.3,
                "ttft_s": 0.14,
            }
            for b, n in cfg.prompt_buckets.items()
        }
        return tp, {"generate_peak_bytes": 19000000000}

    def sweep(self, loaded, cfg):
        self.calls.append("sweep")
        return {"value": 16, "oom_at": 32, "ceiling": cfg.batch_ceiling}


def test_profile_variant_loads_once_sweeps_last_unloads_and_assembles():
    variants = load_variants()
    loaded_box = {}

    def fake_load(variant_id, **kw):
        loaded_box["loaded"] = _FakeLoaded(variants[variant_id])
        return loaded_box["loaded"]

    engine = _FakeEngine()
    res = profile_variant(
        variants["bf16"], ProfileConfig(), load_model=fake_load, engine=engine
    )
    validate_efficiency(res)
    assert res["variant"] == "bf16"
    assert res["memory"]["generate_peak_bytes"] == 19000000000
    assert res["memory"]["load_peak_bytes"] == 15500000000
    assert set(res["throughput"]) == {"short", "medium", "long"}
    # sweep must run AFTER throughput/memory (last GPU measurement)
    assert engine.calls.index("sweep") > engine.calls.index("throughput")
    assert engine.calls.index("sweep") > engine.calls.index("load")
    # model freed even on the happy path
    assert loaded_box["loaded"].unloaded is True


def test_profile_variant_unloads_on_measurement_failure():
    variants = load_variants()
    loaded_box = {}

    def fake_load(variant_id, **kw):
        loaded_box["loaded"] = _FakeLoaded(variants[variant_id])
        return loaded_box["loaded"]

    class _BoomEngine(_FakeEngine):
        def throughput_and_genpeak(self, loaded, cfg):
            raise RuntimeError("boom")

    try:
        profile_variant(
            variants["bf16"],
            ProfileConfig(),
            load_model=fake_load,
            engine=_BoomEngine(),
        )
    except RuntimeError:
        pass
    assert loaded_box["loaded"].unloaded is True
