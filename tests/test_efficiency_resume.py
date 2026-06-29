from __future__ import annotations

from qquant.efficiency.schema import (
    EFFICIENCY_PROVENANCE_KEYS,
    efficiency_active_config,
    efficiency_path,
    is_efficiency_done,
)
from qquant.registry import load_variants


def _runtime() -> dict:
    return {
        "torch_version": "2.12.1+cu126",
        "transformers_version": "5.10.1",
        "cuda_version": "12.6",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "dtype": "bfloat16",
    }


class _Cfg:
    prompt_buckets = {"short": 128, "medium": 1024, "long": 4096}
    decode_tokens = 256
    batch_seq_len = 2048


def test_efficiency_path_layout(tmp_path):
    p = efficiency_path(tmp_path, "bf16")
    assert p == tmp_path / "bf16" / "efficiency.json"


def test_provenance_keys_exact():
    assert EFFICIENCY_PROVENANCE_KEYS == (
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
        "dtype",
        "model_revision",
        "prompt_buckets",
        "decode_tokens",
        "batch_seq_len",
    )


def test_active_config_built_from_cfg_variant_runtime():
    variants = load_variants()
    active = efficiency_active_config(_Cfg(), variants["bf16"], _runtime())
    assert set(active) == set(EFFICIENCY_PROVENANCE_KEYS)
    assert active["model_revision"] == variants["bf16"].revision
    assert active["decode_tokens"] == 256
    assert active["gpu_name"] == "NVIDIA GeForce RTX 4090"


def test_is_efficiency_done_structural():
    assert is_efficiency_done("not a dict") is False
    assert is_efficiency_done({"schema_version": 2}) is False
    valid = {"schema_version": 1, "config": {"decode_tokens": 256}}
    # structurally incomplete (no required blocks) -> not done
    assert is_efficiency_done(valid) is False


def test_is_efficiency_done_provenance_match_and_stale():
    variants = load_variants()
    active = efficiency_active_config(_Cfg(), variants["bf16"], _runtime())
    result = {
        "schema_version": 1,
        "variant": "bf16",
        "disk": {"weights_bytes": 1, "source": "hf-cache"},
        "memory": {
            "weights_resident_bytes": 1,
            "load_peak_bytes": 2,
            "generate_peak_bytes": 3,
        },
        "throughput": {
            "short": {
                "prompt_tokens": 128,
                "gen_tokens": 256,
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "e2e_latency_s": 1.0,
                "ttft_s": 1.0,
            }
        },
        "max_batch_size": {"value": 1, "ceiling": 64},
        "config": dict(active)
        | {"device": "cuda:0", "do_sample": False, "warmup": 2, "repeats": 5},
    }
    assert is_efficiency_done(result, active) is True
    stale = dict(active) | {"decode_tokens": 512}
    assert is_efficiency_done(result, stale) is False
