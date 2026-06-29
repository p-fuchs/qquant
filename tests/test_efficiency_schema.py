from __future__ import annotations

import jsonschema
import pytest

from qquant.efficiency.schema import (
    EFFICIENCY_SCHEMA_VERSION,
    is_valid_efficiency,
    load_efficiency_schema,
    validate_efficiency,
)


def _minimal_result() -> dict:
    return {
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
        "max_batch_size": {"value": 1, "oom_at": 2, "ceiling": 64},
        "config": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "torch_version": "2.12.1+cu126",
            "transformers_version": "5.10.1",
            "cuda_version": "12.6",
            "decode_tokens": 256,
            "prompt_buckets": {"short": 128, "medium": 1024, "long": 4096},
            "batch_seq_len": 2048,
        },
    }


def test_schema_version_constant():
    assert EFFICIENCY_SCHEMA_VERSION == 1
    assert load_efficiency_schema()["properties"]["schema_version"]["const"] == 1


def test_minimal_result_validates():
    validate_efficiency(_minimal_result())  # must not raise
    assert is_valid_efficiency(_minimal_result()) is True


def test_missing_required_top_level_key_fails():
    bad = _minimal_result()
    del bad["throughput"]
    with pytest.raises(jsonschema.ValidationError):
        validate_efficiency(bad)
    assert is_valid_efficiency(bad) is False


def test_wrong_schema_version_fails():
    bad = _minimal_result()
    bad["schema_version"] = 2
    assert is_valid_efficiency(bad) is False
