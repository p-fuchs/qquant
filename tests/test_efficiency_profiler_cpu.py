from __future__ import annotations

from qquant.efficiency.profiler import (
    ProfileConfig,
    build_efficiency_result,
    measure_disk_size,
    median,
    speed_label_and_confounded,
    sum_weight_bytes,
)
from qquant.efficiency.schema import is_efficiency_done, validate_efficiency
from qquant.registry import load_variants

CONFOUNDED = ["prefill_tok_s", "decode_tok_s", "e2e_latency_s", "ttft_s"]


def _runtime() -> dict:
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


def _disk() -> dict:
    return {"weights_bytes": 15231000000, "source": "hf-cache", "weights_gib": 14.19}


def _memory() -> dict:
    return {
        "weights_resident_bytes": 15000000000,
        "load_peak_bytes": 15500000000,
        "generate_peak_bytes": 19000000000,
    }


def _throughput() -> dict:
    return {
        "short": {
            "prompt_tokens": 128,
            "gen_tokens": 256,
            "prefill_tok_s": 900.0,
            "decode_tok_s": 60.0,
            "e2e_latency_s": 4.3,
            "ttft_s": 0.14,
        }
    }


def _max_batch() -> dict:
    return {"value": 16, "oom_at": 32, "ceiling": 64, "ladder": [1, 2, 4, 8, 16]}


def test_profile_config_defaults_are_independent_dicts():
    a, b = ProfileConfig(), ProfileConfig()
    assert a.prompt_buckets == {"short": 128, "medium": 1024, "long": 4096}
    assert a.prompt_buckets is not b.prompt_buckets  # default_factory, not shared
    assert a.decode_tokens == 256 and a.repeats == 5 and a.batch_ceiling == 64


def test_median_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_speed_label_selfquant_vs_comparable():
    variants = load_variants()
    label, conf = speed_label_and_confounded(variants["gptq-selfquant"])
    assert label == "runtime-confounded"
    assert conf == CONFOUNDED
    label2, conf2 = speed_label_and_confounded(variants["bf16"])
    assert label2 == "comparable"
    assert conf2 == []


def test_sum_weight_bytes_follows_symlinks(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "b1").write_bytes(b"x" * 100)
    (blobs / "b2").write_bytes(b"y" * 250)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model-00001.safetensors").symlink_to(blobs / "b1")
    (snap / "model-00002.safetensors").symlink_to(blobs / "b2")
    (snap / "config.json").write_bytes(b"{}")  # not a weight file
    out = sum_weight_bytes(snap)
    assert out["weights_bytes"] == 350
    assert out["snapshot_bytes"] >= 350


def test_measure_disk_size_with_injected_snapshot_dir(tmp_path):
    variants = load_variants()
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"z" * 500)
    out = measure_disk_size(variants["bf16"], snapshot_dir=snap)
    assert out["weights_bytes"] == 500
    assert out["source"] == "hf-cache"


def test_build_result_is_schema_valid_and_labels_selfquant():
    variants = load_variants()
    res = build_efficiency_result(
        variant=variants["gptq-selfquant"],
        cfg=ProfileConfig(),
        runtime=_runtime(),
        disk=_disk(),
        memory=_memory(),
        throughput=_throughput(),
        max_batch=_max_batch(),
    )
    validate_efficiency(res)  # must not raise
    assert res["schema_version"] == 1
    assert res["variant"] == "gptq-selfquant"
    assert res["speed_label"] == "runtime-confounded"
    assert res["confounded_metrics"] == CONFOUNDED
    assert res["config"]["decode_tokens"] == 256
    assert res["config"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    # None for selfquant
    assert res["model_revision"] == variants["gptq-selfquant"].revision


def test_build_result_comparable_for_bf16_and_is_done_roundtrip():
    variants = load_variants()
    cfg = ProfileConfig()
    res = build_efficiency_result(
        variant=variants["bf16"],
        cfg=cfg,
        runtime=_runtime(),
        disk=_disk(),
        memory=_memory(),
        throughput=_throughput(),
        max_batch=_max_batch(),
    )
    assert res["speed_label"] == "comparable"
    assert res["confounded_metrics"] == []
    from qquant.efficiency.schema import efficiency_active_config

    active = efficiency_active_config(cfg, variants["bf16"], _runtime())
    assert is_efficiency_done(res, active) is True
