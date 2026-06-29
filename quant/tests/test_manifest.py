from __future__ import annotations

import json

from selfquant.manifest import (
    QuantManifest,
    read_manifest,
    selfquant_checkpoint_done,
    write_manifest,
)


def _manifest(**overrides) -> QuantManifest:
    base = dict(
        variant_id="gptq-selfquant",
        algorithm="gptq",
        scheme="W4A16_ASYM",
        group_size=128,
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        base_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        calib_id="c4-128x2048-s42",
        calib_sha256="deadbeef",
        llmcompressor_version="0.12.0",
        transformers_version="5.10.1",
        compressed_tensors_version="0.17.1",
        created_at="2026-06-29T00:00:00Z",
    )
    base.update(overrides)
    return QuantManifest(**base)


def test_manifest_roundtrip(tmp_path):
    m = _manifest()
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path) == m


def test_read_manifest_absent_is_none(tmp_path):
    assert read_manifest(tmp_path) is None


def _write_valid_ckpt(tmp_path, m: QuantManifest) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"weights": {"num_bits": 4}}})
    )
    write_manifest(tmp_path, m)


def test_done_true_when_config_and_manifest_match(tmp_path):
    m = _manifest()
    _write_valid_ckpt(tmp_path, m)
    assert (
        selfquant_checkpoint_done(
            tmp_path,
            algorithm="gptq",
            base_revision=m.base_revision,
            scheme="W4A16_ASYM",
            group_size=128,
            calib_sha256="deadbeef",
        )
        is True
    )


def test_done_false_without_quantization_config(tmp_path):
    m = _manifest()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    write_manifest(tmp_path, m)
    assert (
        selfquant_checkpoint_done(
            tmp_path,
            algorithm="gptq",
            base_revision=m.base_revision,
            scheme="W4A16_ASYM",
            group_size=128,
            calib_sha256="deadbeef",
        )
        is False
    )


def test_done_false_on_field_mismatch(tmp_path):
    m = _manifest()
    _write_valid_ckpt(tmp_path, m)
    # different calibration set -> stale
    assert (
        selfquant_checkpoint_done(
            tmp_path,
            algorithm="gptq",
            base_revision=m.base_revision,
            scheme="W4A16_ASYM",
            group_size=128,
            calib_sha256="OTHER",
        )
        is False
    )
    # different algorithm -> stale
    assert (
        selfquant_checkpoint_done(
            tmp_path,
            algorithm="awq",
            base_revision=m.base_revision,
            scheme="W4A16_ASYM",
            group_size=128,
            calib_sha256="deadbeef",
        )
        is False
    )
