"""Efficiency-artifact schema, path, and resume predicate — TORCH-FREE.

Jsonschema only.

This is the per-variant efficiency carve-out (contracts.md §8): its own schema,
path helper, and resume predicate, all owned here. Mirrors qquant.schemas +
qquant.matrix.is_cell_done so the orchestrator's resume/copy model stays uniform.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

from qquant.paths import Paths

EFFICIENCY_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def load_efficiency_schema() -> dict[str, Any]:
    text = (files("qquant.schemas") / "efficiency_result.schema.json").read_text()
    return json.loads(text)


def validate_efficiency(result: dict) -> None:
    """Raise jsonschema.ValidationError if result does not match the v1 schema."""
    jsonschema.validate(instance=result, schema=load_efficiency_schema())


def is_valid_efficiency(result: dict) -> bool:
    try:
        validate_efficiency(result)
    except jsonschema.ValidationError:
        return False
    return True


EFFICIENCY_PROVENANCE_KEYS: tuple[str, ...] = (
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


def efficiency_path(results_root: str | Path, variant: str) -> Path:
    """results_root/<variant>/efficiency.json — the Spec-06 carve-out (contracts §8)."""
    return Paths.from_root(results_root).results_root / variant / "efficiency.json"


def is_efficiency_done(result: object, active_config: dict | None = None) -> bool:
    """Structural validity always; if active_config given, every overlapping
    EFFICIENCY_PROVENANCE_KEYS value must match or the artifact is stale (recompute).
    Mirrors qquant.matrix.is_cell_done.
    """
    if not isinstance(result, dict):
        return False
    if result.get("schema_version") != EFFICIENCY_SCHEMA_VERSION:
        return False
    if not is_valid_efficiency(result):
        return False
    if active_config is not None:
        stored = result.get("config") or {}
        for key in EFFICIENCY_PROVENANCE_KEYS:
            if key in active_config and stored.get(key) != active_config[key]:
                return False
    return True


def efficiency_active_config(cfg: Any, variant: Any, runtime: dict) -> dict:
    """Build provenance block matching EFFICIENCY_PROVENANCE_KEYS.

    ``runtime`` has torch/transformers/cuda versions, gpu_name, dtype.
    ``variant.revision`` is the model SHA (or None for self-quant).
    """
    return {
        "torch_version": runtime["torch_version"],
        "transformers_version": runtime["transformers_version"],
        "cuda_version": runtime["cuda_version"],
        "gpu_name": runtime["gpu_name"],
        "dtype": runtime["dtype"],
        "model_revision": variant.revision,
        "prompt_buckets": dict(cfg.prompt_buckets),
        "decode_tokens": cfg.decode_tokens,
        "batch_seq_len": cfg.batch_seq_len,
    }
