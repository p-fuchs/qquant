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
from typing import Any

import jsonschema

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
