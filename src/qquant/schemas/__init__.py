"""Cell-result JSON schema loader and validator (SSOT for the on-disk cell format).

``validate_cell`` is used by writers and tests for *full* schema validation;
``qquant.matrix.is_cell_done`` is the lighter resume predicate. They must agree on the
required keys. Import-time torch-free (uses ``jsonschema``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    text = (files("qquant.schemas") / "cell_result.schema.json").read_text()
    return json.loads(text)


def validate_cell(cell: dict) -> None:
    """Raise ``jsonschema.ValidationError`` if ``cell`` does not match the v1 schema."""
    jsonschema.validate(instance=cell, schema=load_schema())


def is_valid_cell(cell: dict) -> bool:
    try:
        validate_cell(cell)
    except jsonschema.ValidationError:
        return False
    return True
