"""qquant.efficiency — per-variant efficiency profiler (Spec 06).

Importing this package is torch-free: only ``.schema`` is imported eagerly.
The torch-touching ``profile_variant``/``ProfileConfig``/``build_efficiency_result``
and the CLI ``main`` resolve lazily via ``__getattr__`` so ``import qquant.efficiency``
never pulls torch. NOT part of the torch-free core — the umbrella ``qquant`` CLI must
never import it.
"""

from __future__ import annotations

from typing import Any

from qquant.efficiency.schema import (
    EFFICIENCY_SCHEMA_VERSION,
    is_valid_efficiency,
    load_efficiency_schema,
    validate_efficiency,
)

__all__ = [
    "EFFICIENCY_SCHEMA_VERSION",
    "ProfileConfig",
    "build_efficiency_result",
    "is_valid_efficiency",
    "load_efficiency_schema",
    "main",
    "profile_variant",
    "validate_efficiency",
]

_LAZY = {
    "ProfileConfig": ("qquant.efficiency.profiler", "ProfileConfig"),
    "build_efficiency_result": (
        "qquant.efficiency.profiler",
        "build_efficiency_result",
    ),
    "profile_variant": ("qquant.efficiency.profiler", "profile_variant"),
    "main": ("qquant.efficiency.cli", "main"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
