"""qquant.models — registry-driven variant loaders (Spec 04).

Importing this package is torch-free (heavy imports are deferred into the
builder bodies), so dispatch / source-resolution / greedy-forcing are
unit-testable off-GPU. This is NOT part of the torch-free core: the umbrella
``qquant`` CLI must never import it.
"""

from __future__ import annotations

from qquant.models.greedy import force_greedy
from qquant.models.loaders import (
    BUILDERS,
    LoadedVariant,
    ModelSource,
    load_variant,
    resolve_model_source,
    smoke_test_load,
)

__all__ = [
    "BUILDERS",
    "LoadedVariant",
    "ModelSource",
    "force_greedy",
    "load_variant",
    "resolve_model_source",
    "smoke_test_load",
]
