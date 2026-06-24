"""Unified variant loaders for the 7 canonical Qwen2.5-7B variants (Spec 04).

DEFERRED IMPORTS: torch/transformers/bitsandbytes are imported INSIDE the
builder bodies only — never at module top — so importing this module stays
torch-free and the dispatch table / source resolution / greedy-forcing are
unit-testable on the dev laptop. The actual loads run on the RTX 4090. This
module is NOT the torch-free core; the umbrella ``qquant`` CLI must never
import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qquant.registry import Variant

_LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class ModelSource:
    """Where a variant's weights come from, after resolving ``local:`` ids."""

    path_or_repo: str  # HF repo id, or an absolute/relative local path
    revision: str | None  # pinned SHA for official/baseline; None for local self-quant
    is_local: bool


def resolve_model_source(
    variant: Variant, checkpoints_root: str | Path = "checkpoints"
) -> ModelSource:
    """Resolve a variant's weight source.

    ``local:checkpoints/self-quant/<id>`` -> ``ModelSource(<checkpoints_root>
    /self-quant/<id>, None, True)`` (raising ``FileNotFoundError`` if that
    path is absent — Spec 07 produces it). Anything else ->
    ``ModelSource(model_id, variant.revision, False)``.
    """
    model_id = variant.model_id
    if model_id.startswith(_LOCAL_PREFIX):
        rel = model_id[len(_LOCAL_PREFIX) :].removeprefix("checkpoints/")
        path = Path(checkpoints_root) / rel
        if not path.exists():
            raise FileNotFoundError(
                f"self-quant checkpoint for {variant.id!r} not found at {path} "
                f"(produced by Spec 07; exfil/re-upload owned by Spec 07/08)"
            )
        return ModelSource(str(path), None, True)
    return ModelSource(model_id, variant.revision, False)
