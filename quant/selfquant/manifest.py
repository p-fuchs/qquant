"""Per-checkpoint quant manifest + the idempotency/resume gate. Pure stdlib (torch-free).

The manifest ties a checkpoint to its exact controlled protocol; ``selfquant_checkpoint_done``
is the resume predicate so a re-run never re-quantizes a matching checkpoint, and Spec 08
exfiltrates a done dir as-is. The same fingerprint (``base_revision``/``calib_sha256``/
``algorithm``) feeds Spec 05's self-quant ``RunConfig.model_revision``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class QuantManifest:
    """The provenance of one self-quant checkpoint. ``algorithm`` is the only field that
    differs between the GPTQ and AWQ checkpoints of a controlled pair."""

    variant_id: str  # "gptq-selfquant" | "awq-selfquant"
    algorithm: str  # "gptq" | "awq"  (the ONLY intentional difference)
    scheme: str  # "W4A16_ASYM"
    group_size: int  # 128
    base_model_id: str
    base_revision: str  # pinned baseline SHA
    calib_id: str  # "c4-128x2048-s42"
    calib_sha256: str  # ties the checkpoint to an exact calibration set
    llmcompressor_version: str
    transformers_version: str
    compressed_tensors_version: str
    created_at: str  # UTC ISO-8601


def write_manifest(ckpt_dir: str | Path, m: QuantManifest) -> None:
    """Write ``ckpt_dir/quant_manifest.json`` (sorted keys, atomic via tmp + replace)."""
    path = Path(ckpt_dir) / "quant_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(m), indent=2, sort_keys=True))
    tmp.replace(path)


def read_manifest(ckpt_dir: str | Path) -> QuantManifest | None:
    """Parse ``ckpt_dir/quant_manifest.json`` into a ``QuantManifest``; None if absent/unreadable
    or missing/extra keys."""
    path = Path(ckpt_dir) / "quant_manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    keys = {f.name for f in fields(QuantManifest)}
    if not isinstance(data, dict) or set(data) != keys:
        return None
    return QuantManifest(**data)


def selfquant_checkpoint_done(
    ckpt_dir: str | Path,
    *,
    algorithm: str,
    base_revision: str,
    scheme: str,
    group_size: int,
    calib_sha256: str,
) -> bool:
    """True iff a valid compressed-tensors checkpoint is present (``config.json`` has a
    ``quantization_config``) AND a ``quant_manifest.json`` exists whose
    (algorithm, base_revision, scheme, group_size, calib_sha256) all match. The resume gate."""
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    if not isinstance(config, dict) or "quantization_config" not in config:
        return False
    m = read_manifest(ckpt_dir)
    if m is None:
        return False
    return (m.algorithm, m.base_revision, m.scheme, m.group_size, m.calib_sha256) == (
        algorithm,
        base_revision,
        scheme,
        group_size,
        calib_sha256,
    )
