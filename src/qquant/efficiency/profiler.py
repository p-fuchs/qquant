"""qquant.efficiency.profiler — measurement engine.

The torch-free core (ProfileConfig, median, disk sizing, self-quant labeling, the
``build_efficiency_result`` assembler) is unit-tested off-GPU. The torch-touching
measurements live behind ``TorchEngine`` (added in Task 4); torch is imported only
inside its methods, so importing this module stays torch-free.
"""

from __future__ import annotations

import os  # noqa: F401
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qquant.efficiency.schema import efficiency_active_config, validate_efficiency

_SPEED_METRICS = ["prefill_tok_s", "decode_tok_s", "e2e_latency_s", "ttft_s"]
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_GIB = float(1024**3)


@dataclass(frozen=True)
class ProfileConfig:
    device: str = "cuda:0"
    warmup: int = 2
    repeats: int = 5
    decode_tokens: int = 256
    prompt_buckets: dict[str, int] = field(
        default_factory=lambda: {"short": 128, "medium": 1024, "long": 4096}
    )
    batch_seq_len: int = 2048
    batch_gen_tokens: int = 64
    batch_ceiling: int = 64


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def speed_label_and_confounded(variant: Any) -> tuple[str, list[str]]:
    """Self-quant speed is runtime-confounded (HF generate dequant path on sm_89).

    disk and memory are NEVER confounded — only the speed metrics.
    """
    if variant.source == "selfquant":
        return "runtime-confounded", list(_SPEED_METRICS)
    return "comparable", []


def sum_weight_bytes(snapshot_dir: str | Path) -> dict:
    """Sum real sizes (symlinks resolved) of *.safetensors/*.bin under snapshot_dir.

    Also returns the full resolved snapshot size.
    """
    root = Path(snapshot_dir)
    weights = 0
    snapshot = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size  # Path.stat follows symlinks
        snapshot += size
        if p.name.endswith(_WEIGHT_SUFFIXES):
            weights += size
    return {"weights_bytes": weights, "snapshot_bytes": snapshot}


def measure_disk_size(variant: Any, *, snapshot_dir: str | Path | None = None) -> dict:
    """Disk footprint for a variant.

    ``snapshot_dir`` injects the resolved dir (tests); else it is resolved from the
    HF cache (repo id) or the ``local:`` checkpoint dir (self-quant).
    """
    if snapshot_dir is not None:
        source = "local" if str(variant.model_id).startswith("local:") else "hf-cache"
        out = sum_weight_bytes(snapshot_dir)
        out["source"] = source
        out["path"] = str(snapshot_dir)
        out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
        return out
    if str(variant.model_id).startswith("local:"):
        local = Path(str(variant.model_id).split("local:", 1)[1])
        out = sum_weight_bytes(local)
        out["source"] = "local"
        out["path"] = str(local)
    else:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            variant.model_id, revision=variant.revision, local_files_only=True
        )
        out = sum_weight_bytes(resolved)
        out["source"] = "hf-cache"
        out["path"] = str(resolved)
    out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
    return out


def build_efficiency_result(
    *,
    variant: Any,
    cfg: ProfileConfig,
    runtime: dict,
    disk: dict,
    memory: dict,
    throughput: dict,
    max_batch: dict,
    env_meta: dict | None = None,
    samples: dict | None = None,
    meta_extra: dict | None = None,
) -> dict:
    """Assemble + validate a v1 efficiency result.

    ``config`` carries the comparable provenance block (EFFICIENCY_PROVENANCE_KEYS) plus
    the extra reproducibility keys. Does not persist.
    """
    speed_label, confounded = speed_label_and_confounded(variant)
    active = efficiency_active_config(cfg, variant, runtime)
    config = dict(active) | {
        "device": cfg.device,
        "expandable_segments": runtime.get("expandable_segments"),
        "do_sample": False,
        "warmup": cfg.warmup,
        "repeats": cfg.repeats,
        "batch_gen_tokens": cfg.batch_gen_tokens,
    }
    meta = {
        "timestamp": runtime.get("timestamp"),
        "hostname": runtime.get("hostname"),
    }
    if env_meta:
        meta["env"] = env_meta
    if meta_extra:
        meta.update(meta_extra)
    result = {
        "schema_version": 1,
        "variant": variant.id,
        "quant_method": variant.quant_method,
        "source": variant.source,
        "model_id": variant.model_id,
        "model_revision": variant.revision,
        "speed_label": speed_label,
        "confounded_metrics": confounded,
        "disk": disk,
        "memory": memory,
        "throughput": throughput,
        "max_batch_size": max_batch,
        "config": config,
        "meta": meta,
    }
    if samples is not None:
        result["samples"] = samples
    validate_efficiency(result)
    return result
