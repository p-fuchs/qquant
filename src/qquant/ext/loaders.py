"""Extension variant loader — reuses the v1 builders, adds the ext axes.

``ext_load_variant`` is the drop-in ``load_model`` for ``qquant.eval.runner.EvalRunner``
and the efficiency profiler when running the ext matrix. It:

* dispatches on ``quant_method`` through the v1 ``BUILDERS`` (so EXT-1 Mistral and EXT-3
  W8A8 load with zero new builder code — only different repo ids / checkpoints),
* attaches a PEFT adapter when ``adapter_path`` is set (EXT-5 QLoRA), and
* installs a quantized KV cache on the model's ``generation_config`` when ``kv_cache``
  is set (EXT-4).

DEFERRED IMPORTS: torch / transformers / peft are imported INSIDE the bodies only, so
this module stays import-time torch-free (same discipline as ``qquant.models.loaders``).
The exact transformers v5 quantized-cache API + peft pins are verified on the box at
promotion (see ``docs/specs/13-ext-runbook.md``).
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from qquant.ext.registry import ExtVariant, load_ext_variants
from qquant.models.greedy import force_greedy
from qquant.models.loaders import (
    BUILDERS,
    LoadedVariant,
    resolve_model_source,
)

_LOCAL_PREFIX = "local:"


def resolve_local_path(spec: str, checkpoints_root: str | Path = "checkpoints") -> Path:
    """Resolve a ``local:checkpoints/...`` spec to a concrete path (must exist)."""
    rel = spec[len(_LOCAL_PREFIX) :].removeprefix("checkpoints/")
    path = Path(checkpoints_root) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"adapter/checkpoint not found at {path} (produced by the ext train/quant "
            f"step; see docs/specs/13-ext-runbook.md)"
        )
    return path


def build_kv_cache_generation_kwargs(kv_cache: dict[str, Any] | None) -> dict[str, Any]:
    """Translate a ``{'backend','nbits'}`` knob into transformers generate kwargs.

    Returns ``{}`` when ``kv_cache`` is None. The concrete key names
    (``cache_implementation='quantized'`` + a ``cache_config`` mapping) match the
    transformers v5 quantized-cache API and are re-verified on the box (catalog EXT-4).
    """
    if kv_cache is None:
        return {}
    return {
        "cache_implementation": "quantized",
        "cache_config": {
            "backend": kv_cache["backend"],
            "nbits": int(kv_cache["nbits"]),
        },
    }


def _apply_kv_cache(model: Any, kv_cache: dict[str, Any] | None) -> dict[str, Any]:
    """Install the quantized KV cache on ``model.generation_config`` in place."""
    kwargs = build_kv_cache_generation_kwargs(kv_cache)
    if not kwargs:
        return {}
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        for key, value in kwargs.items():
            setattr(gen, key, value)
    return kwargs


def _attach_adapter(model: Any, adapter_path: Path) -> str:
    """Attach a PEFT/LoRA adapter to a loaded base model; return a fingerprint.

    The adapter stays attached for eval — a 4-bit base cannot be cleanly LoRA-merged
    (catalog EXT-5). The fingerprint is folded into provenance so a re-trained adapter
    makes cells stale under ``is_cell_done``.
    """
    from peft import PeftModel

    PeftModel.from_pretrained(model, str(adapter_path))
    return _fingerprint_path(adapter_path)


def _fingerprint_path(path: str | Path) -> str:
    """Cheap deterministic (name,size) fingerprint of a checkpoint dir. Torch-free."""
    import hashlib
    import os

    digest = hashlib.sha256()
    root = str(path)
    for name in sorted(os.listdir(root)):
        if name.endswith((".json", ".safetensors", ".bin")):
            size = os.stat(os.path.join(root, name)).st_size
            digest.update(name.encode())
            digest.update(str(size).encode())
    return digest.hexdigest()[:16]


def ext_load_variant(
    variant_id: str,
    *,
    variants: dict[str, ExtVariant] | None = None,
    checkpoints_root: str | Path = "checkpoints",
    device_map: Any = None,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
) -> LoadedVariant:
    """Load an ext variant: dispatch on quant_method, force greedy, attach adapter / KV.

    Mirrors ``qquant.models.loaders.load_variant`` but understands ``ExtVariant`` extra
    axes. Raises ``KeyError`` for an unknown id, ``ValueError`` for a disabled variant.
    """
    variants = variants if variants is not None else load_ext_variants()
    if variant_id not in variants:
        raise KeyError(
            f"unknown ext variant id {variant_id!r}; known: {sorted(variants)}"
        )
    variant = variants[variant_id]
    if not variant.enabled:
        raise ValueError(f"ext variant {variant_id!r} is disabled in the registry")

    source = resolve_model_source(variant, checkpoints_root)
    builder = BUILDERS[variant.quant_method]
    opts = {
        "device_map": device_map if device_map is not None else {"": 0},
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }

    t0 = perf_counter()
    model, tokenizer, builder_meta = builder(variant, source, opts)
    load_seconds = perf_counter() - t0

    adapter_fingerprint = None
    if variant.adapter_path:
        adapter_dir = resolve_local_path(variant.adapter_path, checkpoints_root)
        adapter_fingerprint = _attach_adapter(model, adapter_dir)

    force_greedy(model)
    kv_kwargs = _apply_kv_cache(model, variant.kv_cache)

    device_map_resolved = dict(getattr(model, "hf_device_map", {}) or {})
    offloaded = {str(d) for d in device_map_resolved.values()} & {"cpu", "disk"}
    if offloaded:
        raise RuntimeError(
            f"ext variant {variant_id!r} offloaded to {sorted(offloaded)} "
            f"(device_map={device_map_resolved}); refusing — corrupts timings"
        )

    metadata = {
        "variant_id": variant.id,
        "quant_method": variant.quant_method,
        "model_source": source.path_or_repo,
        "model_revision": source.revision,
        "base_model": variant.base_model,
        "is_local": source.is_local,
        "dtype": builder_meta.get("dtype"),
        "attn_implementation": attn_implementation,
        "device_map": device_map_resolved,
        "load_seconds": load_seconds,
        "transformers_version": builder_meta.get("transformers_version"),
        "checkpoint_fingerprint": builder_meta.get("checkpoint_fingerprint"),
        "adapter_fingerprint": adapter_fingerprint,
        "kv_cache": kv_kwargs or None,
    }
    return LoadedVariant(
        variant=variant, model=model, tokenizer=tokenizer, metadata=metadata
    )
