"""Extension registry — the SSOT for Spec 12 extension variants/tasks.

A deliberate sibling of :mod:`qquant.registry`. It owns its OWN id space
(``EXT_VARIANT_IDS`` / ``EXT_TASK_IDS``) so promoting an extension never edits the v1
``VARIANT_IDS`` / ``TASK_IDS`` tuples (and the live v1 run keeps reading the untouched
``registries/variants.yaml`` / ``tasks.yaml``).

``ExtVariant`` / ``ExtTask`` subclass the v1 dataclasses, adding only the extra fields
the extensions need (``base_model`` for EXT-1, ``adapter_path`` for EXT-5, ``kv_cache``
for EXT-4, judge fields for EXT-2). Because the v1 reuse surfaces duck-type on
attributes, the subclasses drop into ``expand_matrix`` / ``is_cell_done`` /
``build_cell`` / ``EvalRunner`` unchanged. Import-time torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from qquant.registry import QUANT_METHODS, Task, Variant

# --------------------------------------------------------------------------------------
# Extension id space (hyphen-separated, no underscores, no `-int4` suffix, no aliases —
# contracts §1/§8). These are NOT added to qquant.registry.VARIANT_IDS/TASK_IDS; they
# live here so the v1 SSOT and its consistency tests are untouched. Promoting an entry
# into a numbered v1 spec (13+) is what would move an id into the canonical tuples.
# --------------------------------------------------------------------------------------

# EXT-3 W8A8/SmoothQuant, EXT-5 QLoRA, EXT-1 Mistral (7 ids), EXT-4 KV-cache (pairs).
EXT_VARIANT_IDS: tuple[str, ...] = (
    # EXT-3 — W8A8 self-quant (compressed-tensors; outside the W4A16 fairness pair).
    "w8a8-selfquant",
    # EXT-5 — QLoRA recovery on the frozen NF4 base.
    "bnb-nf4-qlora",
    # EXT-1 — Mistral-7B-Instruct as a 2nd base model (same 7 quant methods).
    "mistral-bf16",
    "mistral-bnb-int8",
    "mistral-bnb-nf4",
    "mistral-gptq-official",
    "mistral-awq-official",
    "mistral-gptq-selfquant",
    "mistral-awq-selfquant",
    # EXT-4 — quantized KV cache (paired ids; efficiency-only quality sweep optional).
    "bf16-kvq4",
    "bnb-nf4-kvq4",
)

# EXT-2 — JudgeBench (LLM-as-judge meta-eval), a single-cell generative task.
EXT_TASK_IDS: tuple[str, ...] = ("judgebench",)

# Extensions may name a kv-cache backend; the concrete v5 API + backend dep is verified
# on the box at promotion (catalog EXT-4). These are the accepted knob values.
KV_CACHE_BACKENDS: frozenset[str] = frozenset({"quanto", "hqq"})


@dataclass(frozen=True)
class ExtVariant(Variant):
    """A v1 ``Variant`` plus the extra axes the extensions introduce.

    All extra fields default to ``None`` so an ext row that needs none behaves exactly
    like a v1 variant. ``base_model`` (EXT-1) overrides the implicit Qwen base;
    ``adapter_path`` (EXT-5) triggers the loader's PEFT attach step; ``kv_cache``
    (EXT-4) is a ``{"backend": str, "nbits": int}`` knob for the loader/profiler.
    """

    base_model: str | None = None
    adapter_path: str | None = None
    kv_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtTask(Task):
    """A v1 ``Task`` plus optional fields for the JudgeBench custom runner (EXT-2)."""

    # HF dataset id for a custom (non-lm-eval) runner; verified on the box at promotion.
    hf_dataset_id: str | None = None
    # "judge" selects qquant.ext.judge; None means a standard lm-eval simple task.
    runner: str | None = None
    # Average both A/B orders to control position bias (EXT-2). Off => single order.
    position_bias_control: bool = True


def _load_yaml(path: str | Path | None, default_name: str) -> dict[str, Any]:
    if path is None:
        text = (files("qquant") / "registries" / default_name).read_text()
    else:
        text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{default_name}: expected a mapping at the top level")
    return data


def _validate_kv_cache(vid: str, kv: Any) -> dict[str, Any] | None:
    if kv is None:
        return None
    if not isinstance(kv, dict) or "backend" not in kv or "nbits" not in kv:
        raise ValueError(
            f"variant {vid!r}: kv_cache must be a mapping with 'backend' and 'nbits'"
        )
    if kv["backend"] not in KV_CACHE_BACKENDS:
        raise ValueError(
            f"variant {vid!r}: kv_cache backend {kv['backend']!r} not in "
            f"{sorted(KV_CACHE_BACKENDS)}"
        )
    if int(kv["nbits"]) not in (2, 4, 8):
        raise ValueError(f"variant {vid!r}: kv_cache nbits must be one of 2/4/8")
    return {"backend": str(kv["backend"]), "nbits": int(kv["nbits"])}


def load_ext_variants(path: str | Path | None = None) -> dict[str, ExtVariant]:
    """Load + validate the extension variant registry, keyed by ext id.

    Unlike :func:`qquant.registry.load_variants`, this does NOT require all ids to be
    present (extensions are independently promotable/droppable) — it only rejects ids
    outside ``EXT_VARIANT_IDS`` and invalid ``quant_method`` values.
    """
    data = _load_yaml(path, "variants.ext.yaml")
    raw = data.get("variants", {})
    variants: dict[str, ExtVariant] = {}
    for vid, spec in raw.items():
        if vid not in EXT_VARIANT_IDS:
            raise ValueError(f"variants.ext.yaml: unknown ext variant id {vid!r}")
        method = spec["quant_method"]
        if method not in QUANT_METHODS:
            raise ValueError(f"ext variant {vid!r}: invalid quant_method {method!r}")
        variants[vid] = ExtVariant(
            id=vid,
            quant_method=method,
            source=spec["source"],
            enabled=bool(spec["enabled"]),
            model_id=spec["model_id"],
            revision=spec.get("revision"),
            notes=spec.get("notes", ""),
            base_model=spec.get("base_model"),
            adapter_path=spec.get("adapter_path"),
            kv_cache=_validate_kv_cache(vid, spec.get("kv_cache")),
        )
    return variants


def load_ext_tasks(path: str | Path | None = None) -> dict[str, ExtTask]:
    """Load + validate the extension task registry, keyed by ext id."""
    data = _load_yaml(path, "tasks.ext.yaml")
    raw = data.get("tasks", {})
    tasks: dict[str, ExtTask] = {}
    for tid, spec in raw.items():
        if tid not in EXT_TASK_IDS:
            raise ValueError(f"tasks.ext.yaml: unknown ext task id {tid!r}")
        tasks[tid] = ExtTask(
            id=tid,
            lm_eval_task=spec["lm_eval_task"],
            primary_metric=spec["primary_metric"],
            metric_keys=tuple(spec["metric_keys"]),
            per_subject=bool(spec["per_subject"]),
            apply_chat_template=bool(spec["apply_chat_template"]),
            fewshot_as_multiturn=bool(spec["fewshot_as_multiturn"]),
            num_fewshot=int(spec["num_fewshot"]),
            default_batch_size=int(spec["default_batch_size"]),
            batch_overrides={
                str(k): int(v) for k, v in (spec.get("batch_overrides") or {}).items()
            },
            code_exec=bool(spec.get("code_exec", False)),
            max_gen_toks=spec.get("max_gen_toks"),
            hf_dataset_id=spec.get("hf_dataset_id"),
            runner=spec.get("runner"),
            position_bias_control=bool(spec.get("position_bias_control", True)),
        )
    return tasks


def enabled_ext_variants(variants: dict[str, ExtVariant]) -> list[ExtVariant]:
    """Enabled ext variants in canonical (``EXT_VARIANT_IDS``) order."""
    return [
        variants[v] for v in EXT_VARIANT_IDS if v in variants and variants[v].enabled
    ]


# Backwards-friendly alias so callers can ignore the extra field on ExtTask.
__all__ = [
    "EXT_VARIANT_IDS",
    "EXT_TASK_IDS",
    "KV_CACHE_BACKENDS",
    "ExtVariant",
    "ExtTask",
    "load_ext_variants",
    "load_ext_tasks",
    "enabled_ext_variants",
]
