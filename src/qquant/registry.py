"""Variant and task registries — the single source of truth (SSOT).

Every other module/spec imports ``VARIANT_IDS`` and ``MMLU_SUBJECTS`` from here rather
than re-declaring them. The YAML registries (``registries/variants.yaml`` and
``registries/tasks.yaml``) carry the per-entry configuration; the canonical *id* sets
live as Python constants so they cannot drift.

This module is import-time torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------------------
# SSOT: canonical variant ids (hyphen-separated, no aliases — see decision log).
# --------------------------------------------------------------------------------------
VARIANT_IDS: tuple[str, ...] = (
    "bf16",
    "bnb-int8",
    "bnb-nf4",
    "gptq-official",
    "awq-official",
    "gptq-selfquant",
    "awq-selfquant",
)

# Allowed loader dispatch methods. Self-quant variants load via "compressed-tensors".
QUANT_METHODS: frozenset[str] = frozenset(
    {"bf16", "bnb-int8", "bnb-nf4", "gptq", "awq", "compressed-tensors"}
)

# Canonical task ids.
TASK_IDS: tuple[str, ...] = ("mmlu", "gsm8k", "humaneval", "ifeval")

# --------------------------------------------------------------------------------------
# SSOT: the 57 MMLU subjects. lm-eval subtask name is ``mmlu_<subject>``.
# A GPU/lm-eval test asserts this equals the TaskManager expansion of the "mmlu" group.
# --------------------------------------------------------------------------------------
MMLU_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


@dataclass(frozen=True)
class Variant:
    """A model variant in the comparison matrix."""

    id: str
    quant_method: str  # one of QUANT_METHODS
    source: str  # baseline | load-time | official | selfquant
    enabled: bool
    model_id: str  # HF repo id, or "local:<path>" for self-quant checkpoints
    revision: str | None  # pinned commit SHA for official/baseline; None for self-quant
    notes: str = ""


@dataclass(frozen=True)
class Task:
    """A benchmark task and its locked evaluation configuration."""

    id: str  # one of TASK_IDS
    lm_eval_task: str  # base lm-eval task/group name
    primary_metric: str
    metric_keys: tuple[str, ...]  # candidate lm-eval result keys, tried in order
    per_subject: bool  # True for mmlu (expands to 57 per-subject cells)
    apply_chat_template: bool
    fewshot_as_multiturn: bool
    num_fewshot: int
    default_batch_size: int
    batch_overrides: dict[str, int] = field(default_factory=dict)
    code_exec: bool = False
    max_gen_toks: int | None = None

    def batch_size_for(self, variant_id: str) -> int:
        """Per-(variant, task) batch size; falls back to the task default."""
        return self.batch_overrides.get(variant_id, self.default_batch_size)


def _load_yaml(path: str | Path | None, default_name: str) -> dict[str, Any]:
    if path is None:
        text = (files("qquant") / "registries" / default_name).read_text()
    else:
        text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{default_name}: expected a mapping at the top level")
    return data


def load_variants(path: str | Path | None = None) -> dict[str, Variant]:
    """Load and validate the variant registry, keyed by canonical id."""
    data = _load_yaml(path, "variants.yaml")
    raw = data.get("variants", {})
    variants: dict[str, Variant] = {}
    for vid, spec in raw.items():
        if vid not in VARIANT_IDS:
            raise ValueError(f"variants.yaml: unknown variant id {vid!r}")
        method = spec["quant_method"]
        if method not in QUANT_METHODS:
            raise ValueError(f"variant {vid!r}: invalid quant_method {method!r}")
        variants[vid] = Variant(
            id=vid,
            quant_method=method,
            source=spec["source"],
            enabled=bool(spec["enabled"]),
            model_id=spec["model_id"],
            revision=spec.get("revision"),
            notes=spec.get("notes", ""),
        )
    missing = set(VARIANT_IDS) - set(variants)
    if missing:
        raise ValueError(f"variants.yaml is missing variants: {sorted(missing)}")
    return variants


def load_tasks(path: str | Path | None = None) -> dict[str, Task]:
    """Load and validate the task registry, keyed by canonical id."""
    data = _load_yaml(path, "tasks.yaml")
    raw = data.get("tasks", {})
    tasks: dict[str, Task] = {}
    for tid, spec in raw.items():
        if tid not in TASK_IDS:
            raise ValueError(f"tasks.yaml: unknown task id {tid!r}")
        tasks[tid] = Task(
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
        )
    return tasks


def enabled_variants(variants: dict[str, Variant]) -> list[Variant]:
    """Enabled variants in canonical order."""
    return [variants[v] for v in VARIANT_IDS if v in variants and variants[v].enabled]
