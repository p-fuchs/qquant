"""The (variant × task) result matrix and its filesystem-as-state predicates.

MMLU expands to 57 per-subject cells; other tasks are a single cell.
``is_cell_done`` is the SSOT resume predicate: structural validity **plus** an optional
provenance match, so changing seeds/batch/few-shot/template/version recomputes the cell.

Import-time torch-free.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from qquant.paths import cell_path
from qquant.registry import MMLU_SUBJECTS, Task, Variant

# Provenance keys compared on resume; mirror config.cell_provenance and the schema.
PROVENANCE_KEYS: tuple[str, ...] = (
    "seeds",
    "batch_size",
    "num_fewshot",
    "apply_chat_template",
    "fewshot_as_multiturn",
    "max_length",
    "lm_eval_version",
    "model_revision",
)


@dataclass(frozen=True)
class Cell:
    """One unit of work: a (variant, task[, subject]) result."""

    variant: str
    task: str
    subject: str | None
    lm_eval_task: str  # e.g. "mmlu_anatomy", "gsm8k", "humaneval_instruct"
    primary_metric: str

    @property
    def cell_id(self) -> str:
        tail = f"/{self.subject}" if self.subject is not None else ""
        return f"{self.variant}/{self.task}{tail}"

    def path(self, results_root: str | Path) -> Path:
        return cell_path(results_root, self.variant, self.task, self.subject)


def expand_matrix(variants: list[Variant], tasks: list[Task]) -> list[Cell]:
    """Cartesian product of variants × tasks, with MMLU expanded to 57 subjects."""
    cells: list[Cell] = []
    for v in variants:
        for t in tasks:
            if t.per_subject:
                for subject in MMLU_SUBJECTS:
                    cells.append(
                        Cell(
                            variant=v.id,
                            task=t.id,
                            subject=subject,
                            lm_eval_task=f"{t.lm_eval_task}_{subject}",
                            primary_metric=t.primary_metric,
                        )
                    )
            else:
                cells.append(
                    Cell(
                        variant=v.id,
                        task=t.id,
                        subject=None,
                        lm_eval_task=t.lm_eval_task,
                        primary_metric=t.primary_metric,
                    )
                )
    return cells


def is_cell_done(cell: object, active_config: dict | None = None) -> bool:
    """Whether a loaded cell JSON counts as complete.

    Structural checks always apply. If ``active_config`` (a provenance block, e.g. from
    :func:`qquant.config.cell_provenance`) is given, every overlapping provenance key
    must match — otherwise the cell is treated as stale and is recomputed.
    """
    if not isinstance(cell, dict):
        return False
    if cell.get("schema_version") != 1:
        return False
    if not cell.get("cell_id"):
        return False
    value = cell.get("metric_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    n = cell.get("n_samples")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        return False
    if active_config is not None:
        stored = cell.get("config") or {}
        for key in PROVENANCE_KEYS:
            if key in active_config and stored.get(key) != active_config[key]:
                return False
    return True


def missing_cells(
    cells: list[Cell],
    results_root: str | Path,
    active_config: dict | None = None,
) -> list[Cell]:
    """Cells whose result is missing, unreadable, or stale under ``active_config``."""
    missing: list[Cell] = []
    for cell in cells:
        path = cell.path(results_root)
        if not path.exists():
            missing.append(cell)
            continue
        try:
            loaded = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError):
            missing.append(cell)
            continue
        if not is_cell_done(loaded, active_config):
            missing.append(cell)
    return missing
