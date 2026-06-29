"""lm-eval results → v1 cell.

Torch-free; uses cell_provenance + validate_cell + cell_path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from qquant.config import RunConfig, cell_provenance
from qquant.eval.metrics import extract_metric, extract_stderr
from qquant.matrix import Cell
from qquant.registry import Task, Variant
from qquant.schemas import validate_cell

# Per-doc correctness fields tried in order (lm-eval stores the bare metric per sample).
_CORRECTNESS_FIELDS = ("acc", "exact_match", "pass@1", "prompt_level_strict_acc")


def _doc_correct(sample: dict, primary_metric: str) -> int:
    for key in (primary_metric, *_CORRECTNESS_FIELDS):
        if key in sample:
            return int(round(float(sample[key])))
    raise KeyError(
        f"no per-doc correctness field for {primary_metric!r} in sample keys "
        f"{sorted(sample)}"
    )


def cell_inputs_from_results(results: dict, lm_eval_task: str, task: Task) -> dict:
    """Parse one (sub)task's lm-eval output into the inputs build_cell needs."""
    per_task = results["results"][lm_eval_task]
    metric_key, metric_value = extract_metric(per_task, task.metric_keys)
    metric_stderr = extract_stderr(per_task, task.primary_metric)
    samples = results.get("samples", {}).get(lm_eval_task, [])
    item_ids = [s["doc_id"] for s in samples]
    item_correct = [_doc_correct(s, task.primary_metric) for s in samples]
    extra_metrics = {
        k: v
        for k, v in per_task.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != metric_key
    }
    return {
        "metric_key": metric_key,
        "metric_value": metric_value,
        "metric_stderr": metric_stderr,
        "n_samples": len(samples),
        "item_correct": item_correct,
        "item_ids": item_ids,
        "extra_metrics": extra_metrics,
    }


def build_cell(
    *,
    cell: Cell,
    task: Task,
    variant: Variant,
    run: RunConfig,
    metric_value: float,
    metric_stderr: float | None,
    n_samples: int,
    extra_metrics: dict,
    item_correct: list[int],
    item_ids: list,
    meta_extra: dict,
) -> dict:
    """Assemble + validate a v1 cell.

    config == cell_provenance; item vectors live in meta.
    """
    if not (n_samples == len(item_correct) == len(item_ids)):
        raise ValueError(
            f"n_samples ({n_samples}) must equal len(item_correct) "
            f"({len(item_correct)}) and len(item_ids) ({len(item_ids)})"
        )
    meta = {
        **meta_extra,
        "item_correct": item_correct,
        "item_ids": item_ids,
        "n_correct": sum(item_correct),
    }
    doc = {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "variant": variant.id,
        "task": task.id,
        "subject": cell.subject,
        "task_lm_eval": cell.lm_eval_task,
        "primary_metric": task.primary_metric,
        "metric_value": metric_value,
        "metric_stderr": metric_stderr,
        "n_samples": n_samples,
        "extra_metrics": extra_metrics,
        "config": cell_provenance(run, task, variant.id),
        "meta": meta,
    }
    validate_cell(doc)
    return doc


def write_cell(cell_doc: dict, path: Path) -> None:
    """Validate then atomically write the cell (tmp sibling + os.replace)."""
    validate_cell(cell_doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cell_doc, indent=2, sort_keys=True))
    os.replace(tmp, path)
