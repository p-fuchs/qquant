"""Metric extraction from lm-eval result dicts. Torch-free.

wilson_interval is NOT reimplemented here — it is re-exported from
qquant.aggregate.stats (the single SSOT implementation) for the eval-time
HumanEval pass@1 CI.
"""

from __future__ import annotations

from collections.abc import Sequence

from qquant.aggregate.stats import wilson_interval  # re-export; SSOT in aggregate.stats

__all__ = ["extract_metric", "extract_stderr", "wilson_interval"]


def extract_metric(
    results_for_task: dict, metric_keys: Sequence[str]
) -> tuple[str, float]:
    """Return the first (key, float) whose key is present, trying metric_keys in order.

    Raises KeyError (listing available keys) if none match — absorbs lm-eval suffix
    drift.
    """
    for key in metric_keys:
        if key in results_for_task:
            return key, float(results_for_task[key])
    raise KeyError(
        f"none of {list(metric_keys)} in results keys {sorted(results_for_task)}"
    )


def extract_stderr(results_for_task: dict, primary_metric: str) -> float | None:
    """Best-effort '<primary_metric>_stderr[,filter]' lookup.

    Returns None if absent or non-numeric.
    """
    prefix = f"{primary_metric}_stderr"
    for key, value in results_for_task.items():
        if key == prefix or key.startswith(prefix + ","):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None
