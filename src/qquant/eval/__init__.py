"""qquant.eval — GPU-touching quality-eval engine (Spec 05).

Importing this package is torch-free (heavy imports are deferred inside
functions/methods); the umbrella ``qquant`` CLI must never import it.
"""

from __future__ import annotations

from qquant.eval.results import build_cell
from qquant.eval.runner import EvalRunner, RunSummary

__all__ = ["EvalRunner", "RunSummary", "build_cell"]
