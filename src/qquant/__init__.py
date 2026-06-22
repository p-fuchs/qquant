"""qquant — torch-free core for the Qwen2.5-7B quantization evaluation study.

The core (config, registry, paths, matrix, cli) is import-time **torch-free** so it
can run for orchestration and result auditing on machines without a GPU (e.g. macOS).
GPU-touching code lives in submodules imported only on demand by their specs.
"""

from __future__ import annotations

from qquant.config import RunConfig, Seeds, cell_provenance
from qquant.matrix import Cell, expand_matrix, is_cell_done, missing_cells
from qquant.paths import Paths, cell_path
from qquant.registry import (
    MMLU_SUBJECTS,
    TASK_IDS,
    VARIANT_IDS,
    Task,
    Variant,
    enabled_variants,
    load_tasks,
    load_variants,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "VARIANT_IDS",
    "TASK_IDS",
    "MMLU_SUBJECTS",
    "Variant",
    "Task",
    "load_variants",
    "load_tasks",
    "enabled_variants",
    "Cell",
    "expand_matrix",
    "is_cell_done",
    "missing_cells",
    "Paths",
    "cell_path",
    "RunConfig",
    "Seeds",
    "cell_provenance",
]
