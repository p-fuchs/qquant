"""Eval manifest: task-level (variant, task) pairs to run. Torch-free."""

from __future__ import annotations

from pathlib import Path

import yaml

from qquant.registry import TASK_IDS, VARIANT_IDS, Task, Variant


def load_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Parse a JSON/YAML manifest into validated (variant, task) pairs."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "cells" not in data:
        raise ValueError("manifest must be a mapping with a 'cells' list")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in data["cells"]:
        vid, tid = entry["variant"], entry["task"]
        if vid not in VARIANT_IDS:
            raise ValueError(f"manifest: unknown variant id {vid!r}")
        if tid not in TASK_IDS:
            raise ValueError(f"manifest: unknown task id {tid!r}")
        if (vid, tid) in seen:
            raise ValueError(f"manifest: duplicate cell {(vid, tid)!r}")
        seen.add((vid, tid))
        pairs.append((vid, tid))
    return pairs


def default_manifest(
    variants: list[Variant], tasks: list[Task]
) -> list[tuple[str, str]]:
    """Cartesian product of (variant.id, task.id) pairs."""
    return [(v.id, t.id) for v in variants for t in tasks]
