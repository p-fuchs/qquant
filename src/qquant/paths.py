"""Deterministic on-disk layout for the result matrix — the SSOT for paths.

Normative layout (every writer/auditor/aggregator MUST use ``cell_path``)::

    <results_root>/<variant>/<task>/<subject>.json   # per-subject (MMLU)
    <results_root>/<variant>/<task>/result.json      # single-cell tasks
    <results_root>/_meta/env.json                    # run provenance

Import-time torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def cell_path(
    results_root: str | Path,
    variant: str,
    task: str,
    subject: str | None = None,
) -> Path:
    """Path to a single result cell. ``subject`` is set only for per-subject tasks."""
    base = Path(results_root) / variant / task
    leaf = f"{subject}.json" if subject is not None else "result.json"
    return base / leaf


@dataclass(frozen=True)
class Paths:
    """Resolved paths rooted at a results directory."""

    results_root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> Paths:
        return cls(results_root=Path(root))

    @property
    def meta_dir(self) -> Path:
        return self.results_root / "_meta"

    @property
    def env_file(self) -> Path:
        return self.meta_dir / "env.json"

    @property
    def done_manifest(self) -> Path:
        """Listing of completed cell ids the remote run emits before signalling DONE."""
        return self.meta_dir / "done_manifest.json"

    def cell(self, variant: str, task: str, subject: str | None = None) -> Path:
        return cell_path(self.results_root, variant, task, subject)
