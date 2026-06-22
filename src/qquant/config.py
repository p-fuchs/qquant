"""Run configuration and per-cell provenance — torch-free.

``RunConfig`` captures the reproducibility knobs global to a run; combined with a
:class:`~qquant.registry.Task` and a variant id it builds the provenance block stored
in every result cell. :func:`~qquant.matrix.is_cell_done` compares that block against
the active config so a resume never silently reuses a cell from different settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Seeds:
    """The four seeds lm-eval threads through evaluation."""

    random: int = 0
    numpy: int = 1234
    torch: int = 1234
    fewshot: int = 1234

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """Global, reproducibility-relevant run configuration."""

    lm_eval_version: str
    model_revision: str | None = None
    max_length: int | None = None
    seeds: Seeds = field(default_factory=Seeds)


def cell_provenance(run: RunConfig, task, variant_id: str) -> dict:
    """Build the provenance block for a (variant, task) cell.

    Keys must match ``qquant.matrix.PROVENANCE_KEYS`` and the ``config`` block of the
    cell-result JSON schema.
    """
    return {
        "seeds": run.seeds.as_dict(),
        "batch_size": task.batch_size_for(variant_id),
        "num_fewshot": task.num_fewshot,
        "apply_chat_template": task.apply_chat_template,
        "fewshot_as_multiturn": task.fewshot_as_multiturn,
        "max_length": run.max_length,
        "lm_eval_version": run.lm_eval_version,
        "model_revision": run.model_revision,
    }
