from __future__ import annotations

import json

from qquant.config import RunConfig, Seeds, cell_provenance
from qquant.matrix import Cell, expand_matrix, is_cell_done, missing_cells
from qquant.registry import enabled_variants, load_tasks, load_variants


def _matrix():
    variants = enabled_variants(load_variants())
    tasks = list(load_tasks().values())
    return variants, tasks, expand_matrix(variants, tasks)


def test_matrix_cell_count():
    variants, tasks, cells = _matrix()
    # per variant: 57 MMLU + 1 gsm8k + 1 humaneval + 1 ifeval = 60
    assert len(cells) == len(variants) * 60


def test_cell_id_and_lm_eval_task():
    cell = Cell("bf16", "mmlu", "anatomy", "mmlu_anatomy", "acc")
    assert cell.cell_id == "bf16/mmlu/anatomy"
    single = Cell("bf16", "gsm8k", None, "gsm8k", "exact_match")
    assert single.cell_id == "bf16/gsm8k"


def test_is_cell_done_structural():
    good = {
        "schema_version": 1,
        "cell_id": "bf16/gsm8k",
        "metric_value": 0.8,
        "n_samples": 100,
    }
    assert is_cell_done(good)
    assert not is_cell_done(
        {
            "schema_version": 1,
            "cell_id": "x",
            "metric_value": float("nan"),
            "n_samples": 1,
        }
    )
    assert not is_cell_done(
        {"schema_version": 1, "cell_id": "x", "metric_value": 0.5, "n_samples": 0}
    )
    assert not is_cell_done(
        {"cell_id": "x", "metric_value": 0.5, "n_samples": 1}
    )  # no schema_version


def test_is_cell_done_provenance_mismatch():
    run = RunConfig(lm_eval_version="0.4.12", model_revision="abc", seeds=Seeds())
    tasks = load_tasks()
    prov = cell_provenance(run, tasks["gsm8k"], "bf16")
    cell = {
        "schema_version": 1,
        "cell_id": "bf16/gsm8k",
        "metric_value": 0.8,
        "n_samples": 100,
        "config": dict(prov),
    }
    assert is_cell_done(cell, prov)
    # change a provenance knob -> stale -> not done
    stale = dict(prov, num_fewshot=99)
    assert not is_cell_done(cell, stale)


def test_missing_cells(tmp_path):
    variants, tasks, cells = _matrix()
    # nothing written yet -> all missing
    assert len(missing_cells(cells, tmp_path)) == len(cells)
    # write one valid cell -> one fewer missing
    target = cells[0]
    path = target.path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cell_id": target.cell_id,
                "metric_value": 0.5,
                "n_samples": 10,
            }
        )
    )
    assert len(missing_cells(cells, tmp_path)) == len(cells) - 1
