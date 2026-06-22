from __future__ import annotations

import jsonschema
import pytest

from qquant.config import RunConfig, Seeds, cell_provenance
from qquant.registry import load_tasks
from qquant.schemas import is_valid_cell, validate_cell


def _sample_cell():
    run = RunConfig(lm_eval_version="0.4.12", model_revision="abc123", seeds=Seeds())
    task = load_tasks()["mmlu"]
    return {
        "schema_version": 1,
        "cell_id": "bf16/mmlu/anatomy",
        "variant": "bf16",
        "task": "mmlu",
        "subject": "anatomy",
        "task_lm_eval": "mmlu_anatomy",
        "primary_metric": "acc",
        "metric_value": 0.82,
        "metric_stderr": 0.03,
        "n_samples": 135,
        "config": cell_provenance(run, task, "bf16"),
    }


def test_sample_cell_validates():
    validate_cell(_sample_cell())
    assert is_valid_cell(_sample_cell())


def test_missing_required_key_fails():
    cell = _sample_cell()
    del cell["metric_value"]
    assert not is_valid_cell(cell)
    with pytest.raises(jsonschema.ValidationError):
        validate_cell(cell)


def test_config_block_required():
    cell = _sample_cell()
    del cell["config"]["lm_eval_version"]
    assert not is_valid_cell(cell)
