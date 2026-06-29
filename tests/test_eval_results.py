from __future__ import annotations

import json

import pytest

from qquant.config import RunConfig, Seeds
from qquant.eval.results import build_cell, cell_inputs_from_results, write_cell
from qquant.matrix import Cell, is_cell_done
from qquant.registry import load_tasks, load_variants


def _run():
    return RunConfig(
        lm_eval_version="lmv", model_revision="rev123", max_length=4096, seeds=Seeds()
    )


def _gsm8k_results():
    return {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.5,
                "exact_match_stderr,strict-match": 0.1,
            }
        },
        "samples": {
            "gsm8k": [{"doc_id": 0, "exact_match": 1}, {"doc_id": 1, "exact_match": 0}]
        },
    }


def test_cell_inputs_parse_metric_stderr_and_item_vectors():
    tasks = load_tasks()
    out = cell_inputs_from_results(_gsm8k_results(), "gsm8k", tasks["gsm8k"])
    assert out["metric_value"] == 0.5
    assert out["metric_stderr"] == pytest.approx(0.1)
    assert out["n_samples"] == 2
    assert out["item_correct"] == [1, 0]
    assert out["item_ids"] == [0, 1]


def test_cell_inputs_raises_when_no_correctness_field():
    tasks = load_tasks()
    bad = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.5}},
        "samples": {"gsm8k": [{"doc_id": 0, "unexpected": 1}]},
    }
    with pytest.raises(KeyError):
        cell_inputs_from_results(bad, "gsm8k", tasks["gsm8k"])


def test_build_cell_validates_and_config_is_provenance(tmp_path):
    tasks = load_tasks()
    variants = load_variants()
    run = _run()
    cell = Cell(
        variant="bf16",
        task="gsm8k",
        subject=None,
        lm_eval_task="gsm8k",
        primary_metric="exact_match",
    )
    doc = build_cell(
        cell=cell,
        task=tasks["gsm8k"],
        variant=variants["bf16"],
        run=run,
        metric_value=0.5,
        metric_stderr=0.1,
        n_samples=2,
        extra_metrics={},
        item_correct=[1, 0],
        item_ids=[0, 1],
        meta_extra={"gpu": "RTX 4090"},
    )
    from qquant.config import cell_provenance

    assert doc["config"] == cell_provenance(run, tasks["gsm8k"], "bf16")
    assert doc["meta"]["n_correct"] == 1
    assert doc["meta"]["item_correct"] == [1, 0]
    assert doc["schema_version"] == 1


def test_build_cell_rejects_length_mismatch():
    tasks = load_tasks()
    variants = load_variants()
    cell = Cell(
        variant="bf16",
        task="gsm8k",
        subject=None,
        lm_eval_task="gsm8k",
        primary_metric="exact_match",
    )
    with pytest.raises(ValueError):
        build_cell(
            cell=cell,
            task=tasks["gsm8k"],
            variant=variants["bf16"],
            run=_run(),
            metric_value=0.5,
            metric_stderr=None,
            n_samples=3,
            extra_metrics={},
            item_correct=[1, 0],
            item_ids=[0, 1],
            meta_extra={},
        )


def test_write_cell_atomic_and_is_cell_done_roundtrip(tmp_path):
    tasks = load_tasks()
    variants = load_variants()
    run = _run()
    cell = Cell(
        variant="bf16",
        task="gsm8k",
        subject=None,
        lm_eval_task="gsm8k",
        primary_metric="exact_match",
    )
    doc = build_cell(
        cell=cell,
        task=tasks["gsm8k"],
        variant=variants["bf16"],
        run=run,
        metric_value=0.5,
        metric_stderr=0.1,
        n_samples=2,
        extra_metrics={},
        item_correct=[1, 0],
        item_ids=[0, 1],
        meta_extra={},
    )
    path = cell.path(tmp_path)
    write_cell(doc, path)
    assert not list(path.parent.glob("*.tmp"))  # no partial file left
    from qquant.config import cell_provenance

    reread = json.loads(path.read_text())
    assert is_cell_done(reread, cell_provenance(run, tasks["gsm8k"], "bf16")) is True
