"""Ext registry: own id space, validation, gating, and matrix reuse."""

from __future__ import annotations

import textwrap

import pytest

from qquant.ext.registry import (
    EXT_TASK_IDS,
    EXT_VARIANT_IDS,
    ExtTask,
    ExtVariant,
    enabled_ext_variants,
    load_ext_tasks,
    load_ext_variants,
)
from qquant.matrix import expand_matrix, is_cell_done
from qquant.registry import TASK_IDS, VARIANT_IDS, Task, Variant


def test_ext_ids_disjoint_from_v1_ssot():
    """Ext ids must NOT leak into the v1 canonical tuples (the isolation point)."""
    assert set(EXT_VARIANT_IDS).isdisjoint(VARIANT_IDS)
    assert set(EXT_TASK_IDS).isdisjoint(TASK_IDS)


def test_ext_ids_obey_naming_rule():
    """Hyphenated, no underscores, no -int4 suffix, not a bare gptq/awq/bnb."""
    for vid in (*EXT_VARIANT_IDS, *EXT_TASK_IDS):
        assert "_" not in vid
        assert not vid.endswith("-int4")
        assert vid not in {"gptq", "awq", "bnb"}


def test_subclasses_are_v1_compatible():
    assert issubclass(ExtVariant, Variant)
    assert issubclass(ExtTask, Task)


def test_shipped_registry_is_fully_gated():
    """Every shipped ext variant row is enabled:false (nothing joins a run)."""
    variants = load_ext_variants()
    assert variants  # rows exist
    assert all(not v.enabled for v in variants.values())
    assert enabled_ext_variants(variants) == []


def test_shipped_registry_has_known_ids():
    variants = load_ext_variants()
    assert set(variants).issubset(EXT_VARIANT_IDS)
    tasks = load_ext_tasks()
    assert set(tasks).issubset(EXT_TASK_IDS)
    assert "w8a8-selfquant" in variants
    assert "bnb-nf4-qlora" in variants
    assert "judgebench" in tasks


def test_unknown_ext_variant_id_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """
            variants:
              not-a-real-ext-id:
                quant_method: bf16
                source: baseline
                enabled: false
                model_id: x
            """
        )
    )
    with pytest.raises(ValueError, match="unknown ext variant id"):
        load_ext_variants(p)


def test_kv_cache_field_parsed_and_validated(tmp_path):
    p = tmp_path / "kv.yaml"
    p.write_text(
        textwrap.dedent(
            """
            variants:
              bf16-kvq4:
                quant_method: bf16
                source: baseline
                enabled: false
                model_id: x
                kv_cache: { backend: quanto, nbits: 4 }
            """
        )
    )
    v = load_ext_variants(p)["bf16-kvq4"]
    assert v.kv_cache == {"backend": "quanto", "nbits": 4}


def test_kv_cache_bad_backend_rejected(tmp_path):
    p = tmp_path / "kv.yaml"
    p.write_text(
        textwrap.dedent(
            """
            variants:
              bf16-kvq4:
                quant_method: bf16
                source: baseline
                enabled: false
                model_id: x
                kv_cache: { backend: nope, nbits: 4 }
            """
        )
    )
    with pytest.raises(ValueError, match="kv_cache backend"):
        load_ext_variants(p)


def test_adapter_path_parsed(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(
        textwrap.dedent(
            """
            variants:
              bnb-nf4-qlora:
                quant_method: bnb-nf4
                source: recovery
                enabled: false
                model_id: Qwen/Qwen2.5-7B-Instruct
                adapter_path: "local:checkpoints/qlora/bnb-nf4-qlora"
            """
        )
    )
    v = load_ext_variants(p)["bnb-nf4-qlora"]
    assert v.adapter_path == "local:checkpoints/qlora/bnb-nf4-qlora"


def test_ext_variant_expands_in_v1_matrix():
    """ExtVariant + v1 Task drop straight into expand_matrix (duck-typing)."""
    v = ExtVariant(
        id="w8a8-selfquant",
        quant_method="compressed-tensors",
        source="selfquant",
        enabled=True,
        model_id="local:checkpoints/self-quant/w8a8-selfquant",
        revision=None,
    )
    task = Task(
        id="gsm8k",
        lm_eval_task="gsm8k",
        primary_metric="exact_match",
        metric_keys=("exact_match",),
        per_subject=False,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        num_fewshot=5,
        default_batch_size=4,
    )
    cells = expand_matrix([v], [task])
    assert len(cells) == 1
    assert cells[0].cell_id == "w8a8-selfquant/gsm8k"


def test_judge_task_carries_runner_and_dataset():
    task = load_ext_tasks()["judgebench"]
    assert task.runner == "judge"
    assert task.hf_dataset_id
    assert task.position_bias_control is True


def test_resume_predicate_accepts_ext_cell():
    """A built ext cell passes is_cell_done (schema variant/task are plain strings)."""
    doc = {
        "schema_version": 1,
        "cell_id": "w8a8-selfquant/gsm8k",
        "metric_value": 0.8,
        "n_samples": 10,
    }
    assert is_cell_done(doc)
