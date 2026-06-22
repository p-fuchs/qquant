from __future__ import annotations

from qquant.registry import (
    MMLU_SUBJECTS,
    QUANT_METHODS,
    TASK_IDS,
    VARIANT_IDS,
    enabled_variants,
    load_tasks,
    load_variants,
)


def test_variant_ids_count_and_unique():
    assert len(VARIANT_IDS) == 7
    assert len(set(VARIANT_IDS)) == 7


def test_mmlu_subjects_count_and_unique():
    assert len(MMLU_SUBJECTS) == 57
    assert len(set(MMLU_SUBJECTS)) == 57


def test_variants_yaml_loads_and_matches_ssot():
    variants = load_variants()
    assert set(variants) == set(VARIANT_IDS)
    for v in variants.values():
        assert v.quant_method in QUANT_METHODS
        # official/baseline/load-time pin a revision; self-quant are produced locally.
        if v.source in {"official", "baseline", "load-time"}:
            assert v.revision, f"{v.id} must pin a revision SHA"
        else:
            assert v.revision is None


def test_self_quant_uses_compressed_tensors():
    variants = load_variants()
    assert variants["gptq-selfquant"].quant_method == "compressed-tensors"
    assert variants["awq-selfquant"].quant_method == "compressed-tensors"


def test_tasks_yaml_loads():
    tasks = load_tasks()
    assert set(tasks) == set(TASK_IDS)
    assert tasks["mmlu"].per_subject is True
    assert tasks["humaneval"].code_exec is True
    # bf16 gets a smaller MMLU batch than the task default (logits-transient guard).
    assert tasks["mmlu"].batch_size_for("bf16") == 2
    assert tasks["mmlu"].batch_size_for("gptq-official") == 4


def test_enabled_variants_in_canonical_order():
    variants = load_variants()
    enabled = enabled_variants(variants)
    ids = [v.id for v in enabled]
    assert ids == [v for v in VARIANT_IDS if variants[v].enabled]
