"""EXT-2 JudgeBench: pure scoring logic + runner with injected fakes (no torch)."""

from __future__ import annotations

import json

from qquant.config import RunConfig
from qquant.ext.judge import (
    JudgeItem,
    JudgeRunner,
    parse_verdict,
    render_judge_prompt,
    score_pair,
)
from qquant.ext.registry import ExtTask, ExtVariant


def _task(position_bias_control=True):
    return ExtTask(
        id="judgebench",
        lm_eval_task="judgebench",
        primary_metric="accuracy",
        metric_keys=("accuracy",),
        per_subject=False,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        num_fewshot=0,
        default_batch_size=8,
        max_gen_toks=64,
        runner="judge",
        position_bias_control=position_bias_control,
    )


def _variant():
    return ExtVariant(
        id="w8a8-selfquant",
        quant_method="compressed-tensors",
        source="selfquant",
        enabled=True,
        model_id="local:x",
        revision=None,
    )


def test_render_prompt_is_deterministic():
    a = render_judge_prompt("q", "ra", "rb")
    b = render_judge_prompt("q", "ra", "rb")
    assert a == b
    assert "Answer A" in a and "Answer B" in a


def test_parse_verdict():
    assert parse_verdict("A") == "A"
    assert parse_verdict("The better answer is B.") == "B"
    assert parse_verdict("b") == "B"
    assert parse_verdict("neither") is None
    assert parse_verdict("") is None


def test_score_pair_no_position_control():
    ps = score_pair("A", None, "A", position_bias_control=False)
    assert ps.correct == 1 and ps.consistent
    ps = score_pair("B", None, "A", position_bias_control=False)
    assert ps.correct == 0


def test_score_pair_with_position_control_consistent_and_correct():
    # first order says A; swapped order says B  -> swapped maps back to A -> consistent.
    ps = score_pair("A", "B", "A", position_bias_control=True)
    assert ps.consistent and ps.correct == 1


def test_score_pair_position_biased_scores_zero():
    # first says A, swapped also says A (maps back to B) -> inconsistent -> 0.
    ps = score_pair("A", "A", "A", position_bias_control=True)
    assert not ps.consistent and ps.correct == 0


def test_score_items_aggregates():
    items = [
        JudgeItem(id=0, question="q", response_a="x", response_b="y", label="A"),
        JudgeItem(id=1, question="q", response_a="x", response_b="y", label="B"),
    ]
    # Fake judge: always picks "A" in whatever order it sees.
    gen = lambda prompt: "A"  # noqa: E731
    runner = JudgeRunner("unused", RunConfig(lm_eval_version="test"))
    out = runner.score_items(items, gen, _task(position_bias_control=False))
    assert out["n_samples"] == 2
    # item0 gold A -> correct; item1 gold B -> wrong
    assert out["item_correct"] == [1, 0]
    assert out["metric_value"] == 0.5


def test_run_variant_writes_resumable_cell(tmp_path):
    items = [
        JudgeItem(id=0, question="q", response_a="x", response_b="y", label="A"),
        JudgeItem(id=1, question="q", response_a="x", response_b="y", label="A"),
    ]

    class _Loaded:
        model = object()
        tokenizer = object()

        def unload(self):
            pass

    runner = JudgeRunner(
        tmp_path,
        RunConfig(lm_eval_version="test"),
        load_model=lambda vid: _Loaded(),
        generate_fn=lambda loaded, task: lambda prompt: "A",
        dataset_fn=lambda task: items,
    )
    task = _task(position_bias_control=False)
    cell = runner.run_variant(_variant(), task)
    assert cell is not None
    path = cell.path(tmp_path)
    assert path.exists()
    doc = json.loads(path.read_text())
    assert doc["cell_id"] == "w8a8-selfquant/judgebench"
    assert doc["meta"]["item_correct"] == [1, 1]
    assert doc["meta"]["n_correct"] == 2

    # Second run is a resume no-op (matching provenance).
    again = runner.run_variant(_variant(), task)
    assert again is None
