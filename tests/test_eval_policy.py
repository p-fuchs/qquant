from __future__ import annotations

from qquant.config import RunConfig, Seeds
from qquant.eval.policy import greedy_gen_kwargs, simple_evaluate_kwargs
from qquant.registry import load_tasks


def _run():
    return RunConfig(
        lm_eval_version="x",
        model_revision="rev",
        max_length=4096,
        seeds=Seeds(random=1, numpy=2, torch=3, fewshot=4),
    )


def test_greedy_gen_kwargs_contains_do_sample_false_and_max_gen_toks():
    tasks = load_tasks()
    g = greedy_gen_kwargs(tasks["gsm8k"])  # max_gen_toks=512
    assert "do_sample=False" in g
    assert "temperature=0.0" in g
    assert "top_p=1.0" in g
    assert "max_gen_toks=512" in g


def test_greedy_gen_kwargs_omits_max_gen_toks_when_none():
    tasks = load_tasks()
    g = greedy_gen_kwargs(tasks["mmlu"])  # max_gen_toks=None
    assert "do_sample=False" in g
    assert "max_gen_toks" not in g


def test_simple_evaluate_kwargs_forwards_registry_fields():
    tasks = load_tasks()
    kw = simple_evaluate_kwargs(
        tasks["mmlu"], "bf16", _run(), ["mmlu_anatomy"], limit=None
    )
    assert kw["tasks"] == ["mmlu_anatomy"]
    assert kw["num_fewshot"] == 5
    assert kw["apply_chat_template"] is True
    assert kw["fewshot_as_multiturn"] is True
    assert kw["batch_size"] == 2  # bf16 MMLU override
    assert kw["random_seed"] == 1
    assert kw["numpy_random_seed"] == 2
    assert kw["torch_random_seed"] == 3
    assert kw["fewshot_random_seed"] == 4
    assert kw["log_samples"] is True
    assert kw["limit"] is None
    assert "confirm_run_unsafe_code" not in kw  # mmlu is not code_exec


def test_simple_evaluate_kwargs_sets_confirm_unsafe_for_code_exec():
    tasks = load_tasks()
    kw = simple_evaluate_kwargs(
        tasks["humaneval"], "gptq-official", _run(), ["humaneval_instruct"]
    )
    assert kw["confirm_run_unsafe_code"] is True
    assert kw["num_fewshot"] == 0
    assert kw["batch_size"] == 4
