"""Per-task lm-eval policy translation.

Torch-free: pure Task-registry → kwargs mapping.
"""

from __future__ import annotations

from qquant.config import RunConfig
from qquant.registry import Task


def greedy_gen_kwargs(task: Task) -> str:
    """Greedy-determinism gen_kwargs string for generate_until tasks.

    Ignored by MMLU loglik. Mirrors the load-time generation_config Spec 04 already
    forced (do_sample=False, …).
    """
    base = "do_sample=False,temperature=0.0,top_p=1.0"
    if task.max_gen_toks is not None:
        return f"{base},max_gen_toks={task.max_gen_toks}"
    return base


def simple_evaluate_kwargs(
    task: Task,
    variant_id: str,
    run: RunConfig,
    lm_eval_tasks: list[str],
    *,
    limit: int | None = None,
) -> dict:
    """Build lm_eval.simple_evaluate kwargs from Task registry + RunConfig."""
    kwargs = {
        "tasks": list(lm_eval_tasks),
        "num_fewshot": task.num_fewshot,
        "apply_chat_template": task.apply_chat_template,
        "fewshot_as_multiturn": task.fewshot_as_multiturn,
        "gen_kwargs": greedy_gen_kwargs(task),
        "batch_size": task.batch_size_for(variant_id),
        "random_seed": run.seeds.random,
        "numpy_random_seed": run.seeds.numpy,
        "torch_random_seed": run.seeds.torch,
        "fewshot_random_seed": run.seeds.fewshot,
        "log_samples": True,
        "limit": limit,
    }
    if task.code_exec:
        kwargs["confirm_run_unsafe_code"] = True
    return kwargs
