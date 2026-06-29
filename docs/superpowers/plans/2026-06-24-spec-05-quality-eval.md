# Spec 05 — Quality Eval (`qquant.eval` + `qquant-eval`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `qquant.eval` (and the forward-declared `qquant.aggregate.stats`) — the GPU-touching, resumable, manifest-driven quality-eval engine that wraps a Spec-04-loaded model in an lm-eval `HFLM`, runs mmlu/gsm8k/humaneval/ifeval, and writes one schema-valid, provenanced result cell per `(variant, task[, subject])`.

**Architecture:** A stack of small torch-free leaf modules (stats, policy, metrics, datasets, results, manifest) that are CPU-TDD'd on the laptop, plus three lazy modules (hflm, runner, cli) whose `torch`/`lm_eval`/`datasets`/`qquant.models` imports live **inside functions** (mirroring Spec 04's deferred-import pattern). The runner takes injectable `load_model`/`evaluate_fn` so its load-once / skip-when-done / 57-MMLU-cell / code-exec-gating logic is fully tested with fakes; the real lm-eval `HFLM` path is verified on the RTX 4090.

**Tech Stack:** Python 3.11, lm-eval (spike ref `c6491878…`, `0.4.12` fallback), transformers 5.10.1 / torch 2.12.1+cu126 (Linux `gpu` group), `datasets`; pytest; ruff. Consumes Spec-01 core (`qquant.registry/matrix/paths/config/schemas`) and Spec-04 `qquant.models.load_variant`.

## Global Constraints

Every task implicitly includes these (verbatim from `docs/specs/05-quality-eval.md`, `contracts.md`, `decision-log.md`).

- **Deferred imports.** `qquant.aggregate.stats`, `qquant.eval.{policy,metrics,datasets,results,manifest}` import **no** `torch`/`lm_eval`/`datasets` at module top. `qquant.eval.{hflm,runner,cli}` import `torch`/`lm_eval`/`datasets`/`qquant.models` **only inside functions**. `import qquant.eval` stays torch-free; the umbrella `qquant` CLI never imports `qquant.eval`.
- **SSOT reuse — never reimplement.** Cells are written only via `cell_path`/`Paths`; resume is `is_cell_done`/`missing_cells`; provenance is `cell_provenance` (its 8 keys == `PROVENANCE_KEYS` == the schema `config` block); the matrix is `expand_matrix`; validation is `validate_cell`. `wilson_interval` has exactly one implementation (in `qquant.aggregate.stats`), re-exported by `qquant.eval.metrics`.
- **`cell_provenance(run, task, variant_id)` returns exactly** `{seeds, batch_size, num_fewshot, apply_chat_template, fewshot_as_multiturn, max_length, lm_eval_version, model_revision}`. `batch_size` is the **resolved int** `task.batch_size_for(variant_id)`.
- **Per-variant `model_revision`.** Provenance `model_revision` for a variant is `variant.revision` (the pinned SHA for official/baseline; `None` for self-quant — self-quant fingerprinting is **deferred to post-Spec-07**). The runner derives a per-variant `RunConfig` via `dataclasses.replace(run, model_revision=variant.revision)`. Using `variant.revision` (not `resolve_model_source`) keeps `plan()` torch-free and avoids the self-quant `FileNotFoundError`.
- **Per-task policy is registry data, never hardcoded.** `num_fewshot`, `apply_chat_template`, `fewshot_as_multiturn`, `lm_eval_task`, `max_gen_toks`, `metric_keys`, `code_exec`, and `batch_size_for(variant_id)` all come from the `Task`.
- **Greedy determinism.** `greedy_gen_kwargs` → `"do_sample=False,temperature=0.0,top_p=1.0"` (+ `,max_gen_toks=<task.max_gen_toks>` when set). The four lm-eval seeds (`random_seed`, `numpy_random_seed`, `torch_random_seed`, `fewshot_random_seed`) come from `run.seeds.{random,numpy,torch,fewshot}`. `log_samples=True` always (needed for n + item vectors).
- **MMLU = 57 resumable per-subject cells**; one `simple_evaluate` over only the **missing** `mmlu_<subject>` subtasks; emitted subject set == `set(MMLU_SUBJECTS)`; each cell stores per-subject `acc`/`acc_stderr`/`n` **and** that subject's per-item vectors.
- **Item vectors in EVERY cell.** `meta.item_correct` (list of `0/1`) + `meta.item_ids` (lm-eval `doc_id`s) in every cell incl. MMLU; `n_samples == len(item_correct) == len(item_ids)`. `meta.n_correct = sum(item_correct)`.
- **gsm8k id.** `apply_dataset_overrides()` rewrites bare `gsm8k` → `openai/gsm8k` (drops stale revision); the runner calls it before any gsm8k eval. (`datasets>=4` rejects the bare id.)
- **HumanEval double-env gate.** A `code_exec` task runs only when **both** `HF_ALLOW_CODE_EVAL=1` **and** `QQUANT_ALLOW_CODE_EXEC=1`; then `confirm_run_unsafe_code=True`. Else the cell is **skipped** (left missing) with a logged warning — never a hard failure. pass@1 = k/n; store Wilson CI; `metric_stderr=None`; bf16 baseline pass@1 `> 0.5` sanity gate (loud warning + `meta.sanity_failed` if not).
- **IFEval.** Set `langdetect.DetectorFactory.seed = 0` before any ifeval run.
- **`--limit`/smoke refuses to write cells** (builds+validates in memory only). `--plan`/`--list-missing` are torch-free.
- **Entry point.** Add `qquant-eval = "qquant.eval.cli:main"` to `[project.scripts]` (a separate GPU-touching binary, never a subcommand of the torch-free `qquant`). Exit codes: `0` all requested cells `is_cell_done`; `1` any failure; `2` config/manifest error.
- **Tooling.** ruff line-length 88, rules E,F,I,UP,B; `ruff format`. Python `>=3.11,<3.12`. The `gpu` pytest marker already exists (Spec 04); the GPU smoke rides on `@pytest.mark.gpu` (auto-skipped off-CUDA via `tests/conftest.py`).

---

### Task 1: `qquant.aggregate.stats.wilson_interval`

**Files:**
- Create: `src/qquant/aggregate/__init__.py`
- Create: `src/qquant/aggregate/stats.py`
- Test: `tests/test_aggregate_stats.py`

**Interfaces:**
- Consumes: nothing (stdlib `math`, `statistics`).
- Produces: `wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]` — Wilson score interval, clamped to `[0, 1]`; raises `ValueError` for `n <= 0` or `k` outside `[0, n]`. Re-exported later by `qquant.eval.metrics`; **Spec 09 will extend this module, never redefine this function.**

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_stats.py`:

```python
from __future__ import annotations

import pytest

from qquant.aggregate.stats import wilson_interval


def test_wilson_known_reference():
    lo, hi = wilson_interval(140, 164)
    assert 0.79 < lo < 0.81
    assert 0.90 < hi < 0.92


def test_wilson_zero_successes_clamps_low_at_zero():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 0.35


def test_wilson_all_successes_clamps_high_at_one():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert 0.65 < lo < 1.0


def test_wilson_rejects_bad_n_and_k():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.aggregate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/aggregate/__init__.py`:

```python
"""qquant.aggregate — analysis/statistics package.

Spec 05 forward-declares this package and lands ``stats.wilson_interval`` (the single
implementation); Spec 09 EXTENDS this module (MMLU weighting, paired McNemar, plots) without
redefining wilson_interval. Torch-free.
"""
```

Create `src/qquant/aggregate/stats.py`:

```python
"""Statistics for the qquant analysis layer. Torch-free, pure stdlib.

``wilson_interval`` is the SINGLE implementation of the Wilson score interval; Spec 05's
eval-time HumanEval pass@1 CI re-exports it from here, and Spec 09 builds the rest of the
aggregation on top of this module.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Returns (lo, hi) clamped to [0, 1]. Raises ValueError for n <= 0 or k not in [0, n].
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}], got {k}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregate_stats.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/aggregate/__init__.py src/qquant/aggregate/stats.py tests/test_aggregate_stats.py
git commit -m "Spec 05: qquant.aggregate.stats.wilson_interval (SSOT, Spec 09 extends)"
```

---

### Task 2: `qquant.eval` package + `policy`

**Files:**
- Create: `src/qquant/eval/__init__.py` (minimal package marker; finalized in Task 9)
- Create: `src/qquant/eval/policy.py`
- Test: `tests/test_eval_policy.py`

**Interfaces:**
- Consumes: `qquant.registry.Task` (via fixtures), `qquant.config.RunConfig`/`Seeds`.
- Produces:
  - `greedy_gen_kwargs(task: Task) -> str`
  - `simple_evaluate_kwargs(task: Task, variant_id: str, run: RunConfig, lm_eval_tasks: list[str], *, limit: int | None = None) -> dict` — copies policy from the registry; `log_samples=True`; adds `confirm_run_unsafe_code=True` only when `task.code_exec`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_policy.py`:

```python
from __future__ import annotations

from qquant.config import RunConfig, Seeds
from qquant.eval.policy import greedy_gen_kwargs, simple_evaluate_kwargs
from qquant.registry import load_tasks


def _run():
    return RunConfig(lm_eval_version="x", model_revision="rev", max_length=4096,
                     seeds=Seeds(random=1, numpy=2, torch=3, fewshot=4))


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
    kw = simple_evaluate_kwargs(tasks["mmlu"], "bf16", _run(), ["mmlu_anatomy"], limit=None)
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
    kw = simple_evaluate_kwargs(tasks["humaneval"], "gptq-official", _run(), ["humaneval_instruct"])
    assert kw["confirm_run_unsafe_code"] is True
    assert kw["num_fewshot"] == 0
    assert kw["batch_size"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/__init__.py`:

```python
"""qquant.eval — GPU-touching quality-eval engine (Spec 05). Public surface added in Task 9.

torch/lm_eval/datasets are imported lazily inside functions; importing this package stays
torch-free. NOT part of the torch-free core — the umbrella ``qquant`` CLI must never import it.
"""
```

Create `src/qquant/eval/policy.py`:

```python
"""Per-task lm-eval policy translation. Torch-free: pure Task-registry → kwargs mapping."""

from __future__ import annotations

from qquant.config import RunConfig
from qquant.registry import Task


def greedy_gen_kwargs(task: Task) -> str:
    """Greedy-determinism gen_kwargs string for generate_until tasks (ignored by MMLU loglik).

    Mirrors the load-time generation_config Spec 04 already forced (do_sample=False, …).
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
    """Build lm_eval.simple_evaluate kwargs entirely from the Task registry + RunConfig."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_policy.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/__init__.py src/qquant/eval/policy.py tests/test_eval_policy.py
git commit -m "Spec 05: eval package + per-task policy (registry-driven, greedy)"
```

---

### Task 3: `qquant.eval.metrics`

**Files:**
- Create: `src/qquant/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `qquant.aggregate.stats.wilson_interval` (Task 1).
- Produces:
  - `extract_metric(results_for_task: dict, metric_keys: Sequence[str]) -> tuple[str, float]` — first matching key in order; `KeyError` (listing available keys) if none.
  - `extract_stderr(results_for_task: dict, primary_metric: str) -> float | None` — best-effort `<primary>_stderr*`; `None` if absent/non-numeric.
  - `wilson_interval` (re-exported).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_metrics.py`:

```python
from __future__ import annotations

import pytest

from qquant.eval.metrics import extract_metric, extract_stderr, wilson_interval


def test_extract_metric_first_key_in_order_wins():
    res = {"acc,none": 0.81, "acc": 0.79}
    assert extract_metric(res, ("acc,none", "acc")) == ("acc,none", 0.81)


def test_extract_metric_falls_through_to_later_key():
    res = {"pass@1,none": 0.42}
    assert extract_metric(res, ("pass@1,create_test", "pass@1,none", "pass@1")) == ("pass@1,none", 0.42)


def test_extract_metric_raises_listing_keys_when_none_match():
    with pytest.raises(KeyError) as exc:
        extract_metric({"foo": 1.0}, ("acc,none", "acc"))
    assert "foo" in str(exc.value)


def test_extract_stderr_finds_suffixed_key():
    res = {"acc,none": 0.81, "acc_stderr,none": 0.012}
    assert extract_stderr(res, "acc") == pytest.approx(0.012)


def test_extract_stderr_returns_none_when_absent_or_na():
    assert extract_stderr({"pass@1,none": 0.4}, "pass@1") is None
    assert extract_stderr({"acc_stderr,none": "N/A"}, "acc") is None


def test_wilson_interval_is_reexported_here():
    lo, hi = wilson_interval(140, 164)
    assert 0.79 < lo < 0.81 and 0.90 < hi < 0.92
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/metrics.py`:

```python
"""Metric extraction from lm-eval result dicts. Torch-free.

wilson_interval is NOT reimplemented here — it is re-exported from qquant.aggregate.stats
(the single SSOT implementation) for the eval-time HumanEval pass@1 CI.
"""

from __future__ import annotations

from collections.abc import Sequence

from qquant.aggregate.stats import wilson_interval  # re-export (SSOT lives in aggregate.stats)

__all__ = ["extract_metric", "extract_stderr", "wilson_interval"]


def extract_metric(results_for_task: dict, metric_keys: Sequence[str]) -> tuple[str, float]:
    """Return the first (key, float) whose key is present, trying metric_keys in order.

    Raises KeyError (listing available keys) if none match — absorbs lm-eval suffix drift.
    """
    for key in metric_keys:
        if key in results_for_task:
            return key, float(results_for_task[key])
    raise KeyError(
        f"none of {list(metric_keys)} in results keys {sorted(results_for_task)}"
    )


def extract_stderr(results_for_task: dict, primary_metric: str) -> float | None:
    """Best-effort '<primary_metric>_stderr[,filter]' lookup; None if absent/non-numeric."""
    prefix = f"{primary_metric}_stderr"
    for key, value in results_for_task.items():
        if key == prefix or key.startswith(prefix + ","):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/metrics.py tests/test_eval_metrics.py
git commit -m "Spec 05: eval.metrics extract_metric/extract_stderr + wilson re-export"
```

---

### Task 4: `qquant.eval.datasets` (gsm8k → openai/gsm8k rewrite)

**Files:**
- Create: `src/qquant/eval/datasets.py`
- Test: `tests/test_eval_datasets.py`

**Interfaces:**
- Consumes: nothing at import (lazy `datasets`).
- Produces: `apply_dataset_overrides(datasets_module=None) -> module` — idempotently wraps `datasets_module.load_dataset` / `load_dataset_builder` so a bare `"gsm8k"` path becomes `"openai/gsm8k"` (and a `revision=` kwarg is dropped); lazily `import datasets` when no module is passed; returns the (patched) module. Module-level `DATASET_OVERRIDES = {"gsm8k": "openai/gsm8k"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_datasets.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from qquant.eval.datasets import DATASET_OVERRIDES, apply_dataset_overrides


def _fake_datasets():
    calls = []

    def load_dataset(path, *args, **kwargs):
        calls.append(("load_dataset", path, args, kwargs))
        return ("ds", path, args, kwargs)

    def load_dataset_builder(path, *args, **kwargs):
        calls.append(("load_dataset_builder", path, args, kwargs))
        return ("builder", path, args, kwargs)

    return SimpleNamespace(load_dataset=load_dataset,
                           load_dataset_builder=load_dataset_builder), calls


def test_rewrites_bare_gsm8k_and_drops_revision():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    mod.load_dataset("gsm8k", "main", revision="deadbeef")
    name, path, args, kwargs = calls[0]
    assert path == "openai/gsm8k"
    assert args == ("main",)
    assert "revision" not in kwargs


def test_passes_through_non_overridden_ids_unchanged():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    mod.load_dataset_builder("openai/gsm8k", revision="keep")
    name, path, args, kwargs = calls[0]
    assert path == "openai/gsm8k"
    assert kwargs == {"revision": "keep"}


def test_is_idempotent():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    first = mod.load_dataset
    apply_dataset_overrides(mod)
    assert mod.load_dataset is first  # not double-wrapped


def test_overrides_table_maps_gsm8k():
    assert DATASET_OVERRIDES["gsm8k"] == "openai/gsm8k"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.datasets'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/datasets.py`:

```python
"""Runtime dataset-id overrides for lm-eval. Torch-free; imports `datasets` lazily.

lm-eval's gsm8k task references the bare `gsm8k` repo id, which datasets>=4 rejects
(HfUriError); the working id is `openai/gsm8k`. apply_dataset_overrides() monkeypatches the
datasets loaders so the rewrite happens transparently at eval time (mirrors the Spec-02 spike).
"""

from __future__ import annotations

DATASET_OVERRIDES = {"gsm8k": "openai/gsm8k"}

_PATCH_FLAG = "_qquant_dataset_overrides_applied"


def _wrap(orig):
    def wrapped(path, *args, **kwargs):
        if path in DATASET_OVERRIDES:
            path = DATASET_OVERRIDES[path]
            kwargs.pop("revision", None)  # the pinned revision belonged to the old id
        return orig(path, *args, **kwargs)

    return wrapped


def apply_dataset_overrides(datasets_module=None):
    """Idempotently patch datasets.load_dataset/load_dataset_builder to rewrite bare ids."""
    if datasets_module is None:
        import datasets as datasets_module
    if getattr(datasets_module, _PATCH_FLAG, False):
        return datasets_module
    for fn_name in ("load_dataset", "load_dataset_builder"):
        orig = getattr(datasets_module, fn_name)
        setattr(datasets_module, fn_name, _wrap(orig))
    setattr(datasets_module, _PATCH_FLAG, True)
    return datasets_module
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_datasets.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/datasets.py tests/test_eval_datasets.py
git commit -m "Spec 05: eval.datasets apply_dataset_overrides (gsm8k -> openai/gsm8k)"
```

---

### Task 5: `qquant.eval.results` (parse lm-eval results → cell)

**Files:**
- Create: `src/qquant/eval/results.py`
- Test: `tests/test_eval_results.py`

**Interfaces:**
- Consumes: `extract_metric`/`extract_stderr` (Task 3); `qquant.config.cell_provenance`; `qquant.schemas.validate_cell`; `qquant.matrix.Cell`, `is_cell_done`; `qquant.registry.Task`/`Variant`.
- Produces:
  - `cell_inputs_from_results(results: dict, lm_eval_task: str, task: Task) -> dict` — parses lm-eval's `simple_evaluate` output (shape documented below) into `{metric_key, metric_value, metric_stderr, n_samples, item_correct, item_ids, extra_metrics}`. Raises `KeyError` if a per-doc correctness field can't be found (loud, not silent-0).
  - `build_cell(*, cell: Cell, task: Task, variant: Variant, run: RunConfig, metric_value: float, metric_stderr: float | None, n_samples: int, extra_metrics: dict, item_correct: list[int], item_ids: list, meta_extra: dict) -> dict` — assembles a v1 cell; `config = cell_provenance(run, task, variant.id)`; enforces `n_samples == len(item_correct) == len(item_ids)`; `validate_cell` before return.
  - `write_cell(cell_doc: dict, path: Path) -> None` — `validate_cell` + atomic write (`tmp` sibling + `os.replace`), `mkdir parents`.

Expected lm-eval result shape consumed by `cell_inputs_from_results` (documented; the GPU smoke validates it against the real ref):
```python
results = {
  "results": {"<lm_eval_task>": {"<metric_key>": float, "<primary>_stderr,<filter>": float, ...}},
  "samples": {"<lm_eval_task>": [{"doc_id": int, "<primary_metric>": 0|1|bool|float, ...}, ...]},
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_results.py`:

```python
from __future__ import annotations

import json

import pytest

from qquant.config import RunConfig, Seeds
from qquant.eval.results import build_cell, cell_inputs_from_results, write_cell
from qquant.matrix import Cell, is_cell_done
from qquant.registry import load_tasks, load_variants


def _run():
    return RunConfig(lm_eval_version="lmv", model_revision="rev123", max_length=4096,
                     seeds=Seeds())


def _gsm8k_results():
    return {
        "results": {"gsm8k": {"exact_match,strict-match": 0.5,
                              "exact_match_stderr,strict-match": 0.1}},
        "samples": {"gsm8k": [{"doc_id": 0, "exact_match": 1},
                              {"doc_id": 1, "exact_match": 0}]},
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
    bad = {"results": {"gsm8k": {"exact_match,strict-match": 0.5}},
           "samples": {"gsm8k": [{"doc_id": 0, "unexpected": 1}]}}
    with pytest.raises(KeyError):
        cell_inputs_from_results(bad, "gsm8k", tasks["gsm8k"])


def test_build_cell_validates_and_config_is_provenance(tmp_path):
    tasks = load_tasks()
    variants = load_variants()
    run = _run()
    cell = Cell(variant="bf16", task="gsm8k", subject=None,
                lm_eval_task="gsm8k", primary_metric="exact_match")
    doc = build_cell(cell=cell, task=tasks["gsm8k"], variant=variants["bf16"], run=run,
                     metric_value=0.5, metric_stderr=0.1, n_samples=2,
                     extra_metrics={}, item_correct=[1, 0], item_ids=[0, 1],
                     meta_extra={"gpu": "RTX 4090"})
    from qquant.config import cell_provenance
    assert doc["config"] == cell_provenance(run, tasks["gsm8k"], "bf16")
    assert doc["meta"]["n_correct"] == 1
    assert doc["meta"]["item_correct"] == [1, 0]
    assert doc["schema_version"] == 1


def test_build_cell_rejects_length_mismatch():
    tasks = load_tasks()
    variants = load_variants()
    cell = Cell(variant="bf16", task="gsm8k", subject=None,
                lm_eval_task="gsm8k", primary_metric="exact_match")
    with pytest.raises(ValueError):
        build_cell(cell=cell, task=tasks["gsm8k"], variant=variants["bf16"], run=_run(),
                   metric_value=0.5, metric_stderr=None, n_samples=3,
                   extra_metrics={}, item_correct=[1, 0], item_ids=[0, 1], meta_extra={})


def test_write_cell_atomic_and_is_cell_done_roundtrip(tmp_path):
    tasks = load_tasks()
    variants = load_variants()
    run = _run()
    cell = Cell(variant="bf16", task="gsm8k", subject=None,
                lm_eval_task="gsm8k", primary_metric="exact_match")
    doc = build_cell(cell=cell, task=tasks["gsm8k"], variant=variants["bf16"], run=run,
                     metric_value=0.5, metric_stderr=0.1, n_samples=2,
                     extra_metrics={}, item_correct=[1, 0], item_ids=[0, 1], meta_extra={})
    path = cell.path(tmp_path)
    write_cell(doc, path)
    assert not list(path.parent.glob("*.tmp"))  # no partial file left
    from qquant.config import cell_provenance
    reread = json.loads(path.read_text())
    assert is_cell_done(reread, cell_provenance(run, tasks["gsm8k"], "bf16")) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.results'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/results.py`:

```python
"""lm-eval results → v1 cell. Torch-free; uses cell_provenance + validate_cell + cell_path."""

from __future__ import annotations

import json
import os
from pathlib import Path

from qquant.config import RunConfig, cell_provenance
from qquant.eval.metrics import extract_metric, extract_stderr
from qquant.matrix import Cell
from qquant.registry import Task, Variant
from qquant.schemas import validate_cell

# Per-doc correctness fields tried in order (lm-eval stores the bare metric per sample).
_CORRECTNESS_FIELDS = ("acc", "exact_match", "pass@1", "prompt_level_strict_acc")


def _doc_correct(sample: dict, primary_metric: str) -> int:
    for key in (primary_metric, *_CORRECTNESS_FIELDS):
        if key in sample:
            return int(round(float(sample[key])))
    raise KeyError(
        f"no per-doc correctness field for {primary_metric!r} in sample keys "
        f"{sorted(sample)}"
    )


def cell_inputs_from_results(results: dict, lm_eval_task: str, task: Task) -> dict:
    """Parse one (sub)task's lm-eval output into the inputs build_cell needs."""
    per_task = results["results"][lm_eval_task]
    metric_key, metric_value = extract_metric(per_task, task.metric_keys)
    metric_stderr = extract_stderr(per_task, task.primary_metric)
    samples = results.get("samples", {}).get(lm_eval_task, [])
    item_ids = [s["doc_id"] for s in samples]
    item_correct = [_doc_correct(s, task.primary_metric) for s in samples]
    extra_metrics = {
        k: v for k, v in per_task.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != metric_key
    }
    return {
        "metric_key": metric_key,
        "metric_value": metric_value,
        "metric_stderr": metric_stderr,
        "n_samples": len(samples),
        "item_correct": item_correct,
        "item_ids": item_ids,
        "extra_metrics": extra_metrics,
    }


def build_cell(
    *,
    cell: Cell,
    task: Task,
    variant: Variant,
    run: RunConfig,
    metric_value: float,
    metric_stderr: float | None,
    n_samples: int,
    extra_metrics: dict,
    item_correct: list[int],
    item_ids: list,
    meta_extra: dict,
) -> dict:
    """Assemble + validate a v1 cell. config == cell_provenance; item vectors live in meta."""
    if not (n_samples == len(item_correct) == len(item_ids)):
        raise ValueError(
            f"n_samples ({n_samples}) must equal len(item_correct) ({len(item_correct)}) "
            f"and len(item_ids) ({len(item_ids)})"
        )
    meta = {
        **meta_extra,
        "item_correct": item_correct,
        "item_ids": item_ids,
        "n_correct": sum(item_correct),
    }
    doc = {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "variant": variant.id,
        "task": task.id,
        "subject": cell.subject,
        "task_lm_eval": cell.lm_eval_task,
        "primary_metric": task.primary_metric,
        "metric_value": metric_value,
        "metric_stderr": metric_stderr,
        "n_samples": n_samples,
        "extra_metrics": extra_metrics,
        "config": cell_provenance(run, task, variant.id),
        "meta": meta,
    }
    validate_cell(doc)
    return doc


def write_cell(cell_doc: dict, path: Path) -> None:
    """Validate then atomically write the cell (tmp sibling + os.replace)."""
    validate_cell(cell_doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cell_doc, indent=2, sort_keys=True))
    os.replace(tmp, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_results.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/results.py tests/test_eval_results.py
git commit -m "Spec 05: eval.results cell_inputs_from_results + build_cell + atomic write_cell"
```

---

### Task 6: `qquant.eval.manifest`

**Files:**
- Create: `src/qquant/eval/manifest.py`
- Test: `tests/test_eval_manifest.py`

**Interfaces:**
- Consumes: `qquant.registry.VARIANT_IDS`, `TASK_IDS`, `Variant`, `Task` (via fixtures); `yaml` (already a core dep).
- Produces:
  - `load_manifest(path: str | Path) -> list[tuple[str, str]]` — parses JSON/YAML `{"cells": [{"variant","task"}, ...]}`; validates ids ∈ `VARIANT_IDS`/`TASK_IDS`; `ValueError` on unknown or duplicate pair.
  - `default_manifest(variants: list[Variant], tasks: list[Task]) -> list[tuple[str, str]]` — the `(variant.id, task.id)` cartesian product (the enabled matrix when given enabled lists).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_manifest.py`:

```python
from __future__ import annotations

import json

import pytest

from qquant.eval.manifest import default_manifest, load_manifest
from qquant.registry import enabled_variants, load_tasks, load_variants


def test_load_manifest_parses_pairs(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"cells": [{"variant": "bf16", "task": "mmlu"},
                                        {"variant": "gptq-official", "task": "gsm8k"}]}))
    assert load_manifest(p) == [("bf16", "mmlu"), ("gptq-official", "gsm8k")]


def test_load_manifest_rejects_unknown_ids(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"cells": [{"variant": "nope", "task": "mmlu"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)
    p.write_text(json.dumps({"cells": [{"variant": "bf16", "task": "nope"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)


def test_load_manifest_rejects_duplicate_pairs(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"cells": [{"variant": "bf16", "task": "mmlu"},
                                        {"variant": "bf16", "task": "mmlu"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)


def test_default_manifest_is_enabled_matrix():
    variants = load_variants()
    tasks = load_tasks()
    enabled = enabled_variants(variants)
    task_list = list(tasks.values())
    pairs = default_manifest(enabled, task_list)
    assert len(pairs) == len(enabled) * len(task_list)
    assert ("bf16", "mmlu") in pairs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.manifest'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/manifest.py`:

```python
"""Eval manifest: task-level (variant, task) pairs to run. Torch-free."""

from __future__ import annotations

from pathlib import Path

import yaml

from qquant.registry import TASK_IDS, VARIANT_IDS, Task, Variant


def load_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Parse a JSON/YAML manifest into validated (variant, task) pairs."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "cells" not in data:
        raise ValueError("manifest must be a mapping with a 'cells' list")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in data["cells"]:
        vid, tid = entry["variant"], entry["task"]
        if vid not in VARIANT_IDS:
            raise ValueError(f"manifest: unknown variant id {vid!r}")
        if tid not in TASK_IDS:
            raise ValueError(f"manifest: unknown task id {tid!r}")
        if (vid, tid) in seen:
            raise ValueError(f"manifest: duplicate cell {(vid, tid)!r}")
        seen.add((vid, tid))
        pairs.append((vid, tid))
    return pairs


def default_manifest(variants: list[Variant], tasks: list[Task]) -> list[tuple[str, str]]:
    """The (variant.id, task.id) cartesian product (enabled matrix when given enabled lists)."""
    return [(v.id, t.id) for v in variants for t in tasks]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_manifest.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/manifest.py tests/test_eval_manifest.py
git commit -m "Spec 05: eval.manifest load/default manifest (id + duplicate guards)"
```

---

### Task 7: `qquant.eval.hflm` (preloaded HFLM wrapper)

**Files:**
- Create: `src/qquant/eval/hflm.py`
- Test: `tests/test_eval_hflm.py`

**Interfaces:**
- Consumes: nothing at import (lazy `lm_eval`).
- Produces: `build_hflm(model, tokenizer, *, batch_size: int, max_length: int | None = None)` — lazily imports `lm_eval.models.huggingface.HFLM` and returns `HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)`. **Never** passes `device=`/calls `.to()` (Spec 04 already device-placed the model).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_hflm.py`:

```python
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from qquant.eval.hflm import build_hflm


def _install_fake_lm_eval(monkeypatch):
    calls = {}

    class FakeHFLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    pkg = ModuleType("lm_eval")
    models = ModuleType("lm_eval.models")
    hf = ModuleType("lm_eval.models.huggingface")
    hf.HFLM = FakeHFLM
    monkeypatch.setitem(sys.modules, "lm_eval", pkg)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf)
    return calls


def test_build_hflm_passes_preloaded_model_no_device(monkeypatch):
    calls = _install_fake_lm_eval(monkeypatch)
    model = SimpleNamespace(name="m")
    tok = SimpleNamespace(name="t")
    build_hflm(model, tok, batch_size=4, max_length=4096)
    assert calls["pretrained"] is model
    assert calls["tokenizer"] is tok
    assert calls["batch_size"] == 4
    assert calls["max_length"] == 4096
    assert "device" not in calls  # never relocate a quantized model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_hflm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.hflm'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/hflm.py`:

```python
"""Wrap an ALREADY-LOADED (model, tokenizer) in lm-eval's HFLM. Lazy lm_eval import.

Spec 04 owns dtype/device_map/greedy generation_config; this never reloads, casts, or moves
the model (no device=), so the profiled (Spec 06) and evaluated objects are identical.
"""

from __future__ import annotations

from typing import Any


def build_hflm(model: Any, tokenizer: Any, *, batch_size: int, max_length: int | None = None):
    """Return an lm_eval HFLM wrapping the preloaded model. Imports lm_eval lazily."""
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_hflm.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/hflm.py tests/test_eval_hflm.py
git commit -m "Spec 05: eval.hflm build_hflm (preloaded model, never relocates)"
```

---

### Task 8: `qquant.eval.runner` (EvalRunner + RunSummary)

**Files:**
- Create: `src/qquant/eval/runner.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `policy.simple_evaluate_kwargs` (T2), `results.cell_inputs_from_results`/`build_cell`/`write_cell` (T5), `hflm.build_hflm` (T7), `datasets.apply_dataset_overrides` (T4), `metrics.wilson_interval` (T3); `qquant.matrix.{expand_matrix,missing_cells}`, `qquant.config.{RunConfig,cell_provenance}`, `qquant.paths.Paths`, `qquant.registry.{Variant,Task,MMLU_SUBJECTS}`.
- Produces:
  - `@dataclass RunSummary(cells_written: int, cells_skipped: int, variants_loaded: int, failures: list[str])`
  - `class EvalRunner(results_root, run: RunConfig, load_model=None, evaluate_fn=None)` with `plan(manifest, variants, tasks) -> list[Cell]` (torch-free), `run_variant(variant, tasks) -> list[Cell]`, `run_manifest(manifest, variants, tasks) -> RunSummary`, and `run_smoke(variant, task, *, limit) -> dict` (eval at `limit`, build+validate cells **in memory**, never write; returns `{"cells": [{cell_id, metric_value, item_correct}, ...]}` — backs `--limit`/`--determinism-check`).
  - Helpers: `_run_for(variant)` (per-variant `RunConfig` via `replace(run, model_revision=variant.revision)`), `code_exec_allowed() -> bool` (both env vars `== "1"`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_runner.py`:

```python
from __future__ import annotations

import json

import pytest

from qquant.config import RunConfig, Seeds, cell_provenance
from qquant.eval.runner import EvalRunner, RunSummary
from qquant.matrix import expand_matrix
from qquant.paths import cell_path
from qquant.registry import MMLU_SUBJECTS, load_tasks, load_variants


def _run():
    return RunConfig(lm_eval_version="lmv", model_revision=None, max_length=4096, seeds=Seeds())


def _results_for(lm_eval_tasks, primary, value=0.5):
    """Fake lm-eval output covering the given subtasks with 2 samples each."""
    return {
        "results": {t: {f"{primary},none": value, f"{primary}_stderr,none": 0.01}
                    for t in lm_eval_tasks},
        "samples": {t: [{"doc_id": 0, primary: 1}, {"doc_id": 1, primary: 0}]
                    for t in lm_eval_tasks},
    }


def test_plan_returns_missing_cells_only(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    runner = EvalRunner(tmp_path, _run(), load_model=None, evaluate_fn=None)
    cells = runner.plan([("bf16", "gsm8k")], variants, tasks)
    assert [c.cell_id for c in cells] == ["bf16/gsm8k"]


def test_run_variant_skips_load_when_all_present(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    run = _run()
    # Pre-write a done gsm8k cell with matching provenance.
    cell = expand_matrix([variants["bf16"]], [tasks["gsm8k"]])[0]
    prov = cell_provenance(run.__class__(lm_eval_version="lmv", model_revision=None,
                                         max_length=4096, seeds=Seeds()),
                           tasks["gsm8k"], "bf16")
    doc = {"schema_version": 1, "cell_id": cell.cell_id, "variant": "bf16", "task": "gsm8k",
           "primary_metric": "exact_match", "metric_value": 0.5, "n_samples": 2, "config": prov,
           "meta": {"item_correct": [1, 0], "item_ids": [0, 1], "n_correct": 1}}
    p = cell.path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))

    calls = {"load": 0}

    def fake_load(vid):
        calls["load"] += 1
        raise AssertionError("must not load when nothing is missing")

    runner = EvalRunner(tmp_path, run, load_model=fake_load, evaluate_fn=None)
    written = runner.run_variant(variants["bf16"], [tasks["gsm8k"]])
    assert written == []
    assert calls["load"] == 0


def test_run_manifest_loads_once_and_writes_cells(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    load_calls = []

    class FakeLoaded:
        def __init__(self, vid):
            self.model = object()
            self.tokenizer = object()
            self.metadata = {"model_revision": variants[vid].revision}

        def unload(self):
            pass

    def fake_load(vid):
        load_calls.append(vid)
        return FakeLoaded(vid)

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "exact_match")

    runner = EvalRunner(tmp_path, _run(), load_model=fake_load, evaluate_fn=fake_eval)
    summary = runner.run_manifest([("bf16", "gsm8k")], variants, tasks)
    assert isinstance(summary, RunSummary)
    assert load_calls == ["bf16"]  # loaded exactly once
    cell_doc = json.loads(cell_path(tmp_path, "bf16", "gsm8k").read_text())
    assert cell_doc["metric_value"] == 0.5
    assert cell_doc["meta"]["item_correct"] == [1, 0]


def test_mmlu_writes_57_cells_with_item_vectors(tmp_path):
    variants = load_variants()
    tasks = load_tasks()

    class FakeLoaded:
        model = object()
        tokenizer = object()
        metadata = {"model_revision": variants["bf16"].revision}

        def unload(self):
            pass

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "acc")

    runner = EvalRunner(tmp_path, _run(), load_model=lambda vid: FakeLoaded(),
                        evaluate_fn=fake_eval)
    written = runner.run_variant(variants["bf16"], [tasks["mmlu"]])
    subjects = {c.subject for c in written}
    assert subjects == set(MMLU_SUBJECTS)
    assert len(written) == 57
    one = json.loads(cell_path(tmp_path, "bf16", "mmlu", "anatomy").read_text())
    assert one["task_lm_eval"] == "mmlu_anatomy"
    assert one["meta"]["item_correct"] == [1, 0]


def test_code_exec_skipped_unless_both_env_set(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    monkeypatch.delenv("QQUANT_ALLOW_CODE_EXEC", raising=False)

    def fake_eval(model=None, tasks=None, **kw):
        raise AssertionError("must not eval a gated code-exec task")

    runner = EvalRunner(tmp_path, _run(),
                        load_model=lambda vid: _FakeLoaded(variants, vid),
                        evaluate_fn=fake_eval)
    written = runner.run_variant(variants["bf16"], [tasks["humaneval"]])
    assert written == []  # left missing, not errored


def test_code_exec_runs_and_passes_confirm_when_gated(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    monkeypatch.setenv("QQUANT_ALLOW_CODE_EXEC", "1")
    seen = {}

    def fake_eval(model=None, tasks=None, **kw):
        seen.update(kw)
        return _results_for(tasks, "pass@1")

    runner = EvalRunner(tmp_path, _run(),
                        load_model=lambda vid: _FakeLoaded(variants, vid),
                        evaluate_fn=fake_eval)
    runner.run_variant(variants["bf16"], [tasks["humaneval"]])
    assert seen.get("confirm_run_unsafe_code") is True


class _FakeLoaded:
    def __init__(self, variants, vid):
        self.model = object()
        self.tokenizer = object()
        self.metadata = {"model_revision": variants[vid].revision}

    def unload(self):
        pass


def test_ifeval_seeds_langdetect(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    import types

    fake_ld = types.SimpleNamespace(DetectorFactory=types.SimpleNamespace(seed=None))
    monkeypatch.setitem(__import__("sys").modules, "langdetect", fake_ld)

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "prompt_level_strict_acc")

    runner = EvalRunner(tmp_path, _run(),
                        load_model=lambda vid: _FakeLoaded(variants, vid),
                        evaluate_fn=fake_eval)
    runner.run_variant(variants["bf16"], [tasks["ifeval"]])
    assert fake_ld.DetectorFactory.seed == 0


def test_run_smoke_builds_in_memory_without_writing(tmp_path):
    variants = load_variants()
    tasks = load_tasks()

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "exact_match")

    runner = EvalRunner(tmp_path, _run(),
                        load_model=lambda vid: _FakeLoaded(variants, vid),
                        evaluate_fn=fake_eval)
    out = runner.run_smoke(variants["bf16"], tasks["gsm8k"], limit=8)
    assert out["cells"][0]["metric_value"] == 0.5
    assert out["cells"][0]["item_correct"] == [1, 0]
    # nothing persisted to the results tree
    assert not (tmp_path / "bf16").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.runner'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/eval/runner.py`:

```python
"""Manifest-driven, resumable eval runner. lm_eval / qquant.models / datasets / langdetect
are imported lazily inside methods so importing this module stays torch-free.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

from qquant.config import RunConfig, cell_provenance
from qquant.eval.policy import simple_evaluate_kwargs
from qquant.eval.results import build_cell, cell_inputs_from_results, write_cell
from qquant.matrix import Cell, expand_matrix, missing_cells
from qquant.paths import Paths
from qquant.registry import Task, Variant

log = logging.getLogger("qquant.eval")


@dataclass
class RunSummary:
    cells_written: int
    cells_skipped: int
    variants_loaded: int
    failures: list[str]


def code_exec_allowed() -> bool:
    """HumanEval double-env gate: both switches must be exactly '1'."""
    return os.environ.get("HF_ALLOW_CODE_EVAL") == "1" and \
        os.environ.get("QQUANT_ALLOW_CODE_EXEC") == "1"


class EvalRunner:
    def __init__(self, results_root, run: RunConfig, load_model=None, evaluate_fn=None):
        self.results_root = Path(results_root)
        self.run = run
        self._load_model = load_model
        self._evaluate_fn = evaluate_fn

    # --- lazy defaults -------------------------------------------------------------
    def _load(self):
        if self._load_model is not None:
            return self._load_model
        from qquant.models import load_variant
        return load_variant

    def _evaluate(self):
        if self._evaluate_fn is not None:
            return self._evaluate_fn
        import lm_eval
        return lm_eval.simple_evaluate

    def _run_for(self, variant: Variant) -> RunConfig:
        """Per-variant provenance: model_revision = the pinned SHA (None for self-quant)."""
        return replace(self.run, model_revision=variant.revision)

    # --- planning (torch-free) -----------------------------------------------------
    def plan(self, manifest, variants: dict[str, Variant], tasks: dict[str, Task]) -> list[Cell]:
        out: list[Cell] = []
        for vid, tid in manifest:
            variant, task = variants[vid], tasks[tid]
            active = cell_provenance(self._run_for(variant), task, vid)
            expanded = expand_matrix([variant], [task])
            out.extend(missing_cells(expanded, self.results_root, active_config=active))
        return out

    # --- execution -----------------------------------------------------------------
    def run_variant(self, variant: Variant, tasks: list[Task]) -> list[Cell]:
        run = self._run_for(variant)
        per_task_missing: dict[str, list[Cell]] = {}
        for task in tasks:
            active = cell_provenance(run, task, variant.id)
            expanded = expand_matrix([variant], [task])
            per_task_missing[task.id] = missing_cells(
                expanded, self.results_root, active_config=active
            )
        if not any(per_task_missing.values()):
            return []  # never load a fully-present variant

        loaded = self._load()(variant.id)
        written: list[Cell] = []
        try:
            for task in tasks:
                miss = per_task_missing[task.id]
                if not miss:
                    continue
                if task.code_exec and not code_exec_allowed():
                    log.warning("skipping %s/%s: code-exec env gate not set (both "
                                "HF_ALLOW_CODE_EVAL=1 and QQUANT_ALLOW_CODE_EXEC=1)",
                                variant.id, task.id)
                    continue
                if task.id == "ifeval":
                    import langdetect
                    langdetect.DetectorFactory.seed = 0
                if task.id == "gsm8k":
                    from qquant.eval.datasets import apply_dataset_overrides
                    apply_dataset_overrides()
                written.extend(self._eval_task(variant, task, miss, run, loaded))
        finally:
            loaded.unload()
        return written

    def _eval_task(self, variant, task, miss, run, loaded) -> list[Cell]:
        from qquant.eval.hflm import build_hflm

        hflm = build_hflm(loaded.model, loaded.tokenizer,
                          batch_size=task.batch_size_for(variant.id),
                          max_length=run.max_length)
        lm_eval_tasks = [c.lm_eval_task for c in miss]
        kwargs = simple_evaluate_kwargs(task, variant.id, run, lm_eval_tasks, limit=None)
        results = self._evaluate()(model=hflm, **kwargs)

        written: list[Cell] = []
        for cell in miss:
            inp = cell_inputs_from_results(results, cell.lm_eval_task, task)
            meta_extra = self._meta_extra(variant, task, inp)
            doc = build_cell(cell=cell, task=task, variant=variant, run=run,
                             metric_value=inp["metric_value"],
                             metric_stderr=inp["metric_stderr"],
                             n_samples=inp["n_samples"],
                             extra_metrics=inp["extra_metrics"],
                             item_correct=inp["item_correct"],
                             item_ids=inp["item_ids"], meta_extra=meta_extra)
            write_cell(doc, cell.path(self.results_root))
            written.append(cell)
        if task.id == "mmlu":
            self._write_mmlu_group(variant, results)
        return written

    def _meta_extra(self, variant, task, inp) -> dict:
        meta = {"lm_eval_version": self.run.lm_eval_version,
                "model_revision": variant.revision,
                "quant_method": variant.quant_method, "source": variant.source}
        if task.primary_metric == "pass@1":
            from qquant.eval.metrics import wilson_interval
            k, n = sum(inp["item_correct"]), inp["n_samples"]
            meta["wilson_ci"] = list(wilson_interval(k, n)) if n else None
            if variant.id == "bf16" and inp["metric_value"] <= 0.5:
                meta["sanity_failed"] = True
                log.warning("bf16 %s pass@1=%.3f <= 0.5 (sanity gate)",
                            task.id, inp["metric_value"])
        return meta

    def _write_mmlu_group(self, variant, results) -> None:
        import json

        group = results.get("results", {}).get("mmlu", {})
        if not group:
            return
        meta_dir = Paths.from_root(self.results_root).meta_dir
        meta_dir.mkdir(parents=True, exist_ok=True)
        payload = {"variant": variant.id,
                   "acc": group.get("acc,none", group.get("acc")),
                   "acc_stderr": group.get("acc_stderr,none", group.get("acc_stderr")),
                   "n_subjects": 57, "lm_eval_version": self.run.lm_eval_version}
        (meta_dir / f"mmlu_group_{variant.id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True))

    def run_manifest(self, manifest, variants, tasks) -> RunSummary:
        written = skipped = loaded = 0
        failures: list[str] = []
        by_variant: dict[str, list[Task]] = {}
        for vid, tid in manifest:
            by_variant.setdefault(vid, []).append(tasks[tid])
        for vid, task_list in by_variant.items():
            try:
                before = self.plan([(vid, t.id) for t in task_list], variants, tasks)
                if not before:
                    continue
                loaded += 1
                cells = self.run_variant(variants[vid], task_list)
                written += len(cells)
                skipped += max(0, len(before) - len(cells))
            except Exception as exc:  # noqa: BLE001 — record + continue, keep run resumable
                failures.append(f"{vid}: {exc!r}")
                log.exception("variant %s failed", vid)
        return RunSummary(written, skipped, loaded, failures)

    def run_smoke(self, variant: Variant, task: Task, *, limit: int) -> dict:
        """Eval (variant, task) at `limit`, build+validate cells IN MEMORY (never written).

        Backs --limit (smoke) and --determinism-check (run twice, compare). Loads once.
        """
        run = self._run_for(variant)
        if task.code_exec and not code_exec_allowed():
            raise RuntimeError(f"{task.id}: code-exec env gate not set")
        if task.id == "ifeval":
            import langdetect
            langdetect.DetectorFactory.seed = 0
        if task.id == "gsm8k":
            from qquant.eval.datasets import apply_dataset_overrides
            apply_dataset_overrides()
        cells = expand_matrix([variant], [task])
        loaded = self._load()(variant.id)
        try:
            from qquant.eval.hflm import build_hflm

            hflm = build_hflm(loaded.model, loaded.tokenizer,
                              batch_size=task.batch_size_for(variant.id),
                              max_length=run.max_length)
            lm_eval_tasks = [c.lm_eval_task for c in cells]
            kwargs = simple_evaluate_kwargs(task, variant.id, run, lm_eval_tasks, limit=limit)
            results = self._evaluate()(model=hflm, **kwargs)
            out = []
            for cell in cells:
                inp = cell_inputs_from_results(results, cell.lm_eval_task, task)
                build_cell(cell=cell, task=task, variant=variant, run=run,  # validates only
                           metric_value=inp["metric_value"], metric_stderr=inp["metric_stderr"],
                           n_samples=inp["n_samples"], extra_metrics=inp["extra_metrics"],
                           item_correct=inp["item_correct"], item_ids=inp["item_ids"],
                           meta_extra=self._meta_extra(variant, task, inp))
                out.append({"cell_id": cell.cell_id, "metric_value": inp["metric_value"],
                            "item_correct": inp["item_correct"]})
            return {"cells": out}
        finally:
            loaded.unload()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_runner.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/runner.py tests/test_eval_runner.py
git commit -m "Spec 05: EvalRunner (load-once, skip-when-done, 57 MMLU cells, code-exec gate)"
```

---

### Task 9: `cli` + public surface + entry point + GPU smoke + final verification

**Files:**
- Create: `src/qquant/eval/cli.py`
- Modify: `src/qquant/eval/__init__.py` (full re-exports)
- Modify: `pyproject.toml` (add `qquant-eval` entry point)
- Create: `tests/test_eval_cli.py`
- Create: `tests/test_eval_gpu.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `main(argv=None) -> int` (exit codes 0/1/2); `qquant.eval.__all__ = ["EvalRunner", "RunSummary", "build_cell"]`; `qquant-eval` console script.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys


def test_importing_qquant_eval_is_torch_and_lmeval_free():
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant.eval','qquant.eval.policy','qquant.eval.metrics','qquant.eval.results',"
        "'qquant.eval.manifest','qquant.eval.datasets','qquant.aggregate.stats')];"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.') "
        "or m=='lm_eval' or m.startswith('lm_eval.')];"
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_plan_is_torch_free_and_lists_cells(tmp_path, capsys):
    from qquant.eval.cli import main

    rc = main(["--results", str(tmp_path), "--variant", "bf16", "--task", "gsm8k", "--plan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bf16/gsm8k" in out


def test_help_exits_zero():
    from qquant.eval.cli import main

    import pytest as _pytest
    with _pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
```

Create `tests/test_eval_gpu.py`:

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from qquant.eval.cli import main  # noqa: E402

pytestmark = pytest.mark.gpu


def test_gsm8k_smoke_and_determinism(tmp_path):
    rc = main(["--results", str(tmp_path), "--variant", "bf16", "--task", "gsm8k",
               "--limit", "8", "--determinism-check", "bf16", "gsm8k"])
    assert rc == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.eval.cli'`.

- [ ] **Step 3: Implement CLI + finalize package + entry point**

Replace `src/qquant/eval/__init__.py` with:

```python
"""qquant.eval — GPU-touching quality-eval engine (Spec 05).

Importing this package is torch-free (heavy imports are deferred inside functions/methods);
the umbrella ``qquant`` CLI must never import it.
"""

from __future__ import annotations

from qquant.eval.results import build_cell
from qquant.eval.runner import EvalRunner, RunSummary

__all__ = ["EvalRunner", "RunSummary", "build_cell"]
```

Create `src/qquant/eval/cli.py`:

```python
"""qquant-eval entry point. Lazy: only --plan/--list-missing run torch-free."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

from qquant.config import RunConfig, Seeds
from qquant.eval.manifest import default_manifest, load_manifest
from qquant.eval.runner import EvalRunner
from qquant.registry import enabled_variants, load_tasks, load_variants


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qquant-eval")
    p.add_argument("--results", default="results")
    p.add_argument("--manifest")
    p.add_argument("--variant", action="append", default=[], dest="variants")
    p.add_argument("--task", action="append", default=[], dest="tasks")
    p.add_argument("--variants-yaml")
    p.add_argument("--tasks-yaml")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed-random", type=int, default=0)
    p.add_argument("--seed-numpy", type=int, default=1234)
    p.add_argument("--seed-torch", type=int, default=1234)
    p.add_argument("--seed-fewshot", type=int, default=1234)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--list-missing", action="store_true")
    p.add_argument("--determinism-check", nargs=2, metavar=("VARIANT", "TASK"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_lm_eval_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _manifest_from_args(args, variants, tasks) -> list[tuple[str, str]]:
    if args.manifest:
        return load_manifest(args.manifest)
    if args.variants or args.tasks:
        vids = args.variants or [v.id for v in enabled_variants(variants)]
        tids = args.tasks or list(tasks)
        return [(v, t) for v in vids for t in tids]
    return default_manifest(enabled_variants(variants), list(tasks.values()))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    variants = load_variants(args.variants_yaml)
    tasks = load_tasks(args.tasks_yaml)
    run = RunConfig(
        lm_eval_version=_resolve_lm_eval_version(),
        model_revision=None,
        max_length=args.max_length,
        seeds=Seeds(random=args.seed_random, numpy=args.seed_numpy,
                    torch=args.seed_torch, fewshot=args.seed_fewshot),
    )
    try:
        manifest = _manifest_from_args(args, variants, tasks)
    except (ValueError, KeyError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    runner = EvalRunner(args.results, run)

    if args.plan or args.list_missing:
        for cell in runner.plan(manifest, variants, tasks):
            print(cell.cell_id)
        return 0

    if args.determinism_check:
        vid, tid = args.determinism_check
        limit = args.limit if args.limit is not None else 8
        first = runner.run_smoke(variants[vid], tasks[tid], limit=limit)
        second = runner.run_smoke(variants[vid], tasks[tid], limit=limit)
        ok = first == second
        print("determinism: OK" if ok else "determinism: MISMATCH")
        return 0 if ok else 1

    if args.limit is not None:  # smoke: build + validate in memory, never persist
        for vid, tid in manifest:
            runner.run_smoke(variants[vid], tasks[tid], limit=args.limit)
        return 0

    summary = runner.run_manifest(manifest, variants, tasks)
    if args.verbose:
        print(f"written={summary.cells_written} skipped={summary.cells_skipped} "
              f"loaded={summary.variants_loaded}")
    for f in summary.failures:
        print(f"FAILED {f}", file=sys.stderr)
    return 1 if summary.failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

In `pyproject.toml`, under `[project.scripts]`, add the line after the `qquant-spike` entry:

```toml
qquant-eval = "qquant.eval.cli:main"  # Spec 05
```

- [ ] **Step 4: Run the full suite + lint + entry point**

Run: `uv run pytest -v`
Expected: all CPU tests PASS (existing + the new Task 1–9 tests); `tests/test_eval_gpu.py` SKIPPED (no torch/CUDA on the laptop); `test_importing_qquant_eval_is_torch_and_lmeval_free` and the existing core torch-free guard PASS.

Run: `uv sync && uv run qquant-eval --help`
Expected: prints the usage and exits 0.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: "All checks passed!" + "N files already formatted".

Run: `uv run python -c "import qquant.eval; print(sorted(qquant.eval.__all__))"`
Expected: `['EvalRunner', 'RunSummary', 'build_cell']` with no torch/lm_eval import error.

- [ ] **Step 5: Commit**

```bash
git add src/qquant/eval/cli.py src/qquant/eval/__init__.py pyproject.toml tests/test_eval_cli.py tests/test_eval_gpu.py
git commit -m "Spec 05: qquant-eval CLI + public surface + entry point + GPU smoke"
```

---

## GPU verification (on the box)

The CPU plan above is complete and shippable on the laptop. The real lm-eval `HFLM` path — including the **preloaded-model** `HFLM(pretrained=<module>)` construction, which the GO spike did **not** exercise (it ran `pretrained=<repo-id-string>`) — verifies on a rented RTX 4090:

```bash
# on the box, after `uv sync --group gpu` (or the bootstrap-installed lm-eval ref)
uv run pytest tests/test_eval_gpu.py -v          # CUDA present => runs instead of skip
uv run qquant-eval --variant bf16 --task gsm8k --limit 8 --determinism-check bf16 gsm8k
```

This satisfies done-when #12 (gsm8k smoke + determinism) and #15 (MMLU group aggregate file). If the preloaded-HFLM construction or the per-doc sample shape (`cell_inputs_from_results`) disagrees with the resolved lm-eval ref, that surfaces here — adjust `build_hflm` / `_doc_correct` then. Self-quant cells (`gptq-selfquant`, `awq-selfquant`) verify after Spec 07 produces their checkpoints.

---

## Self-Review

**1. Spec coverage** (done-when → task):
- DW1 entry point + `--help` → T9. ✅
- DW2 torch-free imports (`aggregate.stats`, `eval.{metrics,results,manifest,policy,datasets}`) → T9 subprocess test (`test_eval_cli.py`); each module is import-clean by construction. ✅
- DW3 `wilson_interval` values + k/n + `n_correct` → T1 (`test_aggregate_stats`), T3 (re-export), T5 (`n_correct` in build_cell). ✅
- DW4 `extract_metric` order + KeyError; `extract_stderr` None → T3. ✅
- DW5 `build_cell` validates, config==provenance, atomic write, `is_cell_done` round-trip → T5. ✅
- DW6 `plan` == `missing_cells`, no load when present → T8. ✅
- DW7 load once per variant with missing cells → T8 (`test_run_manifest_loads_once...`). ✅
- DW8 57 MMLU cells, subject set, `task_lm_eval`, rerun re-evals only missing → T8 (`test_mmlu_writes_57_cells...`); per-subject missing handled by `missing_cells` + `lm_eval_tasks=[c.lm_eval_task for c in miss]`. ✅
- DW9 `simple_evaluate_kwargs` forwards registry fields; `greedy_gen_kwargs` has `do_sample=False` → T2. ✅
- DW10 code-exec gate skip/confirm → T8 (`test_code_exec_*`). ✅
- DW11 ifeval sets `langdetect…seed == 0` → T8 (`test_ifeval_seeds_langdetect`). ✅
- DW12 GPU smoke + determinism (in-memory, no write) → T8 `run_smoke` + `test_run_smoke_builds_in_memory_without_writing`; T9 wires `--determinism-check` (runs `run_smoke` twice, compares) + `test_eval_gpu.py` (GPU). ✅
- DW13 `n_samples == len(item_correct) == len(item_ids)` for every cell; `--limit` refuses to write → T5 (`build_cell` length guard), T8 `run_smoke` (builds/validates without writing — asserted by `test_run_smoke...` checking no `bf16/` dir is created), T9 (`--limit` routes to `run_smoke`). ✅
- DW14 ruff + pytest green → T9 Step 4. ✅
- DW15 MMLU group aggregate file via `Paths.meta_dir` → T8 (`_write_mmlu_group`). ✅
- DW16 `apply_dataset_overrides` gsm8k→openai/gsm8k → T4. ✅
- DW17 `wilson_interval` single home in `aggregate.stats`, torch-free → T1 + T9 subprocess. ✅

**2. Placeholder scan:** No "TBD/TODO/handle edge cases/similar to Task N". Every code step shows full code; every run step shows the command + expected output. ✅

**3. Type consistency:** `cell_inputs_from_results` return dict keys (`metric_value/metric_stderr/n_samples/item_correct/item_ids/extra_metrics`) match `build_cell`'s params and the runner's `_eval_task`/`run_smoke` call sites. `RunSummary` fields and `EvalRunner` method signatures (`plan`/`run_variant`/`run_manifest`/`run_smoke`) match the runner test's usage and the CLI call sites (`run_smoke(variant, task, *, limit)`; `--determinism-check` → `args.determinism_check` 2-tuple). `simple_evaluate_kwargs(task, variant_id, run, lm_eval_tasks, *, limit)` matches both `_eval_task` and `run_smoke`. `cell_provenance(run, task, variant_id)` 8-key shape matches `build_cell`'s `config` and `is_cell_done`. `build_hflm(model, tokenizer, *, batch_size, max_length)` matches both runner call sites. ✅
