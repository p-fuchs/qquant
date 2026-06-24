# Spec 05 — Quality eval: preloaded HFLM + resumable cells + `qquant-eval`

- **Status:** Draft (ready to implement once Spec 04 lands)
- **Depends on:** 04 (variant loaders). Transitively imports the Spec-01 core
  (`qquant.registry`, `qquant.matrix`, `qquant.paths`, `qquant.config`, `qquant.schemas`).
- **Owns / Produces:** the `qquant.eval` package; the `qquant-eval` entry point (adds its own
  `[project.scripts]` line); the **per-cell result files** written via `cell_path` + the v1 cell
  schema. Produces `<results_root>/<variant>/<task>/<subject>.json` (MMLU) and
  `.../result.json` (gsm8k/humaneval/ifeval) for every requested `(variant, task)`.

## Purpose

Run the quality benchmarks (mmlu, gsm8k, humaneval, ifeval) against each model variant on the
single RTX 4090, and write one schema-valid, fully-provenanced **result cell** per
`(variant, task[, subject])`. The unit of work and the resume predicate are already defined by
the Spec-01 core (`Cell`, `expand_matrix`, `missing_cells`, `is_cell_done`, `cell_path`,
`cell_provenance`); this spec is the GPU-touching layer that:

1. wraps an **already-loaded** `(model, tokenizer)` (from Spec 04) into an `lm-eval` `HFLM`,
2. drives `lm_eval.simple_evaluate` with the **per-task** chat-template / few-shot / generation
   policy taken verbatim from the `Task` registry,
3. extracts the primary metric (absorbing key-suffix drift) plus the per-item correctness vector,
4. writes/validates each cell with `cell_provenance` as its `config` block, and
5. is a **manifest-driven runner** that loads each model **once** and computes only the
   `(variant, task)` cells it is told to, **skipping** anything `is_cell_done` already accepts.

Cost-correctness is the headline constraint: a variant whose cells are all present must never be
loaded, and MMLU must resume at **per-subject** granularity so a crash never reruns 57 subjects.

## Scope (in / out)

**In scope**

- `qquant.eval` package: HFLM wrapper, task→`simple_evaluate` policy translation, metric/stderr
  extraction, eval-time Wilson CI for pass@1, cell construction + atomic write, the
  manifest-driven runner, the `qquant-eval` CLI.
- Per-task chat-template policy (from the registry, not hardcoded), greedy `gen_kwargs`,
  per-(variant,task) batch sizing, HumanEval double-env gating, IFEval `langdetect` seeding.
- Persisting `meta.item_correct` and `meta.item_ids` (the per-item correctness vector + lm-eval
  `doc_id`s) **inside the cell JSON's `meta` block of EVERY cell** — the 57 per-subject MMLU
  cells (loglikelihood `acc` is 0/1 per doc) **and** the generative cells (gsm8k, humaneval,
  ifeval) — the paired input the downstream **paired McNemar** reads **directly from the cell**
  (no `.samples.jsonl` sidecar, no new path contract). *Why all cells:* the decision-log makes
  McNemar gate **all** quality claims and the overview's RQ2 calls for a per-MMLU-subject paired
  significance test, so MMLU cells must carry their item vectors too.
- Persisting lm-eval's `mmlu` group aggregate (`{acc, acc_stderr}`) to
  `results/_meta/mmlu_group_<variant>.json` for Spec 09's MMLU reconciliation.
- A determinism smoke (run a task twice, assert identical output) and a bf16-HumanEval sanity gate.

**Out of scope (owned elsewhere)**

- Model loading / dtype / `generation_config` greedy-clearing → **Spec 04** (consumed here).
- Efficiency/latency/throughput/VRAM profiling and `qquant-profile` → **Spec 06**.
- Self-quant checkpoint production (`local:checkpoints/...`) → **Spec 07** (this spec only *loads*
  them, via Spec 04's `compressed-tensors` path).
- Orchestration, `_meta/env.json`, `_meta/done_manifest.json`, budget guard, exfil/auto-destroy,
  `PYTORCH_CUDA_ALLOC_CONF` export, nltk/dataset pre-download → **Spec 03 / 08 / 11**.
- Cross-variant aggregation: MMLU weighted mean + error-propagated stderr, paired **McNemar**,
  report tables/plots → **Spec 09 / 10**. (This spec only *emits* the per-cell inputs they need.)

## Interface

### Entry point (`[project.scripts]`, this spec adds it)

```toml
# pyproject.toml  [project.scripts]
qquant-eval = "qquant.eval.cli:main"
```

This is a **separate** GPU-touching entry point, never a subcommand of the torch-free `qquant`
umbrella (per `contracts.md §7`). `qquant.eval` and its submodules import `torch`/`lm_eval`
**lazily** (inside functions), so `import qquant` and the torch-free core stay clean.

### CLI — `qquant-eval`

```
qquant-eval [--results DIR]
            [--manifest PATH | (--variant ID … --task ID …)]    # default: full ENABLED matrix
            [--variants-yaml PATH] [--tasks-yaml PATH]
            [--max-length N] [--seed-random N --seed-numpy N --seed-torch N --seed-fewshot N]
            [--limit N]                # smoke only: caps docs/task, REFUSES to write cells
            [--plan | --list-missing]  # print the cells that WOULD run, then exit (no GPU)
            [--determinism-check VARIANT TASK]   # run twice @ small limit, assert identical
            [-v]
```

- Resolves `lm_eval_version` from `importlib.metadata.version("lm_eval")`; builds a `RunConfig`
  from the seed/`max-length` flags; loads `variants.yaml`/`tasks.yaml` via `load_variants` /
  `load_tasks`. `--plan`/`--list-missing` are torch-free (CI-runnable on macOS).
- Exit code `0` if every requested cell ends up `is_cell_done`, `1` on any failure, `2` on a
  configuration/manifest error.

### Manifest format (`qquant.eval.manifest`)

JSON (or YAML), task-level granularity — MMLU still resumes per subject internally:

```json
{ "cells": [ {"variant": "bf16", "task": "mmlu"},
             {"variant": "bf16", "task": "gsm8k"},
             {"variant": "gptq-official", "task": "humaneval"} ] }
```

```python
def load_manifest(path: str | Path) -> list[tuple[str, str]]: ...   # validates ids ∈ VARIANT_IDS/TASK_IDS
def default_manifest(variants: list[Variant], tasks: list[Task]) -> list[tuple[str, str]]: ...
```

Unknown/duplicate ids → `ValueError`. The orchestrator (Spec 08) generates the manifest.

### `qquant.eval.hflm`

```python
def build_hflm(model, tokenizer, *, batch_size: int, max_length: int | None):
    """Wrap an ALREADY-LOADED HF model in lm_eval HFLM (pretrained=<module>, tokenizer=...).
    Imports lm_eval lazily. Does NOT load/move/cast the model — Spec 04 already did, including
    greedy generation_config. Returns an lm_eval.api.model.LM."""
```

### `qquant.eval.policy` (per-task → `simple_evaluate` kwargs; the SSOT-faithful policy)

```python
def simple_evaluate_kwargs(task: Task, variant_id: str, run: RunConfig,
                           lm_eval_tasks: list[str], *, limit: int | None) -> dict:
    """All values come from the Task registry — nothing is hardcoded:
        tasks                = lm_eval_tasks                      # e.g. ["mmlu_anatomy", ...] or ["gsm8k"]
        num_fewshot          = task.num_fewshot
        apply_chat_template  = task.apply_chat_template           # per-task policy
        fewshot_as_multiturn = task.fewshot_as_multiturn          # per-task policy
        gen_kwargs           = greedy_gen_kwargs(task)            # see below
        batch_size           = task.batch_size_for(variant_id)
        random_seed/numpy_random_seed/torch_random_seed/fewshot_random_seed = run.seeds.*
        log_samples          = True                               # needed for n & item vectors
        confirm_run_unsafe_code = task.code_exec                  # only set when gated on
        limit                = limit
    """

def greedy_gen_kwargs(task: Task) -> str:
    """Greedy determinism for generate_until tasks (ignored by loglikelihood MMLU):
       'do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=<task.max_gen_toks>'
       (top_k omitted / cleared). Mirrors the load-time generation_config from Spec 04."""
```

The **per-task chat-template policy** is therefore *data*, not branching: mmlu/gsm8k/ifeval carry
`apply_chat_template=True`; mmlu/gsm8k additionally `fewshot_as_multiturn=True`; humaneval uses the
`humaneval_instruct` lm-eval task (fenced-code extraction) at `num_fewshot=0`. The runner copies
these straight from the registry — see *Implementation notes*.

### `qquant.eval.metrics`

```python
def extract_metric(results_for_task: dict, metric_keys: Sequence[str]) -> tuple[str, float]:
    """Try metric_keys in order against the lm-eval results dict for one (sub)task; return the
    first hit (key, float). Raise KeyError listing the available keys if none match."""

def extract_stderr(results_for_task: dict, primary_metric: str) -> float | None:
    """Best-effort '<metric>_stderr,*' lookup; None if absent (e.g. pass@1)."""

# wilson_interval is NOT reimplemented here. The SINGLE implementation lives in
# qquant.aggregate.stats — Spec 05 CREATES that module now (just wilson_interval; pure-math,
# torch-free) as the forward-declared SSOT home; Spec 09 later EXTENDS the same module (MMLU
# weighting, paired McNemar, …) without redefining wilson_interval. qquant.eval.metrics
# imports/re-exports it for the eval-time HumanEval pass@1 CI (where lm-eval emits no stderr):
from qquant.aggregate.stats import wilson_interval   # (k, n, confidence=0.95) -> (lo, hi)
```

### `qquant.eval.results`

```python
def build_cell(*, cell: Cell, task: Task, variant: Variant, run: RunConfig,
               metric_value: float, metric_stderr: float | None, n_samples: int,
               extra_metrics: dict, item_correct: list[int], item_ids: list, meta_extra: dict) -> dict:
    """Assemble a v1 cell dict. config = cell_provenance(run, task, variant.id) (keys ==
    PROVENANCE_KEYS). meta carries lm-eval task name, model_id/revision/source/quant_method,
    dataset fingerprint, gen_kwargs, gpu name, lib versions, timestamp, the per-item correctness
    vector (item_correct / item_ids) and n_correct. Validated before return."""

def write_cell(cell: dict, path: Path) -> None:
    """validate_cell(cell); atomic write (tmp sibling + os.replace); mkdir parents."""
```

### `qquant.eval.runner`

```python
@dataclass
class RunSummary:
    cells_written: int; cells_skipped: int; variants_loaded: int; failures: list[str]

class EvalRunner:
    def __init__(self, results_root, run: RunConfig,
                 load_model=None,        # default: qquant.models.load_variant (Spec 04), injectable
                 evaluate_fn=None): ...  # default: lm_eval.simple_evaluate, injectable for tests

    def plan(self, manifest, variants, tasks) -> list[Cell]:
        """expand_matrix over the manifest's (variant,task) pairs, then keep only
        missing_cells(..., active_config=cell_provenance(run, task, variant)). Torch-free."""

    def run_variant(self, variant: Variant, tasks: list[Task]) -> list[Cell]:
        """If plan() leaves NO missing cell for this variant -> return [] WITHOUT calling
        load_model. Otherwise load_model(variant) ONCE; per task: build HFLM at the task's batch
        size; for MMLU run a single simple_evaluate over only the MISSING mmlu_<subject> subtasks
        and split into 57 per-subject cells; for single-cell tasks run once; write each cell.
        Free the model (del + torch.cuda.empty_cache) before returning."""

    def run_manifest(self, manifest, variants, tasks) -> RunSummary: ...
```

### File formats produced

- Cells: `cell_path(results_root, variant, task, subject)` — `<subject>.json` (MMLU, subject =
  bare slug e.g. `anatomy`) / `result.json` (others), conforming to `cell_result.schema.json` v1.
- `meta.item_correct` (list of `0/1`) + `meta.item_ids` (lm-eval `doc_id`s) persisted **inside
  every cell's JSON, including each MMLU per-subject cell** — the paired input Spec 09 reads
  **directly from the cell** for McNemar (no `.samples.jsonl` sidecar). `meta.n_correct` (k) for
  HumanEval Wilson CI.

## Files to create

```
src/qquant/aggregate/__init__.py   # NEW package: forward-declared SSOT home for Spec 09; torch-free
src/qquant/aggregate/stats.py      # wilson_interval(k, n, confidence=0.95) -> (lo, hi)  (pure-math, torch-free; Spec 09 EXTENDS this module, never redefines wilson)
src/qquant/eval/__init__.py        # lazy-import package; re-exports EvalRunner, RunSummary, build_cell
src/qquant/eval/hflm.py            # build_hflm  (lazy import lm_eval)
src/qquant/eval/policy.py          # simple_evaluate_kwargs, greedy_gen_kwargs  (torch-free)
src/qquant/eval/metrics.py         # extract_metric, extract_stderr; re-exports wilson_interval from qquant.aggregate.stats  (torch-free)
src/qquant/eval/datasets.py        # apply_dataset_overrides(): rewrite bare gsm8k -> openai/gsm8k (+ drop stale revision) at runtime; lazy import datasets
src/qquant/eval/results.py         # build_cell, write_cell  (torch-free; uses validate_cell + cell_path)
src/qquant/eval/manifest.py        # load_manifest, default_manifest  (torch-free)
src/qquant/eval/runner.py          # EvalRunner, RunSummary  (lazy import lm_eval + Spec-04 loader; calls apply_dataset_overrides before eval)
src/qquant/eval/cli.py             # main()  -> qquant-eval entry point (lazy)
tests/test_aggregate_stats.py      # wilson_interval reference values; k=0 and k=n edge cases  (CPU)
tests/test_eval_metrics.py         # metric_keys order / suffix drift; stderr; wilson re-export reachable  (CPU)
tests/test_eval_datasets.py        # apply_dataset_overrides maps gsm8k -> openai/gsm8k (id + revision rewrite), via a fake datasets loader  (CPU)
tests/test_eval_results.py         # build_cell validates; config==cell_provenance; atomic write; is_cell_done round-trip  (CPU)
tests/test_eval_manifest.py        # parse/validate; id guards; default_manifest == enabled matrix  (CPU)
tests/test_eval_runner.py          # fake load_model + fake evaluate_fn: load-once, skip-when-done, 57 MMLU cells, policy passthrough, code-exec gating  (CPU)
```

Plus the one-line `[project.scripts]` addition to `pyproject.toml`. No change to `contracts.md`,
the schema, the registries, or other specs.

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **Preloaded HFLM (Spec 04 boundary).** Spec 04 owns dtype, `device_map`, NF4
  `bnb_4bit_compute_dtype=bfloat16`, the gptqmodel/optimum and `compressed-tensors` load paths, and
  the greedy `generation_config` clear. `build_hflm` passes `pretrained=<module>` so we never
  reload or re-cast — *why*: a second load would blow the 24 GB budget and could disagree with the
  profiled model in Spec 06. `load_model` is injected (default = Spec 04's `load_variant`) so the
  runner is unit-testable on macOS with a fake.
- **Per-task chat-template policy, from the registry (NOT blanket).** mmlu/gsm8k/ifeval →
  `apply_chat_template=True`; mmlu/gsm8k also `fewshot_as_multiturn=True`; humaneval →
  `humaneval_instruct` (fenced-code extraction) at 0-shot. These are *registry fields*
  (`Task.apply_chat_template`, `.fewshot_as_multiturn`, `.num_fewshot`, `.lm_eval_task`,
  `.max_gen_toks`), so the runner copies them through — *why*: keeps the policy in one SSOT and
  lets `cell_provenance` capture it so a policy change recomputes the affected cells.
- **Greedy determinism.** `gen_kwargs` forces `do_sample=False`, `temperature=0.0`, `top_p=1.0`
  (top_k cleared) for the generate_until tasks; the four lm-eval seeds come from `RunConfig.seeds`.
  *Why*: Qwen ships `do_sample=true`; without both the load-time clear (Spec 04) **and** the
  gen_kwargs clear, the run is nondeterministic and McNemar pairing breaks. `--determinism-check`
  runs a task twice at a small limit and asserts identical metric **and** item vector.
- **MMLU = 57 resumable per-subject cells.** `run_variant` plans missing subjects via
  `missing_cells` (under the active `cell_provenance`), runs ONE `simple_evaluate` over only the
  missing `mmlu_<subject>` subtasks, then writes one cell per subject (subject = bare slug;
  `task_lm_eval = mmlu_<subject>`; per-subject `acc`, `acc_stderr`, n, **and that subject's
  per-item `meta.item_correct`/`meta.item_ids` from `log_samples`** — so paired McNemar on MMLU
  is possible, per the decision-log). *Why*: a crash resumes at subject granularity, not all 57;
  weighted aggregation + a single error-propagated stderr is Spec 09's job and needs per-subject
  n + stderr, and the item vectors feed the per-MMLU-subject significance test. A test asserts
  the emitted subjects equal `set(MMLU_SUBJECTS)`.
- **Persist the lm-eval `mmlu` group number (for Spec 09's reconcile).** A MMLU run also yields
  lm-eval's `mmlu` **group** aggregate (`{acc, acc_stderr}`); this spec writes it to
  `results/_meta/mmlu_group_<variant>.json` (located via `Paths.meta_dir`), recording
  `{variant, acc, acc_stderr, n_subjects, lm_eval_version}`. *Why*: Spec 09's `reconcile_mmlu`
  reads this number to deterministically reconcile its n-weighted aggregate against lm-eval's own
  group figure — so reconciliation is never "skipped with a logged note" for lack of the datum.
- **Per-(variant,task) batch + logits transient.** `task.batch_size_for(variant_id)` gives MMLU
  bf16/bnb-int8 → 2, 4-bit → 4; `max_length` (from `RunConfig`) bounds the
  `batch×seq×vocab(152064)×2B` logits transient. *Why*: the transient (~3.5 GB) is what OOMs, not
  the weights. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the orchestrator
  (Spec 08); the runner only reads `max_length`/batch.
- **HumanEval double-env gate.** A `code_exec` task runs only when **both** `HF_ALLOW_CODE_EVAL=1`
  and `QQUANT_ALLOW_CODE_EXEC=1`; then `simple_evaluate(confirm_run_unsafe_code=True)`. If either is
  unset the cell is **skipped** (left missing) with a clear log, not a hard failure. *Why*: the
  disposable vast.ai container is the sandbox; the project-level `QQUANT_ALLOW_CODE_EXEC` is a
  deliberate second switch so code exec can never fire by accident on a dev box.
- **HumanEval pass@1 + Wilson CI over k/n.** `metric_value = pass@1 = k/n`; store `k` (`n_correct`)
  and the Wilson interval in `extra_metrics`/`meta`; `metric_stderr=None`. *Why*: lm-eval may not
  emit a pass@1 stderr. The eval-time Wilson CI uses the **shared**
  `qquant.aggregate.stats.wilson_interval` (Spec 09 owns the single implementation);
  `qquant.eval.metrics` imports/re-exports it rather than reimplementing. Validate the **bf16
  baseline** pass@1 `> 0.5` (decision-log sanity gate); log a loud warning + a `meta.sanity_failed`
  flag if not.
- **IFEval.** Set `langdetect.DetectorFactory.seed = 0` before any ifeval run. *Why*: langdetect is
  nondeterministic otherwise, perturbing the language-instruction checks. nltk `punkt`/`punkt_tab`
  are pre-downloaded by bootstrap (Spec 03); the runner only asserts availability.
- **gsm8k dataset id (`openai/gsm8k`).** lm-eval's `gsm8k` task references the bare `gsm8k` repo
  id, which `datasets>=4` rejects (`HfUriError`); the working id is `openai/gsm8k`.
  `qquant.eval.datasets.apply_dataset_overrides()` rewrites the id (and drops the stale revision)
  at the `datasets` layer at runtime — mirroring the Spec-02 spike — and the runner calls it once
  before any gsm8k eval. *Why here:* Spec 05 owns the gsm8k run (its own GPU smoke #12 runs
  gsm8k), so the eval is self-contained rather than dependent on an unbuilt Spec-03 bootstrap
  task-yaml patch. `datasets` is imported lazily inside the shim (off the torch-free import path).
- **metric key drift.** `extract_metric` walks `task.metric_keys` in order (e.g.
  `pass@1,create_test → pass@1,none → pass@1`; `acc,none → acc`), so a suffix change in lm-eval
  doesn't silently drop a metric. *Why*: lm-eval result keys carry filter suffixes that drift.
- **Provenance everywhere.** Each cell's `config` is exactly `cell_provenance(run, task, variant)`
  (keys == `PROVENANCE_KEYS`), so `is_cell_done(loaded, active_config)` treats a run under different
  seeds/batch/few-shot/template/`max_length`/`lm_eval_version`/`model_revision` as **stale** and
  recomputes it. `meta` adds the non-keyed provenance (lib versions, dataset fingerprint, gpu name,
  gen_kwargs, timestamp, item vectors).
- **Self-quant provenance (detect re-quantization).** `variants.yaml` has `revision: null` for
  `gptq-selfquant`/`awq-selfquant`, so the Spec-04 loader's `metadata["model_revision"]` is `None`
  and a re-quantized checkpoint would not invalidate its cells. For self-quant variants this spec
  therefore sets `RunConfig.model_revision` to a **checkpoint fingerprint** derived from the
  Spec-07 `quant_manifest.json`, e.g.
  `f"selfquant:{base_revision[:12]}:{calib_sha256[:12]}:{algorithm}"`. Because `model_revision` is
  already a `PROVENANCE_KEYS` entry, a re-quantization (new base/calibration/algorithm) changes the
  fingerprint and `is_cell_done` marks the affected cells **stale** — **no code or
  `PROVENANCE_KEYS` change is needed**. *(Spec 07 emits the manifest; decision-log → self-quant.)*
  **Build order:** Spec 07's `quant_manifest.json` does not exist yet, so this self-quant
  fingerprint path is authored against that interface but **verified post-Spec-07**. The
  official-core variants (the v1-now run) use the pinned-SHA `model_revision` straight from
  Spec 04's `metadata["model_revision"]`; self-quant is a deferrable fast-follow (decision-log →
  Scope), so reusing Spec 04's `metadata["checkpoint_fingerprint"]` is an acceptable fallback
  until Spec 07 lands the structured manifest.
- **lm-eval × transformers v5 risk.** lm-eval `0.4.12` is the verified fallback pin;
  `apply_chat_template` on v5 is open upstream (issue #3537). Spec 02's spike is the go/no-go gate;
  if it pins a different v5-working lm-eval commit, only the resolved `lm_eval_version` string
  changes (captured in provenance) — this spec's code is unaffected.

## Done-when (numbered, testable)

1. `pyproject.toml` registers `qquant-eval = "qquant.eval.cli:main"`; `qquant-eval --help` and
   `python -m qquant.eval.cli --help` both succeed.
2. The torch-free guard stays green: `import qquant` (+ core modules) pulls no `torch`; and
   `qquant.aggregate.stats`, `qquant.eval.metrics`, `qquant.eval.results`, `qquant.eval.manifest`,
   `qquant.eval.policy`, `qquant.eval.datasets` import on macOS **without** `torch` or `lm_eval`
   in `sys.modules` (`datasets` may import only when `apply_dataset_overrides` is actually called).
3. `wilson_interval(k, n)` matches known reference values (e.g. `k=140,n=164` → ≈`[0.797, 0.910]`),
   handles `k=0` and `k=n`; for a HumanEval cell `metric_value == k/n` and `meta.n_correct == k`.
4. `extract_metric` returns the first matching key in `metric_keys` order and raises `KeyError`
   (listing available keys) when none match; `extract_stderr` returns `None` when absent.
5. `build_cell` output passes `validate_cell`; its `config` equals
   `cell_provenance(run, task, variant.id)`; `write_cell` is atomic (no partial file on crash) and
   the reread cell is `is_cell_done(..., active_config)` → `True`.
6. `EvalRunner.plan` returns exactly `missing_cells(...)` for the manifest; a variant whose cells
   are all present yields `[]` and `load_model` is **never called** (asserted via a counting fake).
7. With a fake `load_model`/`evaluate_fn`, `run_manifest` calls `load_model` **once per variant**
   that has missing cells (call count == number of such variants), and writes the expected cells.
8. A faked MMLU run writes **57** per-subject cells with `subject` set, `task_lm_eval ==
   mmlu_<subject>`, paths from `cell_path`; the emitted subject set equals `set(MMLU_SUBJECTS)`; a
   rerun with 56 present re-evaluates only the 1 missing subtask.
9. `simple_evaluate_kwargs` forwards `apply_chat_template`/`fewshot_as_multiturn`/`num_fewshot`/
   `batch_size_for`/`max_gen_toks` from the registry (no literals), and `greedy_gen_kwargs`
   contains `do_sample=False`.
10. A `code_exec` task is skipped (cell stays missing, warning logged) unless both
    `HF_ALLOW_CODE_EVAL=1` and `QQUANT_ALLOW_CODE_EXEC=1`; when both set, `confirm_run_unsafe_code=
    True` is passed to `evaluate_fn` (asserted via fake).
11. Running ifeval sets `langdetect.DetectorFactory.seed == 0` (asserted).
12. **GPU smoke (opt-in `@pytest.mark.gpu`, Linux only):** `qquant-eval --variant bf16 --task
    gsm8k --limit 8` plus `--determinism-check bf16 gsm8k` build a schema-valid cell **in memory**
    (validated, not persisted to the results tree — `--limit`/smoke never write, per #13) and
    yield identical `metric_value` + `item_correct` across two runs.
13. Every written cell has `n_samples == len(meta.item_correct) == len(meta.item_ids)` (the paired
    inputs Spec 09 consumes, read directly from the cell — no sidecar); `--limit`/smoke runs
    refuse to write cells.
14. `ruff check` and `pytest -q` pass for the new package and tests.
15. A MMLU run persists lm-eval's `mmlu` **group** aggregate to
    `results/_meta/mmlu_group_<variant>.json` (via `Paths.meta_dir`) with `{acc, acc_stderr}` plus
    `n_subjects` and `lm_eval_version`; the file is the deterministic input Spec 09's
    `reconcile_mmlu` reads.
16. `apply_dataset_overrides()` rewrites the bare `gsm8k` dataset id to `openai/gsm8k` (and clears
    the stale revision), asserted with a fake `datasets` loader on macOS (no network); the runner
    invokes it before any gsm8k eval so the bare id never reaches `datasets>=4`.
17. `wilson_interval` is defined once in `qquant.aggregate.stats` (pure-math, torch-free) and
    re-exported by `qquant.eval.metrics`; importing `qquant.aggregate.stats` pulls no `torch`.

## Risks & mitigations

- **lm-eval API drift (preloaded HFLM / `simple_evaluate` signature).** Isolate all lm-eval calls
  behind `build_hflm` + `simple_evaluate_kwargs`; inject `evaluate_fn` so unit tests don't need
  lm-eval; the Spec-02 spike validates the real path before any large spend.
- **`apply_chat_template` open on transformers v5 (#3537).** Pin the spike-blessed lm-eval; the
  resolved version is captured in provenance, so a re-pin auto-invalidates stale cells.
- **Code-exec safety.** Double env gate + `confirm_run_unsafe_code` only inside the disposable
  container; default-skip when ungated; never enabled implicitly in tests.
- **VRAM OOM from the logits transient.** Drive batch from `batch_size_for` and bound `seq` with
  `max_length`; rely on the orchestrator's `expandable_segments`. A failed cell is logged and left
  missing (resumable), never written half-formed.
- **McNemar pairing nondeterminism.** Greedy + fixed seeds + recorded dataset revision; pair items
  by stored `doc_id` (`meta.item_ids`), not by position; the determinism smoke guards regressions.
- **MMLU subject/key drift.** Assert emitted subjects == `set(MMLU_SUBJECTS)`; resolve metrics via
  ordered `metric_keys`.
- **Item-vector bloat in cells.** Store compact `0/1` ints; largest is gsm8k (~1.3k) — negligible
  vs the cell count, and keeps everything inside the SSOT cell file (resumable + copied out by the
  existing reconcile, no new path contract).
