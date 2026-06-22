# Spec 01 — Repo scaffold + torch-free `qquant` core

- **Status:** IMPLEMENTED — all Done-when criteria already met (`uv sync --locked`, `ruff check`,
  `ruff format --check`, `pytest`, and the torch-free import guard are green locally and in CI).
- **Depends on:** 00
- **Owns / Produces:** every contract in [`contracts.md`](contracts.md) and the code that backs it:
  the torch-free Python package `src/qquant/` (`config`, `registry`, `paths`, `matrix`, `cli`,
  `schemas`), the two YAML registries, the v1 cell-result JSON Schema, the `qquant` umbrella
  entry point in `[project.scripts]`, the two-uv-environment layout (`./` eval env + `quant/`
  self-quant env), the universal `uv.lock`, ruff/pytest config, and the CI workflow with the
  torch-free import guard.

## Purpose

Establish the project skeleton and the **single source of truth** that every later spec imports
instead of re-declaring: the canonical variant/task ids, the `Variant`/`Task` registry
dataclasses + YAML loaders, the deterministic on-disk path layout (`cell_path`), the v1 cell
schema + validator, the (variant × task) matrix expansion with the SSOT resume predicate
(`is_cell_done` / `missing_cells`), the run-config → per-cell provenance block, and a
**torch-free** umbrella CLI usable on a laptop with no GPU. This module is the contract layer;
GPU-touching specs (02–11) depend on it and never re-define ids, paths, schema keys, or the
resume predicate.

This spec **documents the existing implementation**. It is normative for *what the scaffold
contains and guarantees*; it does not request new work. Treat any divergence between this
document and the code under `src/qquant/` as a bug to reconcile in the same commit as
`contracts.md` (per the SSOT rule).

## Scope (in / out)

**In scope (implemented here):**
- The `qquant` package: `config.py`, `registry.py`, `paths.py`, `matrix.py`, `cli.py`,
  `schemas/__init__.py`, `schemas/cell_result.schema.json`, and the two registries
  `registries/variants.yaml` + `registries/tasks.yaml`.
- All §1–§7 contracts in `contracts.md` (ids, dataclasses, paths, schema, matrix/resume,
  provenance, entry points).
- Packaging: `pyproject.toml` (hatchling, `requires-python >=3.11,<3.12`, torch-free core deps,
  `gpu`/`analysis`/`dev` dependency groups, `[project.scripts] qquant`, universal `uv.lock`).
- The isolated `quant/` self-quant uv project (its `pyproject.toml` + README; its `uv.lock` is
  finalized on the box by Spec 07, so it is intentionally absent from the repo today).
- Dev tooling: ruff (lint + format) and pytest config; `.github/workflows/ci.yml` running
  sync/lint/test/torch-free-guard on Ubuntu + macOS.
- Hygiene: `.env.example`, `.gitignore`, `.python-version`, top-level `README.md`.
- Unit tests (`tests/test_*.py`) proving every contract holds (the torch-free guard included).

**Out of scope (owned by later specs — Spec 01 only registers the torch-free `qquant` entry
point):**
- Any code that imports `torch`, `transformers`, `lm_eval`, `bitsandbytes`, `gptqmodel`,
  `accelerate`, `compressed-tensors`, `optimum`, or `llmcompressor`.
- The vast.ai wrapper / GPU spike (02), remote bootstrap (03), variant loaders (04),
  quality eval `qquant-eval` (05), efficiency profiler `qquant-profile` (06), self-quant
  production (07), orchestration `qquant-orchestrate` (08), aggregation/stats/plots (09),
  `results/REPORT.md` (10), and the extended CI/predownload caching (11). Each of those adds
  its own `[project.scripts]` line when it lands.

## Interface

All public symbols are re-exported from `qquant/__init__.py` (`__version__ == "0.1.0"`).

### `qquant.registry` — canonical ids + dataclasses (contracts §1, §2)

```python
VARIANT_IDS: tuple[str, ...] = (
    "bf16", "bnb-int8", "bnb-nf4", "gptq-official",
    "awq-official", "gptq-selfquant", "awq-selfquant",
)  # hyphens, no aliases
TASK_IDS: tuple[str, ...] = ("mmlu", "gsm8k", "humaneval", "ifeval")
QUANT_METHODS: frozenset[str] = frozenset(
    {"bf16", "bnb-int8", "bnb-nf4", "gptq", "awq", "compressed-tensors"}
)
MMLU_SUBJECTS: tuple[str, ...] = (...)  # the 57 subjects; lm-eval subtask = f"mmlu_{subject}"

@dataclass(frozen=True)
class Variant:
    id: str; quant_method: str; source: str; enabled: bool
    model_id: str; revision: str | None; notes: str = ""

@dataclass(frozen=True)
class Task:
    id: str; lm_eval_task: str; primary_metric: str
    metric_keys: tuple[str, ...]; per_subject: bool
    apply_chat_template: bool; fewshot_as_multiturn: bool
    num_fewshot: int; default_batch_size: int
    batch_overrides: dict[str, int]; code_exec: bool = False
    max_gen_toks: int | None = None
    def batch_size_for(self, variant_id: str) -> int: ...  # override else default

def load_variants(path: str | Path | None = None) -> dict[str, Variant]: ...
def load_tasks(path: str | Path | None = None) -> dict[str, Task]: ...
def enabled_variants(variants: dict[str, Variant]) -> list[Variant]: ...  # canonical order
```

`load_variants` validates that ids ⊆ `VARIANT_IDS`, `quant_method` ∈ `QUANT_METHODS`, and that
**all 7** ids are present; `load_tasks` validates ids ⊆ `TASK_IDS`. Registries are read from the
package via `importlib.resources` (`files("qquant") / "registries" / ...`), so loaders work from
an installed wheel; passing `path=` overrides for tests.

### `qquant.paths` — on-disk layout (contracts §3)

```python
def cell_path(results_root, variant, task, subject=None) -> Path
# <root>/<variant>/<task>/<subject>.json   (subject set)  → per-subject MMLU cell
# <root>/<variant>/<task>/result.json      (subject None) → single-cell task

@dataclass(frozen=True)
class Paths:
    results_root: Path
    @classmethod
    def from_root(cls, root) -> "Paths"
    meta_dir / env_file / done_manifest   # <root>/_meta/{,env.json,done_manifest.json}
    def cell(self, variant, task, subject=None) -> Path   # delegates to cell_path
```

`cell_path` is the **only** path builder; the MMLU subject token is the bare slug (`anatomy`).

### `qquant.schemas` — cell-result schema v1 (contracts §4)

`cell_result.schema.json` (JSON Schema 2020-12). Required top-level: `schema_version (==1)`,
`cell_id`, `variant`, `task`, `primary_metric`, `metric_value`, `n_samples`, `config`; optional
`subject`, `task_lm_eval`, `metric_stderr`, `extra_metrics`, `meta`. Required `config` keys are
exactly the eight `PROVENANCE_KEYS`. API:

```python
def load_schema() -> dict          # lru_cache(1), read from package resources
def validate_cell(cell: dict) -> None      # raises jsonschema.ValidationError
def is_valid_cell(cell: dict) -> bool
```

### `qquant.matrix` — matrix + SSOT resume predicate (contracts §5)

```python
PROVENANCE_KEYS = ("seeds", "batch_size", "num_fewshot", "apply_chat_template",
                   "fewshot_as_multiturn", "max_length", "lm_eval_version", "model_revision")

@dataclass(frozen=True)
class Cell:
    variant: str; task: str; subject: str | None
    lm_eval_task: str; primary_metric: str
    cell_id -> str                 # "<variant>/<task>[/<subject>]"
    def path(self, results_root) -> Path

def expand_matrix(variants, tasks) -> list[Cell]      # MMLU → 57 cells; v1 = 7×60 = 420
def is_cell_done(cell, active_config=None) -> bool    # SSOT resume predicate — never reimplement
def missing_cells(cells, results_root, active_config=None) -> list[Cell]
```

`is_cell_done` enforces structural validity (`schema_version == 1`, non-empty `cell_id`,
finite real `metric_value` that is not a bool, integer `n_samples > 0`) **and**, when an
`active_config` provenance block is supplied, requires every overlapping `PROVENANCE_KEYS`
value to match — otherwise the cell is stale and recomputed. `missing_cells` treats absent,
unreadable, or stale cells as missing.

### `qquant.config` — run config + provenance (contracts §6)

```python
@dataclass(frozen=True)
class Seeds:  random=0; numpy=1234; torch=1234; fewshot=1234;  def as_dict() -> dict
@dataclass(frozen=True)
class RunConfig:  lm_eval_version: str; model_revision: str|None=None;
                  max_length: int|None=None; seeds: Seeds = Seeds()
def cell_provenance(run: RunConfig, task: Task, variant_id: str) -> dict  # keyed by PROVENANCE_KEYS
```

`cell_provenance` round-trips with the schema `config` block and with `is_cell_done`'s
`active_config`.

### `qquant.cli` — torch-free umbrella (contracts §7)

CLI `qquant` (`qquant.cli:main`), `argparse`, subcommands — **never imports torch**:
- `qquant variants` — list the variant registry (enabled flag, method, source, short revision, model_id).
- `qquant tasks` — list the task registry (lm-eval task, metric, per-subject, few-shot, chat-template).
- `qquant matrix` — enabled-variant/task/total-cell counts + per-task breakdown.
- `qquant audit --results <root> [--list]` — done/missing cell counts via `missing_cells`.
- `qquant version` — print `__version__`.

`main(argv=None) -> int` is the entry point; subcommands return process exit codes.

### File formats

- **`registries/variants.yaml`** — top-level `variants:` map keyed by id with
  `quant_method`, `source` (`baseline|load-time|official|selfquant`), `enabled`, `model_id`
  (HF repo id or `local:checkpoints/self-quant/<id>`), `revision` (SHA or `null`), `notes`.
- **`registries/tasks.yaml`** — top-level `tasks:` map keyed by id with `lm_eval_task`,
  `primary_metric`, `metric_keys` (ordered fallback list), `per_subject`, `apply_chat_template`,
  `fewshot_as_multiturn`, `num_fewshot`, `default_batch_size`, `batch_overrides` (per-variant),
  `code_exec`, `max_gen_toks`.
- **Cell JSON** — per `cell_result.schema.json` (above).

## Files to create

All already present and committed; listed for completeness.

```
pyproject.toml                              # eval env: torch-free core deps + gpu/analysis/dev groups + [project.scripts] qquant + uv config
uv.lock                                     # universal lock (darwin-arm64 core; linux-x86_64 full gpu group)
.python-version                             # 3.11
.gitignore                                  # ignores .venv, results/, checkpoints/, hf_cache/, .env, caches
.env.example                                # VAST_API_KEY / HF_TOKEN / WANDB_API_KEY placeholders (NO secrets)
README.md                                   # quickstart + layout + two-env explanation
.github/workflows/ci.yml                    # sync --locked → ruff check → ruff format --check → pytest → torch-free guard (ubuntu + macos)
src/qquant/__init__.py                      # re-exports the public API; __version__ = "0.1.0"
src/qquant/config.py                        # Seeds, RunConfig, cell_provenance
src/qquant/registry.py                      # VARIANT_IDS/TASK_IDS/QUANT_METHODS/MMLU_SUBJECTS, Variant, Task, loaders
src/qquant/paths.py                         # cell_path, Paths
src/qquant/matrix.py                        # PROVENANCE_KEYS, Cell, expand_matrix, is_cell_done, missing_cells
src/qquant/cli.py                           # torch-free umbrella CLI
src/qquant/registries/variants.yaml         # 7 variants (SSOT data)
src/qquant/registries/tasks.yaml            # 4 tasks (SSOT data)
src/qquant/schemas/__init__.py              # load_schema, validate_cell, is_valid_cell
src/qquant/schemas/cell_result.schema.json  # v1 JSON Schema (2020-12)
tests/test_registry.py                      # id counts, YAML↔SSOT, self-quant→compressed-tensors, batch overrides, enabled order
tests/test_paths.py                         # subject vs single-cell paths, meta helpers
tests/test_matrix.py                        # 420-cell count, cell_id, structural + provenance is_cell_done, missing_cells
tests/test_schema.py                        # sample cell validates; missing required keys fail
tests/test_import_torch_free.py             # fresh-interpreter guard: core never imports torch
quant/pyproject.toml                        # isolated self-quant env (llmcompressor==0.12.0); lock finalized on box (Spec 07)
quant/README.md                             # why quant/ is a separate uv project
```

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **One id vocabulary, hyphenated, no aliases** (decision-log "Variant & evaluation decisions";
  contracts §1, §8). The 7 ids live as a Python constant `VARIANT_IDS` *and* are re-validated by
  the YAML loader so the registry can never drift from the SSOT. The bare `gptq`/`awq`/`bnb`
  strings are legal `quant_method`s but forbidden as variant ids. *Why:* downstream specs and
  the consistency-review gate assert zero id drift.
- **Self-quant variants use `quant_method: compressed-tensors`**, `revision: null`, and
  `model_id: local:checkpoints/self-quant/<id>` (decision-log self-quant section; contracts §1–§2).
  *Why:* they load natively in transformers v5 via compressed-tensors, **not** gptqmodel's legacy
  path; the loader (Spec 04) dispatches on `quant_method`.
- **`awq-official` ships `enabled: true` but is CONTINGENT** (decision-log): the sm_89 AWQ-load
  smoke test (run in the **Spec 02 spike / Spec 08 pre-flight**) flips it to `false` if loading
  fails; `awq-selfquant` is the guaranteed AWQ data point. The `enabled` flag +
  `enabled_variants()` make this a one-line registry edit with no code change.
- **Per-task chat-template policy, not blanket** (decision-log): `tasks.yaml` encodes
  `apply_chat_template`/`fewshot_as_multiturn` per task — MMLU/GSM8K with multiturn few-shot,
  HumanEval as `humaneval_instruct` (no chat-template fewshot, fenced-code extraction), IFEval
  zero-shot. *Why:* the registry is where eval policy is declared once and consumed everywhere.
- **MMLU = full 57 per-subject cells** (decision-log; contracts §5). `MMLU_SUBJECTS` is the SSOT
  list; `expand_matrix` fans MMLU to 57 cells → v1 matrix is **7 × 60 = 420**. The bare-slug
  filename + `mmlu_<subject>` lm-eval name keep paths readable while recording the lm-eval task in
  the cell. A GPU/lm-eval test (Spec 02/06) asserts `set(MMLU_SUBJECTS)` equals the lm-eval group
  expansion.
- **Per-(variant, task) batch via `batch_overrides`** (decision-log "VRAM budget includes the
  logits transient"): `tasks.yaml` sets MMLU `bf16` and `bnb-int8` to batch 2 (4-bit variants stay
  at 4) because the `batch×seq×vocab(152064)×2B` logits transient (~3.5 GB) dominates. Exposed via
  `Task.batch_size_for(variant_id)` and folded into the provenance block so a batch change
  invalidates stale cells.
- **Provenance-gated resume** (decision-log greedy-determinism + reproducibility; contracts §5–§6).
  `PROVENANCE_KEYS` is declared once in `matrix.py`, mirrored by `cell_provenance` and the schema
  `config` block; `is_cell_done` is the **only** resume predicate (no parallel `completed:true`
  checks). *Why:* changing seeds/batch/few-shot/template/lm-eval-version/model-revision must force
  recomputation, never a silent reuse.
- **Verified pins live in two uv environments, not one** (decision-log "Two uv environments").
  The root eval env's `gpu` group pins `transformers==5.10.1`, `gptqmodel==7.1.0`, `optimum>=2.2.0`,
  `bitsandbytes==0.49.2`, `accelerate==1.13.0`, `compressed-tensors>=0.17.1`,
  `lm-eval[hf,ifeval,math]==0.4.12`, with `torch` from the cu126 index — all `sys_platform=='linux'`.
  The universal `uv.lock` actually resolved: **torch 2.12.1+cu126, transformers 5.10.1,
  gptqmodel 7.1.0, optimum 2.2.0, bitsandbytes 0.49.2, accelerate 1.13.0, compressed-tensors 0.17.1,
  lm-eval 0.4.12, datasets 5.0.0, numpy 2.2.6** (linux GPU stack) plus core **pyyaml 6.0.3,
  jsonschema 4.26.0** and dev **ruff 0.15.18, pytest 9.1.1**. `quant/pyproject.toml` is the
  separate self-quant project (`llmcompressor==0.12.0`, `datasets>=4.8.4,<=5.0.0`) whose tight
  `transformers<=5.10.1` window cannot co-resolve with the eval stack; its `uv.lock` is finalized
  on the vast.ai box (Spec 07), so it is intentionally not in the repo yet. **`autoawq` is never a
  dependency** (archived; would downgrade transformers).
- **Two-platform universal lock** (decision-log): `[tool.uv].environments` targets
  `darwin-arm64` (core + dev only; torch-free) and `linux-x86_64` (full `gpu` group). *Why:* the
  dev laptop installs zero CUDA wheels while the box gets the GPU stack from one lock file.
- **Torch-free core is enforced, not just intended** (contracts §7; decision-log platform). The
  package docstring states the invariant; `tests/test_import_torch_free.py` imports the whole core
  in a **fresh interpreter** and asserts `torch` never enters `sys.modules`; CI repeats the guard
  as a separate step. *Why:* orchestration and result auditing must run on a no-GPU laptop, and the
  `qquant` umbrella CLI must stay importable there. GPU CLIs are **separate** entry points.
- **`[project.scripts]` ownership** (contracts §7): only `qquant = "qquant.cli:main"` is registered
  now; `qquant-spike` (02), `qquant-eval` (05), `qquant-profile` (06), `qquant-orchestrate` (08),
  `qquant-aggregate` (09), `qquant-report` (10) are reserved and added by their own specs. *Why:* the
  umbrella never gains a torch-importing subcommand.
- **No secrets in the repo / public GitHub delivery** (decision-log platform): `.env.example`
  documents `VAST_API_KEY`/`HF_TOKEN`/`WANDB_API_KEY`; `.gitignore` excludes `.env`, `results/`,
  `checkpoints/`, `hf_cache/`. The account-level spend limit reminder lives in `.env.example`.
- **No PDF/LaTeX** (decision-log reporting): the README points the final deliverable at
  `results/REPORT.md` (markdown); Spec 01 only carries the layout note, not the report.

## Done-when (numbered, testable)

1. `uv sync --locked` succeeds against the committed universal `uv.lock` on both darwin-arm64 and
   linux-x86_64 (CI matrix: macos-latest + ubuntu-latest). *(Verified: "Resolved 142 packages".)*
2. `uv run ruff check .` reports no violations and `uv run ruff format --check .` reports all files
   formatted. *(Verified: "All checks passed!", "12 files already formatted".)*
3. `uv run pytest` passes the full suite. *(Verified: 18 passed — registry, paths, matrix, schema,
   torch-free guard.)*
4. The torch-free import guard passes: importing `qquant` + every core submodule + the CLI in a
   fresh interpreter leaves no `torch*` module in `sys.modules`. *(Verified: "torch-free OK"; CI
   step `torch-free import guard`.)*
5. `uv run qquant variants|tasks|matrix|version` run without importing torch; `qquant matrix`
   prints `enabled variants : 7`, `tasks : 4`, `total cells : 420` (mmlu 399, gsm8k/humaneval/ifeval
   7 each). *(Verified.)*
6. `load_variants()` returns all 7 canonical ids with `quant_method ∈ QUANT_METHODS`; official /
   baseline / load-time variants pin a revision SHA, self-quant variants have `revision is None` and
   `quant_method == "compressed-tensors"`. *(test_registry.py.)*
7. `load_tasks()` returns the 4 task ids; `mmlu.per_subject is True`, `humaneval.code_exec is True`,
   `tasks["mmlu"].batch_size_for("bf16") == 2` and `== 4` for `gptq-official`. *(test_registry.py.)*
8. `len(MMLU_SUBJECTS) == 57` (unique); `expand_matrix(enabled_variants(...), tasks) == 7 × 60`.
   *(test_registry.py + test_matrix.py.)*
9. `cell_path` yields `<root>/<variant>/<task>/<subject>.json` with a subject and
   `<root>/<variant>/<task>/result.json` without; `Paths.env_file`/`done_manifest` resolve under
   `_meta/`. *(test_paths.py.)*
10. A `cell_provenance`-built sample cell validates against the v1 schema; deleting any required
    top-level or `config` key fails `is_valid_cell`/`validate_cell`. *(test_schema.py.)*
11. `is_cell_done` is structural-plus-provenance: a finite-metric cell is done; NaN metric,
    `n_samples == 0`, missing `schema_version`, or a mismatched provenance knob are all not done;
    `missing_cells` decrements by one when a valid cell is written. *(test_matrix.py.)*
12. The repo carries no secrets; `.gitignore` excludes `.env`, `results/`, `checkpoints/`,
    `hf_cache/`; `.env.example` documents the injected env vars and the spend-limit reminder.

## Risks & mitigations

- **Contract drift between this doc, `contracts.md`, and the code.** *Mitigation:* `contracts.md`
  is the SSOT and Spec 01's code mirrors it; ids/paths/schema/predicate change in the same commit;
  the end-of-authoring consistency-review gate asserts zero drift across all specs.
- **A later spec accidentally imports torch into the umbrella CLI (or core).** *Mitigation:* the
  fresh-interpreter `test_import_torch_free.py` plus the dedicated CI guard step fail the build; GPU
  CLIs are registered as separate `[project.scripts]` entry points, never umbrella subcommands.
- **lm-eval metric key suffix drift** (e.g. `pass@1,create_test` vs `pass@1,none`). *Mitigation:*
  `tasks.yaml` `metric_keys` is an ordered fallback list consumed by the eval writer (Spec 05);
  the registry already encodes the candidates so no code change is needed when suffixes move.
- **lm-eval × transformers v5 incompatibility** (`apply_chat_template`, upstream issue). *Mitigation:*
  out of scope here — the `lm-eval==0.4.12` pin is the documented fallback and Spec 02's spike is the
  go/no-go gate before any GPU spend; Spec 01 only fixes the pin in `pyproject.toml`/`uv.lock`.
- **`quant/uv.lock` absent from the repo could look unfinished.** *Mitigation:* intentional — the
  self-quant lock is finalized on the linux/CUDA box (Spec 07); `quant/README.md` and the decision
  log record this. The root universal lock is complete and is what CI uses.
- **MMLU dataset / subject-set mismatch with the pinned lm-eval.** *Mitigation:* `MMLU_SUBJECTS` is
  the SSOT 57-slug list; a GPU/lm-eval test (Spec 02/06) asserts it equals the `TaskManager`
  expansion of the `mmlu` group, catching any upstream subject reshuffle.
