# Spec 11 — CI, determinism contract & reproducible pre-download caching

- **Status:** Draft (ready to implement; extends the green Spec 01 scaffold)
- **Depends on:** 01
- **Owns / Produces:**
  - The **extended** `.github/workflows/ci.yml` (Spec 01 created the first cut; this spec owns its
    final shape): both-OS lint / format / pytest / torch-free guard, a `gpu`-marker auto-skip, and a
    Linux **`lm-eval-contract`** job asserting `set(MMLU_SUBJECTS)` equals the `TaskManager`
    expansion of the `mmlu` group for the **pinned** lm-eval.
  - The pytest marker machinery (`tests/conftest.py` + marker registration in `pyproject.toml`):
    `gpu` (auto-skip without CUDA) and `lmeval` (auto-skip without the installed lm-eval stack).
  - The **determinism contract** as one consolidated, normative statement + a torch-free
    `qquant.repro.determinism` module exposing its canonical values (the four lm-eval seeds via
    `Seeds`, the langdetect seed, the forced-greedy `gen_kwargs`, the fixed per-(variant,task)
    batch) and the CI checks that guard it from regressing.
  - The **`_meta/env.json` reproducibility-provenance writer + schema** (`qquant.repro.env`,
    torch-free) — the file `contracts.md` §3 names but no spec yet *defines*; Specs 03/05/06/08
    *consume* it. **Spec 03's remote `write_env.py` is the GPU-side writer** that uses this module
    (`build_env`/`add_model`/`add_dataset`/`merge_env`/`write_env`) to fill `libraries`/`toolchain`
    from the live GPU stack.
  - The **revision-pinned pre-download** box script `scripts/repro/predownload.py`: snapshots the
    model checkpoints (by SHA, from the registry) and the four task datasets (via lm-eval) into
    `HF_HOME`, fetches the nltk data IFEval needs, and records dataset fingerprints in
    `_meta/env.json`.
  - The CI verification that **no secrets are committed** (`.env` ignored, `.env.example` blank).

This spec does **not** add a `[project.scripts]` entry point (contracts §7 reserves none for
reproducibility tooling); the pre-download is a standalone box script, mirroring Spec 02's
`scripts/spike/spike_remote.py` pattern.

## Purpose

Make every result in the study **cheap to trust and cheap to reproduce**:

1. **CI keeps the contract layer honest on a laptop.** The torch-free core is linted, formatted,
   tested, and import-guarded on macOS **and** Linux on every push, and a dedicated Linux job
   proves the pinned lm-eval still expands the `mmlu` group to exactly the 57 `MMLU_SUBJECTS`
   (the cheapest place to catch an upstream subject reshuffle, complementing Spec 02's on-box
   spike check).
2. **One determinism contract, written once.** The four lm-eval seeds, the langdetect seed, forced
   greedy decoding, and the fixed per-(variant,task) batch are consolidated here, cross-referenced
   to where each is *enforced* (Spec 01 `Seeds`, Spec 04 load-time clearing, Spec 05 `gen_kwargs` +
   langdetect), exposed as torch-free constants, recorded in `_meta/env.json`, and gated on resume
   by `PROVENANCE_KEYS` / `is_cell_done`. The runtime proof is Spec 05's `--determinism-check`; this
   spec owns the *contract surface* and its CI guards.
3. **Pinned, cached artifacts.** A boot-time pre-download fills `HF_HOME` with the SHA-pinned model
   checkpoints and the lm-eval-resolved task datasets so every matrix cell reuses byte-identical
   inputs, never re-downloads mid-run, and the resolved dataset revisions/fingerprints land in
   `_meta/env.json` for the audit trail.
4. **No secrets in a public repo.** `.env` handling is verified in CI so it cannot regress.

## Scope (in / out)

**In scope**
- `.github/workflows/ci.yml`: final shape (core matrix job + `lm-eval-contract` job).
- `tests/conftest.py` + `[tool.pytest.ini_options].markers` for `gpu` / `lmeval` auto-skip.
- `qquant.repro.determinism` — torch-free contract constants + the `Seeds`→lm-eval seed-arg
  mapping; **does not** re-declare `Seeds` (imports it from `qquant.config`).
- `qquant.repro.env` — the `_meta/env.json` schema, builder, merge, validate, write/read (torch-free).
- `scripts/repro/predownload.py` — the box-side, heavy-import pre-download (models + datasets +
  nltk), writing `_meta/env.json`.
- Torch-free tests for the above, a CI-workflow-shape test, a `.env`-hygiene test, the `lmeval`
  MMLU-expansion test, and a `gpu`-marked toolchain test (skipped without CUDA).

**Out of scope (owned elsewhere)**
- The in-eval greedy `gen_kwargs`, the langdetect seeding *call*, and the `--determinism-check`
  twice-and-compare run → **Spec 05** (this spec owns the contract values + CI guards, not the
  eval path). The profiler's greedy/exact-token logic → **Spec 06**.
- The remote bootstrap/`onstart`/`run_matrix` that *calls* `predownload.py` at boot, and the
  `PYTORCH_CUDA_ALLOC_CONF` export → **Spec 03**.
- Orchestration that enriches `_meta/env.json` with the image digest / cost ledger and the
  `done_manifest.json` reconciliation → **Spec 08** (it *consumes* env.json, contracts §3).
- Re-declaring any id / path / schema key / resume predicate → all imported from Spec 01
  (`VARIANT_IDS`, `TASK_IDS`, `MMLU_SUBJECTS`, `Seeds`, `cell_provenance`, `PROVENANCE_KEYS`,
  `is_cell_done`, `Paths.env_file`, `load_variants`/`load_tasks`/`enabled_variants`).

## Interface

### `qquant.repro.determinism` — the determinism contract surface (torch-free)

```python
from qquant.config import Seeds   # the four-seed SSOT (random=0, numpy=1234, torch=1234, fewshot=1234)

# Forced-greedy decoding kwargs — MIRRORS Spec 05's enforced values (do not diverge).
GREEDY_GEN_KWARGS: dict = {"do_sample": False, "temperature": 0.0, "top_p": 1.0}
CLEARED_SAMPLING_KEYS: tuple[str, ...] = ("temperature", "top_p", "top_k")
LANGDETECT_SEED: int = 0

# The PROVENANCE_KEYS subset that pins numerical determinism (the rest pin environment).
DETERMINISM_PROVENANCE_KEYS: tuple[str, ...] = (
    "seeds", "batch_size", "num_fewshot", "apply_chat_template", "fewshot_as_multiturn",
)

def lm_eval_seed_kwargs(seeds: Seeds | None = None) -> dict:
    """Map Seeds → lm-eval `simple_evaluate(...)` seed arguments:
    {"random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"}.
    Defaults to Seeds() (the canonical run seeds)."""
```

`GREEDY_GEN_KWARGS` is the **single written value** of the greedy contract; Spec 04/05 enforce it
(at load and in `gen_kwargs`) and a CI test asserts the contract surface here matches the
decision-log. This module never imports torch/lm-eval; `Seeds` is imported, not redefined.

### `qquant.repro.env` — `_meta/env.json` reproducibility provenance (torch-free)

```python
ENV_SCHEMA_VERSION: int = 1

def build_env(*, lm_eval_ref: str, seeds: Seeds | None = None,
              libraries: dict[str, str] | None = None,
              toolchain: dict | None = None) -> dict:
    """Skeleton env.json with schema_version, the determinism block (seeds + lm_eval_seed_kwargs +
    GREEDY_GEN_KWARGS + LANGDETECT_SEED), empty models/datasets lists, libraries, toolchain."""

def add_model(env: dict, *, variant: str, repo: str, requested_revision: str | None,
              resolved_commit: str, local_dir: str) -> dict:
    """Append/replace (keyed by variant) a model provenance record."""

def add_dataset(env: dict, *, task: str, lm_eval_task: str, subtasks: list[str],
                dataset_path: str, dataset_name: str | None, revision: str | None,
                fingerprint: str | None, num_rows: dict[str, int],
                datasets_version: str, loaded_ok: bool) -> dict:
    """Append/replace (keyed by task) a dataset provenance + load-preflight record."""

def merge_env(existing: dict, new: dict) -> dict:
    """Union models/datasets by their key; shallow-merge libraries/toolchain/determinism
    (new wins). Lets the pre-download, Spec 03 bootstrap, and Spec 08 each contribute without
    clobbering the others."""

def validate_env(env: dict) -> None        # jsonschema (core dep); raises on invalid
def write_env(results_root, env: dict) -> Path     # -> Paths.from_root(results_root).env_file; merges if a file exists
def load_env(results_root) -> dict | None
```

`write_env` writes to **`Paths.env_file`** only (never reimplements `_meta/` layout) and
read-merge-writes so repeated boots and later specs accrete one coherent record. JSON is written
with `sort_keys=True` for stable diffs (a `captured_at` timestamp is optional and excluded from
the resume comparison — cells carry their own `config` provenance; env.json is an audit record).

**`_meta/env.json` shape (schema v1):**

```json
{
  "schema_version": 1,
  "generated_by": "qquant.repro.predownload",
  "lm_eval_ref": "0.4.12",
  "determinism": {
    "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
    "lm_eval_seed_kwargs": {"random_seed": 0, "numpy_random_seed": 1234,
                            "torch_random_seed": 1234, "fewshot_random_seed": 1234},
    "greedy_gen_kwargs": {"do_sample": false, "temperature": 0.0, "top_p": 1.0},
    "langdetect_seed": 0
  },
  "libraries": {"torch": "2.12.1+cu126", "transformers": "5.10.1", "lm_eval": "0.4.12",
                "datasets": "5.0.0", "gptqmodel": "7.1.0", "bitsandbytes": "0.49.2",
                "accelerate": "1.13.0", "optimum": "2.2.0", "compressed_tensors": "0.17.1"},
  "toolchain": {"image_digest": null, "gpu_name": "NVIDIA GeForce RTX 4090",
                "torch_cuda": "12.6", "nvcc_cuda": "12.6"},
  "models": [
    {"variant": "bf16", "repo": "Qwen/Qwen2.5-7B-Instruct",
     "requested_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
     "resolved_commit": "a09a35458c702b33eeacc393d103063234e8bc28", "local_dir": "..."}
  ],
  "datasets": [
    {"task": "mmlu", "lm_eval_task": "mmlu", "subtasks": ["mmlu_anatomy", "..."],
     "dataset_path": "hails/mmlu_no_train", "dataset_name": null, "revision": "<commit-or-null>",
     "fingerprint": "<datasets _fingerprint>", "num_rows": {"test": 14042, "dev": 285},
     "datasets_version": "5.0.0", "loaded_ok": true}
  ]
}
```

### `scripts/repro/predownload.py` — box-side pre-download (heavy; standalone, never imported)

Standalone script (imports `huggingface_hub`, `datasets`, `lm_eval`, `nltk`; **not** importable
from the package, so the core stays torch-free — same rule as `spike_remote.py`). It derives the
artifact set from the SSOT registries so it can never drift:

```
uv run --group gpu python scripts/repro/predownload.py \
    --results-root results \
    [--lm-eval-ref 0.4.12]          # the Spec-02-resolved working ref (default: the locked pin)
    [--hf-home $HF_HOME]            # default: $HF_HOME, else hf_cache/
    [--nltk-data $NLTK_DATA] \
    [--no-models] [--no-datasets] [--no-nltk] [--no-preflight]
```

Behaviour:
1. **Models** — for each `v in enabled_variants(load_variants())` whose `model_id` is an **HF
   repo** (skip `local:` self-quant), `huggingface_hub.snapshot_download(v.model_id,
   revision=v.revision)` into `HF_HOME`. The three distinct repos are the SHA-pinned
   `Qwen/Qwen2.5-7B-Instruct` (bf16/bnb-int8/bnb-nf4), `…-GPTQ-Int4`, `…-AWQ`; `awq-official`
   is downloaded **only if still `enabled`** (the contingency flag drives it — no hardcoding).
   Record `resolved_commit` (the actual fetched SHA) per variant in env.json.
2. **Datasets** — for each `t in load_tasks().values()`, resolve `t.lm_eval_task` through
   `lm_eval.tasks.TaskManager(include_path=…)` (the `mmlu` group → 57 `mmlu_<subject>` subtasks),
   read each subtask's `dataset_path`/`dataset_name`, dedupe, and `datasets.load_dataset(...)`
   under the pinned `datasets` (preflight that the **datasets-4.x loading-script removal** does not
   bite). Record `dataset_path`, `dataset_name`, `revision` (hub commit when available),
   `_fingerprint`, per-split `num_rows`, and `datasets_version` per task.
3. **nltk** — download `punkt` and `punkt_tab` into `--nltk-data` (IFEval needs them; pre-download
   so no mid-run network).
4. Assemble via `qquant.repro.env.build_env(...)/add_model/add_dataset`, capture installed
   `libraries` versions + `torch_cuda`/`nvcc_cuda`, then `write_env(results_root, env)`.
5. Exit `0` iff every required model snapshot resolved and every dataset preflight loaded;
   nonzero (with a one-line reason) otherwise so a boot fails fast rather than mid-matrix.

### `.github/workflows/ci.yml` (final shape)

```yaml
name: ci
on: [push, pull_request]
jobs:
  core:                       # the existing job — both OS, torch-free
    strategy: { fail-fast: false, matrix: { os: [ubuntu-latest, macos-latest] } }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: true }
      - run: uv sync --locked                 # default + dev groups only → NO torch installed
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest                     # gpu/lmeval tests auto-skip here (see conftest)
      - name: torch-free import guard
        run: uv run python -c "import qquant, qquant.repro.env, qquant.repro.determinism, sys; assert not [m for m in sys.modules if m=='torch' or m.startswith('torch.')]"

  lm-eval-contract:           # Linux only; installs the pinned lm-eval stack
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: true }
      - run: uv sync --locked --group gpu      # installs torch(cu126)+lm-eval at the lock pins; CPU runner is fine
      - run: uv run pytest -m lmeval -q        # the MMLU group-expansion assertion
```

The `core` torch-free guard now also imports the new `qquant.repro.*` modules. The
`lm-eval-contract` job is CPU-only (the `TaskManager` group expansion needs no GPU and no dataset
download) and cached by `setup-uv`; it may be path-filtered or `workflow_dispatch` if CI minutes
are tight, but it must run whenever the lm-eval pin or `MMLU_SUBJECTS` changes.

### `tests/conftest.py` — marker auto-skip

```python
import pytest
def pytest_collection_modifyitems(config, items):
    import importlib.util
    has_cuda = False
    try:
        import torch; has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    has_lmeval = importlib.util.find_spec("lm_eval") is not None
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "lmeval" in item.keywords and not has_lmeval:
            item.add_marker(pytest.mark.skip(reason="lm-eval not installed"))
```

## Files to create

```
.github/workflows/ci.yml                    # EDIT: add lm-eval-contract job; extend torch-free guard to qquant.repro.*
pyproject.toml                              # EDIT: register `gpu` + `lmeval` markers under [tool.pytest.ini_options]
src/qquant/repro/__init__.py                # package marker; torch-free
src/qquant/repro/determinism.py             # GREEDY_GEN_KWARGS, LANGDETECT_SEED, lm_eval_seed_kwargs, DETERMINISM_PROVENANCE_KEYS
src/qquant/repro/env.py                      # ENV_SCHEMA_VERSION, build/add_model/add_dataset/merge/validate/write/load
scripts/repro/predownload.py                 # box-side heavy pre-download (models+datasets+nltk → HF_HOME, writes env.json)
tests/conftest.py                            # gpu/lmeval auto-skip hook
tests/test_repro_determinism.py              # contract-surface assertions (torch-free)
tests/test_repro_env.py                      # env.json build/merge/validate/write/load round-trip (torch-free)
tests/test_repro_torch_free.py               # qquant.repro.* is torch-free in a fresh interpreter
tests/test_ci_workflow.py                    # parse ci.yml + .env-hygiene assertions (torch-free)
tests/test_mmlu_group_expansion.py           # @pytest.mark.lmeval: TaskManager mmlu expansion == {mmlu_<s>}
tests/test_gpu_toolchain.py                  # @pytest.mark.gpu: CUDA available + torch.version.cuda == nvcc
tests/test_spec_consistency.py               # OPTIONAL light gate: spec-table variant/task ids ⊆ VARIANT_IDS/TASK_IDS (torch-free)
```

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **Two-platform CI, torch-free guard on both** (decision-log "Two uv environments"; contracts §7).
  `uv sync --locked` installs only the default + `dev` groups, so the `core` job pulls **no torch**
  on Linux or macOS, and the guard proves `import qquant` (now incl. `qquant.repro.*`) never drags
  `torch` into `sys.modules`. *Why:* orchestration and result auditing must run on a no-GPU laptop;
  the guard would still pass even with torch installed (it checks *import behaviour*, not absence).
- **`set(MMLU_SUBJECTS)` == lm-eval `mmlu` expansion, in CI** (contracts §1; decision-log "MMLU =
  full 57"). The `lm-eval-contract` job runs the assertion against the **locked** lm-eval pin so an
  upstream subject reshuffle fails CI on a CPU runner — the cheapest guardian, complementing Spec
  02's on-box `mmlu_subjects_match` (which checks the *chosen* ref before GPU spend). It uses the
  Spec-02-resolved ref (currently the `0.4.12` fallback; decision-log flags lm-eval × transformers
  v5 as OPEN, issue #3537) — whatever the lock holds.
- **Greedy determinism is contractual** (decision-log "Greedy determinism"; Spec 05). The canonical
  `GREEDY_GEN_KWARGS = {do_sample: False, temperature: 0.0, top_p: 1.0}` (+ `top_k` cleared) is
  written once here and **mirrors Spec 05's enforced values**; Spec 04 clears sampling at load,
  Spec 05 passes it in `gen_kwargs`, and Spec 05's `--determinism-check` is the runtime proof.
  *Why one written value:* MMLU is loglikelihood (greedy-irrelevant) but `generate_until` tasks
  (gsm8k/humaneval/ifeval) are nondeterministic without it, which breaks the paired-McNemar pairing
  (decision-log "Significance").
- **The four lm-eval seeds = `Seeds`** (Spec 01 SSOT; contracts §6). `lm_eval_seed_kwargs(Seeds())`
  is the only mapping to lm-eval's `random_seed/numpy_random_seed/torch_random_seed/
  fewshot_random_seed`; the values live in `qquant.config.Seeds` and are **not** redefined here.
  They are recorded in every cell's `config.seeds` (a `PROVENANCE_KEYS` entry) and in env.json, so
  `is_cell_done` recomputes any cell whose seeds drift.
- **langdetect determinism** (decision-log "IFEval"). `LANGDETECT_SEED = 0` is the contract value
  (Spec 05 sets `langdetect.DetectorFactory.seed` before any ifeval run); recorded in env.json.
  The pre-download fetches `punkt`/`punkt_tab` so IFEval never hits the network mid-run.
- **Fixed per-(variant,task) batch** (decision-log "VRAM budget includes the logits transient").
  `batch_overrides` (MMLU `bf16`/`bnb-int8` = 2, 4-bit = 4) lives in `tasks.yaml`/Spec 01 and is in
  `PROVENANCE_KEYS`, so a batch change invalidates stale cells. `DETERMINISM_PROVENANCE_KEYS` names
  the subset of `PROVENANCE_KEYS` that pins numerical determinism; a CI test asserts it is a subset
  and that `Task.batch_size_for` returns the locked values.
- **Pinned, cached, reproducible inputs** (decision-log "Pinned model revisions", "Dataset
  reproducibility"). The pre-download snapshots **by SHA** straight from the registry
  (`enabled_variants(load_variants())`, skipping `local:` self-quant), so the SSOT — not a hardcoded
  list — decides what is fetched, and the `awq-official` contingency flag automatically governs
  whether its repo is pulled. Datasets are resolved through the **pinned lm-eval** `TaskManager` so
  the pre-downloaded data is exactly what the matrix loads; the resolved revision + `_fingerprint`
  go into env.json. Filling `HF_HOME` once at boot means every cell reuses identical artifacts and
  never re-downloads mid-run. (`destroy` wipes the disk — decision-log "vast.ai" — so this runs per
  boot; self-quant checkpoints are exfil'd/re-uploaded by Spec 07/08, not re-fetched here.)
- **`_meta/env.json` has an owner now.** Contracts §3 names it; Spec 05 defers it to "03/08/11",
  Spec 06/08 *consume* it ("does not define"). This spec defines its **schema + writer**, anchored to
  `Paths.env_file`, with `merge_env` so the boot pre-download, Spec 03's bootstrap (image digest),
  and Spec 08 (cost/ledger) accrete one record without clobbering. **Spec 03's remote `write_env.py`
  is the GPU-side writer** that consumes this module (`build_env`/`add_model`/`add_dataset`/
  `merge_env`/`write_env`) to fill `libraries`/`toolchain` from the live GPU stack — it does NOT
  invent a divergent `env` block. Spec 06 reads it best-effort for library/dataset stamps; nothing
  breaks if a field is absent.
- **Datasets 5.0.0 preflight** (decision-log): the pre-download `load_dataset`s all four tasks under
  the pinned `datasets` (5.0.0 in the lock) to confirm the 4.x loading-script removal doesn't bite,
  and records `loaded_ok` per task.
- **No secrets in a public repo** (decision-log "Repo delivery"; Spec 01). `.env` stays git-ignored
  and `.env.example` carries only blank placeholders + the vast.ai spend-limit reminder; a torch-free
  CI test asserts `.env` is in `.gitignore` and every assignment in `.env.example` is empty, so the
  hygiene cannot silently regress.
- **Spec-consistency is an automatable light check, not a manual gate** (resolves Spec 00's
  downgraded Done-when 7). An **OPTIONAL** `tests/test_spec_consistency.py` parses the spec tables
  and asserts every variant/task id they mention is a **subset** of `VARIANT_IDS`/`TASK_IDS` — the
  cheap, machine-checkable form of "no id drift". The broader cross-spec consistency review stays a
  **non-gating** human note in Spec 00. Torch-free; runs in the `core` CI job.
- **No PDF / LaTeX, no new entry point** (decision-log "Reporting"; contracts §7). This spec emits
  no report and registers no `[project.scripts]` line; the pre-download is a standalone box script.

## Done-when (numbered, testable)

1. `uv run pytest` is green locally and in the `core` CI matrix on **both** `ubuntu-latest` and
   `macos-latest`; `gpu`- and `lmeval`-marked tests report **skipped** (no CUDA / lm-eval absent),
   not failed.
2. The `core` torch-free guard passes with the extended import list: a fresh interpreter importing
   `qquant`, `qquant.repro.env`, and `qquant.repro.determinism` leaves no `torch*` in `sys.modules`
   (`tests/test_repro_torch_free.py` mirrors `tests/test_import_torch_free.py`).
3. `lm_eval_seed_kwargs(Seeds())` equals `{"random_seed": 0, "numpy_random_seed": 1234,
   "torch_random_seed": 1234, "fewshot_random_seed": 1234}`, and `Seeds()` carries
   `(0, 1234, 1234, 1234)` (asserted against `qquant.config.Seeds`, not a local copy).
4. `GREEDY_GEN_KWARGS["do_sample"] is False`, `temperature == 0.0`, `top_p == 1.0`, and
   `LANGDETECT_SEED == 0`; `set(DETERMINISM_PROVENANCE_KEYS) <= set(PROVENANCE_KEYS)`; and
   `load_tasks()["mmlu"].batch_size_for("bf16") == 2`, `…("bnb-int8") == 2`,
   `…("gptq-official") == 4` (the four determinism pillars, all torch-free).
5. `qquant.repro.env`: `build_env(...)` then `add_model`/`add_dataset` produces a dict that
   `validate_env` accepts; `write_env(tmp, env)` writes exactly `Paths.from_root(tmp).env_file`;
   `load_env(tmp)` round-trips it; a second `write_env` with a new model/dataset **merges** (union
   by key) rather than clobbering, and shallow-merges `toolchain`. Output JSON is `sort_keys=True`
   stable across two writes of identical content.
6. `validate_env` rejects a payload missing `schema_version` / `determinism`, and rejects
   `schema_version != 1`.
7. `tests/test_mmlu_group_expansion.py` (`@pytest.mark.lmeval`) asserts the pinned lm-eval's
   `TaskManager` expansion of the `mmlu` group, restricted to `mmlu_*` subtasks, equals
   `{f"mmlu_{s}" for s in MMLU_SUBJECTS}` (57 entries); it runs green in the `lm-eval-contract` CI
   job and is auto-skipped where lm-eval is absent.
8. `tests/test_gpu_toolchain.py` (`@pytest.mark.gpu`) asserts `torch.cuda.is_available()` and that
   `torch.version.cuda` major.minor equals the `nvcc --version` major.minor; it is **skipped**
   (not failed) on CI (no CUDA) and is the repeatable on-box form of the spike's `nvcc_matches_torch`.
9. `tests/test_ci_workflow.py` parses `.github/workflows/ci.yml` (PyYAML, a core dep) and asserts:
   the `core` matrix contains both `ubuntu-latest` and `macos-latest`; its steps include `ruff
   check`, `ruff format --check`, `pytest`, and the torch-free guard; and a `lm-eval-contract` job
   exists that installs the `gpu` group and runs `pytest -m lmeval`.
10. `tests/test_ci_workflow.py` also asserts `.env` is listed in `.gitignore` and that every
    non-comment assignment in `.env.example` has an **empty** right-hand side (no committed secret).
11. `python scripts/repro/predownload.py --help` works on a plain (torch-free) machine; importing
    the package never imports `scripts/repro/predownload.py` (it is not part of `qquant`).
12. **On a real RTX 4090 (the actual cache run):** `predownload.py` populates `HF_HOME` with the
    SHA-pinned model snapshots for the enabled non-local variants and successfully `load_dataset`s
    all four tasks, then writes a `validate_env`-valid `results/_meta/env.json` whose `models[*]`
    record the requested + resolved SHAs and whose `datasets[*]` record `revision`, `fingerprint`,
    `num_rows`, and `loaded_ok == true`; a second matrix run re-uses the cache with no re-download.

## Risks & mitigations

- **`lm-eval-contract` job is heavy (installs torch+lm-eval on a CPU runner).** → `setup-uv` cache;
  CPU-only (no GPU, no dataset download — `TaskManager` group expansion reads bundled YAML); may be
  path-filtered / `workflow_dispatch` if minutes are tight, but must run when the lm-eval pin or
  `MMLU_SUBJECTS` changes. It is a **separate** job, so the fast `core` job stays torch-free.
- **lm-eval × transformers v5 still open (issue #3537).** → CI uses whatever the lock pins (the
  Spec-02-resolved ref, `0.4.12` fallback). If the pin moves, the `lm-eval-contract` job and the
  on-box spike both re-validate the `mmlu` expansion before any large spend.
- **MMLU `TaskManager` API drift across lm-eval versions.** → The expansion test reads the public
  `TaskManager` and filters to `mmlu_*`; if the group/subtask access path changes it fails loudly in
  CI (cheap), pointing at the one line to update — never a silent subject-set mismatch.
- **Conftest importing torch could slow/poison the `core` job.** → `pytest_collection_modifyitems`
  imports torch inside a `try/except` only to *detect* CUDA; on macOS/Linux core (no torch) it
  catches `ImportError` and skips `gpu` tests. It runs in pytest, not the torch-free guard
  subprocess, so the guard is unaffected.
- **env.json multi-writer clobber** (pre-download here, Spec 03 image digest, Spec 08 ledger). →
  `merge_env` unions models/datasets by key and shallow-merges metadata; `write_env` read-merges an
  existing file, so order of writers does not matter.
- **Dataset fingerprint instability** (datasets recomputes `_fingerprint` across versions). → The
  record is pinned to the env.json `datasets_version`; resume-time numerical comparison uses
  **cell** `config` provenance (`PROVENANCE_KEYS`/`is_cell_done`), not the env.json fingerprint,
  which is an audit datum. Hub-commit `revision` is recorded when available as the stable anchor.
- **`destroy` wipes `HF_HOME`** (decision-log). → Pre-download is a **per-boot** step (called by
  Spec 03's bootstrap); SHA pins make every re-download byte-reproducible. Self-quant checkpoints
  are persisted/re-uploaded by Spec 07/08 so they are never re-quantized.
- **A future edit silently weakens `.env` hygiene or drops a CI step.** → `tests/test_ci_workflow.py`
  is the regression gate for both the workflow shape and the `.env`/`.env.example` invariants.
- **Drift between `GREEDY_GEN_KWARGS` here and Spec 05's enforced values.** → They are documented as
  identical and Done-when #4 pins the values; the end-of-authoring consistency-review gate
  (contracts §8) is the cross-spec backstop. The runtime proof remains Spec 05's `--determinism-check`.
