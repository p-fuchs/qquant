# Spec 03 — Remote bootstrap scripts (onstart / bootstrap / run_matrix)

- **Status:** Draft (ready to implement; runs ON the vast.ai box, gated by the Spec 02 GO verdict)
- **Depends on:** 02 (vast.ai lifecycle wrapper + go/no-go GPU spike), 11 (the
  `qquant.repro.env` module + `_meta/env.json` schema this spec's `write_env.py` consumes).
  Transitively uses the Spec-01 torch-free core (`qquant.registry`, `qquant.matrix`,
  `qquant.paths`, `qquant.config`) for all matrix/resume logic.
- **Owns / Produces:** the on-instance shell harness under `scripts/remote/` —
  `onstart.sh` (the `<16 KB` vast `--onstart-cmd` payload), `bootstrap.sh` (env install +
  fail-fast toolchain check), `run_matrix.sh` (the phase runner), the shared bash helper
  `lib.sh`, and two **standalone** Python helpers it shells out to: `qquant_state.py`
  (torch-free matrix/manifest reuse) and `write_env.py` (GPU-env provenance capture). It
  **emits** `<results>/_meta/done_manifest.json` (per `contracts.md §3`, via
  `Paths.done_manifest`) and the **completion sentinel** `<results>/_meta/RUN_COMPLETE`, and
  **seeds** the `libraries` + `toolchain` provenance of `<results>/_meta/env.json`
  (`Paths.env_file`) **via Spec 11's `qquant.repro.env`** (the env.json shape is owned by Spec 11).

> **Ownership boundaries (no double-create).** `scripts/remote/deadman.sh` is **owned by
> Spec 08** (the orchestration driver lists it in its Files-to-create); this spec only
> *launches* it from `onstart.sh` if present. The per-(variant,task) **cell files** and the
> per-variant `efficiency.json` are written **atomically by Spec 05 / Spec 06** (`write_cell` /
> the efficiency writer); `run_matrix.sh` invokes those entry points and never writes a cell
> itself. This spec references every SSOT contract by name and re-declares none.

## Purpose

Turn a freshly-rented, bare RTX 4090 container into a reproducible eval box and run the whole
matrix on it, with **zero interactive steps** and **safe resume**. Three scripts, chained:

1. **`onstart.sh`** — the tiny payload vast.ai runs at boot (passed verbatim as the Spec-02
   `create_instance(..., onstart_cmd=…)` string, so it must be **self-contained and `<16 KB`**).
   It clones the **public** GitHub repo over HTTPS at a pinned ref, launches the dead-man's
   switch (Spec 08's `deadman.sh`) if available, and hands off to `bootstrap.sh`.
2. **`bootstrap.sh`** — installs `uv`, runs `uv sync --group gpu` (the Linux/CUDA eval stack),
   **fails fast** if `nvcc` major.minor ≠ `torch.version.cuda`, pre-downloads the nltk
   `punkt`/`punkt_tab` data IFEval needs, points the HF cache at the big disk, and exports the
   three run-time env vars the evaluators require — `HF_ALLOW_CODE_EVAL=1`,
   `QQUANT_ALLOW_CODE_EXEC=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — into a
   persisted runtime env file. It seeds `_meta/env.json` with the resolved library / CUDA /
   image / repo provenance.
3. **`run_matrix.sh`** — runs the **quality phase** (`qquant-eval`, Spec 05) and the
   **efficiency phase** (`qquant-profile`, Spec 06) over the official-core variants, and — when
   enabled — the **self-quant production + eval phase** (Spec 07's `quant/` env + then
   eval/profile of the two self-quant variants). It rebuilds `_meta/done_manifest.json` after
   each phase and on every exit (so the orchestrator's incremental copy always sees a fresh
   manifest), and writes the `_meta/RUN_COMPLETE` sentinel **strictly last**, only when the
   targeted matrix has zero missing cells.

The filesystem is the only state (per `contracts.md §3/§5`); resume is decided **solely** by
`is_cell_done` / `is_efficiency_done` inside the invoked entry points, so re-running any script
is safe and cheap.

## Scope (in / out)

**In scope**

- The four shell scripts (`onstart.sh`, `bootstrap.sh`, `run_matrix.sh`, `lib.sh`) and their
  env-var contract, idempotency markers, logging, bounded retries, and atomic `_meta` writes.
- The fail-fast `nvcc` vs `torch.version.cuda` toolchain gate; nltk data download; HF cache +
  the three exported env vars; `uv sync --group gpu` (with the gptqmodel `--no-build-isolation`
  caveat).
- The phase orchestration (quality → efficiency → optional self-quant), phase gating via env
  flags, and the completion contract: `done_manifest.json` (every exit) then `RUN_COMPLETE`
  (only when complete).
- Two standalone helpers: `qquant_state.py` (torch-free; reuses `expand_matrix`,
  `enabled_variants`, `missing_cells`, `is_cell_done`, `cell_path`, `Paths.done_manifest`) and
  `write_env.py` (GPU-env; introspects installed versions to seed `_meta/env.json`).
- Static + behavioral tests runnable GPU-free on CI (macOS/ubuntu).

**Out of scope (owned elsewhere; invoked or referenced, never reimplemented)**

- vast.ai lifecycle (search/create/poll/ssh/copy/destroy), the GO/NO-GO spike, and the resolved
  v5-working lm-eval ref → **Spec 02** (`onstart.sh` is injected by Spec 02/08 as `onstart_cmd`).
- The dead-man's switch script `deadman.sh`, the local orchestrator, `BudgetGuard`, the cost
  ledger, incremental copy-out / reconcile, the local watchdog, and the choice of *when* to
  launch `run_matrix.sh` → **Spec 08** (`onstart.sh` only *launches* `deadman.sh`).
- Model loading / quant dispatch / greedy-forcing → **Spec 04**; quality cells + `write_cell`
  + lm-eval policy → **Spec 05**; efficiency artifact + `qquant-profile` → **Spec 06**;
  self-quant checkpoint production (`selfquant.quantize`) + calibration → **Spec 07**.
- Aggregation / stats / `REPORT.md` → **Spec 09/10**. Pinning the spike's resolved lm-eval ref
  into the `gpu` group + the determinism/dataset CI contract → **Spec 11**.
- Re-declaring any id / path / schema / predicate → all imported from Spec 01 (`contracts.md`).

## Interface

All paths below are relative to `$QQUANT_HOME` (the cloned repo root) on the box. Scripts use
`#!/usr/bin/env bash` + `set -Eeuo pipefail`.

### Environment-variable contract (injected by the orchestrator via vast `--env`, with defaults)

| Var | Default | Read by | Meaning |
|---|---|---|---|
| `QQUANT_REPO_URL` | (required) | onstart | public GitHub HTTPS clone URL (no secrets) |
| `QQUANT_GIT_REF` | (required) | onstart | commit SHA / branch to check out (must be pushed) |
| `QQUANT_HOME` | `/workspace/qquant` | all | clone target / working root (on the 120 GB disk) |
| `QQUANT_RESULTS` | `$QQUANT_HOME/results` | bootstrap, run_matrix | **remote results root** (SSOT `Paths.from_root`) |
| `HF_HOME` | `/workspace/hf_cache` | bootstrap | HF cache on the big disk (not the small root fs) |
| `NLTK_DATA` | `/workspace/nltk_data` | bootstrap | nltk data dir (`punkt`/`punkt_tab`) |
| `HF_TOKEN` | (unset) | bootstrap | optional; Qwen2.5 is public — only for gated assets |
| `QQUANT_IMAGE_DIGEST` | (unset) | bootstrap | digest of the pinned CUDA 12.6 *devel* image, for provenance |
| `QQUANT_MANIFEST` | (unset) | run_matrix | path to a Spec-05 eval manifest (variant/task pairs); else full enabled matrix |
| `QQUANT_VARIANTS` | (unset) | run_matrix | optional comma list of canonical variant ids to restrict to |
| `QQUANT_TASKS` | (unset) | run_matrix | optional comma list of `mmlu,gsm8k,humaneval,ifeval` to restrict to |
| `QQUANT_RUN_QUALITY` | `1` | run_matrix | run the quality phase |
| `QQUANT_RUN_EFFICIENCY` | `1` | run_matrix | run the efficiency phase |
| `QQUANT_RUN_SELFQUANT` | `0` | run_matrix | run the self-quant production+eval phase (deferrable fast-follow) |
| `QQUANT_AUTORUN` | `0` | onstart | if `1`, onstart chains `run_matrix.sh` (detached) after bootstrap; else it signals readiness and Spec 08 triggers the run |
| `QQUANT_MAX_RUNTIME_S` | `21600` | onstart→deadman | dead-man's-switch TTL (Spec 08's `deadman.sh` reads it) |
| `VAST_API_KEY` | (unset) | onstart→deadman | injected secret for the dead-man's switch self-destroy |
| `QQUANT_FORCE_BOOTSTRAP` | `0` | bootstrap | ignore the `.bootstrap-ok` marker and re-install |

These names resolve Spec 08's open questions on the **remote results root** (`QQUANT_RESULTS`),
the **target-cell handoff** (`QQUANT_MANIFEST` primary; `QQUANT_VARIANTS`/`QQUANT_TASKS`
restrictions; resume correctness comes from `is_cell_done` regardless), and the **completion
sentinel** (`_meta/RUN_COMPLETE`).

### `scripts/remote/onstart.sh` — the `<16 KB` boot payload (self-contained)

```
# Injected verbatim by Spec 02/08 as create_instance(onstart_cmd=<contents of onstart.sh>).
# MUST NOT depend on any repo file before the clone (defines its own log()/retry() inline).
1. set -Eeuo pipefail; redirect all output to /var/log/onstart.log AND $QQUANT_HOME/_logs/onstart.log (tee).
2. Defaults for QQUANT_HOME / QQUANT_GIT_REF etc.; require QQUANT_REPO_URL + QQUANT_GIT_REF (else exit 2).
3. Ensure git (apt-get install -y git || true if absent on the image).
4. retry(5): clone-or-update —
     if [ -d "$QQUANT_HOME/.git" ]: git -C fetch --depth=1 origin "$QQUANT_GIT_REF" && git checkout --detach FETCH_HEAD
     else: git clone --depth=1 "$QQUANT_REPO_URL" "$QQUANT_HOME" && git -C checkout --detach "$QQUANT_GIT_REF"
   (idempotent across container reboots; pins to an exact ref for reproducibility).
5. If scripts/remote/deadman.sh exists AND VAST_API_KEY+QQUANT_MAX_RUNTIME_S set AND no live deadman pidfile:
     setsid bash scripts/remote/deadman.sh >>$QQUANT_HOME/_logs/deadman.log 2>&1 &   # Spec 08 owns deadman.sh
   else log "dead-man's switch unavailable — relying on Spec 08 local handlers + watchdog + account spend limit".
6. exec bash scripts/remote/bootstrap.sh        # hand off; bootstrap is idempotent.
7. If QQUANT_AUTORUN=1: bootstrap tail-calls run_matrix (see bootstrap step 9); else stop after bootstrap
   and let the orchestrator (Spec 08 LAUNCH step) ssh-exec run_matrix.sh.
```

`onstart.sh` writes the readiness marker `$QQUANT_HOME/.bootstrap-ok` (via bootstrap) that the
orchestrator polls with `ssh_exec "test -f …"`.

### `scripts/remote/bootstrap.sh` — env install + fail-fast toolchain gate (idempotent)

```
source scripts/remote/lib.sh   # log/retry/atomic_write/require_env/load_runtime_env

1. If [ -f "$QQUANT_HOME/.bootstrap-ok" ] && [ "$QQUANT_FORCE_BOOTSTRAP" != 1 ]: load_runtime_env; exit 0.
2. Install uv (retry: curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH="$HOME/.local/bin:$PATH".
3. cd "$QQUANT_HOME"; retry(3): uv sync --group gpu --locked
     # gptqmodel JIT-compiles sm_89 kernels at first use (runtime), needs nvcc; its install may need
     # build isolation off — use `uv sync --group gpu --no-build-isolation-package gptqmodel` if the
     # plain sync fails to build gptqmodel (torch must be importable at its build step). See notes.
4. FAIL-FAST toolchain gate:
     nvcc_cuda=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
     torch_cuda=$(uv run python -c 'import torch;print(torch.version.cuda)')
     [ "$nvcc_cuda" = "$torch_cuda" ] || die "nvcc=$nvcc_cuda != torch.version.cuda=$torch_cuda (wrong image)"
5. HF cache on the big disk: export HF_HOME (mkdir -p); optional HF_HUB_ENABLE_HF_TRANSFER=1.
6. nltk data (IFEval): retry: uv run python -c "import nltk;[nltk.download(p) for p in ('punkt','punkt_tab')]"
     into $NLTK_DATA.
7. Write the persisted runtime env file $QQUANT_HOME/.qquant-runtime.sh with EXACTLY:
     export PATH, HF_HOME, NLTK_DATA, HF_TOKEN(if set),
            HF_ALLOW_CODE_EVAL=1, QQUANT_ALLOW_CODE_EXEC=1,
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
            and a `. "$QQUANT_HOME/.venv/bin/activate"` (or `uv run` is used directly).
     This file is the single source of the run-time environment; run_matrix sources it first.
8. Seed provenance: uv run python scripts/remote/write_env.py --results "$QQUANT_RESULTS" \
            --image-digest "$QQUANT_IMAGE_DIGEST" --repo "$QQUANT_HOME"   # merges into _meta/env.json
9. atomic touch "$QQUANT_HOME/.bootstrap-ok"; if QQUANT_AUTORUN=1: exec bash scripts/remote/run_matrix.sh.
```

`PYTORCH_CUDA_ALLOC_CONF` must be set **before** CUDA init; because it lives in the sourced
runtime env file and `run_matrix.sh` sources that file before spawning any GPU process, every
`qquant-eval`/`qquant-profile` process inherits it at launch (Spec 06 only *observes* it).

### `scripts/remote/run_matrix.sh` — phase runner + completion contract (resumable)

```
source scripts/remote/lib.sh; load_runtime_env   # fails if .qquant-runtime.sh absent (bootstrap incomplete)
require_env QQUANT_RESULTS
RESULTS="$QQUANT_RESULTS"; mkdir -p "$RESULTS/_meta/logs"

# Always refresh the manifest on ANY exit so the orchestrator's incremental copy sees fresh state:
trap 'uv run python scripts/remote/qquant_state.py done-manifest --results "$RESULTS" || true' EXIT

CORE=$(uv run python scripts/remote/qquant_state.py enabled-variants --source core   $RESTRICT)
SELF=$(uv run python scripts/remote/qquant_state.py enabled-variants --source selfquant $RESTRICT)
# $RESTRICT = --variants "$QQUANT_VARIANTS" (optional). qquant-eval/-profile do the per-cell resume.

# --- Phase 1: QUALITY (official core) ---
if [ "$QQUANT_RUN_QUALITY" = 1 ]; then
  log_phase quality
  uv run qquant-eval --results "$RESULTS" ${QQUANT_MANIFEST:+--manifest "$QQUANT_MANIFEST"} \
        $(variants_to_flags "$CORE") $(tasks_to_flags "$QQUANT_TASKS") 2>&1 | tee "$RESULTS/_meta/logs/quality.log"
  refresh_manifest
fi

# --- Phase 2: EFFICIENCY (one fresh process per variant — Spec 06 requirement) ---
if [ "$QQUANT_RUN_EFFICIENCY" = 1 ]; then
  log_phase efficiency
  for v in $CORE; do
    uv run qquant-profile --variant "$v" --results "$RESULTS" 2>&1 | tee "$RESULTS/_meta/logs/eff-$v.log" || true
  done
  refresh_manifest
fi

# --- Phase 3: SELF-QUANT production + eval (deferrable fast-follow) ---
if [ "$QQUANT_RUN_SELFQUANT" = 1 ] && [ -n "$SELF" ]; then
  log_phase selfquant
  ( cd "$QQUANT_HOME/quant" && retry(2): uv sync --locked && uv run python -m selfquant.quantize --id all )
  # selfquant.quantize is idempotent (Spec 07 selfquant_checkpoint_done) → resume never re-quantizes.
  uv run qquant-eval --results "$RESULTS" $(variants_to_flags "$SELF") $(tasks_to_flags "$QQUANT_TASKS") \
        2>&1 | tee "$RESULTS/_meta/logs/quality-selfquant.log"
  for v in $SELF; do
    uv run qquant-profile --variant "$v" --results "$RESULTS" 2>&1 | tee "$RESULTS/_meta/logs/eff-$v.log" || true
  done
  refresh_manifest
fi

# --- Completion contract ---
refresh_manifest                                   # final done_manifest.json (atomic)
if uv run python scripts/remote/qquant_state.py complete --results "$RESULTS" $RESTRICT; then
  atomic_write "$RESULTS/_meta/RUN_COMPLETE" "$(date -u +%FT%TZ)"   # DONE sentinel — STRICTLY LAST
  exit 0
else
  log "matrix incomplete — RUN_COMPLETE NOT written; orchestrator will resume"; exit 1
fi
```

`refresh_manifest` = `uv run python scripts/remote/qquant_state.py done-manifest --results "$RESULTS"`.
The DONE sentinel (`_meta/RUN_COMPLETE`) is written **after** the final manifest and **only** on a
complete targeted matrix; Spec 08 treats either the sentinel **or** a non-`running`
`actual_status` as terminal.

### `scripts/remote/lib.sh` — shared bash helpers (sourced post-clone)

```
log MSG                       # timestamped stderr line
retry N CMD...                # bounded retries with exponential backoff (never infinite)
require_env NAME...           # die 2 if any is empty
atomic_write PATH CONTENT     # write to PATH.tmp.$$ then mv -f (atomic, same filesystem)
load_runtime_env              # source $QQUANT_HOME/.qquant-runtime.sh or die "bootstrap incomplete"
variants_to_flags "a b c"     # -> --variant a --variant b --variant c   (matches qquant-eval CLI)
tasks_to_flags "a,b"          # -> --task a --task b   (empty -> no flags = full set)
log_phase NAME                # banner + start time into _meta/logs
die CODE MSG                  # log + exit CODE
```

### `scripts/remote/qquant_state.py` — standalone **torch-free** matrix/manifest helper

Reuses the SSOT (`qquant.registry`, `qquant.matrix`, `qquant.paths`) — it builds **no** ids,
paths, or predicates of its own. Importable on macOS; never pulls `torch`.

```
python scripts/remote/qquant_state.py done-manifest --results DIR
   # scan: cells = expand_matrix(enabled_variants(load_variants()), load_tasks().values());
   # done  = [c.cell_id for c in cells if c not in missing_cells(cells, DIR)]   # i.e. is_cell_done is True
   # atomically write Paths.from_root(DIR).done_manifest (see "File formats").

python scripts/remote/qquant_state.py enabled-variants [--source core|selfquant|all] [--variants a,b]
   # print enabled variant ids (canonical, one per line); "core" = source != "selfquant".

python scripts/remote/qquant_state.py complete --results DIR [--variants a,b] [--tasks a,b]
   # exit 0 iff missing_cells(active_matrix, DIR) == [] for the (optionally restricted) enabled matrix; else 1.
```

### `scripts/remote/write_env.py` — GPU-side `_meta/env.json` writer (consumes `qquant.repro.env`)

Runs in the eval venv (imports torch/transformers/etc. to read versions — **not** torch-free,
by design). It is the **GPU-side writer** for `_meta/env.json`, whose schema + builder/merge are
**owned by Spec 11** (`qquant.repro.env`: `build_env` / `add_model` / `add_dataset` / `merge_env` /
`write_env`). It does **not** invent a divergent `env` block — it introspects the GPU stack, fills
the Spec-11 `libraries` + `toolchain` fields, and emits the Spec-11 shape via
`qquant.repro.env.write_env` (which read-merges `Paths.from_root(results).env_file`, never
clobbering keys other specs own).

```
python scripts/remote/write_env.py --results DIR --image-digest D --repo PATH
   # builds a Spec-11-shaped env dict and MERGES it via qquant.repro.env.merge_env/write_env:
   #   libraries = {torch, transformers, lm_eval, gptqmodel, bitsandbytes, optimum,
   #                compressed_tensors, accelerate, datasets, numpy versions}
   #   toolchain = {torch_cuda (torch.version.cuda), nvcc_cuda, gpu_name, image_digest,
   #                repo_commit (git rev-parse HEAD), uv_lock_sha256}
   # Spec 05/07 later add per-task / calibration dataset records via add_dataset (the `datasets`
   # list). The env.json shape (schema_version/determinism/libraries/toolchain/models/datasets)
   # is owned by Spec 11.
```

### File formats this spec writes (locations are SSOT `Paths`, contents are this spec's contract)

- **`<results>/_meta/done_manifest.json`** (`Paths.done_manifest`) — the remote's view of
  completed cells; refreshed on every `run_matrix.sh` exit:
  ```json
  {
    "schema_version": 1,
    "generated_at": "2026-06-22T12:00:00Z",
    "results_root": "results",
    "complete": false,
    "n_done": 312,
    "n_total": 420,
    "done": ["bf16/mmlu/anatomy", "bf16/gsm8k", "gptq-official/humaneval"]
  }
  ```
  The **load-bearing** field is `done`: a list of `cell_id` strings formatted **exactly** as
  `qquant.matrix.Cell.cell_id` (`<variant>/<task>[/<subject>]`). Spec 08's
  `read_done_manifest(local_results) -> list[str]` reads this `done` array; reconciliation
  re-checks each local file with `is_cell_done` (the manifest is a cross-check, not a substitute
  for the SSOT predicate).
- **`<results>/_meta/RUN_COMPLETE`** — the completion sentinel; a one-line UTC ISO-8601
  timestamp; present **iff** the targeted matrix has zero missing cells. Written strictly after
  the final `done_manifest.json`.
- **`<results>/_meta/env.json`** (`Paths.env_file`) — its `libraries` + `toolchain`
  (CUDA/image/repo) provenance is seeded here via Spec 11's `qquant.repro.env`; the `datasets`
  records are merged by Spec 05/07. **The env.json shape is owned by Spec 11.**
- **`<results>/_meta/logs/*.log`** — per-phase logs (under the results root so the orchestrator's
  incremental copy exfiltrates them for debugging).

## Files to create

```
scripts/remote/onstart.sh         # <16 KB self-contained boot payload (clone + launch deadman + handoff)
scripts/remote/bootstrap.sh       # uv install, uv sync --group gpu, nvcc==torch.cuda gate, nltk, env exports, env.json seed
scripts/remote/run_matrix.sh      # quality / efficiency / self-quant phases; done_manifest each exit; RUN_COMPLETE last
scripts/remote/lib.sh             # shared bash helpers (log/retry/atomic_write/require_env/load_runtime_env/*_to_flags)
scripts/remote/qquant_state.py    # standalone TORCH-FREE: done-manifest / enabled-variants / complete (reuses SSOT)
scripts/remote/write_env.py       # standalone GPU-env: seed _meta/env.json env/libs/image/repo block (merge)
tests/test_remote_scripts.py      # static + behavioral, GPU-free (see Done-when)
```

**Do not create** `scripts/remote/deadman.sh` (Spec 08 owns it), any registry/contract/schema,
or a new `[project.scripts]` line (this spec adds **none** — it invokes the `qquant`,
`qquant-eval`, `qquant-profile` entry points that Spec 01/05/06 register, falling back to
`python -m` if an entry point is not yet on `PATH`).

## Implementation notes (the decision-log points that bind THIS spec, with the "why")

- **Public repo, no secrets, pinned ref.** The box clones over HTTPS at `QQUANT_GIT_REF`;
  `VAST_API_KEY`/`HF_TOKEN` arrive only as injected env, never on disk in the image. Pinning an
  exact ref makes the remote run bit-reproducible. *(decision-log → Platform / Repo delivery.)*
- **`onstart.sh` is the `<16 KB` vast payload.** vast.ai caps the `--onstart-cmd` string, and
  Spec 02's `create_instance` passes that string verbatim — so `onstart.sh` must be small and
  **self-contained** (it cannot source `lib.sh` until *after* it clones the repo). A test asserts
  `wc -c < onstart.sh < 16384`. *(Spec 02 `create_instance(onstart_cmd=…)`.)*
- **CUDA 12.6 *devel* image + nvcc fail-fast.** gptqmodel JIT-compiles Marlin/AWQ sm_89 kernels
  at first use, so `nvcc` must be present and `nvcc` major.minor **must equal**
  `torch.version.cuda` (cu126 → "12.6"); bootstrap dies immediately otherwise — catching an
  image/toolchain mismatch before any expensive load. bitsandbytes ships prebuilt sm_89 cubins
  and needs no nvcc. *(decision-log → vast.ai.)*
- **Two uv envs.** bootstrap uses the **root** eval env (`uv sync --group gpu --locked`); the
  **quant** env (`quant/`, `llmcompressor==0.12.0`) is synced and used **only** inside the
  self-quant phase (`cd quant && uv run python -m selfquant.quantize`), exactly because its
  tight `transformers<=5.10.1` pins cannot co-resolve with the eval stack. *(decision-log → Two
  uv environments; Spec 07.)*
- **gptqmodel `--no-build-isolation`.** If `uv sync` cannot build gptqmodel because torch is not
  visible at build time, bootstrap retries with `--no-build-isolation-package gptqmodel` (never a
  blanket `--no-build-isolation`, which would break other builds). The pinned `torch 2.12.1+cu126`
  comes from the locked root env (reproducible), not the image's torch. **NEVER install autoawq.**
  *(decision-log → root eval env / NEVER autoawq.)*
- **The three exported env vars are real run-time requirements.** Spec 05 skips HumanEval unless
  **both** `HF_ALLOW_CODE_EVAL=1` **and** `QQUANT_ALLOW_CODE_EXEC=1` are set (the disposable
  container is the sandbox; the second switch prevents accidental code-exec on a dev box); Spec
  05/06 rely on `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` being set *before* CUDA init to
  survive the logits transient (`batch×seq×vocab(152064)×2B` ≈ 3.5 GB). bootstrap puts all three
  in the persisted runtime env file that `run_matrix.sh` sources before any GPU process starts.
  *(decision-log → HumanEval code-exec / VRAM budget.)*
- **IFEval nltk data.** bootstrap pre-downloads `punkt` **and** `punkt_tab` into `$NLTK_DATA`;
  Spec 05 only *asserts* their presence and sets `langdetect.DetectorFactory.seed = 0` itself.
  *(decision-log → IFEval.)*
- **Per-task chat-template / greedy / batch policy is data, not bash.** `run_matrix.sh` never
  encodes the MMLU/GSM8K/IFEval-vs-HumanEval template policy, the greedy `gen_kwargs`, or the
  per-(variant,task) batch sizes — those live in `tasks.yaml` and are applied by `qquant-eval`
  (Spec 05). run_matrix only selects *which* variants/tasks and *when*. *(decision-log →
  Per-task chat-template policy; contracts §2.)*
- **MMLU = 57 resumable cells; resume is `is_cell_done` only.** run_matrix delegates all
  resume/skip logic to `qquant-eval`/`qquant-profile` (which call `missing_cells` /
  `is_efficiency_done`); `qquant_state.py` reuses `expand_matrix` + `is_cell_done` + `cell_path`
  to build the manifest and the completeness gate. No parallel `completed:true` flag is invented.
  *(contracts §5/§6; decision-log → MMLU.)*
- **Self-quant is a deferrable fast-follow at the end.** The quality and efficiency phases cover
  the **official core** so a run that drops the self-quant phase (`QQUANT_RUN_SELFQUANT=0`) is
  still a complete, valid v1-core result; the self-quant phase produces W4A16_ASYM checkpoints
  (Spec 07) then evaluates the two compressed-tensors variants. Checkpoints are persisted and
  idempotent (`selfquant_checkpoint_done`) so a resume never re-quantizes; Spec 08 exfiltrates +
  re-uploads them. *(decision-log → Scope / Persist self-quant checkpoints; Spec 07.)*
- **Efficiency = one fresh process per variant.** `qquant-profile` is invoked once per variant in
  the phase loop because gptqmodel JIT buffers and CUDA peak-memory counters are process-global —
  reusing a process would destroy the apples-to-apples comparison. *(Spec 06; decision-log →
  single homogeneous GPU.)*
- **Atomicity & the completion contract.** Per-cell writes are atomic in **Spec 05/06**
  (`write_cell` = tmp sibling + `os.replace`); a truncated cell is treated as missing by
  `is_cell_done` and recomputed, so a killed instance is always resumable. `run_matrix.sh` writes
  its own `_meta` artifacts atomically (`atomic_write`), refreshes `done_manifest.json` on **every
  exit** (EXIT trap) so the orchestrator's incremental copy never reconciles against a stale
  manifest, and writes `RUN_COMPLETE` **strictly last**. *(decision-log → Incremental copy /
  reconcile before destroy; Spec 05/08.)*
- **Dead-man's switch is launched, not owned, here.** `onstart.sh` backgrounds Spec 08's
  `scripts/remote/deadman.sh` (`setsid`, idempotent via a pidfile) with `VAST_API_KEY` +
  `QQUANT_MAX_RUNTIME_S` in env, as the on-box layer of the auto-destroy defense-in-depth (local
  finally/atexit/signal handlers + on-box dead-man's switch + local watchdog + account spend
  limit). If `deadman.sh` is absent (Spec 08 not yet landed), onstart logs a clear warning and
  proceeds. *(decision-log → Auto-destroy defense-in-depth; Spec 08.)*
- **Provenance seeding (Spec-11 shape).** bootstrap runs `write_env.py`, which fills
  `_meta/env.json`'s `libraries` + `toolchain` fields (resolved library versions,
  `torch.version.cuda`, `nvcc`, gpu name, image digest, repo commit, `uv.lock` hash) **via Spec
  11's `qquant.repro.env`** — exactly the data Spec 06 reads best-effort and Spec 09 needs for the
  report; dataset records are merged later (Spec 05/07). The env.json shape is owned by Spec 11.
  *(contracts §3; Spec 11; decision-log → Dataset reproducibility.)*
- **lm-eval × transformers-v5 ref.** bootstrap installs whatever the locked `gpu` group pins for
  `lm-eval` — i.e. the spike-blessed ref (Spec 02 emits it; Spec 11 pins it). run_matrix is
  agnostic to that value; only `RunConfig.lm_eval_version` (captured by Spec 05 in each cell's
  provenance) records it. *(decision-log → lm-eval; Spec 02/11.)*

## Done-when (numbered, testable)

1. `bash -n` parses all of `onstart.sh`, `bootstrap.sh`, `run_matrix.sh`, `lib.sh` with no syntax
   error; `shellcheck` (when available on the runner) reports no errors (warnings allowed). Each
   script begins `#!/usr/bin/env bash` + `set -Eeuo pipefail`.
2. `wc -c < scripts/remote/onstart.sh` is `< 16384`, and `onstart.sh` contains **no**
   `source .../lib.sh` (or `. .../lib.sh`) before its clone step (asserted by ordering in the
   file) — it is self-contained until the repo exists.
3. `bootstrap.sh` contains, verbatim, the fail-fast comparison of `nvcc` release vs
   `torch.version.cuda` that **exits non-zero on mismatch**; `uv sync --group gpu`; the nltk
   `punkt` **and** `punkt_tab` download; and exports of all three of `HF_ALLOW_CODE_EVAL=1`,
   `QQUANT_ALLOW_CODE_EXEC=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` into the
   persisted runtime env file (asserted by grep over the file).
4. `bootstrap.sh` is idempotent: a `.bootstrap-ok` marker short-circuits the heavy steps
   (re-sourcing the runtime env and exiting 0) unless `QQUANT_FORCE_BOOTSTRAP=1` (asserted by the
   guard's presence and a fake-`uv` dry harness, or by static check).
5. `run_matrix.sh` invokes `qquant-eval` and `qquant-profile` (one `--variant` per profile call),
   gates each phase on its `QQUANT_RUN_*` flag, and routes the self-quant phase through
   `cd quant && … selfquant.quantize` before evaluating the self-quant variants (asserted by
   static check of the phase blocks + flag gating).
6. `run_matrix.sh` refreshes `done_manifest.json` in an `EXIT` trap (fires on success **and**
   failure) and writes `_meta/RUN_COMPLETE` **after** the final manifest, **only** when the
   completeness check passes (asserted by ordering: the `RUN_COMPLETE` write appears after the
   final `done-manifest` call and inside the success branch of the `complete` check).
7. `python scripts/remote/qquant_state.py done-manifest --results <fixture>` writes a
   `_meta/done_manifest.json` at `Paths.from_root(...).done_manifest` whose `done` array equals
   the set of `cell_id`s that `is_cell_done` accepts in the fixture tree (one valid + one
   truncated + one missing cell → only the valid one listed), with correct `n_done`/`n_total`.
8. `python scripts/remote/qquant_state.py enabled-variants --source core` prints the enabled,
   non-`selfquant` variant ids (canonical, hyphenated) and `--source selfquant` prints exactly
   `gptq-selfquant`/`awq-selfquant` when enabled; `complete --results <fixture>` exits `0` iff the
   enabled matrix has zero missing cells (1 otherwise) — both verified on fixtures.
9. **Torch-free guard:** a fresh-interpreter test (mirroring `tests/test_import_torch_free.py`)
   confirms importing/executing `scripts/remote/qquant_state.py` pulls **no** `torch` into
   `sys.modules`; the existing core torch-free guard remains green. (`write_env.py` is *exempt* —
   it intentionally introspects the GPU stack and is never imported by the torch-free core.)
10. The `done_manifest.json` produced validates against the documented shape (keys
    `schema_version, generated_at, results_root, complete, n_done, n_total, done`), and every
    entry in `done` matches `Cell.cell_id` formatting (`<variant>/<task>[/<subject>]`, hyphenated
    variant ids, bare-slug subjects) — asserted against a fixture.
11. `ruff check .` and `ruff format --check .` pass for `qquant_state.py`/`write_env.py`/the test;
    `pytest -q` is green on macOS/ubuntu with no GPU, no `vastai`, and no network (fakes/fixtures
    only).
12. **On a real RTX 4090 (the integration gate):** `onstart.sh` clones the repo, `bootstrap.sh`
    passes the `nvcc==torch.cuda` gate and writes `.bootstrap-ok` + the runtime env file, and
    `run_matrix.sh` produces schema-valid cells under `<results>/<variant>/…`, a fresh
    `_meta/done_manifest.json`, and — on a fully-resumed/complete matrix — `_meta/RUN_COMPLETE`;
    re-running `run_matrix.sh` recomputes nothing (every targeted cell already `is_cell_done`).

## Risks & mitigations

- **`onstart.sh` exceeds vast's payload cap.** → Keep it minimal + self-contained (clone + launch
  deadman + handoff); the size test (Done-when 2) fails the build before deploy; all heavy logic
  lives in repo files run *after* the clone.
- **Wrong/drifted image (`nvcc` ≠ torch CUDA).** → bootstrap's **first** post-install action is the
  fail-fast gate; it dies with a clear message rather than JIT-compiling against a mismatched
  toolchain mid-matrix. Image is digest-pinned by Spec 02/08.
- **gptqmodel build failure.** → bounded retry, then `--no-build-isolation-package gptqmodel` with
  torch already installed; surface a clear failure (not a hang). autoawq is never installed.
- **Flaky network (clone / `uv sync` / HF / nltk).** → every network step is wrapped in bounded
  `retry` with backoff (never infinite); HF cache + datasets land on the 120 GB disk; failures
  abort the phase but the EXIT-trap manifest refresh + Spec 08 incremental copy preserve partial
  results for a cheap resume.
- **Instance killed mid-write.** → cells are written atomically by Spec 05/06; a partial file is
  treated as missing by `is_cell_done` and recomputed; `done_manifest.json` is refreshed on every
  exit so the orchestrator reconciles against current truth; `RUN_COMPLETE` only ever appears when
  the matrix is genuinely complete.
- **Stranded instance / runaway cost.** → `onstart.sh` launches Spec 08's `deadman.sh` (on-box TTL
  self-destroy); the rest of the defense-in-depth (local handlers, watchdog, account spend limit)
  is Spec 08. If `deadman.sh` is absent, onstart warns and the other three layers still cover it.
- **Self-quant phase re-quantizes on resume.** → `selfquant.quantize` is idempotent
  (`selfquant_checkpoint_done`, Spec 07); Spec 08 exfiltrates + re-uploads `checkpoints/self-quant/`
  so a resume reuses checkpoints; the phase is fully skippable (`QQUANT_RUN_SELFQUANT=0`).
- **Eval/profile evaluate a self-quant variant before its checkpoint exists.** → the self-quant
  variants are evaluated **only inside** the self-quant phase, after production; the quality and
  efficiency phases run over the **official core** (`enabled-variants --source core`).
- **`run_matrix.sh` launched before bootstrap finished.** → `load_runtime_env` dies if
  `.qquant-runtime.sh` is absent; the orchestrator gates LAUNCH on the `.bootstrap-ok` marker
  (`ssh_exec "test -f"`), or onstart chains them when `QQUANT_AUTORUN=1`.

## Open questions (for the consistency-review gate)

1. **Resolves Spec 08's open questions** on the remote results root (`QQUANT_RESULTS` =
   `/workspace/qquant/results`), the target-cell handoff (`QQUANT_MANIFEST` file primary;
   `QQUANT_VARIANTS`/`QQUANT_TASKS` restrictions; resume via `is_cell_done`), and the completion
   sentinel (`_meta/RUN_COMPLETE`). Spec 08's `read_done_manifest` must read the manifest's `done`
   array — confirm the key name with Spec 08 at the gate.
2. **`deadman.sh` ownership.** Confirmed here as **Spec 08**'s file; `onstart.sh` only launches it.
   Confirm the env var the switch reads to learn its own instance id (Spec 02/08 open question) so
   onstart can also write it to a file as a belt-and-suspenders fallback.
3. **`_meta/env.json` shape & merge — RESOLVED.** The schema + `build_env` / `add_model` /
   `add_dataset` / `merge_env` / `write_env` helpers are owned by **Spec 11** (`qquant.repro.env`,
   torch-free); this spec's `write_env.py` is the GPU-side writer that consumes them to fill the
   `libraries` / `toolchain` fields, and Spec 05/07 add `datasets` records. No divergent `env`
   block; the merge-not-clobber helper lives in Spec 11's `qquant.repro.env`, not here.
4. **Entry-point availability ordering.** `run_matrix.sh` calls `qquant-eval`/`qquant-profile`
   (Spec 05/06). Until those `[project.scripts]` lines land, the scripts fall back to
   `python -m qquant.eval.cli` / `python -m qquant.efficiency.cli`; confirm the module paths at the
   gate.
