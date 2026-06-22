# Spec 08 — `qquant-orchestrate`: lifecycle driver, budget guard, auto-destroy, exfil

- **Status:** Draft (ready to implement; gated by Spec 02 wrapper + Spec 03 remote protocol)
- **Depends on:** 02 (vast.ai lifecycle wrapper + GPU spike), 03 (remote onstart/bootstrap/run_matrix), 05 (`qquant-eval`), 06 (`qquant-profile`), 07 (self-quant `quant/` env + checkpoints)
- **Owns / Produces:** the `qquant-orchestrate` entry point and the orchestration modules it **adds to the EXISTING Spec-02 `qquant.orchestrate` package** — `cli.py`, `driver.py`, `budget.py`, `ledger.py`, `reconcile.py`, `teardown.py` (the driver state machine, `BudgetGuard` + persistent cost ledger, copy-out/reconcile, defense-in-depth teardown, local watchdog) — plus the on-instance dead-man's-switch script. **Spec 02 owns the package `__init__.py` (and `vastai.py`/`spike.py`); this spec does NOT recreate it.** Adds the `qquant-orchestrate` line to `[project.scripts]`.

## Purpose

Replace the old `pf.sh` rsync+tmux harness with a single **local** driver that runs the whole
study end-to-end on rented vast.ai hardware while making it **impossible to silently leak
money or results**. One command:

1. **audits** the local results tree for missing/stale cells (resume — never recompute a
   finished cell);
2. **selects** an affordable on-demand RTX 4090 offer and **creates** one instance (which
   clones the public repo on boot — Spec 03);
3. **triggers** the remote matrix run (quality via Spec 05, efficiency via Spec 06,
   self-quant via Spec 07) over the resume set;
4. **incrementally copies** results out on every poll, **reconciles** the local cell count
   against the remote `_meta/done_manifest.json` before teardown (retrying the copy on a
   shortfall), then **auto-destroys** the instance; and
5. enforces a **$60 budget ceiling** counting compute **+ storage + bandwidth + image-pull**,
   with a **persistent cost ledger** across runs and a **dry-run** estimate.

It is local-only and **import-time torch-free** (it runs on the author's macOS laptop and only
shells out to `vastai`/`rsync` and the Spec 02 wrapper). It never imports the GPU stack.

## Scope (in / out)

**In:**
- The `qquant-orchestrate` CLI: `run`, `estimate` (alias `--dry-run`), `destroy`, `watchdog`,
  `ledger`.
- The lifecycle **state machine**: AUDIT → ESTIMATE/GATE → SEARCH → CREATE → BOOT → LAUNCH →
  POLL(loop: incremental copy + budget check + heartbeat) → RECONCILE → DESTROY → FINALIZE.
- `BudgetGuard` (compute + storage + bandwidth-down + bandwidth-up + image-pull) and a
  **persistent JSON cost ledger** with cumulative actual spend and atomic writes.
- Copy-out + **reconciliation** of local valid cells vs `done_manifest.json` (retry-on-shortfall),
  built on the SSOT `cell_path` / `is_cell_done` / `missing_cells` / `done_manifest`.
- **Defense-in-depth teardown**: local `finally` + `atexit` + SIGINT/SIGTERM handlers
  (idempotent destroy), an **on-instance dead-man's switch** (`sleep MAX_RUNTIME` → self-destroy
  via the vast API), and a **local watchdog** (`qquant-orchestrate watchdog`, schedulable via
  cron/launchd or the harness) that destroys orphaned instances if the driver dies.
- Preflight checks (API key, `vastai` present, public repo + pinned git ref reachable, ledger
  headroom, account-spend-limit reminder).

**Out (owned elsewhere; consumed here):**
- vast.ai API calls / offer search / `ssh`/`scp` plumbing → **Spec 02**
  (`qquant.orchestrate.vastai`, the `VastClient` wrapper).
- Remote boot, env install, and the actual matrix run / `done_manifest.json` writing →
  **Spec 03** (onstart/bootstrap/run_matrix).
- Cell production and metric extraction → **Spec 05 / 06 / 07** (run on the box).
- Aggregation, stats, plots, `REPORT.md` → **Spec 09 / 10**.
- The vast.ai **account-level spending limit** is an **operator pre-requisite** (set in the
  console; `.env.example` already says so) — the driver can only *warn*, never set it.

## Interface

### Entry point (`pyproject.toml` `[project.scripts]`, added by THIS spec)

```toml
qquant-orchestrate = "qquant.orchestrate.cli:main"
```

### CLI

```
qquant-orchestrate run        # full lifecycle (rent → run → exfil → reconcile → destroy)
qquant-orchestrate estimate   # dry-run: print resume plan + cost estimate, create nothing
qquant-orchestrate destroy    # emergency teardown of the active/known instance(s) + finalize ledger
qquant-orchestrate watchdog   # standalone loop: destroy orphaned instances if the driver died
qquant-orchestrate ledger     # print cumulative spend, per-run breakdown, remaining budget
```

`run` flags (all optional except `--repo-url`):

| flag | default | meaning |
|---|---|---|
| `--results PATH` | `results` | local results root (SSOT `Paths.from_root`) |
| `--repo-url URL` | (required) | public GitHub HTTPS URL the box clones on boot |
| `--git-ref REF` | current `HEAD` sha | commit/branch the box checks out (must be pushed) |
| `--image REF` | decision-log `QQUANT_IMAGE` constant (CUDA 12.6 *devel*, digest-pinned by the Spec 02 spike) | container image |
| `--disk-gb N` | `120` | instance disk (decision log) |
| `--budget-ceiling USD` | `60.0` | hard ceiling (decision log; margin under the $75 cap) |
| `--max-runtime-s N` | `21600` (6 h) | local cap **and** dead-man's-switch TTL |
| `--poll-interval-s N` | `120` | poll/copy cadence |
| `--variants a,b` | all `enabled` | restrict the matrix (canonical ids only) |
| `--tasks a,b` | all | restrict tasks (`mmlu,gsm8k,humaneval,ifeval`) |
| `--no-self-quant` | off | skip the Spec 07 phase (self-quant is a deferrable fast-follow) |
| `--dry-run` | off | same as `estimate` |
| `--yes` | off | skip the interactive cost-confirmation prompt |

### Python API (`qquant.orchestrate`)

```python
# driver.py
@dataclass(frozen=True)
class OrchestrateConfig:
    results_root: Path
    repo_url: str
    git_ref: str
    image: str
    disk_gb: int = 120
    budget_ceiling_usd: float = 60.0
    max_runtime_s: int = 21_600
    poll_interval_s: int = 120
    variants: tuple[str, ...] | None = None    # None → all enabled (canonical ids)
    tasks: tuple[str, ...] | None = None
    run_self_quant: bool = True
    assume_yes: bool = False

@dataclass(frozen=True)
class Plan:
    targeted: list[Cell]          # the resume set = missing_cells over the active matrix
    total_cells: int
    estimate: "CostEstimate"
    rates: "CostRates"
    runtime_h: float              # estimated wall time used for the cost estimate

def plan(cfg: OrchestrateConfig, offer: dict | None = None) -> Plan: ...
    # AUDIT + ESTIMATE only; pure, no instance is created. `offer` lets `estimate`
    # price against a real candidate offer (else a conservative default rate is used).

def run(cfg: OrchestrateConfig) -> int: ...
    # full state machine; returns 0 iff every targeted cell is locally valid after reconcile.
```

```python
# budget.py
@dataclass(frozen=True)
class CostRates:
    gpu_usd_hr: float            # on-demand compute rate (offer dph_base)
    storage_usd_gb_month: float
    inet_up_usd_gb: float
    inet_down_usd_gb: float
    @classmethod
    def from_offer(cls, offer: dict) -> "CostRates": ...   # maps Spec 02 offer fields
    @classmethod
    def conservative_default(cls) -> "CostRates": ...      # used by dry-run without an offer

@dataclass(frozen=True)
class CostEstimate:
    compute_usd: float
    storage_usd: float
    bandwidth_down_usd: float
    bandwidth_up_usd: float
    image_pull_usd: float
    @property
    def total_usd(self) -> float: ...

def estimate_cost(rates: CostRates, *, runtime_h: float, disk_gb: int,
                  download_gb: float, upload_gb: float, image_gb: float) -> CostEstimate: ...

class BudgetGuard:
    def __init__(self, ceiling_usd: float, ledger: "CostLedger",
                 rates: CostRates, disk_gb: int): ...
    def remaining_usd(self) -> float: ...                  # ceiling − ledger.cumulative
    def max_affordable_hours(self, *, image_gb: float, download_gb: float) -> float: ...
    def accrued(self, *, elapsed_h: float, downloaded_gb: float,
                uploaded_gb: float, image_pulled_gb: float) -> CostEstimate: ...
    def check(self, accrued: CostEstimate) -> None: ...    # raise BudgetExceeded if over ceiling

class BudgetExceeded(RuntimeError): ...
```

```python
# ledger.py  (persistent, atomic)
@dataclass
class LedgerEntry:
    run_id: str
    instance_id: int | None
    started_at: str            # ISO-8601 UTC
    ended_at: str | None
    offer_dph: float
    estimate: CostEstimate     # final actual-ish accrual at teardown
    note: str = ""

class CostLedger:
    @classmethod
    def load(cls, path: Path) -> "CostLedger": ...         # tolerant: missing → empty ledger
    def cumulative_usd(self) -> float: ...
    def open_entry(self, *, run_id: str, instance_id: int | None,
                   offer_dph: float) -> LedgerEntry: ...
    def commit(self, entry: LedgerEntry) -> None: ...       # append + atomic save (tmp+os.replace)
```

```python
# reconcile.py
@dataclass(frozen=True)
class ReconcileReport:
    targeted: list[str]        # cell_ids requested
    local_valid: list[str]     # cell_ids locally valid (is_cell_done)
    manifest: list[str]        # cell_ids the remote claims done (done_manifest.json)
    still_missing: list[str]   # targeted − local_valid
    @property
    def ok(self) -> bool: ...  # not still_missing

def copy_out(client, instance_id: int, remote_results: str,
             local_results: Path, *, timeout_s: int) -> bool: ...     # incremental rsync; bounded

def read_done_manifest(local_results: Path) -> list[str]: ...          # via Paths.done_manifest

def reconcile(targeted: list[Cell], local_results: Path,
              active_config: dict | None) -> ReconcileReport: ...       # via missing_cells/is_cell_done

def reconcile_with_retry(client, instance_id, remote_results, local_results,
                         targeted, active_config, *, max_retries: int,
                         timeout_s: int) -> ReconcileReport: ...
```

```python
# teardown.py
class TeardownGuard:
    """Idempotent destroy installed as finally + atexit + SIGINT/SIGTERM handler."""
    def __init__(self, client, instance_id_getter: Callable[[], int | None],
                 final_copy: Callable[[], None], finalize_ledger: Callable[[str], None]): ...
    def __enter__(self) -> "TeardownGuard": ...
    def __exit__(self, *exc) -> bool: ...
    def teardown(self, reason: str) -> None: ...   # best-effort bounded final copy → destroy → ledger

def write_active_run(path: Path, state: dict) -> None: ...   # atomic; heartbeat lives here
def clear_active_run(path: Path) -> None: ...
def read_active_run(path: Path) -> dict | None: ...
```

### File formats (THIS spec writes these; locations derive from SSOT `Paths.meta_dir`)

- **Cost ledger** — `<results>/_meta/cost_ledger.json` (persists across runs; operationally
  must not be deleted between runs):
  ```json
  {"schema_version": 1, "ceiling_usd": 60.0, "cumulative_usd": 12.43,
   "entries": [{"run_id":"...","instance_id":1234567,"started_at":"...","ended_at":"...",
                "offer_dph":0.34,
                "estimate":{"compute_usd":1.9,"storage_usd":0.05,"bandwidth_down_usd":0.6,
                            "bandwidth_up_usd":0.01,"image_pull_usd":0.1,"total_usd":2.66},
                "note":"reconciled ok"}]}
  ```
- **Active-run state / heartbeat** — `<results>/_meta/active_run.json` (consumed by the
  watchdog; cleared on clean teardown):
  ```json
  {"run_id":"...","pid":4242,"instance_id":1234567,"started_at":"...",
   "deadline_at":"...","budget_ceiling_usd":60.0,"last_heartbeat_at":"...",
   "results_root":"results","status":"polling"}
  ```
- Consumes (does not define) `_meta/done_manifest.json` and `_meta/env.json` — **contracts §3**.

### Cross-spec consumer contracts (named, not redefined here)

- **Spec 02 `qquant.orchestrate.vastai`** (the `VastClient` wrapper): `search_offers(...)`
  returning offers sorted `dlperf_usd-` for the decision-log query (`gpu_name=RTX_4090 num_gpus=1
  verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.6`) and
  `select_cheapest(offers, *, max_dph, ...)`; `create_instance(ask_id, *, image, disk_gb,
  onstart_cmd, env, ...)`; `show_instance(id)` exposing vast `actual_status`;
  `ssh_exec`/`copy_to`/`copy_from` primitives; `destroy(id)`. THIS spec adds only an
  **affordability filter** on top of `select_cheapest`.
- **Spec 03 remote protocol**: onstart clones `repo_url@git_ref`, runs bootstrap, **launches
  the dead-man's switch** (below) and `run_matrix` over the cell set this driver passes (env
  `QQUANT_TARGET_CELLS` / a written manifest); `run_matrix` writes cells under the **remote
  results root** and appends each finished `cell_id` to `_meta/done_manifest.json`; on
  completion it leaves a **completion sentinel** (`_meta/RUN_COMPLETE`). The driver treats the
  run as finished when the sentinel appears **or** `actual_status` leaves `running`.

## Files to create

```
# NOTE: src/qquant/orchestrate/__init__.py (and vastai.py/spike.py) already exist — Spec 02 owns
# them. This spec ADDS the modules below to that existing package; it does NOT recreate __init__.py.
src/qquant/orchestrate/cli.py          # argparse: run/estimate/destroy/watchdog/ledger → main()
src/qquant/orchestrate/driver.py       # OrchestrateConfig, plan(), run() state machine
src/qquant/orchestrate/budget.py       # CostRates, CostEstimate, estimate_cost, BudgetGuard
src/qquant/orchestrate/ledger.py       # CostLedger, LedgerEntry (atomic JSON persistence)
src/qquant/orchestrate/reconcile.py    # copy_out, read_done_manifest, reconcile[_with_retry]
src/qquant/orchestrate/teardown.py     # TeardownGuard, active_run.json helpers, watchdog loop
scripts/remote/deadman.sh              # on-instance dead-man's switch (sleep TTL → self-destroy)
tests/test_orchestrate_budget.py       # estimate + BudgetGuard ceiling math + ledger persistence
tests/test_orchestrate_reconcile.py    # reconcile via fixture results tree + done_manifest
tests/test_orchestrate_teardown.py     # idempotent destroy on exception/signal (fake client)
tests/test_orchestrate_torch_free.py   # importing qquant.orchestrate.* pulls no torch
```

Edit (1 line): add `qquant-orchestrate = "qquant.orchestrate.cli:main"` to `[project.scripts]`
in `pyproject.toml`.

## Implementation notes (the decision-log points that bind THIS spec, with the "why")

- **On-demand, single homogeneous RTX 4090; resume is a safety net.** Select exactly one
  on-demand offer; the resume audit means an interrupted run resumes cheaply. Poll
  `actual_status=="running"`; treat `exited`/`unknown`/`offline` as terminal → teardown +
  surface, **never loop forever accruing charges** (decision log: vast.ai).
- **Public repo, no secrets.** The box clones over HTTPS; `VAST_API_KEY`/`HF_TOKEN` are
  injected as instance **env**, never committed. Preflight runs `git ls-remote repo_url git_ref`
  so we fail before paying if the ref isn't pushed/public.
- **Digest-pinned CUDA 12.6 *devel* image (nvcc).** `--image` defaults to the decision-log
  `QQUANT_IMAGE` constant (`nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`, whose `sha256` digest the
  Spec 02 spike validates and records on first pull); the image-pull is a real bandwidth cost on
  first pull, so `BudgetGuard` counts it (`image_gb`).
- **`--disk 120` and "destroy wipes the disk".** Therefore **copy-out + reconcile before
  destroy** is mandatory; the disk is the only copy until exfil. Incremental rsync each poll
  bounds data-loss to one interval.
- **No native auto-destroy TTL → defense in depth (all four required):** (1) local
  `finally`/`atexit`/SIGINT-SIGTERM via `TeardownGuard` (idempotent — destroy may be called
  many times, runs once); (2) **on-instance dead-man's switch** `scripts/remote/deadman.sh`
  = `sleep $QQUANT_MAX_RUNTIME_S; vastai destroy instance <self>` so a *dead laptop* can't
  strand the box (launched by Spec 03 onstart in the background, with `VAST_API_KEY` in env);
  (3) **local watchdog** (`qquant-orchestrate watchdog`, schedulable via cron/launchd or the
  harness) reads `active_run.json` and destroys the instance if `now > deadline_at`, the
  heartbeat is stale, or the driver PID is dead; (4) the vast.ai **account spending limit**
  (operator pre-req — only *warned* about, see `.env.example`).
- **BudgetGuard counts compute + storage + bandwidth + image-pull; ceiling $60; persistent
  ledger.** `remaining = ceiling − ledger.cumulative`; the driver refuses to create if the
  estimate exceeds `remaining`, recomputes accrual every poll from elapsed wall time + bytes
  rsynced + bytes the box downloaded, and **tears down on `BudgetExceeded`**. The ledger is
  written atomically (tmp + `os.replace`) and committed on teardown so cumulative spend
  survives crashes and accumulates across runs.
- **Dry-run estimate.** `estimate`/`--dry-run` prints the resume set size and a component cost
  breakdown (priced against the cheapest candidate offer, else a conservative default) and
  exits without creating anything. `run` shows the same and prompts unless `--yes`.
- **Persist self-quant checkpoints (Spec 07).** Self-quant checkpoints are **exfiltrated** to
  the local `checkpoints/self-quant/<id>` after the quant phase and **re-uploaded on resume**
  so a resume never re-quantizes; both transfers are counted in the bandwidth estimate
  (up = exfil, down = re-upload). `--no-self-quant` skips the phase entirely (it is a
  deferrable fast-follow).
- **Reconcile uses only SSOT predicates.** `reconcile` builds the active matrix from
  `enabled_variants(load_variants())` × `load_tasks()`, restricts to `cfg.variants/tasks`,
  computes `targeted = missing_cells(...)` with the active provenance block
  (`cell_provenance`), and after the final copy declares success iff every targeted `cell_id`
  is locally valid by `is_cell_done`. `done_manifest.json` is a **cross-check**: if the remote
  claims a cell done but the local file is absent/invalid, that is a **copy shortfall** →
  retry `copy_out` (bounded `max_retries`) before allowing destroy. Budget/TTL always win over
  reconcile retries (we never keep paying to chase a copy).
- **Torch-free.** `qquant.orchestrate.*` must not import torch (it runs on macOS). It imports
  only `qquant.matrix/registry/paths/config`, stdlib, and the Spec 02 wrapper (itself
  torch-free). A dedicated guard test enforces this, mirroring Spec 01's core guard.

## Done-when (numbered, testable)

1. `uv sync` exposes `qquant-orchestrate`; `qquant-orchestrate --help` lists `run`, `estimate`,
   `destroy`, `watchdog`, `ledger`. The `[project.scripts]` line is present.
2. **Torch-free guard:** a fresh-interpreter test importing `qquant.orchestrate`,
   `.cli`, `.driver`, `.budget`, `.ledger`, `.reconcile`, `.teardown` leaves no `torch*` in
   `sys.modules` (mirrors `tests/test_import_torch_free.py`).
3. **Dry-run, no side effects:** `qquant-orchestrate estimate --results <tmp> --repo-url X`
   prints the resume-set size and a five-component cost breakdown summing to `total_usd`, and
   creates **no** vast instance (verified with a fake client whose `create_instance` asserts it
   is never called).
4. **Resume audit uses the SSOT:** with a fixture results tree where some cells are valid,
   `plan(...).targeted` equals `missing_cells(active_matrix, root, active_config)` exactly
   (same cell_ids); finished cells are never re-targeted; a cell whose provenance differs from
   `cell_provenance` is re-targeted (stale).
5. **BudgetGuard ceiling math:** `estimate_cost` sums compute (`gpu_usd_hr*runtime_h`) +
   storage (`storage_usd_gb_month*disk_gb*runtime_h/730`) + bandwidth (down/up `*_usd_gb*gb`) +
   image-pull (`inet_down_usd_gb*image_gb`); given a ledger with cumulative spend `C`,
   `BudgetGuard(ceiling).check(accrued)` raises `BudgetExceeded` exactly when
   `C + accrued.total_usd > ceiling`, and `remaining_usd() == ceiling − C`.
6. **Ledger persistence & atomicity:** committing an entry, reloading via `CostLedger.load`,
   yields the same `cumulative_usd`; a simulated crash mid-write (tmp file present) never
   corrupts the prior ledger (atomic `os.replace`).
7. **Reconcile semantics:** given a fixture remote-mirrored tree + `done_manifest.json`,
   `reconcile(...).ok` is `True` iff every targeted cell is locally valid; a manifest entry with
   a missing/invalid local file appears in `still_missing` and triggers a `copy_out` retry in
   `reconcile_with_retry` (with a fake client counting calls, bounded by `max_retries`).
8. **Idempotent defense-in-depth teardown:** with a fake client, an exception raised inside the
   `TeardownGuard` context destroys the instance exactly once; a second `teardown()` and the
   `atexit`/SIGTERM paths do **not** issue a second `client.destroy`; `active_run.json` is
   cleared on clean exit and left (for the watchdog) on a hard kill.
9. **Watchdog:** `qquant-orchestrate watchdog` given an `active_run.json` whose `deadline_at`
   is in the past (or whose `pid` is not alive) calls `client.destroy(instance_id)` once and
   appends a teardown note to the ledger; with a healthy/fresh state it destroys nothing.
10. **Dead-man's switch script:** `scripts/remote/deadman.sh` is shell-lint-clean, reads
    `QQUANT_MAX_RUNTIME_S` and `VAST_API_KEY`, resolves its own instance id, and on timeout
    issues a single `vastai destroy instance` (verified by a stubbed `vastai` on `PATH` in a
    fast test with a tiny TTL).
11. **Terminal-status handling:** when the fake client reports `actual_status` in
    `{exited,unknown,offline}`, `run` stops polling, performs a best-effort final copy + reconcile,
    destroys, finalizes the ledger, and returns non-zero (never an infinite poll loop).
12. **Lint/format:** `ruff check .` and `ruff format --check .` pass; new tests run GPU-free in
    CI (Spec 11) and do not require `vastai`/network (fakes only).

## Risks & mitigations

- **Stranded instance / runaway cost** → four independent kill paths (local handlers + atexit,
  on-instance dead-man's switch, local watchdog, account spend limit) + `BudgetGuard` ceiling +
  poll-driven terminal-status teardown. The dead-man's switch survives a dead laptop; the spend
  limit survives a dead switch.
- **Destroy wipes results before exfil** → incremental rsync every poll (bounds loss to one
  interval) + mandatory reconcile-before-destroy with retry; destroy is blocked until reconcile
  passes *or* budget/TTL forces it (then the shortfall is reported loudly so the next `run`
  resumes the missing cells).
- **Dead-man's switch can't self-identify the instance id** (vast self-id env var unverified) →
  belt-and-suspenders: onstart also writes the id to a file once known and the script prefers
  the file; if neither resolves, the local watchdog + spend limit still cover it. *(Open
  question — confirm with Spec 02/03.)*
- **vast pricing-field semantics** (`dph_base` vs `dph_total`, per-GB inet vs included) drift →
  `CostRates.from_offer` is the single mapping point; `conservative_default()` over-estimates
  for dry-run; the ledger records actuals so the ceiling self-corrects across runs.
- **Cost ledger is git-ignored under `results/`** and could be wiped locally, resetting
  cumulative spend → documented operational note (do not delete `_meta/cost_ledger.json`);
  the account spend limit is the durable backstop.
- **lm-eval × transformers v5 go/no-go (Spec 02 spike) fails** → orchestration is unaffected in
  shape; only the remote run command (Spec 03) changes. Keep the driver agnostic to the remote
  command string (passed through, not hard-coded).

## Open questions (for the consistency-review gate)

- Exact vast env var (or file) the dead-man's switch uses to learn its own instance id
  (Spec 02/03).
- Final names of the Spec 03 remote results path, the target-cell handoff
  (`QQUANT_TARGET_CELLS` vs a written manifest), and the completion sentinel
  (`_meta/RUN_COMPLETE`).
