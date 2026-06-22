# Spec 02 — vast.ai lifecycle wrapper + go/no-go GPU spike

- **Status:** Draft (ready to implement)
- **Depends on:** 01
- **Owns / Produces:** the `qquant.orchestrate` subpackage (`qquant.orchestrate.vastai` — a thin,
  mockable wrapper over the `vastai` CLI; `qquant.orchestrate.spike` — the local spike driver),
  the standalone on-box spike script `scripts/spike/spike_remote.py`, the new
  `qquant-spike` `[project.scripts]` entry point (Spec 02 owns this line), and two run-once
  artifacts: the **spike verdict** (`results/_meta/spike.json`, default) and the **resolved
  lm-eval working ref** (a git SHA or the fallback `0.4.12`). The GO/NO-GO verdict is the gate
  that **blocks Specs 04–10** from incurring large GPU spend.

## Purpose

Two coupled deliverables, both required before any large GPU run:

1. **A thin, mockable vast.ai lifecycle wrapper.** Every interaction with the `vastai` CLI
   (`search offers`, pick the cheapest acceptable offer, `create instance` with the pinned
   CUDA 12.6 *devel* image, poll `actual_status==running` with non-happy-path handling, get the
   ssh url, run remote commands, `copy` up/down, `destroy`) goes through one Python module so the
   orchestration driver (Spec 08) never shells out ad hoc. The module is **import-time torch-free**
   so it runs on the macOS dev laptop, and **every CLI call funnels through one injectable seam**
   so the whole wrapper is unit-testable on CI with **no network and no GPU**.

2. **The first-in-line go/no-go spike.** Stand up one real RTX 4090 *through the wrapper*, run an
   **inline minimal bootstrap** (deliberately independent of Spec 03's production bootstrap, which
   does not exist yet), and prove the single riskiest assumption in the whole project: that
   **`lm-eval` @ some pin works against `transformers==5.10.1`** end to end — it renders
   `apply_chat_template`, scores one real MMLU subtask, runs one `generate_until` task, and loads
   the bnb / gptq / awq stacks on sm_89 — and that **`nvcc` major.minor == `torch.version.cuda`**.
   The spike emits a **pinned v5-working lm-eval commit SHA** (fallback `0.4.12`) plus a **GO/NO-GO
   verdict**. This is the cheap experiment that gates the expensive ones.

## Scope (in / out)

**In scope**
- `qquant.orchestrate.vastai`: `VastClient` (one injectable runner), `Offer`/`Instance`
  dataclasses, `search_offers`, `select_cheapest`, `create_instance`, `show_instance`,
  `wait_until_running` (non-happy-path handling, bounded, never infinite), `ssh_url`,
  `ssh_exec`, `copy_to`, `copy_from`, `destroy`, and a typed error hierarchy.
- `qquant.orchestrate.spike`: the local driver that sequences the wrapper calls, **guarantees
  teardown on every exit path**, copies the verdict back, and writes `results/_meta/spike.json`.
- `scripts/spike/spike_remote.py`: the on-box checks (the only GPU-touching file in this spec).
- The `qquant-spike` entry point and unit tests with a fake runner.

**Out of scope (owned elsewhere)**
- The production remote bootstrap / `onstart` / `run_matrix` scripts → **Spec 03** (the spike uses
  its *own* tiny inline bootstrap and must not depend on Spec 03).
- The variant loaders themselves → **Spec 04**; the spike only does a *load-smoke*, not the real
  loaders.
- The full orchestration driver, `BudgetGuard`, cost ledger, watchdog `CronCreate`, incremental
  exfil/reconcile → **Spec 08** (the spike implements only a minimal, bounded teardown guard and
  references Spec 08 for the generalization).
- Re-declaring any contract id/path/schema/predicate → all imported from Spec 01 (`contracts.md`).

## Interface

### `qquant.orchestrate.vastai` (torch-free; subprocess + json + dataclasses only)

```python
from __future__ import annotations
import subprocess, time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# The single mock seam. Default runs the real CLI; tests inject a fake.
Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]

def default_runner(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(argv), capture_output=True, text=True)

# Verified offer query (decision-log "vast.ai"): RTX 4090, 1 GPU, verified+rentable,
# >=1 direct port, CUDA >= 12.6, ordered cheapest-by-dlperf first.
DEFAULT_OFFER_QUERY: str = (
    "gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true "
    "direct_port_count>=1 cuda_max_good>=12.6"
)
DEFAULT_ORDER: str = "dlperf_usd-"
RUNNING: str = "running"
TERMINAL_BAD: frozenset[str] = frozenset({"exited", "offline", "unknown"})

@dataclass(frozen=True)
class Offer:
    ask_id: int
    dph_total: float          # $/hr (compute); used for select_cheapest
    gpu_name: str
    num_gpus: int
    cuda_max_good: float
    reliability: float
    raw: dict                 # full --raw record, kept for forward-compat

@dataclass(frozen=True)
class Instance:
    id: int
    actual_status: str | None     # the field we gate on
    cur_state: str | None
    status_msg: str | None
    ssh_host: str | None
    ssh_port: int | None
    raw: dict

class VastError(RuntimeError): ...           # any nonzero rc / unparseable --raw
class VastTimeout(VastError): ...            # deadline reached, still not running
class VastNonRunning(VastError): ...         # actual_status in TERMINAL_BAD

class VastClient:
    def __init__(self, runner: Runner = default_runner, *, binary: str = "vastai") -> None: ...
    # internal: build argv [binary, *args], run via self.runner, raise VastError on rc!=0,
    # json.loads(stdout) for --raw calls. ALL public methods go through this one path.

    def search_offers(self, query: str = DEFAULT_OFFER_QUERY,
                      *, order: str = DEFAULT_ORDER, limit: int = 32) -> list[Offer]: ...
    def create_instance(self, ask_id: int, *, image: str, disk_gb: int = 120,
                        onstart_cmd: str | None = None, env: dict[str, str] | None = None,
                        label: str | None = None, ssh: bool = True,
                        direct: bool = True) -> int: ...   # returns new instance id
    def show_instance(self, instance_id: int) -> Instance: ...
    def ssh_url(self, instance_id: int) -> str: ...        # ssh://root@host:port
    def ssh_exec(self, instance_id: int, command: str, *,
                timeout_s: float | None = None) -> "subprocess.CompletedProcess[str]": ...
    def copy_to(self, instance_id: int, local_path: str, remote_path: str) -> None: ...
    def copy_from(self, instance_id: int, remote_path: str, local_path: str) -> None: ...
    def destroy(self, instance_id: int) -> None: ...       # idempotent: swallow "already gone"

def select_cheapest(offers: list[Offer], *, max_dph: float,
                    min_reliability: float = 0.95, min_cuda: float = 12.6) -> Offer:
    """Cheapest offer (min dph_total) meeting price/reliability/CUDA gates; raises VastError if none."""

def wait_until_running(client: VastClient, instance_id: int, *, timeout_s: float = 900.0,
                      poll_interval_s: float = 15.0,
                      sleep: Callable[[float], None] = time.sleep,
                      now: Callable[[], float] = time.monotonic) -> Instance:
    """Poll show_instance until actual_status == RUNNING *and* ssh_host is populated.
    Raise VastNonRunning if actual_status enters TERMINAL_BAD; VastTimeout at the deadline.
    `sleep`/`now` are injected so tests never sleep for real and never loop forever."""
```

Mapped `vastai` invocations (the wrapper builds exactly these; tests assert the argv):
- `vastai search offers "<query>" -o <order> --limit <n> --raw`
- `vastai create instance <ask_id> --image <digest> --disk <gb> [--onstart-cmd <sh>] [--env "-e K=V ..."] [--label <l>] --ssh --direct --raw`
- `vastai show instance <id> --raw`
- `vastai ssh-url <id>`
- `vastai copy <local> <id>:<remote>` / `vastai copy <id>:<remote> <local>`
- `vastai destroy instance <id>`

### `qquant.orchestrate.spike` (torch-free local driver)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SpikeVerdict:
    verdict: str                  # "GO" | "NO-GO"
    lm_eval_ref: str              # resolved git SHA, or "0.4.12" (fallback pin)
    lm_eval_version: str
    transformers_version: str
    torch_version: str
    torch_cuda: str               # e.g. "12.6"
    nvcc_cuda: str                # e.g. "12.6"
    checks: dict[str, bool]       # see remote spec below
    awq_official_loads: bool      # NON-gating: flips variants.yaml awq-official.enabled
    notes: list[str]
    raw: dict                     # full remote verdict json

# Candidate refs tried in order; first to pass the gating checks wins. Fallback is last.
DEFAULT_LM_EVAL_CANDIDATES: tuple[str, ...] = ("main", "0.4.12")

def run_spike(client: VastClient, *, image: str,
             candidates: tuple[str, ...] = DEFAULT_LM_EVAL_CANDIDATES,
             out_path: str = "results/_meta/spike.json",
             max_runtime_s: int = 1800, hf_token: str | None = None) -> SpikeVerdict: ...

def main(argv: list[str] | None = None) -> int:
    """qquant-spike CLI. Flags: --image REF (default: the decision-log `QQUANT_IMAGE` constant),
    --candidates a,b,c, --out PATH, --max-runtime-s N, --keep-alive (skip destroy for debug),
    --offer-query STR, --max-dph FLOAT. Exit 0 iff verdict == GO."""
```

`run_spike` sequence (each step wrapped so teardown is guaranteed — see Implementation notes):
search → `select_cheapest` → `create_instance(image=…, disk_gb=120, onstart_cmd=<dead-man switch>)`
→ `wait_until_running` → `copy_to(spike_remote.py)` → `ssh_exec(<inline bootstrap> && python spike_remote.py …)`
→ `copy_from(/workspace/spike_verdict.json)` → write `out_path` → **`destroy` (finally)**.

### `scripts/spike/spike_remote.py` (the only GPU-touching file; runs on the 4090)

Standalone script — **never imported** by the package (keeps the core torch-free). CLI:

```
python spike_remote.py --lm-eval-ref <ref> \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --gptq-model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --gptq-revision e9c932ac… \
  --awq-model  Qwen/Qwen2.5-7B-Instruct-AWQ        --awq-revision  b250375… \
  --out /workspace/spike_verdict.json
```

Writes the verdict JSON (matching `SpikeVerdict.raw`) to `--out` **and** prints it to stdout;
exits 0 iff GO. `checks` keys (all booleans), in priority order:

| key | what it proves | gating? |
|---|---|---|
| `nvcc_matches_torch` | `nvcc --version` major.minor == `torch.version.cuda` | **GATING** (fail-fast first) |
| `mmlu_subjects_match` | `set(MMLU_SUBJECTS)` == `TaskManager` expansion of the `mmlu` group | **GATING** (contracts §1) |
| `chat_template_renders` | Qwen tokenizer `apply_chat_template` produces a non-empty prompt under transformers 5.10.1 | **GATING** |
| `mmlu_subtask_scored` | lm-eval scores `mmlu_anatomy` `--limit 4` with `apply_chat_template=True, fewshot_as_multiturn=True` and returns a finite `acc` | **GATING** |
| `generate_until_ok` | one `generate_until` task (`gsm8k --limit 2`, **greedy**) yields non-empty generations | **GATING** |
| `bnb_loads` | `BitsAndBytesConfig` int8 **and** NF4 (`bnb_4bit_compute_dtype=bfloat16`) load + 1-token generate | **GATING** |
| `gptq_loads` | official GPTQ-Int4 loads via gptqmodel+optimum (JIT kernels compile) + 1-token generate | **GATING** |
| `awq_official_loads` | official AWQ loads on sm_89 with only gptqmodel+optimum (no autoawq) | **NON-gating** |

`verdict == "GO"` iff every **gating** check passes. `awq_official_loads` is reported separately
and feeds the contingency decision (flip `variants.yaml` `awq-official.enabled`).

## Files to create

- `src/qquant/orchestrate/__init__.py` — package marker; torch-free.
- `src/qquant/orchestrate/vastai.py` — the wrapper (`VastClient`, dataclasses, `select_cheapest`,
  `wait_until_running`, errors).
- `src/qquant/orchestrate/spike.py` — local driver (`SpikeVerdict`, `run_spike`, `main`).
- `scripts/spike/spike_remote.py` — on-box checks (GPU; standalone, not importable from the package).
- `scripts/spike/bootstrap.sh` — the inline minimal bootstrap the driver runs via `ssh_exec`
  (installs the pinned eval stack + the chosen lm-eval ref; torch comes from the image). Kept
  separate from Spec 03's production bootstrap.
- `tests/test_vastai_wrapper.py` — fake-runner unit tests (no network, no GPU, no real sleep).
- `tests/test_spike_driver.py` — driver teardown-guarantee + verdict-parsing tests (fake runner).
- Edit `pyproject.toml`: add `qquant-spike = "qquant.orchestrate.spike:main"` under
  `[project.scripts]` (the one `[project.scripts]` line this spec owns; contracts §7).

## Implementation notes (the decision-log points that bind THIS spec, and why)

- **Image = CUDA 12.6 *devel* PyTorch image, pinned by digest (with `nvcc`).** gptqmodel JIT-compiles
  Marlin/AWQ sm_89 kernels on first use, so `nvcc` must be present and its CUDA must equal
  `torch.version.cuda`; bitsandbytes ships prebuilt sm_89 cubins and needs no nvcc. The spike's very
  first check is `nvcc_matches_torch` — **fail fast** here saves an expensive wrong-image run.
  The default `--image` is the decision-log `QQUANT_IMAGE` constant
  (`nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`); this spike is the place that **validates the
  exact tag, SSH/`--direct` compatibility, and the `sha256` digest on first pull** and records the
  resolved digest (and the resolved torch/nvcc CUDA) in the verdict so any image drift is caught.
- **lm-eval × transformers v5 is the project's biggest open risk** (decision-log: lm-eval v5 support
  is OPEN, issue #3537, `apply_chat_template`). That is *exactly* why this spike is first-in-line and
  gates 04–10. The spike tries each candidate ref in order and **emits the first that passes the
  gating checks**; if only the fallback works under v5, it emits `0.4.12`; if none render
  `apply_chat_template` under transformers 5.10.1, the verdict is **NO-GO** — cheaply blocking the
  large spend rather than discovering the break mid-matrix. The resolved ref is the value Spec 03/11
  pin into the `gpu` group's `lm-eval` requirement.
- **Per-task chat-template policy (not blanket).** The MMLU subtask check uses
  `apply_chat_template=True` + `fewshot_as_multiturn=True` exactly as `tasks.yaml` declares for `mmlu`,
  so the spike exercises the real policy path, not a generic one. (HumanEval's `humaneval_instruct` +
  fenced-extraction and IFEval's nltk/langdetect setup are out of the spike's scope; full task wiring
  is Specs 05/09.)
- **Greedy determinism.** The `generate_until` smoke forces `do_sample=False` and clears
  `temperature/top_p/top_k`, matching the locked greedy decision; the spike does not run the full
  twice-and-compare determinism smoke (that is Spec 05/11).
- **NF4 compute dtype.** The bnb load-smoke sets `bnb_4bit_compute_dtype=torch.bfloat16` and uses
  `dtype=` (not the removed `torch_dtype=`/`load_in_4bit` shortcuts) so the smoke reflects the v5
  canonical `quantization_config=` path the real loaders (Spec 04) will use.
- **`awq-official` is CONTINGENT.** The AWQ load is attempted with **only** gptqmodel+optimum
  (decision-log: **never install autoawq** — it pins `transformers~=4.5x` and would break the stack).
  Its result is **non-gating**: a GO still stands if AWQ fails, and the operator flips
  `variants.yaml` `awq-official.enabled=false` (the guaranteed AWQ data point is `awq-selfquant`).
- **`MMLU_SUBJECTS` must equal lm-eval's `mmlu` group expansion** (contracts §1). The spike performs
  this assertion on the box under the *chosen* lm-eval ref, confirming the 57-cell expansion is real
  for the pin we actually ship — the cheapest possible place to catch a subject-set mismatch.
- **vast.ai polling discipline.** Poll `actual_status==running` via `--raw`; treat
  `exited`/`offline`/`unknown` as terminal-bad (`VastNonRunning`) and `loading`/`created` as
  keep-polling; **never loop forever accruing charges** — `wait_until_running` is bounded by
  `timeout_s` and raises `VastTimeout`. `vastai>=1.1.1`.
- **`destroy` wipes the disk**, so the driver copies the verdict out *before* destroying; but the
  spike produces no matrix cells, so there is no `done_manifest` reconciliation here (that is Spec 08).
- **Auto-destroy defense-in-depth (minimal subset for the spike).** Local `try/finally` + `atexit` +
  SIGINT/SIGTERM handlers guarantee `destroy` on every exit path **and** the `create_instance`
  `onstart_cmd` installs a tiny on-instance **dead-man's switch** (`setsid sh -c 'sleep
  $MAX_RUNTIME; <self-destroy via vast API using injected $VAST_API_KEY>'`). Operator pre-req: a
  vast.ai **account spending limit**. Spec 08 generalizes this into the watchdog/BudgetGuard; the
  spike just must not leak an instance.
- **Torch-free boundary.** `qquant.orchestrate.vastai`/`spike` only use `subprocess`/`json`/
  `dataclasses` and must never import torch (so orchestration runs on macOS); the GPU imports live
  exclusively in `scripts/spike/spike_remote.py`, which is copied to the box and run as a script,
  never imported. The umbrella `qquant` CLI is untouched (contracts §7).
- **Repo delivery.** Public GitHub clone over HTTPS; no secrets in the repo. `VAST_API_KEY`/`HF_TOKEN`
  are read from the environment (`.env.example`) and injected — never written to disk in the image.

## Done-when (numbered, testable)

1. `python -c "import qquant.orchestrate.vastai, qquant.orchestrate.spike"` succeeds, and a
   fresh-interpreter check (mirroring `tests/test_import_torch_free.py`) confirms **neither module
   pulls `torch` into `sys.modules`**.
2. With a fake `Runner` returning canned `vastai search offers --raw` JSON, `search_offers` returns
   `Offer`s with the right fields, and `select_cheapest` returns the **minimum `dph_total`** offer
   that meets `max_dph`/`min_reliability`/`min_cuda`, and raises `VastError` when none qualify.
3. With a fake runner scripted `loading → loading → running`, `wait_until_running` returns the
   running `Instance` using the **injected `sleep`/`now`** (test asserts no real sleeping); a script
   yielding `exited`/`offline`/`unknown` raises `VastNonRunning`; a script that never reaches
   `running` before `timeout_s` raises `VastTimeout` (bounded, never infinite).
4. `create_instance` builds exactly `vastai create instance <ask_id> --image <digest> --disk 120
   --onstart-cmd <sh> --ssh --direct --raw` (asserted on captured argv) and returns the parsed new
   instance id; `destroy`, `copy_to`, `copy_from`, `ssh_url`, `show_instance` likewise build their
   documented argv (asserted).
5. The driver **guarantees teardown**: a test where `ssh_exec`/`wait_until_running` raises still
   results in exactly one `destroy <id>` call (via `finally`); a `--keep-alive` run skips it.
6. `qquant-spike --help` works (entry point importable, torch-free); the resolved
   `[project.scripts]` contains `qquant-spike = "qquant.orchestrate.spike:main"`.
7. `scripts/spike/spike_remote.py --help` works on a plain machine; its verdict JSON validates
   against the documented `checks` keys + `SpikeVerdict` fields (schema-shape test with a fixture).
8. **On a real RTX 4090 (the actual gate run):** the spike completes, copies
   `results/_meta/spike.json` back, and the verdict contains `verdict ∈ {GO, NO-GO}`,
   `lm_eval_ref` (a git SHA or `0.4.12`), `torch_cuda == nvcc_cuda`, `mmlu_subjects_match == true`,
   and `awq_official_loads` recorded. The instance is destroyed on exit (verified via
   `vastai show instances`).
9. A **GO** verdict is the recorded precondition for Specs 04–10; a **NO-GO** documents the failing
   gating check(s) and blocks them. (Recorded in `results/_meta/spike.json`; the chosen `lm_eval_ref`
   is handed to Spec 03/11 to pin.)

## Risks & mitigations

- **lm-eval breaks under transformers 5.10.1 (issue #3537).** → The spike is the cheap gate: try
  candidate refs, fall back to `0.4.12`, and return **NO-GO** rather than failing mid-matrix; the
  verdict names the exact failing check.
- **AWQ official won't load on sm_89 without autoawq.** → `awq_official_loads` is **non-gating**;
  GO still possible; operator flips `awq-official.enabled=false`; `awq-selfquant` is the guaranteed
  AWQ point. autoawq is never installed.
- **Image/toolchain drift (`nvcc` ≠ torch CUDA).** → Image pinned by digest; first gating check is
  `nvcc_matches_torch`, fail-fast.
- **Runaway instance cost.** → Bounded `wait_until_running`; never loop on bad status; local
  finally/atexit/signal `destroy`; on-box dead-man's switch; operator account spend limit. (Full
  BudgetGuard/ledger/watchdog = Spec 08.)
- **`vastai --raw` field/schema drift.** → Parse defensively, keep the full record in `Offer.raw`/
  `Instance.raw`, pin `vastai>=1.1.1`, and centralize parsing in one place so a field rename is a
  one-line fix.
- **`actual_status==running` but ssh not yet ready.** → `wait_until_running` also requires
  `ssh_host` populated, and `ssh_exec` retries a bounded number of times before failing.
- **Flaky on-box installs.** → `bootstrap.sh` uses bounded retries / `--no-cache-dir` and
  `--no-build-isolation` for gptqmodel; a bootstrap failure surfaces as a NO-GO with a clear note,
  not a hang.
- **Spike depending on Spec 03 (which doesn't exist yet).** → The spike ships its **own** minimal
  `bootstrap.sh`; it must not import or call any Spec 03 artifact.
