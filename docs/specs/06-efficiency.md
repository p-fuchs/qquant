# Spec 06 — Efficiency profiler (`qquant.efficiency` + `qquant-profile`)

- **Status:** ready to implement
- **Depends on:** 04 (unified variant loaders)
- **Owns / Produces:** the `qquant.efficiency` package; the `qquant-profile` GPU entry point;
  the per-variant efficiency artifact `results/<variant>/efficiency.json` and its schema
  `src/qquant/schemas/efficiency_result.schema.json`. Consumed by Spec 08 (orchestration runs
  one `qquant-profile` per variant) and Spec 09 (aggregation/plots for RQ#3 — speed/memory
  trade-offs).

## Purpose

Answer **RQ#3 (efficiency)**: for each enabled variant, measure the real cost/throughput
profile on the single homogeneous RTX 4090 (24 GB, sm_89) so the report can state the
memory-vs-speed trade-offs apples-to-apples. Exactly the metrics in the scope brief:

1. **model size on disk** (weights bytes),
2. **peak GPU memory at load**,
3. **peak GPU memory during generation**,
4. **prefill throughput** (prompt tokens/s),
5. **decode throughput** (generated tokens/s, steady state),
6. **end-to-end latency** (full `generate`, seconds),
7. **max batch size before OOM** (a sweep), and
8. **behaviour across short / medium / long prompts** (every speed metric reported per prompt
   bucket).

Each variant is profiled in **one fresh OS process** (the orchestrator spawns it), so CUDA
context, gptqmodel JIT kernel buffers, and the allocator's peak-memory counters are never
polluted by a previously-loaded variant. Output is a single schema-valid JSON per variant under
`results/`, reusing the `<results_root>/<variant>/…` + `_meta/` conventions.

## Scope (in / out)

**In**

- The `qquant.efficiency` package: a torch-free schema/predicate/path submodule, a
  torch-touching measurement engine, and the `qquant-profile` argparse CLI.
- Deterministic measurement protocol: warmup (discarded) → N measured repeats → **median**
  (plus min/max for transparency); identical synthetic prompts across variants; greedy decode
  forced to an exact token count.
- A new artifact + JSON schema for the per-variant efficiency result, with a resume predicate
  that mirrors the SSOT style (`is_cell_done`) but for the efficiency artifact.
- Adding the `qquant-profile` line to `[project.scripts]`.

**Out** (owned elsewhere — do not implement here)

- Model loading / quant dispatch / greedy-forcing at load → **Spec 04** (imported, not reimplemented).
- Quality metrics, lm-eval, the `<variant>/<task>/…` cell files, `cell_path`, `is_cell_done`,
  `cell_result.schema.json`, `cell_provenance` → **Spec 01/05** (untouched).
- Spawning one process per variant, incremental copy-out, budget/auto-destroy, `_meta/env.json`
  creation → **Spec 03/08**.
- Aggregation, cross-variant tables, plots, the markdown report → **Spec 09/10**.
- Building self-quant checkpoints → **Spec 07** (this spec only *reads* their local dir for disk size).

## Interface

### CLI — `qquant-profile`

```
qquant-profile --variant <id> [options]

  --variant ID            REQUIRED. Exactly one of qquant.registry.VARIANT_IDS.
                          (One variant per fresh process — passing >1 is an error.)
  --results PATH          results root (default: "results")
  --variants-file PATH    override registry path (tests); default = packaged registry
  --device STR            torch device (default: "cuda:0")
  --warmup INT            discarded warmup iterations per measurement (default: 2)
  --repeats INT           measured iterations; median reported (default: 5)
  --decode-tokens INT     forced new tokens for throughput/latency (default: 256)
  --prompt-lens S,M,L     short,medium,long prompt token lengths (default: 128,1024,4096)
  --batch-seq-len INT     prompt length used for the OOM sweep (default: 2048)
  --batch-gen-tokens INT  new tokens per item in the OOM sweep (default: 64)
  --batch-ceiling INT     largest batch tried in the OOM sweep (default: 64)
  --force                 recompute even if a done artifact exists
  --dry-run               print the resolved plan + output path; do NOT import torch/load model
  -h/--help

Exit codes: 0 = success or up-to-date skip; 2 = bad args (unknown/>1 variant);
            3 = load failure or OOM-at-load (no artifact written);
            non-zero otherwise. (--force always recomputes; absent it, a done artifact => skip, exit 0.)
```

### Module layout & public API

```
src/qquant/efficiency/
  __init__.py     # re-exports torch-free helpers; `main`/`profile_variant` via lazy __getattr__
  schema.py       # TORCH-FREE: schema load, validators, resume predicate, path + provenance
  profiler.py     # torch-touching engine; torch imported lazily INSIDE functions
  cli.py          # argparse `main`
```

`src/qquant/efficiency/schema.py` (import-time **torch-free**, jsonschema only):

```python
EFFICIENCY_SCHEMA_VERSION = 1

# Subset of the result that, if changed, must invalidate a prior artifact on resume.
EFFICIENCY_PROVENANCE_KEYS = (
    "torch_version", "transformers_version", "cuda_version", "gpu_name",
    "dtype", "model_revision", "prompt_buckets", "decode_tokens", "batch_seq_len",
)

def load_efficiency_schema() -> dict: ...            # lru_cache; reads efficiency_result.schema.json
def validate_efficiency(result: dict) -> None: ...   # raises jsonschema.ValidationError
def is_valid_efficiency(result: dict) -> bool: ...

def efficiency_path(results_root: str | Path, variant: str) -> Path:
    """results_root/<variant>/efficiency.json — built via qquant.paths.Paths.

    Spec-06-owned artifact path. This is the SANCTIONED carve-out in contracts.md §8: the
    per-variant efficiency artifact is the ONE exception to "one path builder / one schema /
    one resume predicate", with its own path helper, schema, and resume predicate all owned by
    qquant.efficiency. Intentionally OUTSIDE cell_path: an efficiency profile is per-variant,
    not a (variant,task[,subject]) quality cell. "efficiency" is not in TASK_IDS and this is a
    *file* directly under <variant>/ (tasks live in <variant>/<task>/ dirs), so it never
    collides with the quality matrix."""

def is_efficiency_done(result: object, active_config: dict | None = None) -> bool:
    """Resume predicate, same shape/spirit as qquant.matrix.is_cell_done:
    structural validity ALWAYS; if active_config given, every overlapping
    EFFICIENCY_PROVENANCE_KEYS value must match or the artifact is stale (recompute)."""

def efficiency_active_config(cfg: "ProfileConfig", variant: Variant, runtime: dict) -> dict:
    """Build the comparable provenance block (the EFFICIENCY_PROVENANCE_KEYS) from the run
    config + the runtime facts (torch/transformers/cuda versions, gpu_name, dtype)."""
```

`src/qquant/efficiency/profiler.py` (torch imported lazily inside the functions):

```python
@dataclass(frozen=True)
class ProfileConfig:
    device: str = "cuda:0"
    warmup: int = 2
    repeats: int = 5
    decode_tokens: int = 256
    prompt_buckets: dict[str, int] = {"short": 128, "medium": 1024, "long": 4096}
    batch_seq_len: int = 2048
    batch_gen_tokens: int = 64
    batch_ceiling: int = 64

def profile_variant(variant: Variant, cfg: ProfileConfig,
                    env_meta: dict | None = None) -> dict:
    """Load `variant` in THIS process via the Spec-04 loader, run every measurement in a fixed
    order, and return a schema-valid efficiency result dict. The caller persists it. The OOM
    sweep runs LAST so it cannot pollute the load/throughput/memory peaks. env_meta is the
    parsed results/_meta/env.json (best-effort) used to stamp library/dataset revisions."""

# Internal measurement helpers (all greedy, deterministic, sync-bounded):
def measure_disk_size(variant: Variant) -> dict          # torch-free; HF cache snapshot or local: dir
def make_synthetic_inputs(tokenizer, n_tokens, batch, device) -> dict  # exact-length deterministic ids
def timed_generate(model, inputs, *, min_new, max_new) -> tuple[float, float, int]  # (ttft_s, total_s, n_new)
def measure_memory_at_load(load_fn) -> dict              # reset_peak_memory_stats around the load
def measure_throughput(model, tokenizer, cfg) -> dict    # per bucket: prefill/decode tok_s, e2e latency
def sweep_max_batch(model, tokenizer, cfg) -> dict       # largest ok batch + first OOM batch
def median(xs: list[float]) -> float
```

`qquant-profile = "qquant.efficiency.cli:main"` is added to `[project.scripts]` (this spec's
own line; the torch-free umbrella `qquant` is untouched and still never imports torch).

### Output file format — `results/<variant>/efficiency.json` (schema_version 1)

Required top level: `schema_version` (==1), `variant`, `disk`, `memory`, `throughput`,
`max_batch_size`, `config`. Optional: `quant_method`, `source`, `model_id`, `model_revision`,
`speed_label`, `confounded_metrics`, `samples`, `meta`.

```jsonc
{
  "schema_version": 1,
  "variant": "bf16",
  "quant_method": "bf16",          // from the Variant (registry)
  "source": "baseline",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",

  "speed_label": "comparable",     // "runtime-confounded" for source=="selfquant"
  "confounded_metrics": [],        // self-quant => the SPEED metrics only (see notes)

  "disk": {                        // size on disk
    "weights_bytes": 15231000000,  // sum of *.safetensors/*.bin (real sizes, symlinks resolved)
    "snapshot_bytes": 15240000000, // full resolved snapshot/local dir
    "weights_gib": 14.19,
    "source": "hf-cache",          // "hf-cache" | "local"
    "path": "/root/.cache/huggingface/.../snapshots/<rev>"
  },

  "memory": {                      // GiB and raw bytes; allocated + reserved
    "weights_resident_bytes": ..., // allocated immediately after load, before any generate
    "load_peak_bytes": ...,        // max_memory_allocated across the load
    "load_peak_reserved_bytes": ...,
    "generate_peak_bytes": ...,    // peak during the reference generation (KV + logits transient)
    "generate_peak_reserved_bytes": ...,
    "load_peak_gib": ..., "generate_peak_gib": ...
  },

  "throughput": {                  // per prompt bucket; medians over `repeats`
    "short":  { "prompt_tokens": 128, "gen_tokens": 256,
                "prefill_tok_s": ..., "decode_tok_s": ..., "e2e_latency_s": ...,
                "ttft_s": ... },
    "medium": { "prompt_tokens": 1024, ... },
    "long":   { "prompt_tokens": 4096, ... }
  },

  "max_batch_size": {              // OOM sweep, run last
    "value": 16,                   // largest batch that succeeded (>=1)
    "oom_at": 32,                  // first batch that OOMed, or null if ceiling reached cleanly
    "ceiling": 64,
    "seq_len": 2048,
    "gen_tokens": 64,
    "ladder": [1, 2, 4, 8, 16]     // batches actually attempted that succeeded
  },

  "config": {                      // reproducibility block; carries EFFICIENCY_PROVENANCE_KEYS
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "device": "cuda:0",
    "dtype": "bfloat16",           // bnb-nf4 compute dtype = bfloat16; official ckpts = fp16
    "torch_version": "2.12.1+cu126",
    "transformers_version": "5.10.1",
    "cuda_version": "12.6",
    "expandable_segments": true,   // PYTORCH_CUDA_ALLOC_CONF observed at process start
    "do_sample": false,
    "warmup": 2, "repeats": 5,
    "decode_tokens": 256,
    "prompt_buckets": {"short": 128, "medium": 1024, "long": 4096},
    "batch_seq_len": 2048, "batch_gen_tokens": 64,
    "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234}
  },

  "samples": { /* optional: per-repeat raw timings, for audit */ },
  "meta": { "timestamp": "2026-06-22T...Z", "hostname": "...",
            "note": "self-quant W4A16 on HF generate may take a slow dequant path on sm_89 ..." }
}
```

## Files to create

1. `src/qquant/efficiency/__init__.py` — package init. Imports the **torch-free** helpers from
   `.schema` at import time; exposes `main`, `profile_variant`, `ProfileConfig` via module-level
   `__getattr__` so `import qquant.efficiency` (and `qquant.efficiency.schema`) stay torch-free.
2. `src/qquant/efficiency/schema.py` — schema loader + `validate_efficiency`/`is_valid_efficiency`,
   `efficiency_path`, `is_efficiency_done`, `EFFICIENCY_PROVENANCE_KEYS`,
   `efficiency_active_config`. Torch-free.
3. `src/qquant/efficiency/profiler.py` — the measurement engine (torch imported lazily inside
   functions; consumes the Spec-04 loader).
4. `src/qquant/efficiency/cli.py` — `build_parser()` + `main(argv=None) -> int`.
5. `src/qquant/schemas/efficiency_result.schema.json` — JSON Schema 2020-12 for the artifact
   above (additive resource; does **not** modify `schemas/__init__.py` or the cell schema).
6. `tests/test_efficiency.py` — torch-free unit tests (see Done-when). GPU integration tests are
   marked `@pytest.mark.gpu` and skipped where CUDA is absent.
7. **Edit** `pyproject.toml`: add `qquant-profile = "qquant.efficiency.cli:main"` to
   `[project.scripts]` (only line touched).

## Implementation notes (the decision-log points that bind THIS spec, with the "why")

- **One fresh process per variant.** gptqmodel JIT-compiles Marlin/AWQ kernels on first forward,
  and `torch.cuda.max_memory_allocated()` / `reset_peak_memory_stats()` are process-global. A
  second variant in the same process would inherit a warm/fragmented allocator and stale peaks,
  destroying the apples-to-apples comparison the single-GPU platform decision buys us. So the CLI
  profiles exactly **one** `--variant`; Spec 08 spawns the loop.
- **Load via Spec 04 only.** Spec 04 owns dispatch (bf16 / bnb-int8 / bnb-nf4 / gptq-official /
  awq-official / compressed-tensors self-quant) and the greedy-forcing at load. This spec imports
  that loader and never re-creates `BitsAndBytesConfig`, `quantization_config=`, device_map, or
  the `dtype=` (not `torch_dtype=`) wiring. NF4's `bnb_4bit_compute_dtype=bfloat16` and the
  official-checkpoint fp16 are loader concerns; we only *record* the resulting `dtype` for the
  report's compute-dtype caveat.
- **Greedy determinism, exact token counts.** Per the greedy-determinism decision, `do_sample`
  is forced `False` and `temperature/top_p/top_k` cleared. For *fair* throughput we must generate
  a fixed number of tokens regardless of EOS, so generation uses
  `min_new_tokens == max_new_tokens == decode_tokens` (and disables early EOS). This makes
  decode-token counts identical across variants and prompts.
- **Self-quant speed is runtime-confounded.** compressed-tensors W4A16_ASYM on the HF `generate()`
  backend may hit a slow dequant path on sm_89 (fast Marlin targets vLLM), and GPTQ-asym disables
  Marlin so even official-vs-self GPTQ speed differs. Therefore for `source == "selfquant"` we set
  `speed_label = "runtime-confounded"` and list the **speed** metrics in `confounded_metrics`
  (`prefill_tok_s`, `decode_tok_s`, `e2e_latency_s`, `ttft_s`). **disk** and **memory** are NOT
  confounded — they are real and fair for every variant — so they are never in that list. Spec
  09/10 must surface this label next to any self-quant speed number and use self-quant only for the
  calibration-controlled *quality* comparison.
- **VRAM budget includes the logits transient** (`batch×seq×vocab(152064)×2B` ≈ 3.5 GB @ batch 4 /
  seq 3k). The *generate-peak* memory measurement deliberately captures this transient (and the KV
  cache) by resetting peak stats immediately before the reference generation. We also set/record
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: it must be set **before** CUDA init, so the
  launcher (Spec 03/08 onstart) sets it and the profiler only *observes & records* it (warns if
  unset — it cannot retroactively enable it).
- **Deterministic, fair prompts.** All variants see byte-identical synthetic inputs of exact target
  token length (built from a fixed token sequence, seeded). Short/medium/long buckets stress the
  prefill compute, the KV cache, and any dequant-path length sensitivity respectively.
- **Warmup + repeats + medians.** Warmup iterations are discarded (absorb JIT compile, cuBLAS
  autotune, allocator warmup); the median over `repeats` measured iterations is robust to GC/jitter.
  Timing uses `torch.cuda.synchronize()` + `time.perf_counter()` for end-to-end and CUDA events
  (`torch.cuda.Event(enable_timing=True)`) for prefill/decode; all under `torch.inference_mode()`
  with `model.eval()`.
- **Prefill vs decode separation.** Prefill (TTFT) = a `generate(..., max_new_tokens=1)`;
  `prefill_tok_s = prompt_tokens / ttft_s`. Decode = a `generate` of `decode_tokens`;
  `decode_tok_s = (decode_tokens - 1) / (total_s - ttft_s)`. `e2e_latency_s = total_s`.
- **Provenance reuse.** The profiler reads `results/_meta/env.json` (best-effort) for pinned
  library/dataset revisions + image digest, falling back to runtime introspection
  (`torch.__version__`, `transformers.__version__`, `torch.version.cuda`,
  `torch.cuda.get_device_name`). `model_revision` comes from the `Variant` (the pinned SHA; `null`
  for self-quant). Seeds come from `qquant.config.Seeds` so the block round-trips the project's
  reproducibility vocabulary.
- **Artifact, not a cell (the contracts §8 carve-out).** Efficiency results are per-variant, so
  they cannot use `cell_path` / `cell_result.schema.json` / `is_cell_done` (those are
  per-(variant,task[,subject]) quality cells). Rather than silently diverging from the "one path
  builder / one schema / one resume predicate" rule, this is the **explicit sanctioned exception
  recorded in `contracts.md` §8**: the per-variant efficiency artifact
  (`results/<variant>/efficiency.json`) is the ONE allowed addition beyond the cell SSOT, with its
  own path helper (`efficiency_path`), schema (`efficiency_result.schema.json`), and resume
  predicate (`is_efficiency_done`), all owned by `qquant.efficiency`. It reuses the
  `results_root/<variant>/` and `_meta/` path conventions and the `is_cell_done` design (structural
  validity + optional provenance match) so the orchestrator's resume/incremental-copy model is
  uniform. `results/<variant>/efficiency.json` does not collide with the matrix because "efficiency"
  is not a task id and it is a file (not a `<task>/` directory).

## Done-when (numbered, testable)

1. `qquant-profile --variant bf16` on a 4090 box writes `results/bf16/efficiency.json` for which
   `qquant.efficiency.schema.validate_efficiency(...)` passes and `schema_version == 1`.
2. **Resume:** with an existing done artifact and matching provenance, re-running **without**
   `--force` is a no-op skip that exits 0 and does not load the model; `--force` recomputes; a
   change to any `EFFICIENCY_PROVENANCE_KEYS` value makes `is_efficiency_done(result, active)` →
   `False` (stale → recompute).
3. **Self-quant labelling:** for `gptq-selfquant`/`awq-selfquant`, `speed_label ==
   "runtime-confounded"` and `confounded_metrics` is exactly the speed set
   (`prefill_tok_s, decode_tok_s, e2e_latency_s, ttft_s`); `disk` and `memory` keys are **not**
   listed. For all other variants `speed_label == "comparable"` and `confounded_metrics == []`.
4. **Throughput buckets:** `throughput` has `short/medium/long`, each with finite, positive
   `prefill_tok_s`, `decode_tok_s`, `e2e_latency_s`, `ttft_s`, reported as the **median** over
   `--repeats` after discarding `--warmup` iterations; `gen_tokens == decode_tokens` exactly.
5. **Memory ordering:** `generate_peak_bytes >= load_peak_bytes >= weights_resident_bytes > 0`,
   all finite and `<= 24 GiB`; both allocated and reserved are recorded.
6. **OOM sweep:** `max_batch_size.value >= 1`; `oom_at` is either `> value` or `null` (ceiling
   reached); the sweep runs **after** all other measurements; on `torch.cuda.OutOfMemoryError`/
   `RuntimeError("out of memory")` it `empty_cache()`s, records the boundary, and continues
   (no crash, no leak into the persisted earlier metrics).
7. **One variant per process:** missing `--variant` or more than one variant → exit 2; an id not
   in `qquant.registry.VARIANT_IDS` → exit 2 with a clear message.
8. **Determinism:** the reference generation produces exactly `decode_tokens` new tokens
   (`min_new_tokens == max_new_tokens`), `do_sample` is `False`, and two consecutive measured
   generations on the same prompt yield identical output tokens.
9. **Torch-free boundary:** `python -c "import qquant.efficiency, qquant.efficiency.schema"` does
   not pull `torch` into `sys.modules` (a fresh-interpreter test like
   `tests/test_import_torch_free.py`); the existing core torch-free guard still passes
   unchanged. `pyproject.toml` `[project.scripts]` gains `qquant-profile`.
10. **Disk size:** `disk.weights_bytes > 0`, equals the summed real sizes (symlinks resolved) of
    `*.safetensors`/`*.bin` in the resolved snapshot (HF cache for repo ids, the `local:` dir for
    self-quant), and `disk.source` ∈ {`hf-cache`, `local`}.
11. `--dry-run` prints the resolved `ProfileConfig`, the target output path, and the resume
    decision **without** importing torch or loading a model (exit 0).
12. `ruff` clean and `pytest -q` green for the new torch-free tests (GPU tests skipped off-CUDA).

## Risks & mitigations

- **OOM corrupts the CUDA context mid-sweep.** Run the sweep **last**; bound it by
  `--batch-ceiling`; catch `torch.cuda.OutOfMemoryError`/`RuntimeError("out of memory")`,
  `torch.cuda.empty_cache()` + `reset_peak_memory_stats()` between trials; if the context becomes
  unrecoverable, persist the already-computed metrics with `max_batch_size.value` = last success
  and `oom_at` set, then exit non-zero so the orchestrator sees the boundary.
- **`expandable_segments` set too late.** It must precede CUDA init; the profiler only records the
  observed value and warns if unset — the launcher (Spec 03/08) exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the process env.
- **gptqmodel JIT compile / cuBLAS autotune inflates the first iteration.** Warmup iterations
  discard it; `load_peak` may miss kernel scratch buffers, but `generate_peak` (measured after a
  real forward) captures them — that is the figure the report uses for the VRAM headroom claim.
- **Prefill/decode conflation.** Forcing `min_new_tokens == max_new_tokens` removes EOS-length
  variance; TTFT is measured with a separate `max_new_tokens=1` call so prefill and steady-state
  decode are cleanly separated.
- **Self-quant slowness misread as "INT4 is slow."** The `speed_label`/`confounded_metrics` flags
  travel with the artifact; Spec 09/10 must print the caveat. disk/memory stay un-confounded.
- **Compute-dtype confound (official fp16 vs bf16 baseline; NF4 compute dtype).** Recorded in
  `config.dtype` per variant so the report can footnote it; loading itself is Spec 04's concern.
- **HF cache layout (blobs/symlinks).** Resolve real file sizes via `os.stat` following symlinks
  and sum weight files only; locate the snapshot with `huggingface_hub.snapshot_download(...,
  revision=variant.revision, local_files_only=True)` (the snapshot is guaranteed present post-load).
- **Spec 04 loader symbol name not finalised.** This spec imports the Spec-04 loader entry symbol;
  if Spec 04 names it differently, only the single import line changes (see open questions).

---

### Open questions (for the consistency-review gate)

1. **Spec 04 loader contract.** Confirm the exact import: assumed
   `qquant.models.load_variant(variant: Variant, *, device) -> Loaded` exposing `.model`
   (HF `PreTrainedModel`, `eval()`, greedy-forced) and `.tokenizer`. Adjust the one import line in
   `profiler.py` to match Spec 04's final symbol/return type.
2. **Promote `efficiency_path` into `qquant.paths`?** Kept local to `qquant.efficiency.schema` to
   avoid editing Spec 01's module now. If the consistency gate prefers all path-building in one
   place, add `Paths.efficiency(variant)` to Spec 01 and have this module delegate to it.
3. **Per-variant batch for throughput.** The decision-log per-(variant,task) batch sizes target
   MMLU logits transients; for the throughput buckets we use batch 1 (latency-style) and report the
   max-batch sweep separately. Confirm batch 1 is the intended throughput basis for the report.
