# Decision Log

Locked, verified decisions for the qquant study. Pins were **verified against live PyPI on
2026-06-22** by resolving the universal `uv.lock` (the listed eval-env versions are the ones
uv actually selected). Every spec must honour these; deviations require updating this file.

## Platform

- **GPU provider: vast.ai** (not the MIMUW lab cluster). **One on-demand RTX 4090 (24 GB,
  Ada, sm_89)**. All 7 variants — including the BF16 baseline (~15 GB weights) — run on this
  **single homogeneous GPU**, so the speed comparison is apples-to-apples (no cross-hardware
  caveat). Budget ceiling **$25–75**.
- **Instances: on-demand** (stable, per-second billing). The resume harness is a safety net,
  not load-bearing.
- **Repo delivery: public GitHub repo.** The instance clones it over HTTPS on boot. No
  secrets in the repo; `VAST_API_KEY`/`HF_TOKEN` etc. are injected via env (`.env.example`).

## Two uv environments (NOT one)

`llm-compressor 0.12.0` caps `transformers<=5.10.1` with tight transitive pins; co-resolving
it with `gptqmodel` + `lm-eval` in one lock is brittle. So there are two **separate** uv
projects:

### root eval env (`./pyproject.toml`, `./uv.lock`) — resolved & verified

| Package | Resolved version | Notes |
|---|---|---|
| torch | **2.12.1+cu126** | gptqmodel 7.1 floor is `>=2.8`; no upper cap hit. cu126 wheels run on sm_89. |
| transformers | **5.10.1** | v5: `quantization_config=` is the canonical path; `load_in_8bit/4bit` shortcut kwargs removed; `torch_dtype`→`dtype`; default dtype `auto`. |
| gptqmodel | **7.1.0** | GPTQ backend (autogptq removed). JIT-compiles Marlin/AWQ kernels for sm_89 → needs nvcc. Install `--no-build-isolation`. |
| optimum | **2.2.0** | GPTQ/AWQ glue transformers imports. |
| bitsandbytes | **0.49.2** | LLM.int8() + NF4; ships prebuilt sm_89 cubins (no nvcc needed). Linux-only. |
| accelerate | **1.13.0** | device_map loading. |
| compressed-tensors | **0.17.1** | to **load** self-quant checkpoints natively in transformers v5. |
| lm-eval | **`c6491878…`** (main) `[hf,ifeval,math]` | ✅ Spec-02 spike (2026-06-23) confirmed lm-eval `main` @ `c6491878fb0a9d627dff50336c98bbf39e3b56e4` works with transformers 5.10.1 (scores MMLU + runs `generate_until`). `0.4.12` remains the fallback pin. |
| torchvision | **0.27.1+cu126** | gptqmodel imports it at model-load time but doesn't declare it (Spec-02 spike). |
| ninja | **1.13.0** | gptqmodel JIT-compiles its sm_89 Marlin extension at first load and hard-requires ninja **on PATH**. |
| numpy | **2.2.6** | pinned by gptqmodel. |

Universal lock covers `macos-arm64` (core+dev only; torch-free) and `linux-x86_64` (full
`gpu` group). `bitsandbytes`/`gptqmodel`/`optimum`/`torch(cu126)` are `sys_platform=='linux'`.

### Spec-02 go/no-go spike outcome — **GO** (2026-06-23, real RTX 4090)

All 8 on-box checks passed (`results/_meta/spike.json`, archived at
`docs/specs/artifacts/spike-verdict.json`): `nvcc_matches_torch` (12.6==12.6),
`mmlu_subjects_match` (57), `chat_template_renders`, `mmlu_subtask_scored`,
`generate_until_ok`, `bnb_loads` (int8+NF4), `gptq_loads` (Marlin sm_89 JIT), and
`awq_official_loads` (**so `awq-official` stays enabled** — the contingency did not fire).
Resolved working **lm-eval ref = `c6491878fb0a9d627dff50336c98bbf39e3b56e4`** (main) with
transformers 5.10.1 + torch 2.12.1+cu126. The gate that blocks Specs 04–10 is cleared.

**Bare-image bootstrap requirements discovered on the box** (folded into
`scripts/spike/bootstrap.sh` + the `gpu` group; Spec 03 must mirror them):

- `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` has no python/pip/torch and **no
  `/workspace`** — install everything via `uv`; `mkdir -p /workspace` before use.
- `setuptools>=77.0.1,<83`: gptqmodel + its sdist-only deps `tokenicer`/`logbar` use the
  PEP 639 string license and fail to build under `setuptools<77`.
- **`torchvision`** (cu126) — gptqmodel imports it at load but doesn't declare it.
- **`ninja`** + a C/C++ toolchain (`build-essential`) on PATH — gptqmodel JIT-compiles its
  sm_89 Marlin kernel at first load; torch path-looks-up `ninja` (export `venv/bin`).
- ssh sessions don't inherit the image PATH (`export /usr/local/cuda/bin`), and
  `vastai copy` is a no-op local→instance (use `scp -i`); `vastai destroy` needs `-y`.
- **gsm8k**: lm-eval's gsm8k task uses the bare `gsm8k` id (renamed to `openai/gsm8k`);
  `datasets>=4` rejects it. **The real GSM8K eval (Spec 05/06) must use `openai/gsm8k`.**

### quant env (`./quant/pyproject.toml`, `./quant/uv.lock`) — finalized & committed on the box (Spec 07)

`llmcompressor==0.12.0` → pulls `transformers` 5.9–5.10.1, `compressed-tensors==0.17.1`,
`torch` 2.10–2.12+cu126, `datasets` 4.8.4–5.0, `accelerate<=1.13.0`, `auto-round` 0.10–0.13.
The eval env only needs `compressed-tensors` to load what this env produces.

- **NEVER install `autoawq`** (archived; pins `transformers~=4.5x` → breaks the whole stack).

## Pinned model revisions (commit SHAs)

| Variant | HF repo | revision |
|---|---|---|
| `bf16`, `bnb-int8`, `bnb-nf4` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `gptq-official` | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | `e9c932ac1893a49ae0fc497ad6e1e86e2e39af20` |
| `awq-official` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `b25037543e9394b818fdfca67ab2a00ecc7dd641` |

## vast.ai

- CLI `vastai>=1.1.1`. Poll **`actual_status==running`** (via `--raw`), and handle
  `exited`/`unknown`/`offline` → destroy + retry (never loop forever accruing charges).
- **Image: a CUDA 12.6 *devel* PyTorch image (with `nvcc`), pinned by digest.** gptqmodel
  JIT-compiles sm_89 kernels on first use; bootstrap **fails fast** if
  `nvcc` major.minor ≠ `torch.version.cuda`. (bitsandbytes does not need nvcc.) Default image
  constant: `QQUANT_IMAGE = "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"` (CUDA 12.6 devel →
  `nvcc` 12.6 to match torch's cu126 build; uv installs Python 3.11 + torch cu126 into the
  venv). The exact available tag, SSH/`--direct` compatibility, and the `sha256` digest are
  **VALIDATED AND RECORDED by the Spec 02 go/no-go spike on first pull** (the spike writes the
  resolved digest back here / into a constant). Spec 02 and Spec 08 default their `--image` to
  `QQUANT_IMAGE`.
- Offer query: `gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true
  direct_port_count>=1 cuda_max_good>=12.6`, sort `dlperf_usd-`.
- `--disk 120`. **`destroy` wipes the disk** → results must be copied out + reconciled first.
- There is **no native auto-destroy TTL**; client-side guards + an on-instance dead-man's
  switch + an account spend limit are all required.

## Variant & evaluation decisions

- **7 variants** (canonical ids, hyphenated, no aliases): `bf16`, `bnb-int8`, `bnb-nf4`,
  `gptq-official`, `awq-official`, `gptq-selfquant`, `awq-selfquant`. Each has an `enabled` flag.
- **`awq-official` is CONTINGENT**: smoke-test the load on sm_89 with only `gptqmodel`+`optimum`
  (no autoawq); if it fails, set `enabled=false`. The **guaranteed** AWQ data point is
  **`awq-selfquant`** (compressed-tensors, loaded natively).
- **Self-quant = fair *quality*, confounded *speed*.** compressed-tensors W4A16 on the HF
  `generate()` backend may take a slow dequant path on sm_89 (fast Marlin targets vLLM). Label
  self-quant speed "runtime-confounded"; use self-quant for the calibration-controlled
  **quality** comparison. Scheme = **W4A16_ASYM for both** GPTQ and AWQ (controlled axis =
  calibration set + group_size). GPTQ-asym disables Marlin → official-vs-self GPTQ speed differs.
- **Greedy determinism.** Qwen ships `generation_config` with `do_sample=true`; force
  `do_sample=False` and clear `temperature/top_p/top_k` at load **and** in `gen_kwargs`. A
  determinism smoke runs a task twice and asserts identical output.
- **Per-task chat-template policy** (NOT blanket): MMLU/GSM8K → `apply_chat_template=True` +
  `fewshot_as_multiturn=True`; IFEval → `apply_chat_template=True`, 0-shot; HumanEval →
  `humaneval_instruct` + fenced extraction. Validate baseline HumanEval pass@1 is sane (>0.5).
- **NF4** sets `bnb_4bit_compute_dtype=torch.bfloat16` (defaults to fp32 → unfair speed). Use
  `dtype=` not `torch_dtype=`.
- **MMLU** = full, **57 per-subject cells**; aggregate weighted by subject n; **one** stderr
  method (error-propagation of per-subject binomial stderr), reconciled with lm-eval. Confirm
  which MMLU dataset lm-eval resolves.
- **VRAM budget includes the logits transient** (`batch×seq×vocab(152064)×2B` ≈ 3.5 GB @
  batch 4 / seq 3k). Per-(variant,task) batch: **bf16 & bnb-int8 MMLU batch = 2**, 4-bit
  variants batch = 4. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Significance** = paired **McNemar** on per-item correct vectors (from `log_samples`), not
  unpaired z. Gate all quality claims.
- **HumanEval code-exec**: the disposable container *is* the sandbox; export **both**
  `HF_ALLOW_CODE_EVAL=1` and `QQUANT_ALLOW_CODE_EXEC=1`; report Wilson CI over k/n (lm-eval
  may not emit a pass@1 stderr).
- **IFEval**: set `langdetect.DetectorFactory.seed=0`; bootstrap downloads nltk `punkt`/`punkt_tab`.
- **Dataset reproducibility**: record each task dataset's resolved revision/fingerprint in
  `results/_meta/env.json`; pre-flight that all four task datasets load under the pinned
  `datasets` version (4.x removed loading scripts).

## Orchestration durability & cost

- **Incremental copy** each poll; **best-effort final copy** on every teardown path (bounded
  timeout); **reconcile** local cell-count vs the remote `done_manifest.json` before destroy.
- **Auto-destroy defense-in-depth**: local `finally`/atexit/signal handlers **+** an on-instance
  dead-man's switch (`sleep MAX_RUNTIME` → self-destroy via vast API) **+** a local watchdog
  (harness `CronCreate`) **+** a vast.ai **account spending limit** (operator pre-req).
- **Persist self-quant checkpoints** (exfil + re-upload on resume) so a resume never re-quantizes.
- **BudgetGuard** counts compute **+ storage + bandwidth + image-pull** (not compute only),
  ceiling **$60** (margin under the $75 hard cap); a **cost ledger** persists cumulative actual
  spend across runs.

## Reporting

- **No PDF / LaTeX.** The final deliverable is a **markdown** results document
  (`results/REPORT.md`) with tables inline and plots linked, which the authors rewrite into
  their report. (User decision, 2026-06-22.)

## Scope

- **v1 = "Core + self-quant fairness"** (the 7 variants above). The 5-variant core (official
  checkpoints) is valid and self-contained on its own; the 2 self-quant variants are a
  **deferrable fast-follow** (Spec 08) that can be dropped if budget/time runs short.
- **Spec-only extensions** (documented, not implemented): Mistral-7B 2nd model, JudgeBench,
  W8A8/SmoothQuant, quantized KV cache, QLoRA recovery.

## Expected results (priors for RQ#3)

bnb int8 is typically *slower* than bf16 (outlier decomposition); official GPTQ/AWQ
(Marlin/GEMM) are the faster INT4 paths; NF4 is slower (dequant) — i.e. memory savings, not
speed. Official checkpoints are fp16 (minor compute-dtype confound vs the bf16 baseline).
