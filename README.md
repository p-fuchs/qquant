# qquant — Quantization for Efficient LLM Inference

Reproducible evaluation of post-training quantization for **Qwen2.5-7B-Instruct**: a
BF16 baseline vs. 8-bit, 4-bit NF4, GPTQ, and AWQ variants (plus calibration-controlled
self-quantized GPTQ/AWQ), measured on **quality** (MMLU, GSM8K, HumanEval, IFEval) and
**efficiency** (memory, throughput, latency). Runs on a single **RTX 4090 (24 GB)** rented
on-demand from **vast.ai**.

> Course project — Hubert Michalski & Przemysław Fuchs. See `proposal.tex` for the original
> proposal and `docs/specs/` for the implementation specification set.

## Layout

| Path | What |
|---|---|
| `src/qquant/` | **torch-free core**: config, registries, paths, matrix/state, CLI. Importable anywhere (incl. macOS, no GPU). |
| `quant/` | **isolated uv env** for producing self-quant checkpoints with `llm-compressor` (separate from the eval env — see decision log). |
| `docs/specs/` | the spec set (`00`–`12`) + `contracts.md` (single source of truth) + `decision-log.md` (verified pins, SHAs, risks). |
| `results/` | experiment outputs (git-ignored). Final analysis is written to `results/REPORT.md` (markdown — no PDF). |

## Quickstart (local, no GPU)

```bash
uv sync                 # installs the torch-free core + dev tools
uv run qquant variants  # list the variant matrix
uv run qquant tasks     # list the benchmark tasks
uv run qquant matrix    # total cell count (variants × tasks, MMLU expanded to 57)
uv run ruff check . && uv run pytest
```

The GPU stack (`transformers`, `gptqmodel`, `bitsandbytes`, `lm-eval`, …) lives in the
`gpu` dependency group and is installed only on the Linux instance:

```bash
uv sync --group gpu     # Linux + CUDA only
```

## Environments

Two **separate** uv projects by design (see `docs/specs/decision-log.md`):

- **root** (`./pyproject.toml`) — eval/inference: `transformers` v5 + `gptqmodel` + `bitsandbytes` + `lm-eval`, plus `compressed-tensors` to *load* self-quant checkpoints.
- **`quant/`** (`./quant/pyproject.toml`) — *produce* self-quant checkpoints with `llm-compressor` (its tight `transformers<=5.10.1` pins are incompatible with a single shared lock).

## Secrets

No secrets in the repo. Copy `.env.example` → `.env` and set `VAST_API_KEY` etc. Also set an
account-level **spending limit** in the vast.ai console as a server-side cost backstop.
