# qquant specification set

Implementation specs for the Qwen2.5-7B quantization study. Each `NN-*.md` is **self-contained**
so an independent agent can execute it without this conversation. Read these two first:

- [`contracts.md`](contracts.md) — single source of truth (ids, paths, schema, predicates, entry points).
- [`decision-log.md`](decision-log.md) — verified pins/SHAs, platform, and every locked decision.

## Specs

| # | Spec | Depends on |
|---|---|---|
| [00](00-overview.md) | Overview, matrix/state model, contracts index, glossary | — |
| [01](01-scaffold.md) | Repo scaffold + torch-free `qquant` core (**implemented**) | 00 |
| [02](02-vastai-wrapper-and-spike.md) | vast.ai lifecycle wrapper + go/no-go GPU spike | 01 |
| [03](03-remote-scripts.md) | Remote bootstrap scripts (onstart / bootstrap / run_matrix) | 02, 11 |
| [04](04-variant-loaders.md) | Unified variant loaders (bf16, bnb, gptq/awq official, compressed-tensors) | 01, 02 |
| [05](05-quality-eval.md) | Quality eval: preloaded HFLM + resumable cells + `qquant-eval` | 04 |
| [06](06-efficiency.md) | Efficiency profiler `qquant-profile` | 04 |
| [07](07-self-quant.md) | Self-quant fairness: isolated `quant/` env + C4 calibration + checkpoints | 01, 04 |
| [08](08-orchestration-driver.md) | `qquant-orchestrate`: run + budget guard + auto-destroy + exfil | 02, 03, 05, 06, 07 |
| [09](09-aggregation.md) | Aggregation + stats (MMLU stderr, Wilson CI, paired McNemar) + plots | 05, 06, 07 |
| [10](10-results-markdown.md) | `results/REPORT.md` (markdown — **no PDF/LaTeX**), mapped to the RQs | 09 |
| [11](11-ci-and-repro.md) | CI, determinism contract, dataset/model pre-download caching | 01 |
| [12](12-extensions-catalog.md) | Extensions catalog (SPEC-ONLY): Mistral, JudgeBench, W8A8, KV-cache, QLoRA | — |

> Numbering note: this is the **single monotonic** scheme. The earlier draft collided ids
> (two `03`s/`05`s/`06`s by merging an orchestration family with an eval family); they are
> renumbered here. `depends_on` uses these ids only.

## Status

- **Spec 01 is implemented** and green locally (`uv sync`, ruff, pytest, torch-free guard, CI).
- **Spec 02's spike is the go/no-go gate** for lm-eval × transformers v5 before any large GPU spend.
