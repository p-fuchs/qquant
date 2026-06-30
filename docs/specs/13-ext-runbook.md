# Spec 12 extensions — box promotion runbook (GATED)

How to turn ONE extension on and run it on the RTX 4090. The code is implemented and
CPU-tested on branch `spec-12-extensions`; every extension is **gated OFF** (registry rows
`enabled: false`, separate `qquant-ext` entry point, separate `results-ext/` root). Nothing
here touches the v1 SSOT or the v1 `results/` tree.

## Golden rules

- **Promote ONE extension at a time.** Flip a single row to `enabled: true`, run, exfil,
  then disable it again. Never bundle (catalog + decision-log budget discipline).
- **Reuse the disk.** `stop` the box between sessions; `destroy` only when the whole study
  is done (wiping the disk re-pays all downloads). The dead-man's switch stays DISABLED for
  the reuse workflow — tear down manually.
- **Verify library APIs on the box before a full run.** Each extension below lists the
  exact "verify-at-implementation" items the catalog flagged. Smoke them first ($0–cheap).
- All `qquant-ext` runs write to `results-ext/` (override with `--results`). Exfil that tree
  the same way the v1 run exfils `results/`.

## Common setup (eval env, box)

```bash
# v1 eval env + the gated ext deps (peft for the QLoRA adapter attach):
uv sync --group gpu --group ext
# torch-free dry run of the enabled ext matrix (no GPU):
uv run qquant-ext plan
```

---

## EXT-3 — W8A8 / SmoothQuant  (cheapest; recommended first)

**Verify on box:** the llm-compressor SmoothQuant modifier import path + the `W8A8` preset
name (`quant/selfquant/recipes.py::smoothquant_w8a8_recipe`).

```bash
# 1) Produce the checkpoint in the ISOLATED quant env (reuses the SAME C4 calibration):
cd quant && uv run python -m selfquant.quantize --id w8a8-selfquant
#    -> checkpoints/self-quant/w8a8-selfquant  (+ quant_manifest.json, scheme=W8A8)
# 2) Enable the row:
#    registries/variants.ext.yaml: w8a8-selfquant.enabled: true
# 3) Quality sweep + efficiency (eval env):
uv run qquant-ext eval --variant w8a8-selfquant
uv run qquant-ext profile --variant w8a8-selfquant
```
Report W8A8 on the **separate bit-width axis** — NOT inside the W4A16 GPTQ-vs-AWQ pair —
and label its speed **runtime-confounded** (HF backend, sm_89).

## EXT-2 — JudgeBench (LLM-as-judge meta-eval)

**Verify on box:** the HF dataset id + field names (`tasks.ext.yaml::judgebench.hf_dataset_id`,
default `ScalerLab/JudgeBench`) — adjust `qquant.ext.judge._normalize_label` /
`load_judgebench` field accessors to the real schema, then run a `--limit`-style smoke.

```bash
# Judge the v1 CORE quant variants (the headline RQ: does quantization degrade judging?):
uv run qquant-ext judge --registry v1
#   writes results-ext/<variant>/judgebench/result.json with meta.item_correct (McNemar-ready)
# Or judge ext variants:
uv run qquant-ext judge --registry ext --variant w8a8-selfquant
```
Position-bias control is on by default (both A/B orders; inconsistent → scored 0;
`meta.consistency_rate` recorded).

## EXT-4 — Quantized KV cache (efficiency sub-study)

**Verify on box:** the transformers v5 quantized-cache API (`cache_implementation="quantized"`
+ `cache_config`) and the backend dep (quanto/HQQ) — see
`qquant.ext.loaders.build_kv_cache_generation_kwargs`. Install the backend in the eval env.

```bash
# Enable bf16-kvq4 / bnb-nf4-kvq4 in variants.ext.yaml, then:
uv run qquant-ext profile --variant bf16-kvq4     # vs the bf16 fp16-KV baseline
uv run qquant-ext profile --variant bnb-nf4-kvq4
# Optional lossy-cache quality check:
uv run qquant-ext eval --variant bf16-kvq4 --task mmlu --task gsm8k
```
Quantized KV is **lossy** → greedy output may differ from fp16-KV; that delta IS the
measurement. Do not compare kvq quality to v1 numbers as if loss-free.

## EXT-1 — Mistral-7B-Instruct as a 2nd model

**Verify on box:** pin a Mistral base SHA (`variants.ext.yaml` `revision: null` → real SHA);
confirm whether official Mistral GPTQ/AWQ checkpoints exist (set `model_id`/`revision`, else
keep those rows disabled and rely on the Mistral self-quants — mirrors v1 awq-official).

```bash
# Self-quant Mistral via the SAME C4 protocol (quant env; needs a base-model flag):
cd quant && uv run python -m selfquant.quantize --id gptq-selfquant \
    # (extend the quant driver's base-model arg for Mistral at promotion)
# Enable mistral-* rows, then sweep into a SEPARATE results root:
uv run qquant-ext eval --results results-ext/mistral --variant mistral-bf16 ...
```
Budget: ~1× the whole v1 GPU spend → run as a separate budgeted campaign, never bundled.

## EXT-5 — QLoRA recovery (only extension with a training cost)

**Verify on box:** resolve + pin `peft` (eval env, already in `--group ext`) and `trl` +
`datasets` (quant/training env); choose a GENERAL instruction dataset.

```bash
# Train the adapter in the isolated env (CONTAMINATION GUARD fails fast on eval-task data):
cd quant && uv run python -m selfquant.train_qlora --dataset <general-instruct-id>
#   -> checkpoints/qlora/bnb-nf4-qlora  (+ qlora_manifest.json)
# Enable bnb-nf4-qlora, then evaluate (loader attaches the adapter over the frozen NF4 base):
uv run qquant-ext eval --variant bnb-nf4-qlora
```
The report MUST state the training corpus + the disjointness argument (recovery, not
leakage). The recovery claim (`bnb-nf4` → `bnb-nf4-qlora`) is gated by paired McNemar.

---

## Exfil + teardown

```bash
# pull results-ext/ (and any new checkpoints/) back to the laptop, then:
uv run vastai stop instance <id>      # preserves the disk for the next extension
# uv run vastai destroy instance <id> -y   # ONLY when the whole study is finished
```
