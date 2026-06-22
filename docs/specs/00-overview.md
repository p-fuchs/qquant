# Spec 00 — Project Overview, Matrix & State Model

- **Status:** stable (map/index document; no code)
- **Depends on:** — (none)
- **Owns / Produces:** this document (`docs/specs/00-overview.md`). It is the orientation
  map for the whole spec set. It **owns no ids, paths, schema, or predicates** — those live
  in [`contracts.md`](contracts.md) (SSOT) and [`decision-log.md`](decision-log.md); this
  page only references them by name.

## Purpose

Give an independent agent the one-page mental model of the **qquant** study before they open
any other spec: what we are measuring, against which variants and tasks, where results live on
disk, how resume works, and which spec owns each piece. Everything normative (ids, file
layout, cell schema, the resume predicate, entry points) is defined once in `contracts.md` and
the verified pins/decisions in `decision-log.md`; this overview must never contradict or
re-declare them — when in doubt those two files win.

**One sentence:** qquant is a post-training-quantization evaluation of
**Qwen2.5-7B-Instruct** across **7 variants × 4 task families (60 cells each) = 420 result
cells**, run on a **single on-demand RTX 4090 (24 GB, sm_89) on vast.ai**, comparing quality
and inference efficiency, with the filesystem itself as the unit of state so any run is
fully resumable.

## Scope (in / out)

**In scope (this document):**
- The 3 core research questions (RQs) from `proposal.tex` + the optional QLoRA RQ, each mapped
  to the spec / metric that answers it.
- The full **(variant × task) matrix** and its arithmetic (420 cells), the `enabled` flags, and
  the **v1 = "Core + self-quant fairness"** tiering (5-variant official core is a valid
  standalone minimal version; the 2 self-quant variants are a deferrable fast-follow).
- The **filesystem-as-state** model (per-cell JSON + `_meta`) and how `is_cell_done` /
  `missing_cells` make runs resumable — referenced, not redefined.
- A one-line index of specs 01–12 and a glossary.

**Out of scope (owned elsewhere):**
- Defining `VARIANT_IDS` / `TASK_IDS` / `MMLU_SUBJECTS`, the `cell_path` layout,
  `cell_result.schema.json`, `PROVENANCE_KEYS`, `is_cell_done`, or `[project.scripts]`
  → **`contracts.md` + Spec 01**.
- Verified version pins / commit SHAs / platform / budget / determinism / batch sizing /
  significance method → **`decision-log.md`** (and the spec that implements each).
- Any executable code, registry edits, or CLI behaviour → the numbered specs below.

## Interface

This document's "interface" is the canonical orientation tables. They mirror the SSOT
(`src/qquant/registry.py`, `registries/*.yaml`, `contracts.md`); on any mismatch the SSOT
governs.

### Research questions → evidence map

From `proposal.tex` (verbatim intent), mapped to the spec and metric that answers each:

| RQ | Question | Answered by | Primary evidence |
|----|----------|-------------|------------------|
| **RQ1** | Which quantization method gives the best **quality–memory–speed trade-off** for a 7B-class instruction-tuned LLM? | Spec 05 (quality) + Spec 06 (efficiency) → Spec 09/10 (synthesis) | quality-vs-VRAM and quality-vs-throughput trade-off tables/plots in `results/REPORT.md` |
| **RQ2** | Are some **capabilities more sensitive** to quantization than others (reasoning / coding / QA / instruction following)? | Spec 05 + Spec 09 | per-task degradation vs `bf16`, per-MMLU-subject breakdown, paired **McNemar** significance gate |
| **RQ3** | Does quantization **improve real inference speed**, or mainly reduce **memory**? | Spec 06 (`qquant-profile`) + Spec 09 | model size on disk, peak VRAM (load vs generate), prefill/decode throughput, max batch before OOM — priors: `bnb-int8` often slower; NF4 = memory-not-speed; official GPTQ/AWQ are the faster INT4 paths |
| **RQ4** *(optional)* | Can light **QLoRA-style adaptation** recover quality lost after 4-bit quantization? | Spec 12 (**spec-only**, not implemented in v1) | documented design only — out of the v1 run |

### Variant roster (the 7 canonical ids)

Ids are exactly `VARIANT_IDS` from `qquant.registry`; flags mirror
`registries/variants.yaml`. Do not invent aliases or `-int4` suffixes.

| Variant id | `quant_method` | `source` | Tier | `enabled` |
|------------|----------------|----------|------|-----------|
| `bf16` | `bf16` | baseline | **official core** | true |
| `bnb-int8` | `bnb-int8` | load-time | **official core** | true |
| `bnb-nf4` | `bnb-nf4` | load-time | **official core** | true |
| `gptq-official` | `gptq` | official | **official core** | true |
| `awq-official` | `awq` | official | **official core** | true *(CONTINGENT — see below)* |
| `gptq-selfquant` | `compressed-tensors` | selfquant | self-quant fast-follow | true |
| `awq-selfquant` | `compressed-tensors` | selfquant | self-quant fast-follow | true |

- **`awq-official` is contingent**: a sm_89 load smoke-test (gptqmodel+optimum, **no autoawq**)
  in the **Spec 02 spike / Spec 08 pre-flight** decides it; if it fails, set `enabled=false` and
  rely on `awq-selfquant` as the guaranteed AWQ data point.
- **Self-quant variants** load natively via `compressed-tensors` (W4A16_ASYM from the shared
  C4 calibration). They give a **fair quality** comparison but **runtime-confounded speed**
  (HF `generate()` may take a slow dequant path on sm_89).

### Task roster (the 4 canonical ids)

Ids are exactly `TASK_IDS`; per-task eval config is owned by `registries/tasks.yaml` (shown
here for orientation only):

| Task id | lm-eval task | Primary metric | Cells / variant | Chat template / few-shot |
|---------|--------------|----------------|-----------------|--------------------------|
| `mmlu` | `mmlu` (group) | `acc` | **57** (one per `MMLU_SUBJECTS`) | `apply_chat_template=True`, `fewshot_as_multiturn=True`, 5-shot |
| `gsm8k` | `gsm8k` | `exact_match` | 1 | chat-template + multiturn, 5-shot |
| `humaneval` | `humaneval_instruct` | `pass@1` | 1 | instruct + fenced-code extraction, 0-shot, code-exec |
| `ifeval` | `ifeval` | `prompt_level_strict_acc` | 1 | chat-template, 0-shot |

### The matrix (arithmetic)

```
per variant   = 57 (MMLU subjects) + 1 (gsm8k) + 1 (humaneval) + 1 (ifeval) = 60 cells
v1 full       = 7 variants × 60 = 420 cells            # all 7 enabled
official core = 5 variants × 60 = 300 cells            # bf16, bnb-int8, bnb-nf4, gptq-official, awq-official
self-quant    = 2 variants × 60 = 120 cells            # gptq-selfquant, awq-selfquant  (fast-follow)
if awq-official disabled → core = 4 × 60 = 240 cells   # awq-selfquant becomes the only AWQ point
```

`qquant matrix` prints these counts from the enabled registry; `expand_matrix(variants,
tasks)` produces the `Cell` list (MMLU → 57 per-subject cells). MMLU is aggregated **weighted
by per-subject n** downstream (Spec 09), never as a flat mean.

### Filesystem-as-state model

The on-disk tree (built **only** via `cell_path` / `Paths`, defined in `contracts.md` §3) *is*
the run state — there is no database:

```
results/<variant>/<task>/<subject>.json   # MMLU: 57 per-subject cells (subject = bare slug, e.g. anatomy)
results/<variant>/<task>/result.json      # single-cell tasks (gsm8k, humaneval, ifeval)
results/_meta/env.json                    # run provenance: lib + dataset revisions, image digest
results/_meta/done_manifest.json          # cell ids the remote run completed (copy-out reconciliation)
```

- Each cell is a JSON document validated against `cell_result.schema.json` (schema_version 1).
- **Resume** is decided solely by `is_cell_done(cell, active_config)` (the SSOT predicate):
  structural validity always applies, and when an `active_config` provenance block is supplied,
  every overlapping `PROVENANCE_KEYS` value must match — otherwise the cell is **stale** and
  recomputed. `missing_cells(...)` returns the work still to do. No spec adds a parallel
  `completed:true` / `status=='complete'` flag.
- Provenance for a cell is produced by `cell_provenance(run, task, variant_id)` and stored in
  the cell's `config` block, so it round-trips with the schema and the resume predicate.

### Spec index (01–12)

One line each; `depends_on` and the authoritative table live in [`README.md`](README.md).

| # | Spec | What it lands |
|---|------|---------------|
| 00 | Overview | this map: RQs, matrix, state model, glossary (depends on: —) |
| 01 | Scaffold | repo + torch-free `qquant` core + the SSOT contracts (**implemented**) |
| 02 | vast.ai wrapper + spike | lifecycle wrapper and the **go/no-go** lm-eval × transformers-v5 GPU spike |
| 03 | Remote scripts | onstart / bootstrap / run_matrix bootstrap on the box |
| 04 | Variant loaders | unified loaders: bf16, bnb int8/nf4, gptq/awq official, compressed-tensors |
| 05 | Quality eval | preloaded HFLM, resumable cells, `qquant-eval` |
| 06 | Efficiency | `qquant-profile`: size/VRAM/prefill/decode/latency/max-batch |
| 07 | Self-quant | isolated `quant/` env + shared C4 calibration + W4A16_ASYM checkpoints |
| 08 | Orchestration driver | `qquant-orchestrate`: run + BudgetGuard + auto-destroy + incremental exfil |
| 09 | Aggregation + stats | MMLU weighted stderr, Wilson CI, paired McNemar, plots |
| 10 | Results markdown | `results/REPORT.md` (**no PDF/LaTeX**), mapped back to the RQs |
| 11 | CI & repro | CI, determinism contract, dataset/model pre-download caching |
| 12 | Extensions catalog | **spec-only**: Mistral, JudgeBench, W8A8, KV-cache, QLoRA |

## Files to create

- `docs/specs/00-overview.md` — **this document only.** Spec 00 writes no code and edits no
  registry, contract, README, or other spec.

## Implementation notes (the decisions that bind THIS spec, with the "why")

- **v1 = "Core + self-quant fairness."** The 5-variant official core (`bf16`, `bnb-int8`,
  `bnb-nf4`, `gptq-official`, `awq-official`) is exactly the proposal's "minimal successful
  version" and is **valid and self-contained on its own**; the 2 self-quant variants are a
  **deferrable fast-follow** (Spec 07/08) that may be dropped if budget/time runs short. The
  overview must present the tiering this way so a reader knows the core can ship without
  self-quant. *(decision-log → Scope.)*
- **Single homogeneous GPU.** All 7 variants — including the ~15 GB BF16 baseline — run on one
  RTX 4090, so the speed comparison is apples-to-apples with no cross-hardware caveat. The
  overview states this because it is the precondition that makes RQ3's speed claims meaningful.
  *(decision-log → Platform.)*
- **Self-quant = fair quality, confounded speed.** W4A16_ASYM for both GPTQ and AWQ from the
  **same** C4 calibration controls the quality axis; HF-`generate()` dequant on sm_89 confounds
  the speed axis. The overview labels self-quant speed "runtime-confounded" so RQ3 is read
  correctly. *(decision-log → Variant & evaluation.)*
- **awq-official is contingent.** The overview must flag it as such (not as a guaranteed point)
  and name `awq-selfquant` as the guaranteed AWQ fallback. *(decision-log → Variant.)*
- **MMLU is 57 per-subject cells, weighted aggregate.** The matrix arithmetic (60/variant, 420
  total) depends on this; the overview shows the count, never a 1-cell MMLU. *(decision-log →
  MMLU.)*
- **Filesystem-as-state + one resume predicate.** Resume is `is_cell_done` only; the overview
  reinforces "no parallel completion flag" so downstream specs don't invent one.
  *(contracts.md §5.)*
- **No PDF / LaTeX.** The deliverable is a markdown `results/REPORT.md`; the overview names it
  so RQ→deliverable mapping is unambiguous. *(decision-log → Reporting.)*
- **QLoRA (RQ4) is optional and spec-only.** It maps to Spec 12, not the v1 run, so the
  overview lists it as optional and excluded from the 420-cell matrix. *(decision-log → Scope.)*

## Done-when (numbered, testable)

1. The file exists at `docs/specs/00-overview.md` and is referenced by `README.md`'s spec
   table (row 00, `depends_on` empty).
2. It lists all 3 core RQs **plus** the optional QLoRA RQ, matching `proposal.tex` §"Research
   Questions", and maps each RQ to a spec and a metric.
3. The matrix arithmetic is stated as **7 × 60 = 420** cells and is consistent with
   `qquant matrix` (all 7 enabled) and with `qquant matrix` showing **300** cells when only the
   5-variant official core is enabled; it explicitly notes the 240-cell case if `awq-official`
   is disabled.
4. It uses **only** the canonical ids — `bf16`, `bnb-int8`, `bnb-nf4`, `gptq-official`,
   `awq-official`, `gptq-selfquant`, `awq-selfquant` and `mmlu`, `gsm8k`, `humaneval`, `ifeval`
   (hyphens, no aliases, no `-int4`) — matching `VARIANT_IDS` / `TASK_IDS`.
5. It describes the `results/<variant>/<task>/...` + `_meta/{env,done_manifest}.json` layout and
   the `is_cell_done` / `missing_cells` resume model by **reference** to `contracts.md`, without
   re-declaring any path, schema key, or predicate.
6. It contains a one-line index for **every** spec 00–12 consistent with `README.md`, and a
   glossary defining at least: **variant, cell, provenance, prefill/decode, W4A16_ASYM,
   dead-man's switch**.
7. *(Non-gating review note — NOT a testable gate.)* A manual consistency review **should**
   confirm this document does not drift from the SSOT on variant ids, task ids, cell paths,
   schema keys, and entry points. This is a review aid, not a pass/fail criterion. The
   automatable subset — that the spec tables' variant/task ids are a **subset** of
   `VARIANT_IDS`/`TASK_IDS` — is an optional light CI check owned by Spec 11
   (`tests/test_spec_consistency.py`).

## Risks & mitigations

- **Drift between this overview and the SSOT** (someone edits a count/flag here instead of in
  the registry). *Mitigation:* this doc declares the registries + `contracts.md` authoritative
  and presents tables as mirrors; a non-gating consistency review (Done-when 7) plus Spec 11's
  optional `tests/test_spec_consistency.py` (ids ⊆ `VARIANT_IDS`/`TASK_IDS`) catch drift.
- **Reader mistakes the 420-cell number as fixed.** *Mitigation:* the arithmetic block shows
  the count is a function of `enabled` variants (420 / 300 / 240) and that `qquant matrix`
  recomputes it, so a contingent/dropped variant changes the total predictably.
- **`awq-official` read as guaranteed.** *Mitigation:* flagged CONTINGENT in the roster and in
  implementation notes, with `awq-selfquant` named as the guaranteed AWQ point.
- **RQ3 misread (memory vs speed).** *Mitigation:* the RQ map carries the priors (bnb-int8
  slower, NF4 memory-not-speed) and labels self-quant speed "runtime-confounded", steering the
  reader to Spec 06's measured efficiency rather than assuming 4-bit ⇒ faster.

## Glossary

- **Variant** — one of the 7 canonical model builds in `VARIANT_IDS` (a precision/quantization
  configuration of Qwen2.5-7B-Instruct, e.g. `bnb-nf4`, `gptq-official`). Defined in
  `registries/variants.yaml`; loaded by `quant_method`.
- **Task** — one of the 4 benchmark families in `TASK_IDS` (`mmlu`, `gsm8k`, `humaneval`,
  `ifeval`) with locked eval config in `registries/tasks.yaml`.
- **Cell** — the atomic unit of work and of state: one `(variant, task[, subject])` result,
  identified by `cell_id` (`<variant>/<task>[/<subject>]`) and persisted to one JSON file via
  `cell_path`. MMLU produces 57 cells (one per subject); other tasks produce one (`result.json`).
- **Provenance** — the reproducibility block (`PROVENANCE_KEYS`: seeds, batch_size, num_fewshot,
  apply_chat_template, fewshot_as_multiturn, max_length, lm_eval_version, model_revision) stored
  in each cell's `config`, produced by `cell_provenance(...)`. On resume, a mismatch marks the
  cell **stale** so it is recomputed rather than silently reused.
- **Prefill / decode** — the two phases of autoregressive generation profiled for RQ3: *prefill*
  is the parallel forward pass over the whole prompt (compute-bound; the prefill-throughput
  number); *decode* is generating output tokens one at a time (memory-bandwidth-bound; the
  decode-throughput number). Quantization can change these very differently, which is the heart
  of "memory savings vs real speedup".
- **W4A16_ASYM** — the self-quant scheme: **W**eights in **4** bits, **A**ctivations in **16**
  bits, **asym**metric (non-zero zero-point) quantization, applied with group_size from the
  shared C4 calibration. Used identically for `gptq-selfquant` and `awq-selfquant` so the
  controlled axis is only the calibration algorithm. (GPTQ-asym disables the Marlin kernel,
  which is part of why self-quant speed is runtime-confounded.)
- **Dead-man's switch** — an on-instance safety timer (`sleep MAX_RUNTIME` then self-destroy via
  the vast API) that tears the rented box down even if the local orchestrator dies. It is one
  layer of the auto-destroy defense-in-depth (local `finally`/atexit/signal handlers + on-box
  dead-man's switch + a local watchdog + a vast.ai account spend limit) that prevents an
  abandoned instance from accruing charges. Owned by Spec 03/08.
