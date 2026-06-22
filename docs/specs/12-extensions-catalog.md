# Spec 12 — Extensions catalog (SPEC-ONLY)

- **Status:** Documented (spec-only) — **NOT implemented in v1**. This file adds **no** runtime
  code, **no** `[project.scripts]` line, **no** registry/schema/contract edits. It is a catalog of
  five deferred extensions, each described precisely enough that a future agent can *promote* it
  into its own numbered spec (13+) and implement it without re-deriving the design.
- **Depends on:** — (none). The catalog introduces **no** edge into the v1 dependency graph. Each
  entry *references* existing specs/contracts by name; promoting an entry creates its own
  `depends_on`.
- **Owns / Produces:** this catalog document only. Per entry it lists the files/edits a *future*
  spec would produce; it produces **none** of them now.

## Purpose

The v1 scope is locked at **"Core + self-quant fairness"** — the 7 canonical variants
(`bf16`, `bnb-int8`, `bnb-nf4`, `gptq-official`, `awq-official`, `gptq-selfquant`,
`awq-selfquant`) × the 4 tasks (`mmlu`, `gsm8k`, `humaneval`, `ifeval`) on one RTX 4090
(decision-log "Scope"). The decision log also names five **spec-only extensions** that are
deliberately out of v1: *Mistral-7B as a 2nd model, JudgeBench, W8A8/SmoothQuant, quantized KV
cache, QLoRA recovery.* This catalog turns that one-line list into actionable, scoped entries so
that (a) reviewers see the intended growth path and its blast radius on the SSOT, and (b) whoever
later has budget can promote one entry at a time without scope-creeping v1.

Every entry answers the same three questions in the same shape:
1. **what it adds** (the research question + scope in/out),
2. **exactly what would change** in the **registries / harnesses / matrix** (named SSOT modules
   and contracts, reusing `cell_path` / `cell_result.schema.json` / `is_cell_done` /
   `cell_provenance` / `PROVENANCE_KEYS` rather than redefining them), and
3. a **cost / effort** note (GPU $, engineering effort, and whether it forces a `contracts.md`
   change vs. being purely additive).

## Scope (in / out)

**In (this spec):**
- An index of the five extensions and a fixed **entry template**.
- One entry each, with the change-surface mapped onto the *named* v1 modules
  (`qquant.registry`, `src/qquant/registries/{variants,tasks}.yaml`, `qquant.models` /
  `BUILDERS`, `qquant.eval`, `qquant.efficiency`, `quant/selfquant`, `qquant.aggregate`, the
  Spec-10 report).
- A **promotion checklist** for converting an entry into the next monotonic spec.

**Out (this spec):**
- Any implementation: no `src/` code, no `quant/` code, no tests, no schema, no registry rows, no
  `[project.scripts]` lines, no `contracts.md`/README/decision-log edits. Each entry's "would
  change" lists are **proposals**, written as `# proposed — NOT added in v1`.
- Verified dependency pins. The decision-log pins (transformers 5.10.1, gptqmodel 7.1.0,
  bitsandbytes 0.49.2, llm-compressor 0.12.0, compressed-tensors 0.17.1, lm-eval 0.4.12, torch
  2.12.1+cu126) are honoured as the *baseline* an extension builds on; any **new** dependency
  (peft/trl, a quantized-cache backend, a JudgeBench loader) must be resolved + verified against
  live PyPI in the appropriate uv env at promotion time, exactly as the decision log did for v1.
- Library API names below are **illustrative** (marked "verify at implementation"); the project's
  own pattern (lm-eval v5 pin, awq-official contingency) is to confirm the exact API on the box.

## Interface

### A. Catalog entry template (every entry below fills exactly these fields)

```
### EXT-<n> — <Title>            (proposed spec id: <NN>; canonical, no aliases)
- RQ / what it adds
- Scope (in / out)
- Registry changes      # variants.yaml / tasks.yaml / VARIANT_IDS / TASK_IDS / QUANT_METHODS / new fields
- Harness changes       # qquant.models(BUILDERS) / qquant.eval / qquant.efficiency / quant.selfquant / qquant.aggregate / report
- Matrix impact         # cells added (× tasks=60/variant), new axis, contracts.md change? (yes/no)
- Cost / effort         # GPU $ (vs the $60 BudgetGuard / $75 hard cap), eng effort (LOW/MED/HIGH)
- Caveats / why deferred
```

### B. Index

| # | Extension | Proposed id(s) | New SSOT axis? | `contracts.md` change? | GPU cost | Eng effort |
|---|---|---|---|---|---|---|
| EXT-1 | Mistral-7B-Instruct as a 2nd model | (model axis; same 7 variant ids) | **model** | **yes** (or 2nd results root → no) | ~1× v1 (~$60) | HIGH |
| EXT-2 | JudgeBench (LLM-as-judge meta-eval) | task `judgebench` | task | no (additive task id) | low (7 cells) | MED |
| EXT-3 | W8A8 / SmoothQuant (same C4 set) | variant `w8a8-selfquant` | variant | no (additive variant id) | low (1 quant + 60 cells) | LOW–MED |
| EXT-4 | Quantized KV cache (long-context knob) | profiler axis (or `*-kvq4` ids) | knob/axis | no (efficiency-only) / yes (if paired ids) | low (profiling) | MED |
| EXT-5 | QLoRA recovery on best 4-bit | variant `bnb-nf4-qlora` | variant + training phase | no (additive variant id) | training hrs + 60 cells | HIGH |

> **Naming rule (binding on every proposed id above).** Any id an extension introduces MUST obey
> contracts §1/§8: **hyphen separators, no underscores, no `-int4`-style suffixes, no aliases**,
> and the bare strings `gptq`/`awq`/`bnb` remain forbidden as variant ids. Adding an id to
> `VARIANT_IDS`/`TASK_IDS` is an **SSOT change** — per contracts §8 it must land in `contracts.md`
> **and** Spec-01 code/registry **in the same commit**. This catalog adds none of them.

---

### EXT-1 — Mistral-7B-Instruct as a 2nd base model   (proposed spec id: 13)

- **RQ / what it adds.** Do the quantization findings (quality degradation ranking; memory/speed
  trade-offs; GPTQ-vs-AWQ calibration fairness) **generalize across model families**, or are they
  Qwen2.5-specific? Adds `mistralai/Mistral-7B-Instruct-v0.3` (pin a SHA at promotion) as a second
  base, re-running the same 7-variant × 4-task design.
- **Scope (in / out).**
  - *In:* a `model` namespace; per-model base SHA + (if they exist) official GPTQ/AWQ checkpoints;
    per-model self-quant on the **same** C4 protocol; the full quality + efficiency sweep; a
    cross-model section in the report.
  - *Out:* changing the v1 Qwen study; any third model.
- **Registry changes (`# proposed — NOT added in v1`).**
  - `variants.yaml` is currently single-model (`model_base: Qwen/Qwen2.5-7B-Instruct` + a Qwen
    `model_id`/`revision` per row). The 7 **variant ids stay identical** (they name quantization
    *methods*, which are model-agnostic), so `VARIANT_IDS`/`QUANT_METHODS` are **unchanged**.
  - Need a per-model registry. Two options:
    - **(lighter, recommended)** a second registry file `registries/variants.mistral.yaml` selected
      by a `--variants-yaml` flag, written to a **second results root** `results-mistral/`. No
      `cell_path`/`Cell` change → **no `contracts.md` change**.
    - **(heavier)** a real `model` dimension on `Cell`/`cell_path` (`<model>/<variant>/<task>/…`)
      and `cell_id` — that **is** a `contracts.md` change (cell layout, `Cell`, `expand_matrix`,
      `cell_provenance` gains a `base_model` provenance key).
  - Mistral has a **different vocab** (~32k vs Qwen's 152064) and a different chat template
    (`[INST]…[/INST]`), so the `tasks.yaml` `batch_overrides` (which exist to bound the logits
    transient) can be **larger** for Mistral; the per-task chat-template policy itself is reused
    verbatim (lm-eval applies the model's own template).
  - `gptq-official`/`awq-official` are **contingent on official Mistral checkpoints existing** at a
    pinned SHA; if absent, those rows get `enabled=false` and the guaranteed 4-bit points are the
    Mistral self-quants (mirrors the v1 `awq-official` contingency).
- **Harness changes.** `qquant.models` dispatch is already keyed on `quant_method`, so the loaders
  work unchanged once given Mistral ids/SHAs. `qquant.eval`/`qquant.efficiency` are model-agnostic;
  the only real touch is `quant/selfquant` (run the **same** `build_calibration` C4 artifact through
  Mistral) and `qquant.aggregate`/report (add the cross-model comparison + an MMLU-subjects
  re-assert, since the lm-eval `mmlu` group is model-independent so `MMLU_SUBJECTS` is unchanged).
- **Matrix impact.** Doubles the design: 2 models × 7 variants × 60 = **840 cells** (fewer if
  Mistral official checkpoints are absent). New axis = **model**. `contracts.md` change **only** if
  the heavier `model`-dimension option is chosen.
- **Cost / effort.** ~**1× the whole v1 GPU budget again** (≈ another $60) → it alone can blow the
  $75 hard cap, so it would run as a *separate budgeted campaign*, not bundled. Effort **HIGH**
  (registry namespacing + 2 self-quant runs + report rework; the cell-layout option adds SSOT work).
- **Caveats / why deferred.** Biggest blast radius of the five; the only one that pressures the
  single-model `cell_path` contract; and the only one that re-spends the entire compute budget.

---

### EXT-2 — JudgeBench (LLM-as-judge meta-eval)   (proposed spec id: 14)

- **RQ / what it adds.** Does quantization degrade a model's ability to act as a **judge** (a
  capability distinct from generation quality)? Evaluate each variant as a pairwise judge on
  **JudgeBench** (objective-correctness preference labels over knowledge/reasoning/math/coding;
  HF dataset id to confirm at promotion, e.g. `ScalerLab/JudgeBench`), scoring **agreement with the
  gold label** (pairwise judge accuracy).
- **Scope (in / out).**
  - *In:* a new generative task that prompts the variant to pick the better of two responses,
    parses the verdict, and scores against gold; **position-bias control** (swap A/B and require
    consistency, or average both orders).
  - *Out:* using an *external* judge (e.g. a frontier API) — the **judge is the quantized variant
    itself**; building new preference data; reward-model training.
- **Registry changes (`# proposed`).** Add task id `judgebench` to `TASK_IDS` (SSOT change) and a
  `tasks.yaml` row: `primary_metric: accuracy`, `apply_chat_template: true`,
  `fewshot_as_multiturn: false`, `num_fewshot: 0`, `per_subject: false`, `code_exec: false`,
  `max_gen_toks` ≈ 64 (verdict only), `metric_keys: ["accuracy", "acc"]`. It is a **single-cell**
  task (`result.json`), reusing `cell_path` unchanged.
- **Harness changes.** lm-eval ships **no** JudgeBench task, so this needs a **custom runner**
  (lives alongside `qquant.eval`, e.g. `qquant.eval.judge`, NOT a `simple_evaluate` row): load the
  dataset, render the judge prompt, run the variant via the **Spec-04 loader** with the project's
  greedy-determinism, parse `A/B`, score vs gold, and write a **standard v1 cell** via the existing
  `build_cell`/`write_cell` — `config = cell_provenance(...)` (keys `== PROVENANCE_KEYS`), and the
  per-pair correctness vector stored in `meta.item_correct`/`meta.item_ids` so `qquant.aggregate`'s
  **paired McNemar** vs `bf16` works with **zero** stats changes. The Wilson CI path (small n) is
  already in `qquant.aggregate.stats`. `qquant.efficiency` is untouched.
- **Matrix impact.** +1 task → **+7 cells** (one per enabled variant): 7×(60+1)=427. No new
  per-subject expansion, **no `contracts.md` change** beyond the additive `TASK_IDS` entry.
- **Cost / effort.** GPU **low** (7 short generative cells). Effort **MED** (custom dataset
  loader + verdict parser + position-bias control + a determinism smoke; everything downstream is
  reused).
- **Caveats / why deferred.** This is a **meta-eval** (judge capability), a different RQ from v1's
  generation quality — keep it clearly labelled. Self-preference / contamination risk (a model
  judging in a domain it was trained on) is mitigated by JudgeBench's *objective* (not preference)
  labels and by reporting per-category accuracy + position-bias consistency.

---

### EXT-3 — W8A8 / SmoothQuant on the SAME C4 set   (proposed spec id: 15)

- **RQ / what it adds.** How does **8-bit weights + 8-bit activations** (W8A8 via SmoothQuant) trade
  off against the v1 **4-bit weight-only** points on quality and memory? Natural sibling of Spec 07:
  reuse the **identical** C4 calibration artifact (128×2048, seed 42) and the isolated `quant/` env.
- **Scope (in / out).**
  - *In:* one new self-quant checkpoint produced with `SmoothQuantModifier` + a `W8A8` scheme on the
    **same** calibration object as `gptq-selfquant`/`awq-selfquant`; loaded natively via
    compressed-tensors; full quality + efficiency sweep.
  - *Out:* the calibration-controlled **W4A16 GPTQ-vs-AWQ** fairness pair (W8A8 is a *different
    bit-width axis*, reported alongside, not inside, that controlled comparison).
- **Registry changes (`# proposed`).** Add variant id `w8a8-selfquant` to `VARIANT_IDS` (SSOT
  change; hyphenated, no underscore, no `-int4` suffix) with a `variants.yaml` row:
  `quant_method: compressed-tensors`, `source: selfquant`,
  `model_id: local:checkpoints/self-quant/w8a8-selfquant`, `revision: null`. **No new
  `QUANT_METHODS`** — it loads through the existing `compressed-tensors` builder.
- **Harness changes.** `quant/selfquant/recipes.py` (Spec 07) gains
  `smoothquant_w8a8_recipe()` → `[SmoothQuantModifier(smoothing_strength=0.5),
  QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]` (verify the exact
  llm-compressor preset name at implementation); `quantize.py`'s `SELF_QUANT_IDS` extends to
  include `w8a8-selfquant` and feeds it the **same** `build_calibration` object (so `calib_sha256`
  ties all self-quants to one set). The Spec-04 `compressed-tensors` builder, `qquant.eval`, and
  `qquant.efficiency` are **unchanged** (it is just another compressed-tensors checkpoint). In
  `qquant.aggregate`/report, W8A8 is a new row; because its `source == selfquant`, the
  **runtime-confounded** speed label (decision-log) applies to its throughput exactly as for the
  4-bit self-quants — W8A8's theoretical INT8-GEMM speedup is a vLLM/kernel story, not guaranteed on
  the HF `generate()` backend on sm_89, so disk/memory are fair but speed is flagged.
- **Matrix impact.** +1 variant → **+60 cells** (57 MMLU + gsm8k + humaneval + ifeval) + 1
  efficiency artifact: 8×60=480 + efficiency. **No `contracts.md` change** beyond the additive
  `VARIANT_IDS` entry.
- **Cost / effort.** One extra `oneshot` (~10–20 min GPU) + 60 eval cells + 1 profile. GPU **low**.
  Effort **LOW–MED** — almost entirely a recipe + registry row; calibration, loader, eval, profiler,
  aggregation are reused.
- **Caveats / why deferred.** Naming (`w8a8-selfquant` scheme-style vs `smoothquant-selfquant`
  algorithm-style) is an open question for the consistency gate. W8A8 is **not** part of the
  controlled W4A16 fairness pair — the report must place it on the separate bit-width axis.

---

### EXT-4 — Quantized KV cache (long-context inference knob)   (proposed spec id: 16)

- **RQ / what it adds.** This is **not** a weight-quantization variant; it is an **inference-time
  knob**: quantize the **KV cache** (e.g. transformers `cache_implementation="quantized"` with a
  quanto/HQQ backend at int4/int2 — verify the exact v5 API + backend dep at implementation) to free
  VRAM at long context, letting the 24 GB card hold longer sequences / larger batch. Quantifies the
  long-context **memory/throughput headroom** and any **quality** cost of a lossy KV cache.
- **Scope (in / out).**
  - *In:* an orthogonal `kv_cache` axis applied **on top of** an existing base variant; primarily an
    **efficiency** sub-study (the v1 profiler already measures `generate_peak` = KV + logits
    transient, the `long` (4096) prompt bucket, and the max-batch OOM sweep — KV-quant should visibly
    raise max-batch / extend the long bucket); plus a small quality check that lossy KV does not move
    `mmlu`/`gsm8k` accuracy materially.
  - *Out:* training; weight quantization changes; serving on vLLM.
- **Registry changes (`# proposed`).** Best as a **knob, not a row**:
  - **(lighter, recommended)** add an optional `Variant`/profiler field
    `kv_cache: {backend, nbits} | null` consumed only by the loader's `generation_config` / the
    `qquant.efficiency` profiler; run as an **efficiency-only sub-study** over a couple of base
    variants × {fp16 KV, int4 KV}. **No `VARIANT_IDS`/`cell_path` change.**
  - **(heavier)** paired ids `bf16-kvq4` / `bnb-nf4-kvq4` if a full 60-cell *quality* sweep per KV
    setting is wanted — that adds rows to `VARIANT_IDS` (additive SSOT change, still hyphenated /
    no underscore / no `-int4`).
- **Harness changes.** A pass-through for the cache config through the Spec-04 loader's
  `force_greedy`/`generation_config` and Spec-05 `greedy_gen_kwargs` (so generation actually uses the
  quantized cache); the main work is in `qquant.efficiency` (a KV-cache dimension on the throughput +
  max-batch + long-prompt measurements, with new memory rows). `qquant.eval` only needs the
  gen-config pass-through if the optional quality check runs.
- **Matrix impact.** Cleanest as **efficiency artifacts only** (no quality cells) → no matrix growth,
  no `contracts.md` change. If paired ids are used: +60 cells per id.
- **Cost / effort.** GPU **low** (profiling runs; no training). Effort **MED** (transformers v5
  quantized-cache API + backend dependency in the eval env, a profiler axis, and a gen-config
  pass-through).
- **Caveats / why deferred.** A quantized KV cache is **lossy**, so greedy output may differ from the
  fp16-KV run — that delta is itself the measurement, but it means KV-quant points must **not** be
  compared to v1 quality numbers as if loss-free. Run-to-run determinism must still hold (it is a
  fixed lossy transform), so the project's determinism smoke still applies. Like self-quant speed,
  any throughput gain on the HF backend is **runtime-confounded** vs a vLLM kernel path.

---

### EXT-5 — QLoRA recovery on the best 4-bit variant   (proposed spec id: 17)

- **RQ / what it adds.** Can a cheap **PEFT** pass recover the quality lost to 4-bit quantization?
  Train a **QLoRA** adapter (LoRA on the frozen NF4 base, the canonical QLoRA recipe) on top of the
  best 4-bit variant and re-evaluate, reporting the recovered gap vs the un-adapted 4-bit point and
  vs the `bf16` baseline.
- **Scope (in / out).**
  - *In:* a LoRA adapter trained on **general instruction data** over the frozen NF4 base; an
    adapter-aware load path; the full quality sweep of the adapted model; a recovery table
    (`bnb-nf4` → `bnb-nf4-qlora`, paired McNemar across the **same** items).
  - *Out:* full fine-tuning; RLHF; training the official-checkpoint variants (LoRA-on-4-bit is the
    QLoRA design and targets the NF4 base specifically); training on any eval-task data.
- **Registry changes (`# proposed`).** Add variant id `bnb-nf4-qlora` to `VARIANT_IDS` (SSOT change)
  with a `variants.yaml` row: `quant_method: bnb-nf4` (base load path is reused), a new
  `source: recovery`, an **`adapter_path: local:checkpoints/qlora/bnb-nf4-qlora`** field, and
  `revision: null`. The "best 4-bit" target is chosen from v1 results; if it is `gptq-official`
  rather than NF4, note QLoRA's bnb-NF4 base is the standard, so `bnb-nf4` is the recovery substrate.
- **Harness changes.** The Spec-04 loader gains an **adapter-attach** step (after the NF4 base loads,
  `PeftModel.from_pretrained(base, adapter_path)`; keep the adapter attached for eval — a 4-bit base
  cannot be cleanly LoRA-merged), keyed on the new `adapter_path` field — a **loader** change, not a
  new `quant_method`. A **new training pipeline** (peft/trl in the `quant/` env or a new training
  env; new deps to resolve+verify) produces and **persists** the adapter exactly as Spec 07/08
  persist+exfil self-quant checkpoints (so a resume never re-trains). `qquant.eval` is unchanged once
  the loader returns the adapted model; `qquant.aggregate`/report add the recovery comparison
  (`bnb-nf4` vs `bnb-nf4-qlora`) and gate the recovery claim with paired **McNemar**.
- **Matrix impact.** +1 variant → **+60 cells**, **plus** a training phase (GPU hours, a new
  dependency surface, and a training/eval split). No `contracts.md` change beyond the additive
  `VARIANT_IDS` entry; `cell_provenance` should fold the adapter fingerprint into `model_revision`
  so a re-trained adapter makes cells **stale** under `is_cell_done` (same provenance discipline
  Spec 07 flags for self-quant `revision: null`).
- **Cost / effort.** Training GPU hours (the only extension with a *training* cost) + 60 eval cells.
  Effort **HIGH** — a training pipeline, a dataset, hyperparameters, checkpoint persistence/exfil,
  and a **contamination guard** are all net-new.
- **Caveats / why deferred.** **Contamination is the headline risk:** QLoRA recovery must train on
  *general* instruction data, **never** on MMLU/GSM8K/HumanEval/IFEval content, or the "recovery"
  is leakage; the report must state the training corpus + the disjointness argument. Determinism
  extends to a *seeded training* run. Biggest engineering surface of the five.

---

### C. Promotion checklist (turning one entry into the next monotonic spec)

When budget/time allows, promote **one** entry at a time:
1. Allocate the next monotonic spec number (13+) and add a README row + `depends_on` (e.g. EXT-3
   → `depends_on: [04, 07]`; EXT-2 → `[04, 05, 09]`; EXT-5 → `[04, 05, 09]` + a training spec;
   EXT-1 → `[04, 05, 06, 07, 09]`).
2. If the entry adds an id/path/schema axis, land the SSOT change in **`contracts.md` + Spec-01
   code/registry in the same commit** (contracts §8); flag whether it is *additive* (most) or a
   *cell-layout* change (EXT-1 heavy option).
3. Resolve + **verify** any new dependency in the correct uv env (eval env vs `quant/` env vs a new
   training env), re-lock, and record the pin — mirroring the decision-log verification discipline.
4. Add a per-variant/-task **BudgetGuard** estimate against the $60 ceiling / $75 hard cap and
   sequence the campaign; do **not** bundle extensions.
5. Reuse `cell_path` / `cell_result.schema.json` / `is_cell_done` / `cell_provenance` /
   `PROVENANCE_KEYS` / `expand_matrix` / `enabled_variants` — never re-declare them.

## Files to create

- **This file only:** `docs/specs/12-extensions-catalog.md` (the catalog). **Nothing else.**
- Per entry, the files a *future* promoted spec would create are listed inline above as
  `# proposed — NOT added in v1` (e.g. a `qquant.eval.judge` runner for EXT-2; a
  `smoothquant_w8a8_recipe` in `quant/selfquant/recipes.py` for EXT-3; a training pipeline +
  adapter-aware loader step for EXT-5). This spec creates **none** of them and edits **no**
  `src/`, `quant/`, `pyproject.toml`, registry, schema, `contracts.md`, README, or decision-log.

## Implementation notes (the decision-log points that bind THIS spec, with the "why")

- **"Spec-only extensions (documented, not implemented)"** (decision-log "Scope"). This catalog is
  precisely that list; its job is to be implementable-later, not implemented-now. *Why:* v1 must
  stay at the locked 7×4 design under the $60/$75 budget; the extensions are the agreed growth path.
- **"v1 = Core + self-quant fairness; self-quant is a deferrable fast-follow"** (decision-log
  "Scope"). The catalog inherits that discipline: each entry is independently promotable and
  independently droppable; none is a v1 dependency (hence `depends_on: []`).
- **Self-quant = fair quality, confounded speed** (decision-log). Binds **EXT-3** (W8A8 self-quant
  speed is runtime-confounded on the HF backend) and **EXT-4** (KV-quant throughput gains are a
  kernel/vLLM story) — both must carry the `runtime-confounded` label that `qquant.aggregate`
  already applies to `source == selfquant`.
- **Same C4 calibration set / W-scheme controlled axis** (decision-log; Spec 07). Binds **EXT-3**:
  reuse Spec 07's `build_calibration` artifact and `calib_sha256` so W8A8 sits on a fair calibration
  footing, while staying *outside* the controlled W4A16 GPTQ-vs-AWQ pair.
- **Paired McNemar gates all quality claims** (decision-log "Significance"). Binds **EXT-2** and
  **EXT-5**: both emit the per-item correctness vector into `meta.item_correct`/`meta.item_ids` so
  `qquant.aggregate`'s McNemar works unchanged (judge accuracy; recovery vs the un-adapted 4-bit
  point).
- **`awq-official` contingency pattern** (decision-log). Binds **EXT-1**: Mistral official
  GPTQ/AWQ checkpoints are *contingent*; absent them, `enabled=false` and the Mistral self-quants
  are the guaranteed 4-bit data points.
- **Greedy determinism** (decision-log). Binds **EXT-2/4/5**: judge verdicts, KV-quant generation,
  and QLoRA eval all run through the Spec-04 greedy-forced load + Spec-05 `greedy_gen_kwargs`; a
  lossy KV cache is still a deterministic transform, so the determinism smoke still applies.
- **Two uv envs (NOT one)** (decision-log). Binds **EXT-3/5**: quantization/training deps
  (llm-compressor SmoothQuant; peft/trl) live in the isolated `quant/` (or a new training) env; the
  eval env only loads the produced artifact via `compressed-tensors` / a PEFT adapter attach.
- **Persist + exfil checkpoints; resume never recomputes** (decision-log; Spec 07/08). Binds
  **EXT-3** (the W8A8 checkpoint) and **EXT-5** (the LoRA adapter) — both persisted and reconciled
  before destroy, with provenance folded into `model_revision` so `is_cell_done` invalidates stale
  cells.
- **No PDF / markdown report** (decision-log "Reporting"). Any promoted extension's results fold
  into `results/REPORT.md` (Spec 10), never a PDF.
- **Naming & one-vocabulary rules** (contracts §1/§8). Every proposed id is hyphenated, alias-free,
  underscore-free, `-int4`-suffix-free, and only enters `VARIANT_IDS`/`TASK_IDS` via a single
  SSOT-plus-code commit.

## Done-when (numbered, testable)

1. `docs/specs/12-extensions-catalog.md` exists, opens with the locked v1-scope statement and an
   **index table** listing exactly the five extensions (EXT-1…EXT-5: Mistral, JudgeBench,
   W8A8/SmoothQuant, quantized KV cache, QLoRA recovery), plus the **entry template** and the
   **promotion checklist**.
2. Each of the five entries fills **every** template field: RQ/what-it-adds, Scope (in/out),
   Registry changes, Harness changes, Matrix impact, Cost/effort, Caveats.
3. Each entry names the **specific** SSOT/harness surfaces it would touch — drawn only from the real
   modules: `qquant.registry` / `VARIANT_IDS` / `TASK_IDS` / `QUANT_METHODS` / `MMLU_SUBJECTS`,
   `registries/{variants,tasks}.yaml`, `qquant.models`+`BUILDERS`, `qquant.eval`,
   `qquant.efficiency`, `quant/selfquant`, `qquant.aggregate`, and the Spec-10 report — and reuses
   `cell_path` / `cell_result.schema.json` / `is_cell_done` / `cell_provenance` / `PROVENANCE_KEYS`
   where applicable instead of redefining them.
4. Every **proposed id** in the doc (`judgebench`, `w8a8-selfquant`, `bnb-nf4-qlora`, any `*-kvq4`)
   is hyphen-separated, has no underscore and no `-int4` suffix, is not a bare `gptq`/`awq`/`bnb`,
   and the doc states adding it is an SSOT change (contracts §8) — i.e. **none is added by this
   spec**. The existing 7 variant ids and 4 task ids are referenced verbatim and **not redefined**.
5. Each entry states explicitly whether it forces a **`contracts.md` change** (EXT-1 heavy option:
   yes; all others: additive / no) and carries a **GPU-cost** note relative to the $60 BudgetGuard /
   $75 hard cap and an **effort** rating.
6. The doc states unambiguously that these are **documented, not implemented in v1**, creates only
   this one file, and a check confirms it touches no `src/`, `quant/`, `pyproject.toml`, registry,
   schema, `contracts.md`, README, or decision-log content (`[project.scripts]` unchanged).
7. `depends_on` is empty and the catalog introduces **no** new edge into the v1 dependency graph;
   all cross-references to specs/contracts are **by name**.

## Risks & mitigations

- **Readers implement straight from the catalog (scope creep).** *Mitigation:* the Status line, the
  index, and every entry are stamped "spec-only / NOT v1"; the promotion checklist is the only
  sanctioned path to code, and it forces a numbered spec + budget estimate first.
- **Ad-hoc id naming drift.** *Mitigation:* the binding naming rule under the index (hyphens, no
  underscores, no `-int4`, no aliases) plus the contracts §8 "SSOT change in one commit" requirement
  are restated per entry.
- **Hidden contract pressure.** *Mitigation:* the index column and each Matrix-impact line flag
  exactly which extension needs a `contracts.md` change (only EXT-1's heavy `model`-axis option) so
  reviewers see the blast radius before promoting.
- **Budget blowout past $60/$75.** *Mitigation:* per-entry GPU-cost notes; EXT-1 (~1× the whole v1
  budget) and EXT-5 (training hours) are explicitly flagged as separate budgeted campaigns, never
  bundled.
- **Confound mislabeling.** *Mitigation:* each entry names its confound up front — EXT-2
  self-preference/meta-eval framing, EXT-3/EXT-4 runtime-confounded speed, EXT-5 train/eval
  contamination — so a promoted spec inherits the caveat rather than rediscovering it.
- **Stale library APIs.** *Mitigation:* all library API names (SmoothQuant preset, transformers
  quantized-cache config, JudgeBench dataset id, peft/trl) are marked "verify at implementation,"
  and new deps must be resolved+verified in the correct uv env at promotion — matching the
  decision-log's live-PyPI verification discipline rather than asserting unverified pins here.
