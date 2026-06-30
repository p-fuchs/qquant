# Spec 12 extensions — implementation design (gated `qquant.ext`)

- **Status:** approved design, implemented on branch `spec-12-extensions` (worktree),
  **gated OFF** — nothing runs until the operator flips an `enabled` flag and invokes
  `qquant-ext`.
- **Promotes:** the five spec-only entries in `docs/specs/12-extensions-catalog.md`
  (EXT-1 Mistral, EXT-2 JudgeBench, EXT-3 W8A8/SmoothQuant, EXT-4 quantized KV cache,
  EXT-5 QLoRA recovery) into real, runnable code.
- **Non-goal:** changing the v1 SSOT. `contracts.md`, `qquant.registry.VARIANT_IDS` /
  `TASK_IDS`, `registries/variants.yaml`, `registries/tasks.yaml` are **untouched**, so
  the live v1 run and its consistency tests are unaffected.

## Why a sibling package (the key decision)

`qquant.registry.load_variants/load_tasks` validate every id against the hardcoded
`VARIANT_IDS`/`TASK_IDS` tuples **and require all v1 ids to be present**. So a separate
`*.ext.yaml` cannot pass through the v1 loaders without editing the SSOT tuples — exactly
the blast radius we are avoiding while a live run reads those files.

Resolution: a sibling **`qquant.ext`** package with its **own** id space
(`EXT_VARIANT_IDS`, `EXT_TASK_IDS`) and `ExtVariant`/`ExtTask` dataclasses that **subclass**
the v1 `Variant`/`Task`. Because the v1 reuse surfaces duck-type on attributes
(`expand_matrix`, `is_cell_done`, `missing_cells` use `.id`/`.per_subject`/`.lm_eval_task`;
`build_cell`/`cell_provenance` use `variant.id` + task fields; the cell schema's `variant`
/`task` are plain strings with no enum), the subclasses drop straight into the existing
matrix / cell-writer / resume / aggregation machinery with **zero** v1 edits.

## Architecture

```
src/qquant/ext/
  __init__.py
  registry.py    # EXT_VARIANT_IDS, EXT_TASK_IDS; ExtVariant(+base_model, adapter_path, kv_cache);
                 # ExtTask(+ judge fields); load_ext_variants / load_ext_tasks (own id space)
  loaders.py     # ext_load_variant: reuses qquant.models.loaders {resolve_model_source, BUILDERS,
                 # force_greedy, LoadedVariant}; adds Mistral base_model, adapter-attach (EXT-5),
                 # KV-cache generation_config (EXT-4). torch deferred-import (import-time torch-free).
  judge.py       # EXT-2 custom JudgeBench runner: dataset load, prompt render, A/B verdict parse,
                 # position-bias swap+consistency; writes a STANDARD v1 cell via build_cell/write_cell
                 # with meta.item_correct/item_ids so qquant.aggregate McNemar works unchanged.
  kvcache.py     # EXT-4 profiler-axis helper: builds the quantized-cache generation kwargs.
  cli.py         # qquant-ext: subcommands eval | judge | profile | train-qlora, rooted at results-ext/.
registries/
  variants.ext.yaml   # w8a8-selfquant, mistral-* rows, bnb-nf4-qlora   (ALL enabled: false)
  tasks.ext.yaml      # judgebench
quant/selfquant/
  recipes.py     # += smoothquant_w8a8_recipe()  (EXT-3, additive; NOT in the W4A16 fairness pair)
  quantize.py    # generalized scheme/recipe dispatch to also produce w8a8-selfquant (additive)
results-ext/          # separate state root (gitignored)
checkpoints/qlora/    # EXT-5 adapter output (gitignored)
```

**Reuse, never redefine:** `cell_path`, `Paths`, `cell_result.schema.json`,
`is_cell_done`, `cell_provenance`, `PROVENANCE_KEYS`, `expand_matrix`, `missing_cells`,
`build_cell`/`write_cell`, `EvalRunner` (injected `load_model=ext_load_variant`,
`results_root=results-ext`), the C4 `build_calibration` artifact, and the McNemar/Wilson
helpers in `qquant.aggregate`. The ext package **imports** these.

## Per-extension surface (all gated)

| Ext | New code | Registry row (enabled:false) | Runtime path |
|-----|----------|------------------------------|--------------|
| EXT-3 W8A8 | `smoothquant_w8a8_recipe()` + generalized `quantize.py` | `w8a8-selfquant` | existing `compressed-tensors` builder + `EvalRunner` |
| EXT-2 JudgeBench | `qquant.ext.judge` | task `judgebench` | custom runner → `build_cell` |
| EXT-4 KV cache | `qquant.ext.kvcache` + loader `kv_cache` field | `*-kvq4` (efficiency-only) | `EvalRunner`/profiler with quantized-cache gen kwargs |
| EXT-1 Mistral | `base_model` field + ext loader | `mistral-*` (7 ids) | reuse v1 builders, `results-ext/mistral` |
| EXT-5 QLoRA | adapter-attach in loader + `qquant.ext.train` | `bnb-nf4-qlora` | NF4 base + PeftModel attach |

## Isolation & gating

- All work on a **git worktree** off `90bb630` (branch `spec-12-extensions`); the other
  agent's uncommitted v1-run changes in the primary checkout are never touched.
- No new id enters `VARIANT_IDS`/`TASK_IDS`/v1 registries/`contracts.md`. The live
  `qquant-eval`/`qquant-orchestrate` run is structurally incapable of picking these up.
- Every ext variant ships `enabled: false`; `qquant-ext` is a **separate** entry point that
  writes only under `results-ext/`.

## Verify-now vs verify-on-the-box

- **Now (laptop, CPU, $0):** all torch-free plumbing — ext registry load/validate, matrix
  expansion, cell write/resume against `results-ext/`, judge prompt render + verdict parser
  + position-bias logic, recipe object construction, KV/adapter generation-kwargs builders,
  CLI wiring — covered by `pytest` and the torch-free guard extended to `qquant.ext.*`.
- **On the box (GPU, on operator signal):** the library-API-bound items the catalog marks
  "verify at implementation" — SmoothQuant preset name, transformers v5 quantized-cache API
  + backend dep (quanto/HQQ), JudgeBench HF dataset id, peft/trl pins, Mistral checkpoint
  existence — smoke-tested then run. See `docs/specs/13-ext-runbook.md`.

## Testing & discipline

- Extend `tests/test_import_torch_free.py` to import every `qquant.ext.*` module and assert
  no torch in `sys.modules` (deferred imports, same pattern as `qquant.models.loaders`).
- New unit tests mirror existing ones: ext registry validation, matrix counts on the ext id
  space, resume predicate on `results-ext/`, verdict parsing, position-bias consistency,
  recipe construction, KV/adapter gen-kwargs.
- `ruff check`/`format` clean.

## Build order

EXT-3 → EXT-2 → EXT-4 → EXT-1 → EXT-5 (cheap→expensive). Deliverable: branch with all five
implemented + gated + CPU-tested, plus `docs/specs/13-ext-runbook.md` describing how to flip
each on and run it on the 4090.
