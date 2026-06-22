# Contracts (Single Source of Truth)

**Spec 01 owns every contract on this page; all other specs import them — they never
re-declare ids, paths, schema keys, or predicates.** These mirror the code already in
`src/qquant/` (the scaffold). If a contract must change, change it here and in Spec 01's code
in the same commit; downstream specs follow.

## 1. Canonical ids — `qquant.registry`

```python
VARIANT_IDS = ("bf16", "bnb-int8", "bnb-nf4", "gptq-official",
               "awq-official", "gptq-selfquant", "awq-selfquant")   # hyphens, no aliases
TASK_IDS    = ("mmlu", "gsm8k", "humaneval", "ifeval")
QUANT_METHODS = {"bf16", "bnb-int8", "bnb-nf4", "gptq", "awq", "compressed-tensors"}
MMLU_SUBJECTS = (...)   # the 57 subjects; lm-eval subtask = f"mmlu_{subject}"
```

- The same `gptq`/`awq`/`bnb` strings are **forbidden as bare variant ids**; use the full ids.
- Self-quant variants have `quant_method == "compressed-tensors"` (they load natively in
  transformers v5 — NOT via gptqmodel's legacy GPTQ/AWQ path).
- A GPU/lm-eval test (Spec 02/06) asserts `set(MMLU_SUBJECTS)` equals the `TaskManager`
  expansion of the `mmlu` group for the pinned lm-eval.

## 2. Registry dataclasses — `qquant.registry`

`Variant(id, quant_method, source, enabled, model_id, revision, notes)` and
`Task(id, lm_eval_task, primary_metric, metric_keys, per_subject, apply_chat_template,
fewshot_as_multiturn, num_fewshot, default_batch_size, batch_overrides, code_exec,
max_gen_toks)` with `Task.batch_size_for(variant_id)`.

Loaders: `load_variants(path=None)`, `load_tasks(path=None)`, `enabled_variants(variants)`.
The YAML registries live at `src/qquant/registries/{variants,tasks}.yaml` (the **only**
registry location/format — do not create `configs/variants.yaml`, `variants/*.json`, etc.).

- `variants.yaml` entry: `quant_method`, `source` (`baseline|load-time|official|selfquant`),
  `enabled`, `model_id` (HF repo id, or `local:<path>` for self-quant), `revision` (SHA or
  `null`), `notes`. `model_id` for self-quant is `local:checkpoints/self-quant/<id>`.
- `tasks.yaml` entry: includes `metric_keys` (tried in order against the lm-eval results dict
  to absorb suffix drift like `pass@1,create_test` vs `pass@1,none`) and `batch_overrides`
  (per-variant batch size; e.g. `bf16: 2` for MMLU).

## 3. On-disk layout — `qquant.paths`

```
<results_root>/<variant>/<task>/<subject>.json   # per-subject (MMLU)
<results_root>/<variant>/<task>/result.json      # single-cell tasks
<results_root>/_meta/env.json                     # run provenance (lib + dataset revisions, image digest)
<results_root>/_meta/done_manifest.json           # cell ids the remote run completed (copy-out reconciliation)
```

`cell_path(results_root, variant, task, subject=None) -> Path` is the **only** way to build a
cell path. Every writer/auditor/aggregator imports it; none reimplement the layout. `Paths`
provides `.cell(...)`, `.env_file`, `.done_manifest`. The MMLU subject token is the **bare
slug** (`anatomy`), with the lm-eval name (`mmlu_anatomy`) recorded inside the cell.

## 4. Cell result schema (v1) — `qquant.schemas`

`src/qquant/schemas/cell_result.schema.json` (JSON Schema 2020-12). Required top-level:
`schema_version (==1)`, `cell_id`, `variant`, `task`, `primary_metric`, `metric_value`,
`n_samples`, `config`. Optional: `subject`, `task_lm_eval`, `metric_stderr`, `extra_metrics`,
`meta`. Required `config` block: `seeds`, `batch_size`, `num_fewshot`, `apply_chat_template`,
`fewshot_as_multiturn`, `max_length`, `lm_eval_version`, `model_revision`.

`validate_cell(cell)` / `is_valid_cell(cell)` do full schema validation (writers + tests).

## 5. Matrix & resume — `qquant.matrix`

`Cell(variant, task, subject, lm_eval_task, primary_metric)` with `.cell_id` (`<variant>/<task>[/<subject>]`)
and `.path(results_root)`. `expand_matrix(variants, tasks)` (MMLU → 57 cells). For the v1
matrix that is **7 × 60 = 420 cells** (57 MMLU + gsm8k + humaneval + ifeval per variant).

`is_cell_done(cell, active_config=None)` is the **SSOT resume predicate** — never reimplement
it (no separate `completed:true`/`status=='complete'` checks). Structural validity always
applies; if `active_config` (a provenance block) is given, every overlapping
`PROVENANCE_KEYS` value must match or the cell is treated as **stale** and recomputed.

```python
PROVENANCE_KEYS = ("seeds", "batch_size", "num_fewshot", "apply_chat_template",
                   "fewshot_as_multiturn", "max_length", "lm_eval_version", "model_revision")
```

`missing_cells(cells, results_root, active_config=None)` returns cells whose file is absent,
unreadable, or not done.

## 6. Run config & provenance — `qquant.config`

`Seeds(random, numpy, torch, fewshot)`, `RunConfig(lm_eval_version, model_revision,
max_length, seeds)`, and `cell_provenance(run, task, variant_id) -> dict` which produces a
block keyed by exactly `PROVENANCE_KEYS` (so it round-trips with the schema `config` block and
`is_cell_done`).

## 7. Entry points — `[project.scripts]` (Spec 01 owns)

- Registered now: `qquant = "qquant.cli:main"` — the **torch-free umbrella** (subcommands
  `variants|tasks|matrix|audit|version`). It must never import torch.
- Reserved for their specs (GPU-touching → **separate** entry points, not subcommands of the
  torch-free umbrella): `qquant-spike` (Spec 02), `qquant-eval` (Spec 05),
  `qquant-profile` (Spec 06), `qquant-orchestrate` (Spec 08), `qquant-aggregate` (Spec 09),
  `qquant-report` (Spec 10). Each spec adds its own `[project.scripts]` line when it lands;
  until then use `python -m`.

## 8. Naming & cross-spec rules

- One variant-id vocabulary (§1); one task-id vocabulary; hyphen separators; no underscores or
  `-int4` suffixes anywhere.
- One registry location (§2), one path builder (§3), one schema (§4), one resume predicate (§5).
- **Efficiency-artifact carve-out (the ONE sanctioned exception).** The per-variant efficiency
  artifact `results/<variant>/efficiency.json` is the only addition beyond the cell SSOT: it has
  its **own** path helper (`efficiency_path`), its **own** schema
  (`efficiency_result.schema.json`), and its **own** resume predicate (`is_efficiency_done`), all
  owned by `qquant.efficiency` (Spec 06). It is per-variant (not a `(variant,task[,subject])`
  quality cell), `"efficiency"` is not a `TASK_ID`, and it is a file directly under `<variant>/`
  (tasks live in `<variant>/<task>/` dirs), so it never collides with the quality matrix. No
  other path/schema/predicate may be added beyond the cell SSOT and this carve-out.
- The consistency-review gate (end of Spec authoring) asserts zero drift across all specs on
  variant ids, cell paths, schema keys, and entry points.
