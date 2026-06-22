# Spec 09 — Aggregation + Stats (MMLU stderr, Wilson CI, paired McNemar) + Plots

- **Status:** draft (ready to implement once 05/06/07 produce cells (with per-item vectors in `meta`) and efficiency JSON)
- **Depends on:** 05 (quality cells), 06 (efficiency JSON), 07 (self-quant variant cells/efficiency)
- **Owns / Produces:** the `qquant.aggregate` package — **one** aggregation pipeline + **one** stats module —
  the `qquant-aggregate` entry point, and the `<results_root>/_analysis/` artifacts
  (`aggregate.json`, master + degradation tables in `.md`/`.csv`, and the two PNG/SVG plots) that Spec 10 consumes.

## Purpose

Turn the on-disk result matrix (the per-cell quality JSONs written by Spec 05/07 plus the per-variant
efficiency JSONs written by Spec 06/07) into the study's analysis layer:

1. **Re-aggregate MMLU** from its 57 per-subject cells into one overall accuracy, **weighted by subject n**,
   with **one documented stderr method** (error-propagation of the per-subject binomial stderr), reconciled
   against lm-eval's own group number.
2. Attach a **Wilson score CI** to HumanEval `pass@1` (lm-eval may emit no pass@1 stderr).
3. Establish **significance vs the `bf16` baseline** with a **paired McNemar test** on the per-item
   correctness vectors read **directly from each generative cell's** `meta.item_correct` /
   `meta.item_ids` (written by Spec 05) — never an unpaired z-test.
4. Emit the deliverables Spec 10 (the markdown report) reads: the **master comparison table**
   (all variants × all benchmarks + efficiency), the **per-capability degradation table**, and the
   **quality-vs-memory** / **quality-vs-throughput** plots.

This runs **locally** (laptop/CI), not on the GPU box: it only reads JSON/JSONL off the copied-out
`results/` tree using the `analysis` dependency group (pandas/numpy/matplotlib). It is **torch-free and
lm-eval-free** at import time.

## Scope (in / out)

**In scope**
- Reading + structurally validating every present cell via `cell_path` / `is_valid_cell`; never reimplement
  the layout or the resume predicate (`is_cell_done`).
- The **single** stats module: Wilson CI, weighted MMLU aggregate + error-propagated stderr, paired McNemar.
- The **single** aggregation pipeline: cell discovery → per-(variant,task) score → significance → tables →
  `aggregate.json`.
- Plot rendering (matplotlib, Agg backend) → PNG **and** SVG.
- `qquant-aggregate` CLI; `--strict` completeness gate driven by `missing_cells`.

**Out of scope** (owned elsewhere)
- Producing cells, efficiency JSON, or `log_samples` (Spec 05/06/07).
- The prose report / RQ narrative (Spec 10 — `qquant-report`, reads our `aggregate.json` + tables + plots).
- Anything GPU/torch/lm-eval (this spec must import neither).
- Re-resolving lm-eval `metric_keys` suffix drift — that happened in Spec 05 when it wrote `metric_value`;
  we read `primary_metric` / `metric_value` straight off the cell.

## Interface

### Entry point (`pyproject.toml` → `[project.scripts]`)

Per contracts §7 the `qquant-aggregate` name is reserved for this spec (Spec 09). This spec adds
**one** line; it is a **separate** entry point, not a subcommand of the torch-free `qquant` umbrella:

```toml
qquant-aggregate = "qquant.aggregate.cli:main"
```

### CLI

```
qquant-aggregate --results-root results
                 [--baseline bf16]            # significance reference variant
                 [--confidence 0.95]          # Wilson + McNemar alpha = 1 - confidence
                 [--quality-metric mmlu]      # plot y-axis: mmlu|gsm8k|humaneval|ifeval|mean
                 [--out-dir results/_analysis]
                 [--strict]                   # nonzero exit if any enabled-variant cell is missing
```

Exit codes: `0` success; `2` strict-mode gap (enabled-variant cells missing); `1` usage/IO error.
Stable, deterministic output (no timestamps in artifacts, canonical `VARIANT_IDS` row order).

### `qquant/aggregate/stats.py` — THE one stats module (no other module reimplements these)

```python
def z_for_confidence(confidence: float = 0.95) -> float:
    """Two-sided normal quantile via statistics.NormalDist().inv_cdf (stdlib; no scipy)."""

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score CI (lo, hi) for the binomial proportion k/n. n>0; 0<=k<=n."""

def wilson_from_rate(rate: float, n: int, confidence: float = 0.95) -> tuple[float, float, int]:
    """k = round(rate * n); returns (lo, hi, k). Used for HumanEval pass@1 over k/n."""

@dataclass(frozen=True)
class WeightedMetric:
    value: float       # micro-average (Σ n_g·p_g / Σ n_g)
    stderr: float      # error-propagated: sqrt(Σ (n_g/N)^2 · se_g^2)
    n_total: int
    n_groups: int

def weighted_aggregate(
    values: Sequence[float], ns: Sequence[int],
    stderrs: Sequence[float] | None = None,
) -> WeightedMetric:
    """Weighted-by-n micro-average; se_g defaults to sqrt(p_g(1-p_g)/n_g) when stderrs is None."""

@dataclass(frozen=True)
class McNemarResult:
    n_pairs: int; b: int; c: int        # b: base correct & variant wrong; c: base wrong & variant correct
    statistic: float; p_value: float
    method: str                          # "exact-binomial" | "chi2-continuity"
    significant: bool                    # p_value < alpha

def mcnemar(
    baseline_correct: Sequence[bool], variant_correct: Sequence[bool],
    alpha: float = 0.05, exact_threshold: int = 25,
) -> McNemarResult:
    """Paired McNemar on aligned per-item correctness. Exact binomial (math.comb) when
    b+c <= exact_threshold, else continuity-corrected chi^2 (p via math.erfc). No scipy."""
```

### `qquant/aggregate/pipeline.py` — THE one aggregation pipeline

```python
@dataclass(frozen=True)
class CellRecord:
    variant: str; task: str; subject: str | None
    primary_metric: str; metric_value: float
    metric_stderr: float | None; n_samples: int; raw: dict

@dataclass(frozen=True)
class TaskScore:
    variant: str; task: str; metric: str
    value: float; stderr: float | None
    ci_low: float | None; ci_high: float | None
    n: int; cells_present: int; cells_expected: int; complete: bool

@dataclass(frozen=True)
class EfficiencyRecord:
    variant: str
    peak_vram_gb: float | None; weights_vram_gb: float | None
    throughput_tok_s: float | None; latency_ms_per_tok: float | None
    model_disk_gb: float | None; batch_size: int | None
    runtime_confounded: bool                     # True for self-quant (decision-log)
    raw: dict

@dataclass(frozen=True)
class AggregateResult:
    baseline: str; confidence: float
    task_scores: dict[str, dict[str, TaskScore]]      # variant -> task -> score
    efficiency: dict[str, EfficiencyRecord]
    significance: dict[str, dict[str, McNemarResult]]  # variant -> task -> McNemar vs baseline
    master_table: "pandas.DataFrame"
    degradation_table: "pandas.DataFrame"
    gaps: list[str]                                   # cell_ids missing for enabled variants

def load_cells(results_root, variants, tasks) -> list[CellRecord]
def aggregate_task(variant_id, task, records) -> TaskScore   # MMLU->weighted; humaneval->Wilson; else point
def load_efficiency(results_root, variant_id) -> EfficiencyRecord | None
def load_per_item(results_root, cell) -> dict[str, bool] | None  # doc-keyed correctness from the cell JSON's meta.item_correct/meta.item_ids
def reconcile_mmlu(weighted: WeightedMetric, lm_eval_group: dict) -> dict   # diagnostic delta; lm_eval_group loaded from _meta/mmlu_group_<variant>.json
def build_master_table(...) -> "pandas.DataFrame"
def build_degradation_table(..., baseline="bf16") -> "pandas.DataFrame"
def aggregate(results_root, *, baseline="bf16", confidence=0.95,
              quality_metric="mmlu", out_dir=None, strict=False) -> AggregateResult
```

### `qquant/aggregate/plots.py`

```python
def quality_vs_memory(result: AggregateResult, quality_metric: str, out_stem: Path) -> list[Path]
def quality_vs_throughput(result: AggregateResult, quality_metric: str, out_stem: Path) -> list[Path]
# Each writes <out_stem>.png and <out_stem>.svg (Agg backend); returns the written paths.
```

### Input contracts consumed (owned by 05/06/07 — read defensively)

- **Quality cells** — `cell_path(results_root, variant, task, subject)`; fields per `cell_result.schema.json`:
  `primary_metric`, `metric_value`, `metric_stderr?`, `n_samples`. MMLU subject token = bare slug.
- **Per-item correctness (inside the cell JSON)** — read **directly from each generative cell's**
  `meta.item_correct` (a list of `0/1`) aligned positionally with `meta.item_ids` (lm-eval `doc_id`s),
  as written by Spec 05. **There is no `.samples.jsonl` sidecar.** `load_per_item` zips
  `item_ids` → `item_correct` into a doc-keyed `{doc_id: bool}` map. Pairing key = `doc_id` within a
  (task[/subject]); MMLU McNemar concatenates subjects via `(subject, doc_id)`. (The
  `meta.item_correct`/`meta.item_ids` schema is owned by Spec 05.)
- **Efficiency JSON** — per-variant, `<results_root>/<variant>/efficiency.json` (built on `results_root`,
  not a redefinition of `cell_path`). Fields above are read defensively; missing → blank cell + a recorded gap.
  **Exact filename/shape owned by Spec 06** (open question below).
- **lm-eval MMLU group number** — Spec 05 persists the lm-eval `mmlu` group `{acc, acc_stderr}` to
  `<results_root>/_meta/mmlu_group_<variant>.json` (via `Paths.meta_dir`); `reconcile_mmlu` reads it from
  there and compares against our n-weighted aggregate. This is a **required Spec-05 output**, so the
  reconciliation is **deterministic** — there is no "skipped with a logged note" degradation.

### Output artifacts — `<results_root>/_analysis/`

- `aggregate.json` — machine-readable consolidation (the ONLY structured handoff to Spec 10): `schema`,
  `baseline`, `confidence`, per-variant per-task `{metric, value, stderr, ci_low, ci_high, n, complete}`,
  per-variant `efficiency`, per-variant per-task `mcnemar {b, c, p_value, method, significant}`, `gaps`,
  and an `mmlu_reconciliation` diagnostic block.
- `master_table.md` + `master_table.csv` — rows = `VARIANT_IDS` order; columns = mmlu / gsm8k / humaneval /
  ifeval (value ± stderr or [CI]) + efficiency (peak VRAM GB, throughput tok/s, model size GB); significance
  marker vs baseline on quality columns; self-quant throughput tagged "runtime-confounded".
- `degradation_table.md` + `degradation_table.csv` — one row per non-baseline variant × benchmark: `Δabs`,
  `Δrel%` vs `bf16`, McNemar `p`, significance flag.
- `plots/quality_vs_memory.{png,svg}`, `plots/quality_vs_throughput.{png,svg}`.

## Files to create

- `src/qquant/aggregate/__init__.py`
- `src/qquant/aggregate/stats.py` — the one stats module (Wilson / weighted MMLU / McNemar)
- `src/qquant/aggregate/pipeline.py` — the one aggregation pipeline (cells, efficiency, tables, `aggregate.json`)
- `src/qquant/aggregate/plots.py` — matplotlib (Agg) PNG+SVG renderers
- `src/qquant/aggregate/cli.py` — `qquant-aggregate` entry point (`main(argv=None) -> int`)
- `tests/test_aggregate_stats.py` — Wilson / weighted-aggregate / McNemar reference-value tests
- `tests/test_aggregate_pipeline.py` — fixture results tree → tables / `aggregate.json` / strict gate / determinism
- `tests/test_aggregate_torch_free.py` — importing `qquant.aggregate.*` imports neither `torch` nor `lm_eval`
- **Edit** `pyproject.toml` → add the one `qquant-aggregate` `[project.scripts]` line (contracts §7).

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **MMLU = 57 per-subject cells, weighted by subject n, ONE stderr method** (decision-log "MMLU"). Overall
  acc is the micro-average `Σ n_s p_s / N`; stderr is error-propagation of the per-subject binomial stderr,
  `se = sqrt(Σ (n_s/N)^2 · se_s^2)`, with `se_s = sqrt(p_s(1-p_s)/n_s)` (or the cell's stored `metric_stderr`).
  *Why one method:* prevents the ambiguity of mixing macro/micro and bootstrap/binomial. The point estimate
  equals lm-eval's group acc **by construction** (same n-weighting); `reconcile_mmlu` records any stderr
  delta vs lm-eval's pooled group formula as a **diagnostic**, not a failure (lm-eval may add a between-group
  term). Partial MMLU (some of 57 missing) aggregates over present subjects and is flagged `complete=False`.
- **Wilson CI for HumanEval pass@1** (decision-log "HumanEval code-exec": "report Wilson CI over k/n").
  *Why:* lm-eval may not emit a pass@1 stderr; Wilson is well-behaved near 0/1 and at small n (164 problems).
  `k = round(pass@1 · n_samples)`.
- **Paired McNemar vs `bf16`, gating all quality claims** (decision-log "Significance"). *Why:* variants run
  on the **same items** (paired); an unpaired z-test wastes the pairing and inflates variance. Discordant
  pairs only (`b`,`c`); exact binomial for small `b+c`, continuity-corrected χ² otherwise. If a cell lacks
  `meta.item_correct`/`meta.item_ids`, significance is **skipped with a recorded reason** and the claim is
  marked unverified — the gate is explicit, never silently assumed.
- **Self-quant speed is "runtime-confounded"** (decision-log "Self-quant = fair quality, confounded speed").
  *Why:* W4A16 on HF `generate()` may hit a slow dequant path on sm_89. `EfficiencyRecord.runtime_confounded`
  is `True` for self-quant; throughput cells and the quality-vs-throughput plot tag these points so no speed
  claim is drawn from them. Self-quant remains a first-class point on the **quality** axes.
- **Single homogeneous RTX 4090** (decision-log "Platform"): all efficiency numbers are apples-to-apples;
  no cross-hardware caveat needed in tables.
- **`awq-official` is contingent** (decision-log): iterate `enabled_variants(load_variants())`; a disabled
  variant simply produces no rows. Aggregation must never crash on its absence.
- **Per-variant batch differences are provenance, not noise** (decision-log VRAM/batch): record each cell's
  `config.batch_size` in `aggregate.json` so the report can note efficiency was measured at each variant's
  configured batch.
- **No PDF / LaTeX** (decision-log "Reporting"): this spec emits only `.md`/`.csv`/`.json` and PNG/SVG; the
  prose `results/REPORT.md` is Spec 10.
- **Torch-free + lm-eval-free** (contracts §7, the umbrella rule): aggregation uses only the `analysis` group
  (pandas/numpy/matplotlib) + stdlib; an import-guard test enforces it. matplotlib forced to the **Agg**
  backend for headless laptop/CI.
- **No scipy** (Dependencies Guideline): Wilson, McNemar, and the normal quantile are implemented with stdlib
  (`statistics.NormalDist().inv_cdf`, `math.erfc`, `math.comb`) + numpy. scipy is explicitly **not** added.
- **SSOT reuse:** import `VARIANT_IDS`/`TASK_IDS`/`MMLU_SUBJECTS`, `load_variants`/`load_tasks`/
  `enabled_variants`, `expand_matrix`/`Cell`/`missing_cells`, `cell_path`/`Paths`, `is_valid_cell`. Do not
  re-declare ids, paths, schema keys, or the resume predicate.

## Done-when (numbered, testable)

1. `qquant-aggregate --results-root <fixture>` (and `python -m qquant.aggregate.cli ...`) runs end-to-end on a
   synthetic 7-variant matrix and exits 0, writing every `_analysis/` artifact listed above.
2. `stats.weighted_aggregate` reproduces a hand-computed micro-average and error-propagated stderr on a fixture;
   the point estimate equals the n-weighted mean within `1e-12`.
3. `stats.wilson_interval(82, 164)` matches the published Wilson 95% CI for 0.5 over n=164 within `1e-6`, and is
   contained in `[0,1]` for edge cases `k=0` and `k=n`.
4. `stats.mcnemar` returns the correct `b`/`c` orientation (baseline-correct→variant-wrong is `b`), uses the
   exact-binomial path for `b+c <= 25` (matches a reference binomial two-sided p) and the χ²-continuity path
   above it, and sets `significant = p < 1 - confidence`.
5. `master_table.{md,csv}` contains all enabled variants (rows in `VARIANT_IDS` order) × the 4 benchmarks +
   the efficiency columns; quality cells carry stderr/CI and a significance marker vs `bf16`.
6. `degradation_table.{md,csv}` has one row per non-baseline variant × benchmark with `Δabs`, `Δrel%`,
   McNemar `p`, and a significance flag; the `bf16` row is omitted (or all-zero by construction).
7. `plots/quality_vs_memory.{png,svg}` and `plots/quality_vs_throughput.{png,svg}` are written; self-quant
   points are visibly tagged "runtime-confounded" on the throughput plot.
8. `aggregate.json` is written and round-trips (valid JSON with the documented top-level keys); a test loads it
   and asserts Spec 10 can read scores, efficiency, significance, gaps, and the MMLU reconciliation block from
   it **alone**.
9. Import guard: importing any `qquant.aggregate` submodule does not import `torch` or `lm_eval`
   (assert both absent from `sys.modules`).
10. `--strict` exits `2` when an enabled variant has missing cells (computed via `missing_cells`/`is_cell_done`);
    non-strict aggregates the present cells and lists every gap in `aggregate.json.gaps`.
11. Determinism: two runs over identical inputs produce byte-identical `.md`/`.csv`/`.json`
    (no timestamps/PYTHONHASHSEED dependence; stable row ordering).
12. Single Wilson/McNemar/weighted-mean implementation: the canonical implementations live **only** in
    `qquant.aggregate.stats`, and **Spec 05's eval-time pass@1 CI imports `wilson_interval` from here**
    (a shared helper) rather than reimplementing it. The #12 test checks the **import graph** —
    `qquant.aggregate.pipeline`/`plots`/`cli` and `qquant.eval.metrics` all import these from
    `qquant.aggregate.stats`, none define their own — not a global definition count.

## Risks & mitigations

- **Per-item vectors missing from a cell (Spec 05).** `load_per_item` reads `meta.item_correct` /
  `meta.item_ids` defensively (tolerates a cell that omits them); when correctness can't be recovered,
  McNemar is **skipped with a recorded reason** and the claim is flagged unverified rather than fabricated.
- **Efficiency JSON shape owned by Spec 06.** `load_efficiency` is field-tolerant (accepts `*_gb` or `*_bytes`,
  missing keys → `None`), records a gap, and never crashes. Tracked as an open question.
- **lm-eval group stderr ≠ our error-propagation.** We declare the error-propagation method authoritative,
  reconcile the **point estimate** exactly, and surface any stderr delta as a diagnostic in
  `mmlu_reconciliation` — never a hard failure.
- **McNemar exact-binomial overflow at large discordant n** (MMLU concatenated ≈ thousands of items).
  Hybrid threshold (`exact_threshold`, default 25) switches to the continuity-corrected χ² with a `math.erfc`
  tail; documented in `McNemarResult.method`.
- **matplotlib backend on headless CI.** Force `matplotlib.use("Agg")` before `pyplot` import; tests render to
  a tmp dir and assert non-empty PNG+SVG.
- **`awq-official` disabled / partial matrix.** Iterate `enabled_variants`; aggregate present cells; mark
  `complete=False` and record gaps; never assume all 420 cells exist.
- **Self-quant speed misread as a real win.** `runtime_confounded` flag propagates into the table label and the
  throughput plot so the report cannot draw a speed conclusion from confounded points.

## Open questions (cross-spec, to reconcile before implementation)

- **Spec 05 — RESOLVED.** Per-item correctness is read from each generative cell's
  `meta.item_correct` / `meta.item_ids` (no sidecar), and the lm-eval `mmlu` group `{acc, acc_stderr}`
  is persisted by Spec 05 to `<results_root>/_meta/mmlu_group_<variant>.json` (read by `reconcile_mmlu`).
- **Spec 06:** exact filename + field names of the per-variant efficiency JSON (proposed:
  `<results_root>/<variant>/efficiency.json`), including the `runtime_confounded` flag for self-quant.
- Confirm whether `_analysis/` is acceptable as the analysis output dir (distinct from `_meta/`, which holds
  run provenance) so Spec 10 reads a stable location.
