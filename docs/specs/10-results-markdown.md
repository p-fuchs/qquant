# Spec 10 — Results Markdown (`results/REPORT.md`, no PDF/LaTeX), mapped to the RQs

- **Status:** draft (ready to author once Spec 09 emits the `_analysis/` artifacts)
- **Depends on:** 09 (aggregation: `aggregate.json` + master/degradation tables + the two trade-off plots)
- **Owns / Produces:** the `qquant.report` package — **one** markdown assembler — the `qquant-report`
  entry point, and the single human-facing deliverable `results/REPORT.md`. It owns **no** numbers,
  tables, plots, ids, paths, schema keys, or predicates: it **assembles** Spec 09's `_analysis/`
  artifacts and `_meta/env.json` into a prose document and maps every section back to the research
  questions. The authors rewrite this markdown into their final report.

## Purpose

Spec 09 produces the analysis layer (scores, significance, efficiency, tables, plots) as machine and
table artifacts under `<results_root>/_analysis/`. Spec 10 turns that into **one** narrative
`results/REPORT.md` that a reader (and the authors) can act on without opening any JSON:

1. **Experimental setup** — hardware (single on-demand RTX 4090, 24 GB, sm_89, vast.ai), the two uv
   environments + CUDA 12.6 *devel* image (nvcc) provenance, the verified library pins and model
   commit SHAs, the seeds, the per-task chat-template / few-shot policy, and the resolved dataset
   revisions — sourced from `_meta/env.json` (falling back to the decision-log pins when a field is
   absent).
2. **RQ#1 (best trade-off)** — embed Spec 09's **master comparison table** (all variants × the 4
   benchmarks + efficiency) and link the two trade-off plots (quality-vs-memory, quality-vs-throughput).
3. **RQ#2 (capability sensitivity)** — embed Spec 09's **per-capability degradation table** and narrate
   which capabilities degrade, **gated by the paired McNemar significance** carried in `aggregate.json`.
4. **RQ#3 (speed vs memory)** — read the efficiency columns and state the priors (bnb-int8 often slower,
   NF4 = memory-not-speed, official GPTQ/AWQ are the faster INT4 paths), explicitly labelling
   **self-quant throughput "runtime-confounded"** so no speed conclusion is drawn from it.
5. **RQ#4 (optional QLoRA)** — documented as **spec-only** (Spec 12), out of the v1 run.
6. A **best-default-under-24 GB recommendation**, computed from `aggregate.json` by a documented,
   deterministic rule and emitted as a **DRAFT** for the authors to confirm, with the full candidate
   ranking so a human can override.

This runs **locally** (laptop/CI), not on the GPU box. It is **torch-free and lm-eval-free** at import
time and reads only `.json`/`.md` text + links `.png`/`.svg` — it renders no plots and recomputes no
statistics (Spec 09 owns those).

## Scope (in / out)

**In scope**
- Loading Spec 09's `_analysis/aggregate.json` (the **only** structured handoff), embedding
  `_analysis/master_table.md` and `_analysis/degradation_table.md` **verbatim**, and linking
  `_analysis/plots/quality_vs_{memory,throughput}.{png,svg}` with relative paths.
- Loading run provenance from `<results_root>/_meta/env.json` via `Paths.env_file`.
- Mapping each section explicitly to RQ#1–3 (+ the optional QLoRA RQ#4) using the exact `proposal.tex`
  wording.
- A documented, deterministic **recommendation** rule (best default under a VRAM ceiling), emitted as a
  reviewable draft with a candidate ranking.
- The `qquant-report` CLI; a `--strict` gate that fails when required `_analysis/` inputs are missing.

**Out of scope** (owned elsewhere)
- Computing any score, CI, significance, table, or plot → **Spec 09**. Spec 10 never re-renders a table
  or re-runs a stat; it embeds Spec 09's output.
- Producing cells (incl. their per-item `meta.item_correct`/`meta.item_ids`) / efficiency JSON → Spec 05/06/07.
- PDF / LaTeX / HTML rendering of any kind → **explicitly excluded** (decision-log "Reporting").
- Anything GPU / torch / lm-eval (this spec imports none).

## Interface

### Entry point (`pyproject.toml` → `[project.scripts]`)

Per contracts §7, `qquant-report` is reserved for this spec (README maps `qquant-report` → Spec 10).
This spec adds **one** line; it is a **separate** entry point, not a subcommand of the torch-free
`qquant` umbrella:

```toml
qquant-report = "qquant.report.cli:main"
```

### CLI

```
qquant-report --results-root results
              [--analysis-dir results/_analysis]   # where Spec 09 wrote its artifacts (default: <root>/_analysis)
              [--out results/REPORT.md]            # output markdown path (default: <root>/REPORT.md)
              [--baseline bf16]                    # reference variant (must match aggregate.json.baseline)
              [--max-vram-gb 24.0]                 # VRAM ceiling for the recommendation (the 4090's 24 GB)
              [--quality-metric mean]              # headline metric for the recommendation: mmlu|gsm8k|humaneval|ifeval|mean
              [--title "Qwen2.5-7B-Instruct quantization study"]
              [--strict]                           # nonzero exit if any required _analysis input is missing
```

Exit codes: `0` success; `2` strict-mode missing input (a required `_analysis/` artifact absent);
`1` usage / IO error. Output is **deterministic** (no wall-clock timestamps; any date shown is read from
`env.json`; canonical `VARIANT_IDS` row order inherited from the embedded tables).

### `qquant/report/render.py` — the markdown assembler

```python
@dataclass(frozen=True)
class ReportInputs:
    results_root: Path
    analysis_dir: Path
    aggregate: dict                       # parsed _analysis/aggregate.json (Spec 09 SSOT handoff)
    master_table_md: str | None           # verbatim _analysis/master_table.md (None + recorded gap if absent)
    degradation_table_md: str | None      # verbatim _analysis/degradation_table.md
    env: dict | None                      # parsed _meta/env.json (run provenance) or None
    plots: dict[str, dict[str, Path]]     # {"quality_vs_memory": {"png":P,"svg":P}, "quality_vs_throughput": {...}}
    missing: list[str]                    # required inputs that were absent (drives --strict)

def load_inputs(results_root: Path, analysis_dir: Path | None = None) -> ReportInputs:
    """Read aggregate.json (required), the two table .md files, env.json (via Paths.env_file),
    and resolve plot paths. Never raises on a missing optional file — records it in .missing."""

# Pure section renderers — each returns a markdown fragment; deterministic; no I/O, no wall clock.
def render_header(inp: ReportInputs, *, title: str) -> str
def render_experimental_setup(inp: ReportInputs) -> str        # hardware, 2 uv envs, image+nvcc, pins, SHAs, seeds, dataset revs, chat-template policy
def render_matrix_and_completeness(inp: ReportInputs) -> str   # enabled variants, 420/300/240 arithmetic, awq-official status, gaps from aggregate.json
def render_rq1_tradeoff(inp: ReportInputs) -> str              # embeds master_table.md + links both plots + trade-off narrative
def render_rq2_capability(inp: ReportInputs) -> str            # embeds degradation_table.md + McNemar-gated per-capability narrative
def render_rq3_speed_vs_memory(inp: ReportInputs) -> str       # efficiency narrative + priors + self-quant "runtime-confounded"
def render_rq4_qlora(inp: ReportInputs) -> str                 # optional, spec-only (Spec 12), out of v1
def render_recommendation(inp: ReportInputs, rec: "Recommendation") -> str   # DRAFT + candidate ranking table
def render_caveats(inp: ReportInputs) -> str                   # self-quant confound, awq-official contingency, partial-matrix flags
def render_provenance_footer(inp: ReportInputs) -> str         # artifact paths, aggregate.json schema id, exact reproduce command

def build_report(inp: ReportInputs, *, title: str, baseline: str,
                 max_vram_gb: float, quality_metric: str) -> str:
    """Concatenate the sections in fixed order; returns the full REPORT.md text."""

def write_report(markdown: str, out_path: Path) -> Path:
    """Write atomically (tmp + replace), trailing newline; returns out_path."""
```

### `qquant/report/recommend.py` — the deterministic recommendation rule

```python
@dataclass(frozen=True)
class Candidate:
    variant: str
    fits_vram: bool
    peak_vram_gb: float | None
    throughput_tok_s: float | None
    speed_confounded: bool                 # True for self-quant (runtime-confounded)
    headline_quality: float | None         # the --quality-metric value (mmlu|...|mean composite)
    quality_delta_vs_baseline: float | None
    quality_significant_loss: bool         # any quality task with McNemar p < alpha and a drop
    eligible: bool                         # fits_vram and not a significant quality regression

@dataclass(frozen=True)
class Recommendation:
    quality_metric: str
    max_vram_gb: float
    pick: str | None                       # chosen variant id, or None if no eligible candidate
    rationale: str
    candidates: list[Candidate]            # ranked, full slate for human override

def recommend_default(aggregate: dict, *, baseline: str = "bf16",
                      max_vram_gb: float = 24.0,
                      quality_metric: str = "mean") -> Recommendation:
    """Deterministic draft recommendation from aggregate.json. Rule (documented in the report):
    candidate = enabled, non-baseline variant; eligible = peak_vram_gb is known and <= max_vram_gb
    AND no quality task shows a *significant* drop vs baseline (McNemar p < 1-confidence with Δ<0).
    Rank eligible candidates by (headline_quality desc, peak_vram_gb asc, throughput asc only for
    non-confounded variants). Self-quant throughput is IGNORED in tie-breaks (runtime-confounded).
    If no candidate is eligible, pick=None and the rationale names the blocking constraint."""
```

The `mean` headline composite is the arithmetic mean of the present primary metrics
(`mmlu` acc, `gsm8k` exact_match, `humaneval` pass@1, `ifeval` prompt_level_strict_acc), each already in
[0,1]; documented in the report as a coarse aggregate, with per-task numbers always shown alongside.

### `qquant/report/cli.py`

```python
def main(argv: list[str] | None = None) -> int:
    """Parse args → load_inputs → recommend_default → build_report → write_report.
    --strict: return 2 if ReportInputs.missing contains a *required* artifact (aggregate.json,
    master_table.md, degradation_table.md, or either plot). Otherwise render with explicit
    '(artifact not found)' placeholders and return 0."""
```

### Output deliverable — `results/REPORT.md` (markdown only)

Fixed top-level section order (the format contract authors rely on):

```
# <title>
> Auto-generated DRAFT from results/_analysis. Authors rewrite into the final report.

## 1. Experimental setup
   - Hardware: single on-demand RTX 4090 (24 GB, Ada, sm_89), vast.ai; budget ceiling.
   - Environments: root eval env + isolated quant env; CUDA 12.6 devel image (digest) + nvcc.
   - Library pins (torch/transformers/gptqmodel/optimum/bitsandbytes/accelerate/
     compressed-tensors/lm-eval/numpy) and model commit SHAs (table).
   - Seeds (random/numpy/torch/fewshot) and greedy determinism (do_sample=False).
   - Per-task eval policy table (chat-template / few-shot / metric) and resolved dataset revisions.
## 2. Variant & task matrix (completeness)
   - Enabled variants, 7×60=420 / 300 / 240 arithmetic, awq-official status, listed gaps.
## 3. RQ#1 — Best quality–memory–speed trade-off
   - Embedded master comparison table; linked quality-vs-memory and quality-vs-throughput plots.
## 4. RQ#2 — Capability sensitivity
   - Embedded per-capability degradation table; McNemar-gated narrative per benchmark.
## 5. RQ#3 — Real speed vs memory savings
   - Efficiency narrative; priors; self-quant throughput flagged runtime-confounded.
## 6. RQ#4 (optional) — QLoRA recovery
   - Spec-only (Spec 12); out of the v1 run.
## 7. Recommendation — best default under 24 GB  [DRAFT]
   - Picked variant + rationale + full candidate ranking table.
## 8. Caveats & confounds
   - self-quant speed confound; awq-official contingency; partial-matrix flags; single-GPU note.
## Appendix — provenance & reproduction
   - Source artifact paths, aggregate.json schema id, exact `qquant-aggregate`/`qquant-report` commands.
```

Plots are linked relative to `results/REPORT.md`, e.g.
`![Quality vs peak VRAM](_analysis/plots/quality_vs_memory.png)` with a text link to the `.svg`.

### Inputs consumed (owned by Spec 09 / 03·08 — read defensively)

- `<analysis-dir>/aggregate.json` — **required**. Read per Spec 09's documented top-level keys:
  `baseline`, `confidence`, per-variant per-task `{metric, value, stderr, ci_low, ci_high, n, complete}`,
  per-variant `efficiency` (`peak_vram_gb`, `throughput_tok_s`, `model_disk_gb`, `batch_size`,
  `runtime_confounded`), per-variant per-task `mcnemar {b, c, p_value, method, significant}`, `gaps`,
  and `mmlu_reconciliation`. Spec 10 reads these straight off — it never recomputes them.
- `<analysis-dir>/master_table.md`, `<analysis-dir>/degradation_table.md` — embedded **verbatim** inside
  fenced sections (Spec 09 owns the column set, row order, significance markers, and the
  "runtime-confounded" self-quant tag).
- `<analysis-dir>/plots/quality_vs_{memory,throughput}.{png,svg}` — linked, not regenerated.
- `<results_root>/_meta/env.json` (via `Paths.env_file`) — run provenance: resolved library versions,
  dataset revisions/fingerprints, and the pinned image digest. Optional; missing fields fall back to the
  decision-log pins/SHAs (which the report labels "(from spec pins, env.json absent)").

## Files to create

- `src/qquant/report/__init__.py`
- `src/qquant/report/render.py` — `load_inputs`, the section renderers, `build_report`, `write_report`.
- `src/qquant/report/recommend.py` — `recommend_default` + `Candidate`/`Recommendation` (the one rule).
- `src/qquant/report/cli.py` — `qquant-report` entry point (`main(argv=None) -> int`).
- `tests/test_report_render.py` — fixture `_analysis/` + `_meta/env.json` → `REPORT.md`; section presence,
  verbatim table embedding, relative plot links, RQ mapping, `--strict` gate, byte-determinism.
- `tests/test_report_recommend.py` — `recommend_default` on a synthetic `aggregate.json`: VRAM filtering,
  significant-loss exclusion, self-quant tie-break exclusion, and the no-eligible-candidate path.
- `tests/test_report_torch_free.py` — importing any `qquant.report.*` imports neither `torch` nor `lm_eval`.
- **Edit** `pyproject.toml` → add the one `qquant-report` `[project.scripts]` line (contracts §7).

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **No PDF / LaTeX — markdown only** (decision-log "Reporting", user decision 2026-06-22). The sole
  deliverable is `results/REPORT.md` with tables inline (embedded from Spec 09) and plots linked. *Why:*
  the authors rewrite this markdown into their own report; a PDF/LaTeX pipeline is wasted effort and was
  explicitly cut. No HTML either.
- **`aggregate.json` is the only structured handoff** (Spec 09 §"Output artifacts" + its open question
  confirming `_analysis/`). Spec 10 reads scores/efficiency/significance/gaps/reconciliation from it
  **alone** and embeds the pre-rendered tables/plots; it never re-derives a number. *Why:* one renderer
  per table (no drift between a "report table" and the "analysis table"); Spec 09's Done-when 8 already
  guarantees `aggregate.json` carries everything Spec 10 needs.
- **Paired McNemar gates all quality claims** (decision-log "Significance"). The RQ#2 narrative may call a
  degradation "real/significant" **only** where `aggregate.json.…mcnemar.significant` is true; where a
  cell's per-item vectors (`meta.item_correct`/`meta.item_ids`) were absent, Spec 09 marked significance
  skipped/unverified and the report must phrase the change as "not significance-tested" rather than
  asserting it. *Why:* variants run on the same items;
  unpaired claims are invalid and the gate must survive into the prose.
- **Self-quant speed is "runtime-confounded"** (decision-log "Self-quant = fair quality, confounded
  speed"). The RQ#3 narrative, the recommendation tie-break, and the caveats section all label
  `gptq-selfquant`/`awq-selfquant` throughput as runtime-confounded (HF `generate()` W4A16 dequant on
  sm_89), and **no speed conclusion** is drawn from them; they remain first-class on the **quality** axes.
  *Why:* prevents the report from reading a slow dequant path as a speed result.
- **`awq-official` is contingent** (decision-log "Variant & evaluation"). The matrix/completeness section
  states whether it was enabled; if `enabled_variants(load_variants())` shows it disabled (or
  `aggregate.json` has no rows for it), the report names **`awq-selfquant` as the guaranteed AWQ data
  point** and never treats AWQ-official as assumed. *Why:* the sm_89 AWQ load may fail; the report must not
  imply a data point that doesn't exist.
- **Single homogeneous RTX 4090** (decision-log "Platform"): all efficiency numbers are apples-to-apples,
  so the report states this once and adds **no cross-hardware caveat** to the speed/memory comparison.
- **Per-variant batch differences are provenance, not noise** (decision-log VRAM/batch): the setup/RQ#3
  text notes efficiency was measured at each variant's configured batch (read from `aggregate.json`
  `efficiency.batch_size`), so a reader does not mistake a batch difference for a method difference.
- **Greedy determinism + per-task chat-template policy** (decision-log): the setup section records
  `do_sample=False` and the per-task chat-template/few-shot table (MMLU/GSM8K/IFEval chat-template +
  multiturn; HumanEval `humaneval_instruct` + fenced-code extraction) so the eval configuration is
  unambiguous and reproducible.
- **MMLU = 57 weighted cells, one stderr method** (decision-log "MMLU"): the setup states MMLU is the
  n-weighted aggregate over 57 subjects; the actual value/stderr and any `mmlu_reconciliation` delta come
  from `aggregate.json` — the report surfaces the reconciliation note as a diagnostic, never as a failure.
- **Two uv envs + CUDA 12.6 devel image (nvcc) + verified pins/SHAs** (decision-log "Platform"/"Two uv
  environments"/"Pinned model revisions"): the setup table is populated from `_meta/env.json` when present,
  else from the decision-log pins (`torch 2.12.1+cu126`, `transformers 5.10.1`, `gptqmodel 7.1.0`,
  `lm-eval 0.4.12`, …) and the three Qwen commit SHAs, clearly labelled by source. *Why:* reproducibility
  is a graded deliverable; the report is where pins/SHAs/seeds become legible.
- **QLoRA (RQ#4) is optional and spec-only** (decision-log "Scope"): mapped to Spec 12, excluded from the
  420-cell v1 run; the report lists it as optional/future, never as a measured result.
- **Best default under limited VRAM** (proposal "Final Deliverables"): the recommendation is the proposal's
  asked-for deliverable, computed deterministically and flagged **DRAFT** with the full candidate slate so
  the authors confirm/override rather than trust an auto-pick.
- **Torch-free + lm-eval-free** (contracts §7 umbrella rule): the report uses only stdlib (`json`,
  `pathlib`, `argparse`, `dataclasses`) and the SSOT torch-free core; an import-guard test enforces it. It
  does **not** import matplotlib (it links Spec 09's PNG/SVG, it does not render).
- **SSOT reuse:** import `VARIANT_IDS`/`TASK_IDS`, `load_variants`/`enabled_variants`, and `Paths`
  (`.env_file`). Read `gaps`/completeness from `aggregate.json` (do **not** reimplement `is_cell_done` /
  `missing_cells`). Do not re-declare ids, paths, schema keys, or the resume predicate.

## Done-when (numbered, testable)

1. `qquant-report --results-root <fixture>` (and `python -m qquant.report.cli ...`) runs end-to-end on a
   synthetic `_analysis/` tree (+ `_meta/env.json`) and exits 0, writing `results/REPORT.md`.
2. `REPORT.md` contains the eight fixed top-level sections in order, each RQ section labelled with the
   matching `proposal.tex` question text for RQ#1, RQ#2, RQ#3, and the optional RQ#4.
3. The **experimental setup** section renders a pins table (the decision-log libraries) and the **three
   Qwen commit SHAs**, the four seeds, `do_sample=False`, and the per-task chat-template/few-shot policy;
   values come from `_meta/env.json` when present and are labelled "(from spec pins, env.json absent)"
   otherwise — verified by a fixture with and without `env.json`.
4. The RQ#1 section embeds `_analysis/master_table.md` **verbatim** (byte-for-byte substring) and links
   both plots with **relative** paths (`_analysis/plots/quality_vs_memory.png` and `…throughput.png`),
   each with a text link to its `.svg`.
5. The RQ#2 section embeds `_analysis/degradation_table.md` verbatim and the narrative calls a degradation
   "significant" **only** for `(variant, task)` pairs where `aggregate.json` McNemar `significant == true`;
   pairs with skipped/unverified significance are phrased as "not significance-tested" (asserted on a
   fixture containing both a significant and an unverified pair).
6. The RQ#3 section labels every self-quant variant's throughput **"runtime-confounded"** and draws no
   speed conclusion from it; it states the bnb-int8/NF4/official-INT4 priors.
7. `recommend_default` on a synthetic `aggregate.json`: (a) excludes a variant whose `peak_vram_gb >
   max_vram_gb`; (b) excludes a variant with a significant quality drop; (c) never tie-breaks on a
   self-quant variant's confounded throughput; (d) returns `pick=None` with a blocking-constraint rationale
   when no candidate is eligible. The chosen `pick` and ranking appear in section 7 marked **DRAFT**.
8. The caveats section states the self-quant speed confound, the `awq-official` contingency (naming
   `awq-selfquant` as the guaranteed AWQ point when awq-official is absent/disabled), and lists the
   `aggregate.json.gaps` (partial-matrix flag); the single-homogeneous-GPU note appears exactly once.
9. `--strict` exits `2` when a required `_analysis/` artifact (aggregate.json, either table `.md`, or
   either plot) is missing; non-strict renders an explicit "(artifact not found)" placeholder and exits 0.
10. Import guard: importing any `qquant.report` submodule imports neither `torch` nor `lm_eval` (assert
    both absent from `sys.modules`); it also does not import `matplotlib`.
11. Determinism: two runs over identical inputs produce a **byte-identical** `REPORT.md` (no wall-clock
    timestamps; any date is read from `env.json`; row order inherited from the embedded tables).
12. No recomputation: a test asserts `qquant.report` defines no Wilson/McNemar/weighted-mean/table-builder
    function (those live only in `qquant.aggregate`); the report reads `aggregate.json` and embeds the
    pre-rendered tables.

## Risks & mitigations

- **`_analysis/` artifacts missing or partial** (Spec 09 not run, or a partial matrix). *Mitigation:*
  `load_inputs` records every absent file in `.missing`; `--strict` fails fast (exit 2), non-strict emits
  "(artifact not found)" placeholders so a partial report is still produced; `aggregate.json` is the only
  hard requirement and its absence is always a strict failure.
- **`aggregate.json` shape drift vs Spec 09.** *Mitigation:* read defensively with `.get(...)` and explicit
  per-key fallbacks; a fixture `aggregate.json` mirroring Spec 09's documented keys is the contract test, and
  Spec 09's Done-when 8 guarantees the keys the report reads.
- **Significance over-claimed in prose.** *Mitigation:* the RQ#2 narrative is driven solely by the boolean
  `mcnemar.significant`; unverified/skipped pairs are phrased as "not significance-tested" (Done-when 5),
  never asserted — the McNemar gate survives into the text.
- **Self-quant speed misread as a real win.** *Mitigation:* the `runtime_confounded` flag from
  `aggregate.json` drives an explicit label in RQ#3, the caveats, and the recommendation tie-break, which
  ignores confounded throughput (Done-when 6, 7c).
- **`awq-official` disabled.** *Mitigation:* the report iterates `enabled_variants` and the present
  `aggregate.json` variants; when AWQ-official is absent it names `awq-selfquant` as the guaranteed AWQ
  point and makes no AWQ-official claim (Done-when 8).
- **Recommendation looks authoritative.** *Mitigation:* section 7 is marked **DRAFT**, states the exact
  deterministic rule, and shows the full ranked candidate slate so the authors confirm or override.
- **Non-deterministic output (timestamps / dict ordering).** *Mitigation:* no wall-clock; dates come from
  `env.json`; table/row order inherited from Spec 09's deterministic embeds; byte-identity test (Done-when 11).

## Open questions (cross-spec, to reconcile before implementation)

- **Spec 09:** confirm the final `aggregate.json` key names the report reads
  (`efficiency.peak_vram_gb` / `throughput_tok_s` / `model_disk_gb` / `batch_size` / `runtime_confounded`,
  and `…mcnemar.significant`), and the exact filenames `master_table.md` / `degradation_table.md` and the
  `plots/` subdir, so the embed/link paths match byte-for-byte.
- **Spec 03/08:** confirm the `_meta/env.json` field names for resolved library versions, dataset
  revisions/fingerprints, and the pinned **image digest**, so the setup table reads them rather than
  always falling back to the decision-log pins.
- Confirm `results/REPORT.md` (alongside `_analysis/` and `_meta/`, not inside them) is the agreed output
  location the authors expect.
</content>
</invoke>
