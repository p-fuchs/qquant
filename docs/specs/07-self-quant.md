# Spec 07 — Self-quant fairness: isolated `quant/` env + C4 calibration + checkpoints

- **Status:** Draft (ready to implement on the box)
- **Depends on:** 01 (scaffold: `qquant.registry`, `qquant.paths`), 04 (compressed-tensors variant loader — used by the kernel smoke and by eval)
- **Owns / Produces:**
  - the isolated `quant/` uv project (package `selfquant`) — finalizes the skeleton at `quant/pyproject.toml`
  - a reproducible **C4 calibration artifact** (128 samples × 2048 tokens, deterministic) under `checkpoints/self-quant/_calib/`
  - the two self-quant **compressed-tensors checkpoints** `checkpoints/self-quant/gptq-selfquant/` and `checkpoints/self-quant/awq-selfquant/` (the `local:` targets that `variants.yaml` already names)
  - a per-checkpoint **quant manifest** (idempotency / resume) and a **kernel smoke** record (which W4A16 kernel + decode tok/s the HF backend uses)
  - `docs/selfquant_fairness.md`

> **Numbering note.** This spec is **Spec 07** per `README.md` (the renumbered, monotonic scheme).
> The `quant/README.md`, `quant/pyproject.toml` comments, and a couple of `decision-log.md`
> lines still say "Spec 08" — those are pre-renumbering labels for *this* work; do not let them
> mislead. (Spec 08 in the current scheme is the orchestration driver.) Updating those stale
> labels is an open question below; this spec does not edit them.

## Purpose

Produce the two **self-quantized** variants — `gptq-selfquant` and `awq-selfquant` — under a
single controlled protocol so that the GPTQ-vs-AWQ comparison is **calibration-controlled**:
**W4A16_ASYM for both**, the **same** C4 calibration set, and the **same** `group_size`. The
**only** intentional difference is the quantization *algorithm* (GPTQ error-compensation vs AWQ
activation-aware scaling). This is the guaranteed AWQ data point (per the decision log, official
AWQ on sm_89 is contingent) and the backbone of the **quality** fairness claim. Self-quant
**speed** is explicitly **runtime-confounded** (W4A16_ASYM disables Marlin; the HF `generate()`
backend may take a slow dequant path on sm_89 — fast Marlin targets vLLM), so this spec also
records a **kernel smoke** to substantiate that caveat, and writes `docs/selfquant_fairness.md`.

Checkpoints are **persisted** (and exfiltrated by Spec 08) so a resume **never re-quantizes**.

## Scope (in / out)

**In:**
- Finalize the isolated `quant/` uv env (`llmcompressor==0.12.0`, `datasets`, transformers/torch
  via llm-compressor's transitive pins) and commit `quant/uv.lock` resolved **on the box**.
- Build a deterministic, reproducible **C4 calibration artifact** (128 × 2048 tokens) once and
  reuse the **identical object** for both algorithms.
- Run `llm-compressor` `oneshot` twice (GPTQ, then AWQ) with `scheme="W4A16_ASYM"`,
  `group_size=128`, `ignore=["lm_head"]`, writing **compressed-tensors** checkpoints with
  `save_compressed=True` (loadable natively by transformers v5 + `compressed-tensors==0.17.1`).
- Per-checkpoint **quant manifest** + an **idempotency predicate** so resume skips done work.
- A **kernel smoke** (eval env) that loads each checkpoint via the **Spec 04** loader and records
  the W4A16 linear kernel/module class, a rough decode tok/s, and peak VRAM.
- `docs/selfquant_fairness.md`.

**Out (owned elsewhere — do not implement here):**
- Loading self-quant checkpoints for evaluation → **Spec 04** (`quant_method == "compressed-tensors"`).
- The authoritative efficiency numbers and `qquant-profile` → **Spec 06** (the kernel smoke here is
  diagnostic only, not the speed number that goes in the table).
- Exfil / re-upload of checkpoints on resume, budget guard, auto-destroy → **Spec 08**.
- Aggregation, paired McNemar, plots, REPORT.md → **Specs 09/10**.
- Any change to ids / paths / cell schema / resume predicate — those are SSOT (`contracts.md`).
  `variants.yaml` **already** points `gptq-selfquant`/`awq-selfquant` `model_id` at
  `local:checkpoints/self-quant/<id>`; this spec must match that, not redefine it.

## Interface

### A. Calibration — `quant/selfquant/calibration.py` (quant env)

```python
C4_DATASET = "allenai/c4"     # datasets 4.x: parquet config (loading scripts removed)
C4_CONFIG  = "en"
C4_SPLIT   = "train"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 2048
CALIB_SEED  = 42

@dataclass(frozen=True)
class CalibrationSpec:
    dataset: str = C4_DATASET
    config: str = C4_CONFIG
    split: str = C4_SPLIT
    num_samples: int = NUM_CALIBRATION_SAMPLES
    max_seq_len: int = MAX_SEQ_LEN
    seed: int = CALIB_SEED
    @property
    def calib_id(self) -> str: ...        # e.g. "c4-128x2048-s42"

def build_calibration(tokenizer, spec: CalibrationSpec, out_dir: Path) -> "datasets.Dataset":
    """Deterministically select C4 documents, tokenize, and truncate to EXACTLY
    `spec.max_seq_len` tokens, yielding exactly `spec.num_samples` rows with an
    `input_ids` (+ `attention_mask`) column. Persists the tokenized Dataset
    (`save_to_disk`) and a calibration manifest (selected source indices, resolved
    dataset revision/fingerprint, tokenizer name+revision, seed, sha256 of all
    input_ids). Idempotent: if a valid artifact + manifest already exist for this
    `calib_id`, load and return it instead of rebuilding."""

def load_calibration(out_dir: Path) -> "datasets.Dataset": ...
def calibration_sha256(ds: "datasets.Dataset") -> str:   # sha256 over concatenated input_ids
```

Determinism: fixed `seed`; deterministic candidate stream (e.g. `load_dataset(..., split, streaming=True).take(POOL)` or non-streaming `.shuffle(seed)` then deterministic filter for docs with ≥ `max_seq_len` tokens); take the first `num_samples` that pass; truncate to exactly `max_seq_len`. The selected source indices and the input-ids sha256 are recorded so two builds are bit-identical.

### B. Recipes — `quant/selfquant/recipes.py` (quant env)

```python
GROUP_SIZE = 128
SCHEME     = "W4A16_ASYM"     # 4-bit weights, fp16 activations, ASYMMETRIC, per-group
IGNORE     = ["lm_head"]

def gptq_recipe(group_size: int = GROUP_SIZE) -> list:
    # [GPTQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]
    # (GPTQModifier carries the quant config itself)

def awq_recipe(group_size: int = GROUP_SIZE) -> list:
    # [AWQModifier(),
    #  QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]
    # (AWQ smooths first, then a separate QuantizationModifier applies the scheme)

def resolved_quant_args(recipe: list) -> dict:
    """Resolve the recipe to its concrete weight QuantizationArgs
    (num_bits, symmetric, group_size, strategy) for the equality assertion."""

def assert_schemes_match(a: list, b: list) -> None:
    """Fail fast unless both recipes resolve to IDENTICAL weight quant args
    (num_bits=4, symmetric=False, strategy='group', group_size=128). This is the
    machine-checked fairness guarantee that the ONLY difference is the algorithm."""
```

### C. Quantizer + CLI — `quant/selfquant/quantize.py` (quant env)

```python
SELF_QUANT_IDS = ("gptq-selfquant", "awq-selfquant")   # mirrors VARIANT_IDS, no aliases
BASE_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION  = "a09a35458c702b33eeacc393d103063234e8bc28"   # decision-log baseline SHA

def quantize_variant(variant_id: str, calib_ds, out_dir: Path, *,
                     base_model_id: str = BASE_MODEL_ID,
                     base_revision: str = BASE_REVISION,
                     force: bool = False) -> Path:
    """Load the base model at the pinned SHA, run `oneshot(model=..., dataset=calib_ds,
    recipe=<gptq|awq>_recipe(), max_seq_length=2048, num_calibration_samples=128)`,
    then model.save_pretrained(out_dir, save_compressed=True) + tokenizer.save_pretrained.
    Writes a QuantManifest. Skips (returns out_dir) if selfquant_checkpoint_done(...) and
    not force. One variant at a time; free the model between runs (24 GB budget)."""

def main(argv=None) -> int:
    # python -m selfquant.quantize  [--id gptq-selfquant|awq-selfquant|all]
    #                               [--checkpoints-root checkpoints/self-quant]
    #                               [--base-revision SHA] [--calib-only] [--force]
    # Default: --id all. Builds the calibration ONCE, passes the SAME object to both runs.
```

Canonical invocation: `cd quant && uv run python -m selfquant.quantize --id all`.
(Optional convenience console script `qquant-selfquant = "selfquant.quantize:main"` MAY be added
to `quant/pyproject.toml` — it is a *separate* uv project, so it does not touch the root
`[project.scripts]` reservations in `contracts.md` §7. This spec adds **no** root entry point;
`qquant-profile` belongs to Spec 06.)

### D. Manifest + idempotency — `quant/selfquant/manifest.py` (quant env)

```python
@dataclass(frozen=True)
class QuantManifest:
    variant_id: str            # "gptq-selfquant" | "awq-selfquant"
    algorithm: str             # "gptq" | "awq"   (the ONLY field that differs)
    scheme: str                # "W4A16_ASYM"
    group_size: int            # 128
    base_model_id: str
    base_revision: str         # pinned baseline SHA
    calib_id: str              # "c4-128x2048-s42"
    calib_sha256: str          # ties the checkpoint to an exact calibration set
    llmcompressor_version: str
    transformers_version: str
    compressed_tensors_version: str
    created_at: str            # UTC ISO-8601

def write_manifest(ckpt_dir: Path, m: QuantManifest) -> None      # ckpt_dir/quant_manifest.json
def read_manifest(ckpt_dir: Path) -> QuantManifest | None

def selfquant_checkpoint_done(ckpt_dir: Path, *, algorithm: str, base_revision: str,
                              scheme: str, group_size: int, calib_sha256: str) -> bool:
    """True iff a valid compressed-tensors checkpoint is present (config.json has a
    quantization_config) AND quant_manifest.json exists AND (algorithm, base_revision,
    scheme, group_size, calib_sha256) all match. This is the resume/idempotency gate —
    Spec 08 exfiltrates the dir; a matching dir is never re-quantized."""
```

### E. Kernel smoke — `src/qquant/selfquant_smoke.py` (root **eval** env, GPU-touching)

```python
def run_kernel_smoke(variant_id: str, ckpt_dir: Path, *,
                     prompt_tokens: int = 512, gen_tokens: int = 128,
                     out_path: Path | None = None) -> dict:
    """Load the checkpoint via the Spec 04 compressed-tensors loader, then record:
      - the module/kernel class used for the quantized Linear layers (e.g. the
        compressed-tensors CompressedLinear / which W4A16 path; whether Marlin or a
        dequant-and-matmul fallback fires on sm_89),
      - compressed-tensors `format` from the loaded config,
      - decode_tok_s (rough single-stream greedy decode of `gen_tokens`),
      - peak_vram_gb, dtype.
    Writes JSON to ckpt_dir/kernel_smoke.json (so it travels with the checkpoint)."""

def main(argv=None) -> int:
    # python -m qquant.selfquant_smoke --variant gptq-selfquant \
    #         --ckpt checkpoints/self-quant/gptq-selfquant
```

This module is **GPU-touching** and **must not** be imported by `qquant.cli` (the torch-free
umbrella). It is invoked via `python -m`, never as a `qquant` subcommand.

### File formats

- **Calibration manifest** `checkpoints/self-quant/_calib/<calib_id>.manifest.json`:
  `{dataset, config, split, revision, num_samples, max_seq_len, seed, tokenizer, tokenizer_revision, selected_indices, calib_sha256}`.
- **Tokenized calibration set** `checkpoints/self-quant/_calib/<calib_id>/` (`datasets.save_to_disk`).
- **Checkpoint** `checkpoints/self-quant/<id>/`: standard compressed-tensors layout
  (`config.json` with `quantization_config`, `*.safetensors`, tokenizer files) + `quant_manifest.json` + `kernel_smoke.json`.

## Files to create

**`quant/` (isolated uv project, package `selfquant`):**
- `quant/selfquant/__init__.py`
- `quant/selfquant/calibration.py`
- `quant/selfquant/recipes.py`
- `quant/selfquant/quantize.py`
- `quant/selfquant/manifest.py`
- `quant/tests/test_calibration_determinism.py` (pure-Python determinism of the selection/sha logic on a fake corpus — no model download)
- `quant/uv.lock` (resolved & committed **on the box**)
- (edit) `quant/pyproject.toml` — only if adding the optional `[project.scripts]` console entry; the package layout `packages = ["selfquant"]` is already correct.

**root eval env:**
- `src/qquant/selfquant_smoke.py`
- `tests/test_selfquant_smoke_torch_free.py` (assert importing `qquant`/`qquant.cli` does **not** import `qquant.selfquant_smoke`, so the umbrella stays torch-free)

**docs:**
- `docs/selfquant_fairness.md`

**Do not create** `configs/variants.yaml`, alternate registries, or any new id/path/schema —
those are SSOT in Spec 01.

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **Two uv envs (NOT one).** `llmcompressor==0.12.0` caps `transformers<=5.10.1` with tight
  transitive pins that cannot co-resolve with the eval stack (`gptqmodel` + `lm-eval`). So the
  quantizer lives in `quant/` and only ever *produces* checkpoints; the eval env loads them with
  `compressed-tensors==0.17.1`. *Why:* avoids a brittle single lock and keeps eval reproducible.
- **W4A16_ASYM for both; controlled axis = calibration + group_size.** Verified against
  llm-compressor docs: `W4A16_ASYM` is a real compressed-tensors preset; GPTQ →
  `GPTQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])`; AWQ →
  `[AWQModifier(), QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]`.
  `assert_schemes_match` machine-checks that both resolve to **identical** weight quant args so the
  *only* difference is the algorithm. *Why:* this is the entire fairness claim for RQ on
  GPTQ-vs-AWQ quality.
- **Self-quant = fair quality, confounded speed.** GPTQ-asym disables Marlin and the HF
  `generate()` backend may take a slow dequant path on sm_89 (fast Marlin targets vLLM). So this
  spec's `kernel_smoke.json` only *documents* the kernel path + a rough tok/s; the authoritative
  speed numbers and the "runtime-confounded" label in the report come from Spec 06/10. *Why:*
  prevents mislabeling a kernel artifact as an algorithmic speed result.
- **Persist checkpoints; resume never re-quantizes.** `selfquant_checkpoint_done` + the per-dir
  `quant_manifest.json` (carrying `base_revision`, `scheme`, `group_size`, `calib_sha256`,
  `algorithm`) make quantization idempotent; Spec 08 exfiltrates `checkpoints/self-quant/` and
  re-uploads on resume. The same manifest fingerprint (`base_revision`/`calib_sha256`/`algorithm`)
  feeds **Spec 05**'s self-quant `RunConfig.model_revision` so re-quantization invalidates eval
  cells. *Why:* each oneshot run is ~10–20 min of GPU time we must not repeat.
- **Pinned base model SHA.** Quantize `Qwen/Qwen2.5-7B-Instruct` at
  `a09a35458c702b33eeacc393d103063234e8bc28` (same baseline weights as `bf16`/`bnb-*`). *Why:*
  removes a base-weights confound between the baseline and the self-quant variants.
- **Dataset reproducibility.** C4 under `datasets` 4.x uses the parquet config (`allenai/c4`/`en`);
  loading scripts are gone. Record the resolved dataset revision/fingerprint + selected indices +
  input-ids sha256 in the calibration manifest, and pre-flight that C4 loads under the pinned
  `datasets`. *Why:* the calibration set is the controlled variable — it must be reconstructible.
- **`compressed-tensors` native load.** Save with `save_compressed=True`; self-quant variants load
  natively in transformers v5 (NOT via gptqmodel's legacy GPTQ/AWQ path) — matches
  `quant_method == "compressed-tensors"` and `model_id == local:checkpoints/self-quant/<id>`.
- **Greedy determinism (cross-check).** After loading via Spec 04, the determinism smoke (force
  `do_sample=False`, clear `temperature/top_p/top_k`) must give identical output twice; the kernel
  smoke decodes greedily for the same reason. *Why:* keeps these checkpoints on the project's
  determinism contract.
- **No PDF.** `docs/selfquant_fairness.md` is markdown; Spec 10 folds its conclusions into
  `results/REPORT.md`.

## Done-when (numbered, testable)

1. `cd quant && uv sync` resolves on the Linux/CUDA box; `quant/uv.lock` is committed;
   `uv run python -m selfquant.quantize --help` prints the documented flags.
2. **Calibration determinism:** two `build_calibration` runs with the same `CalibrationSpec`
   produce identical `calib_sha256` and identical `selected_indices`; the manifest records the
   dataset revision, seed, `num_samples=128`, and `max_seq_len=2048`.
   (`quant/tests/test_calibration_determinism.py` proves the selection/sha logic deterministic on
   a fake corpus without any download.)
3. `uv run python -m selfquant.quantize --id all` creates
   `checkpoints/self-quant/gptq-selfquant/` and `checkpoints/self-quant/awq-selfquant/`, each a
   valid compressed-tensors checkpoint whose `config.json.quantization_config` has weights
   `num_bits=4`, `symmetric=false`, `strategy="group"`, `group_size=128`.
4. **Controlled axis:** both `quant_manifest.json` files share the same `base_revision`, `scheme`,
   `group_size`, and `calib_sha256`; the **only** differing field is `algorithm`
   (`gptq` vs `awq`). `assert_schemes_match(gptq_recipe(), awq_recipe())` passes.
5. **Idempotency:** a second `--id all` invocation without `--force` re-quantizes nothing
   (`selfquant_checkpoint_done` returns True for both) and exits 0; `--force` recomputes.
6. The **Spec 04** compressed-tensors loader loads each checkpoint in the eval env, and a greedy
   determinism smoke yields identical output on two runs (cross-checked here; loader owned by 04).
7. `python -m qquant.selfquant_smoke --variant <id> --ckpt <dir>` writes
   `checkpoints/self-quant/<id>/kernel_smoke.json` with the W4A16 linear kernel/module class,
   compressed-tensors `format`, a `decode_tok_s`, and `peak_vram_gb`; the JSON states whether
   Marlin or a dequant fallback is used on sm_89.
8. `docs/selfquant_fairness.md` exists and states: the controlled axes (W4A16_ASYM, group_size
   128, identical C4 calibration, identical base SHA), that the only difference is the algorithm,
   and that self-quant **speed is runtime-confounded** — referencing the calib manifest and both
   quant manifests.
9. Importing `qquant` / `qquant.cli` does **not** import `qquant.selfquant_smoke`
   (`tests/test_selfquant_smoke_torch_free.py`); no new root `[project.scripts]` line is added.
10. `variants.yaml` is unchanged: `gptq-selfquant`/`awq-selfquant` already map to
    `local:checkpoints/self-quant/<id>` with `quant_method: compressed-tensors` — the produced
    dirs match those paths exactly.

## Risks & mitigations

- **AWQ auto-mappings for Qwen2 fail.** `AWQModifier()` relies on architecture mappings for the
  smoothing pairs. *Mitigation:* if auto-detection errors, pass explicit Qwen2 `mappings`
  (attention q/k/v + gate/up/down) per the llm-compressor Qwen example; validate on the box.
- **`W4A16_ASYM` group_size assumption.** The preset default is per-group 128, but we must not
  rely on it silently. *Mitigation:* `assert_schemes_match` + the Done-when #3 config check fail
  fast unless both resolve to `num_bits=4, symmetric=False, strategy="group", group_size=128`.
- **C4 availability / format under `datasets` 4.x.** Loading scripts removed; C4 may be slow or
  gated. *Mitigation:* use the parquet config; record the resolved revision; pre-flight the load;
  keep `POOL` modest (stream + `.take`) so only a small slice is fetched; fall back to a pinned
  mirror revision if the default is unavailable, recording it in the manifest.
- **llm-compressor rejects a pre-tokenized Dataset.** *Mitigation:* prefer passing the
  pre-tokenized `input_ids` Dataset (zero tokenization ambiguity); if the data pipeline requires
  text, pass the **same** 128 text rows + same tokenizer + `max_seq_length=2048` to both runs
  (identical tokenization) and compute `calib_sha256` over the resulting tokens. Either way the
  **same object** is handed to both `oneshot` calls.
- **24 GB VRAM during quantization.** 7B GPTQ/AWQ oneshot can spike. *Mitigation:* quantize one
  variant at a time, free the model + `torch.cuda.empty_cache()` between runs, rely on
  llm-compressor's sequential pipeline / CPU offload; set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Eval-cell provenance blind to re-quantization — resolved via the quant manifest.**
  `variants.yaml` sets `revision: null` for self-quant, so an eval cell's `model_revision` would
  not change if the checkpoint is rebuilt. This spec's `quant_manifest.json` carries
  `base_revision`, `calib_sha256`, `algorithm`, and `group_size`; **Spec 05 derives the effective
  `RunConfig.model_revision` for self-quant from this manifest** (e.g.
  `f"selfquant:{base_revision[:12]}:{calib_sha256[:12]}:{algorithm}"`) so `is_cell_done` treats a
  re-quantized checkpoint as **stale**. This spec does not alter the SSOT (no `PROVENANCE_KEYS`
  change) — it only emits the manifest fingerprint that feeds Spec 05.
- **Stale "Spec 08" labels.** `quant/README.md`, `quant/pyproject.toml` comments, and two
  `decision-log.md` lines call this work "Spec 08". *Mitigation:* leave them (out of this spec's
  write scope) and flag for a docs sweep; the canonical numbering is `README.md` (Spec 07).
- **Kernel-smoke tok/s misread as the speed result.** *Mitigation:* `docs/selfquant_fairness.md`
  and the JSON both label it diagnostic; Spec 06 owns the authoritative profile.
