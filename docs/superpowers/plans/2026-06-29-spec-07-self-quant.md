# Spec 07 — Self-quant checkpoints (`quant/selfquant` + kernel smoke) Implementation Plan

> **For agentic workers:** Lean plan — Spec 07's core is GPU/box-only (the 7B `oneshot` quantization, the `llmcompressor` recipes, the kernel smoke, `quant/uv.lock`), verified on the rented RTX 4090, NOT on the laptop. CPU-testable units (manifest idempotency, calibration determinism on a fake corpus, the torch-free guard) get tests now; box-only modules are authored against the Context7-confirmed `llm-compressor` API and run on the box. Steps use checkbox (`- [ ]`).

**Goal:** Produce the two calibration-controlled self-quant checkpoints `checkpoints/self-quant/{gptq-selfquant,awq-selfquant}/` (W4A16_ASYM, group_size 128, identical C4 calibration, same base SHA — only the algorithm differs), idempotently, with a kernel smoke, runnable today via `cd quant && uv run python -m selfquant.quantize --id all`.

**Architecture:** An isolated `quant/` uv project (package `selfquant`, `llmcompressor==0.12.0`, Linux/CUDA) that builds a deterministic C4 calibration set ONCE and runs `oneshot` twice (GPTQ, AWQ) into compressed-tensors checkpoints + per-dir `quant_manifest.json` (idempotency). A root-env GPU smoke (`qquant.selfquant_smoke`) loads each checkpoint via the Spec-04 loader and records the W4A16 kernel path. `variants.yaml` already names the `local:` targets — unchanged.

**Tech Stack:** `quant/` env: Python 3.11, `llmcompressor==0.12.0`, `datasets>=4.8.4` (Linux+CUDA, lock resolved on the box). Root eval env: torch/transformers + the Spec-04 `compressed-tensors` loader. pytest; ruff.

## Global Constraints (verbatim from `docs/specs/07-self-quant.md`, `contracts.md`, `decision-log.md`)

- **Two uv envs.** The quantizer lives in `quant/` (package `selfquant`, `llmcompressor==0.12.0`); it only PRODUCES checkpoints. The eval env LOADS them via `compressed-tensors==0.17.1`. Never co-resolve them.
- **W4A16_ASYM for BOTH; controlled axis = calibration + group_size.** GPTQ = `GPTQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])`; AWQ = `[AWQModifier(), QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]`. The ONLY intentional difference is the algorithm. `assert_schemes_match` machine-checks both resolve to identical weight quant args (`num_bits=4, symmetric=False, strategy="group", group_size=128`).
- **`oneshot` call (Context7-confirmed):** `oneshot(model=…, dataset=calib_ds, recipe=…, max_seq_length=2048, num_calibration_samples=128, shuffle_calibration_samples=False)`; then `model.save_pretrained(out_dir, save_compressed=True)` + `tokenizer.save_pretrained(out_dir)`. Pass the **pre-tokenized** `input_ids` Dataset (deterministic); `shuffle_calibration_samples=False` because we pre-select.
- **Pinned base model.** `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28`.
- **Deterministic C4 calibration.** 128 samples × 2048 tokens, seed 42, parquet config (`allenai/c4`/`en`/`train`). Record selected source indices + `input_ids` sha256 + resolved dataset revision + tokenizer name/revision in a calib manifest. Two builds bit-identical → identical `calib_sha256`.
- **Persist checkpoints; resume never re-quantizes.** `selfquant_checkpoint_done(...)` + per-dir `quant_manifest.json` (carrying `algorithm`, `base_revision`, `scheme`, `group_size`, `calib_sha256`) gate quantization. `--force` recomputes.
- **One variant at a time; free VRAM between** (24 GB budget). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA init (launcher concern; quantize.py only relies on it).
- **Self-quant speed is runtime-confounded.** `kernel_smoke.json` is DIAGNOSTIC only (kernel path + rough tok/s); the authoritative speed number is Spec 06. `docs/selfquant_fairness.md` states the controlled axes + the speed caveat.
- **No SSOT changes.** No new ids/paths/cell-schema/resume predicate; `variants.yaml` unchanged (already `local:checkpoints/self-quant/<id>`, `quant_method: compressed-tensors`). No root `[project.scripts]` line (the optional `qquant-selfquant` console script lives in `quant/pyproject.toml`, a separate uv project).
- **Torch-free umbrella.** `qquant.selfquant_smoke` is GPU-touching and MUST NOT be imported by `qquant.cli`; invoked via `python -m`.
- **Tooling.** ruff line-length 88, rules E,F,I,UP,B. `from __future__ import annotations`. `quant/` env has its own ruff/pytest (dev group, not platform-gated → runs on laptop).

---

### Task 1: `quant/selfquant` package + `manifest.py` (idempotency) — CPU-testable

**Files:** Create `quant/selfquant/__init__.py`, `quant/selfquant/manifest.py`, `quant/tests/test_manifest.py`. Edit `quant/pyproject.toml` (console script + stale-label fix), `quant/README.md` (stale-label fix).

**Interfaces produced:** `QuantManifest` (frozen dataclass, 12 fields per spec §D), `write_manifest(ckpt_dir, m)`, `read_manifest(ckpt_dir) -> QuantManifest | None`, `selfquant_checkpoint_done(ckpt_dir, *, algorithm, base_revision, scheme, group_size, calib_sha256) -> bool`.

- [ ] **Step 1: failing test** — `quant/tests/test_manifest.py`: write a `QuantManifest`, read it back (round-trip equality); `selfquant_checkpoint_done` returns False when no `config.json`/manifest, False when fields mismatch (e.g. different `calib_sha256` or `algorithm`), True when a `config.json` with a `quantization_config` key AND a matching manifest are present (use `tmp_path`, write a fake `config.json = {"quantization_config": {...}}`).
- [ ] **Step 2:** run → FAIL (no module).
- [ ] **Step 3: implement** `manifest.py` — `QuantManifest` is a `@dataclass(frozen=True)` with the 12 fields (`variant_id, algorithm, scheme, group_size, base_model_id, base_revision, calib_id, calib_sha256, llmcompressor_version, transformers_version, compressed_tensors_version, created_at`). `write_manifest` → `json.dump(asdict(m))` to `ckpt_dir/quant_manifest.json` (sorted keys, indent 2). `read_manifest` → parse + `QuantManifest(**data)` or None. `selfquant_checkpoint_done`: `config = ckpt_dir/config.json`; require it exists with a top-level `"quantization_config"`; `m = read_manifest(...)`; require `m` and `(m.algorithm, m.base_revision, m.scheme, m.group_size, m.calib_sha256) == (algorithm, base_revision, scheme, group_size, calib_sha256)`. `__init__.py`: package docstring + `from __future__ import annotations`.
- [ ] **Step 4:** `cd quant && uv run --group dev pytest tests/test_manifest.py -v` → PASS. (If `quant/` has no synced env yet on the laptop, run with the root env: `uv run --project . pytest quant/tests/test_manifest.py` — the manifest module is pure stdlib, importable anywhere.)
- [ ] **Step 5: edits** — `quant/pyproject.toml`: add `[project.scripts]\nqquant-selfquant = "selfquant.quantize:main"`; fix the header comment `Spec 08` → `Spec 07`. `quant/README.md`: title `(Spec 08)` → `(Spec 07)`, and the trailing `docs/specs/08-*.md` → `docs/specs/07-self-quant.md`.
- [ ] **Step 6: commit** `Spec 07: selfquant package + quant_manifest idempotency + quant/ label fixes`.

---

### Task 2: `quant/selfquant/calibration.py` — deterministic C4 set (CPU-testable selection/sha)

**Files:** Create `quant/selfquant/calibration.py`, `quant/tests/test_calibration_determinism.py`.

**Interfaces produced:** `CalibrationSpec` (frozen dataclass, defaults C4/en/train/128/2048/seed 42; `.calib_id` → `"c4-128x2048-s42"`), `select_indices(doc_token_counts, spec) -> list[int]` (PURE: deterministic selection of the first `num_samples` doc indices with `token_count >= max_seq_len`, in a seeded-shuffled order — no datasets/torch import), `input_ids_sha256(rows) -> str` (PURE: sha256 over concatenated input_ids), `build_calibration(tokenizer, spec, out_dir) -> Dataset` (GPU/box: load C4, tokenize, select via `select_indices`, truncate to exactly `max_seq_len`, `save_to_disk`, write calib manifest; idempotent), `load_calibration(out_dir) -> Dataset`.

**Design for CPU-testability:** the determinism lives in `select_indices` + `input_ids_sha256`, which take plain Python inputs (a list of token counts; a list of int-lists) and import NOTHING heavy. `build_calibration` imports `datasets` lazily inside the function (box-only). The test exercises ONLY the pure functions on a fake corpus.

- [ ] **Step 1: failing test** — `test_calibration_determinism.py`: `CalibrationSpec().calib_id == "c4-128x2048-s42"`; `select_indices` on a fake list of token counts (mix of <2048 and ≥2048) returns exactly `num_samples` indices all pointing at ≥2048-token docs, and is identical across two calls (deterministic) and order-stable for a fixed seed; `input_ids_sha256([[1,2,3],[4,5]])` is stable and changes if a token changes. Use a small `CalibrationSpec(num_samples=4, max_seq_len=8, seed=42)` for the fake corpus.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** `calibration.py`. `select_indices`: build the list of candidate indices `[i for i,c in enumerate(doc_token_counts) if c >= spec.max_seq_len]`, deterministically order them with `random.Random(spec.seed).shuffle(candidates)`, return `candidates[:spec.num_samples]` (raise `ValueError` if fewer than `num_samples` candidates). `input_ids_sha256`: `hashlib.sha256()`; for each row update with `",".join(map(str,row)).encode()` + a `b"|"` separator; return hexdigest. `build_calibration` (lazy `import datasets`): stream `load_dataset("allenai/c4","en",split="train",streaming=True)`, pull a POOL (e.g. 4×num_samples or until enough), tokenize each `text` with the tokenizer, record token counts, `select_indices`, re-tokenize/truncate the selected to exactly `max_seq_len` (`input_ids` + `attention_mask`), build a `datasets.Dataset.from_dict`, `save_to_disk(out_dir/spec.calib_id)`, write `out_dir/<calib_id>.manifest.json` (dataset+config+split+resolved revision+num_samples+max_seq_len+seed+tokenizer name/revision+selected_indices+calib_sha256). Idempotent: if the saved dir + manifest exist, `load_calibration` and return.
- [ ] **Step 4:** run the pure-function test → PASS (no download). `build_calibration`/`load_calibration` verify on the box.
- [ ] **Step 5: commit** `Spec 07: deterministic C4 calibration (pure selection/sha + box build)`.

---

### Task 3: `quant/selfquant/recipes.py` — the fairness guarantee (box-verified)

**Files:** Create `quant/selfquant/recipes.py`. (No laptop test — needs `llmcompressor`; `assert_schemes_match` runs on the box and inside `quantize.main` as a pre-flight.)

**Interfaces produced:** `GROUP_SIZE=128`, `SCHEME="W4A16_ASYM"`, `IGNORE=["lm_head"]`, `gptq_recipe(group_size=GROUP_SIZE) -> list`, `awq_recipe(group_size=GROUP_SIZE) -> list`, `resolved_quant_args(recipe) -> dict`, `assert_schemes_match(a, b) -> None`.

- [ ] **Step 1: implement** (lazy imports inside functions): `gptq_recipe` → `[GPTQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]` (`from llmcompressor.modifiers.quantization import GPTQModifier` — fall back to `from llmcompressor.modifiers.gptq import GPTQModifier` if the first path errors). `awq_recipe` → `[AWQModifier(), QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]` (`from llmcompressor.modifiers.awq import AWQModifier`, `from llmcompressor.modifiers.quantization import QuantizationModifier`). `resolved_quant_args(recipe)`: find the modifier that carries the quant scheme (the `GPTQModifier` for gptq, the `QuantizationModifier` for awq), resolve its config to the concrete weight `QuantizationArgs` (`num_bits, symmetric, strategy, group_size`) and return them as a dict — read the modifier's `.resolve_quantization_config()` / `.scheme` attributes (verify exact attribute on the box; the goal is the 4 comparable values). `assert_schemes_match(a, b)`: `ra, rb = resolved_quant_args(a), resolved_quant_args(b)`; assert `ra == rb` AND `ra == {"num_bits":4,"symmetric":False,"strategy":"group","group_size":128}`; raise `AssertionError` with both dicts on mismatch.
- [ ] **Step 2: box verification (deferred):** on the 4090, `cd quant && uv run python -c "from selfquant.recipes import gptq_recipe, awq_recipe, assert_schemes_match; assert_schemes_match(gptq_recipe(), awq_recipe()); print('schemes match')"`. If `resolved_quant_args` can't read the args off the modifier as written, adjust to the real `llmcompressor==0.12.0` API there (this is the one place the exact attribute path must be confirmed against the installed lib).
- [ ] **Step 3: commit** `Spec 07: W4A16_ASYM recipes + assert_schemes_match fairness guarantee`.

---

### Task 4: `quant/selfquant/quantize.py` — oneshot orchestration + CLI (box)

**Files:** Create `quant/selfquant/quantize.py`.

**Interfaces produced:** `SELF_QUANT_IDS=("gptq-selfquant","awq-selfquant")`, `BASE_MODEL_ID`, `BASE_REVISION`, `recipe_for(variant_id) -> list`, `quantize_variant(variant_id, calib_ds, out_dir, *, base_model_id=…, base_revision=…, calib_sha256, calib_id, force=False) -> Path`, `main(argv=None) -> int`.

- [ ] **Step 1: implement.** `main`: argparse `--id {gptq-selfquant,awq-selfquant,all}` (default `all`), `--checkpoints-root checkpoints/self-quant`, `--base-revision`, `--calib-only`, `--force`. Pre-flight `assert_schemes_match(gptq_recipe(), awq_recipe())`. Build calibration ONCE: `tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=base_revision)`; `calib_ds = build_calibration(tok, CalibrationSpec(), Path(checkpoints_root)/"_calib")`; `calib_sha = input_ids_sha256(calib_ds["input_ids"])` (or read from the calib manifest); `calib_id = CalibrationSpec().calib_id`. If `--calib-only`, stop. For each requested id: `quantize_variant(id, calib_ds, Path(checkpoints_root)/id, base_revision=…, calib_sha256=calib_sha, calib_id=calib_id, force=force)`. Return 0; non-zero on failure.
  `quantize_variant`: if `selfquant_checkpoint_done(out_dir, algorithm=<gptq|awq>, base_revision=…, scheme="W4A16_ASYM", group_size=128, calib_sha256=…)` and not force → return out_dir (skip). Else: `model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, revision=base_revision, dtype="auto")`; `oneshot(model=model, dataset=calib_ds, recipe=recipe_for(id), max_seq_length=2048, num_calibration_samples=128, shuffle_calibration_samples=False)`; `model.save_pretrained(out_dir, save_compressed=True)`; `tok.save_pretrained(out_dir)`; `write_manifest(out_dir, QuantManifest(...))` (fill versions via `importlib.metadata.version`); free the model (`del model; gc.collect(); torch.cuda.empty_cache()`). All heavy imports (`torch`, `transformers`, `llmcompressor`) lazy inside the functions.
- [ ] **Step 2: box verification (deferred):** `cd quant && uv sync && uv run python -m selfquant.quantize --id all` produces both checkpoints with matching manifests (only `algorithm` differs); a second run without `--force` skips both; `--help` lists the flags.
- [ ] **Step 3: commit** `Spec 07: quantize.py oneshot orchestration + CLI (build calib once, idempotent)`.

---

### Task 5: kernel smoke (root eval env) + torch-free guard + fairness doc

**Files:** Create `src/qquant/selfquant_smoke.py`, `tests/test_selfquant_smoke_torch_free.py`, `docs/selfquant_fairness.md`.

**Interfaces produced:** `run_kernel_smoke(variant_id, ckpt_dir, *, prompt_tokens=512, gen_tokens=128, out_path=None) -> dict`, `main(argv=None) -> int`.

- [ ] **Step 1: failing test** — `tests/test_selfquant_smoke_torch_free.py`: fresh-interpreter subprocess imports `qquant`, `qquant.cli` and asserts `qquant.selfquant_smoke` is NOT in `sys.modules` and no `torch` (mirror `tests/test_eval_cli.py`'s torch-free subprocess pattern).
- [ ] **Step 2:** run → FAIL (module missing) or PASS-by-accident; ensure the module exists but is not imported by the umbrella.
- [ ] **Step 3: implement** `selfquant_smoke.py` (torch imported lazily inside functions). `run_kernel_smoke`: `from qquant.models import load_variant`; `loaded = load_variant(variant_id, checkpoints_root=Path(ckpt_dir).parent.parent)`; introspect a quantized Linear submodule's class name (walk `loaded.model.named_modules()`, pick the first whose class name contains `"Linear"` and lives under a quantized layer — record `module_class`), read `loaded.model.config.quantization_config` `format`; do a greedy decode of `gen_tokens` (do_sample=False, min_new==max_new) timing `decode_tok_s`; record `peak_vram_gb` via `torch.cuda.max_memory_allocated`, `dtype`. Determinism cross-check: generate twice, assert identical tokens. Write JSON to `out_path or ckpt_dir/kernel_smoke.json`; `loaded.unload()` in finally. `main`: argparse `--variant`, `--ckpt`, optional `--out`.
- [ ] **Step 4:** `uv run pytest tests/test_selfquant_smoke_torch_free.py -v` → PASS; full suite still green; ruff clean. Box: `python -m qquant.selfquant_smoke --variant gptq-selfquant --ckpt checkpoints/self-quant/gptq-selfquant` writes `kernel_smoke.json`.
- [ ] **Step 5: docs** — `docs/selfquant_fairness.md`: controlled axes (W4A16_ASYM, group_size 128, identical C4 calibration `c4-128x2048-s42`, identical base SHA), the only difference = algorithm, self-quant **speed is runtime-confounded** (references the calib manifest + both quant manifests + the kernel smoke).
- [ ] **Step 6: commit** `Spec 07: kernel smoke + torch-free guard + selfquant_fairness doc`.

---

## Box runbook (the deliverable you run today)

```bash
# on the rented RTX 4090, repo synced, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd quant && uv sync                                   # resolves quant/uv.lock (commit it)
uv run python -m selfquant.quantize --id all          # builds both checkpoints (~10-20 min each)
cd .. && uv sync --group gpu                           # eval env
python -m qquant.selfquant_smoke --variant gptq-selfquant --ckpt checkpoints/self-quant/gptq-selfquant
python -m qquant.selfquant_smoke --variant awq-selfquant  --ckpt checkpoints/self-quant/awq-selfquant
git add quant/uv.lock checkpoints/self-quant/*/quant_manifest.json checkpoints/self-quant/*/kernel_smoke.json  # provenance (NOT the weights)
```
Resume safety: re-running `--id all` skips done checkpoints (manifest match); `--force` recomputes.

## Self-Review

**Spec coverage (done-when → task):** DW1 (uv sync + --help) → box runbook + T4. DW2 (calib determinism, fake corpus) → T2. DW3 (compressed-tensors checkpoint, config quant args) → T4 box. DW4 (controlled axis, assert_schemes_match) → T3. DW5 (idempotency) → T1 + T4. DW6 (Spec-04 load + determinism) → T5 smoke. DW7 (kernel_smoke.json) → T5. DW8 (fairness doc) → T5. DW9 (torch-free guard, no root script) → T5 + T1. DW10 (variants.yaml unchanged) → respected (no edit). ✅
**Placeholder scan:** box-only modules show the exact API + the one attribute to confirm on the box (`resolved_quant_args`); no TBD. ✅
**Type consistency:** `QuantManifest` fields ↔ `write/read_manifest` ↔ `selfquant_checkpoint_done` args ↔ `quantize_variant`'s manifest write; `select_indices`/`input_ids_sha256` signatures ↔ `build_calibration` + the test; `calib_sha256`/`calib_id` threaded quantize→manifest. ✅
