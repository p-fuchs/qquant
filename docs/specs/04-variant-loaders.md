# Spec 04 — Unified variant loaders (`qquant.models`)

- **Status:** Planned (not yet implemented).
- **Depends on:** 01 (registry/paths/config contracts + torch-free core), 02 (vast.ai
  wrapper + GPU go/no-go spike that validates `lm-eval × transformers v5` and the CUDA-12.6
  *devel* image with `nvcc`).
- **Owns / Produces:** the `src/qquant/models/` package — the **only** place a Qwen2.5-7B
  checkpoint is turned into a ready-to-evaluate `(model, tokenizer)` pair. Public API:
  `load_variant`, `LoadedVariant`, `smoke_test_load`, `force_greedy`, `resolve_model_source`,
  and the `BUILDERS` dispatch table. Consumed by Spec 05 (quality eval), Spec 06 (efficiency
  profiler) and Spec 07 (self-quant). This module **imports torch/transformers** and is
  therefore *not* part of the torch-free core (§ contracts 7); the umbrella `qquant` CLI must
  never import it.

## Purpose

Provide one registry-driven function

```python
load_variant(variant_id) -> LoadedVariant(model, tokenizer, metadata)
```

that loads any of the seven canonical variants (`VARIANT_IDS`) correctly and identically for
every downstream consumer, so that quality (Spec 05) and efficiency (Spec 06) measure the
*same* object. The loader centralises every quantization-specific `from_pretrained` detail,
the greedy-determinism enforcement, the single-GPU device placement, and the
`awq-official` contingency smoke test, so no other spec re-implements (or drifts on) loading.

## Scope (in / out)

**In scope**

- Dispatch on `Variant.quant_method` ∈ `QUANT_METHODS`
  (`bf16`, `bnb-int8`, `bnb-nf4`, `gptq`, `awq`, `compressed-tensors`) to the matching builder.
- bf16 baseline, bitsandbytes int8 / NF4 (load-time), official GPTQ and AWQ checkpoints
  (gptqmodel + optimum, **no autoawq**), and self-quant W4A16 checkpoints loaded natively via
  `compressed-tensors`.
- Pinning the verified revision SHAs from `variants.yaml`; resolving `local:` model ids to a
  checkpoints directory.
- Forcing greedy decoding at load time (`do_sample=False`; clear `temperature/top_p/top_k`).
- Single-GPU placement that inherits the quantizer's `device_map` and **never** calls `.to()`.
- The `awq-official` contingency: a `smoke_test_load` that loads + runs a 1-token greedy
  generate on sm_89 and reports `(ok, message)` so the operator/orchestrator can flip
  `enabled=false` in `variants.yaml` on failure.
- A `metadata` dict that carries the **resolved** model revision (for Spec 05 to thread into
  `RunConfig.model_revision`) and the device map (for the no-offload assertion).
- VRAM release (`LoadedVariant.unload()` / context-manager) so variants can be loaded
  sequentially within one process.

**Out of scope (owned elsewhere)**

- Producing self-quant checkpoints (calibration, `llm-compressor`, the isolated `quant/`
  env) — **Spec 07**. This spec only *loads* what Spec 07 writes to
  `checkpoints/self-quant/<id>`.
- Wrapping the model in lm-eval's `HFLM`, building `gen_kwargs`, writing result cells, and
  setting `RunConfig` — **Spec 05**. This spec exposes the resolved revision; Spec 05 owns
  `cell_provenance`.
- Latency/throughput/VRAM measurement and `PYTORCH_CUDA_ALLOC_CONF` env wiring — **Spec 06 /
  08** (the loader assumes the orchestrator already set the env).
- Defining ids, paths, the cell schema, or the resume predicate — those are SSOT contracts
  imported from Spec 01 (`VARIANT_IDS`, `Variant`, `load_variants`, `cell_provenance`, …).

## Interface

All code lives under `src/qquant/models/`. Type hints for torch/transformers objects are
written as strings/`Any` so the file parses without those libraries installed. **The heavy
imports (`torch`, `transformers`, `bitsandbytes`, …) are *deferred* — performed inside the
builder function bodies, never at module top.** Importing `qquant.models` therefore pulls in
only stdlib + the torch-free `qquant.registry`, so the dispatch table (`BUILDERS` keys),
`resolve_model_source`, and `force_greedy` are unit-testable on the macOS dev laptop (where
the `gpu` group is not installed); a builder only touches torch when it actually runs, on the
GPU box. This is *not* the torch-free core (§ contracts 7) — the umbrella `qquant` CLI must
still never import `qquant.models` — but deferring the imports keeps the CPU-only test plan
(`test_models_dispatch.py` / `test_models_greedy.py`) runnable off-GPU.

```python
# src/qquant/models/loaders.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qquant.registry import Variant, load_variants, QUANT_METHODS


@dataclass(frozen=True)
class ModelSource:
    """Where a variant's weights come from, after resolving `local:` ids."""
    path_or_repo: str        # HF repo id, or an absolute/relative local path
    revision: str | None     # pinned SHA for official/baseline; None for local self-quant
    is_local: bool


@dataclass
class LoadedVariant:
    """A loaded, greedy-forced, GPU-resident model ready for HFLM / profiling."""
    variant: Variant
    model: Any               # transformers PreTrainedModel (already device-placed)
    tokenizer: Any           # PreTrainedTokenizerBase
    metadata: dict[str, Any]

    def unload(self) -> None: ...          # del model, gc.collect(), torch.cuda.empty_cache()
    def __enter__(self) -> "LoadedVariant": ...
    def __exit__(self, *exc: object) -> None: ...   # calls unload()


def resolve_model_source(variant: Variant, checkpoints_root: str | Path = "checkpoints") -> ModelSource:
    """`local:checkpoints/self-quant/<id>` -> ModelSource(local path under checkpoints_root,
    revision=None, is_local=True); otherwise (repo_id, variant.revision, is_local=False)."""


def force_greedy(model: Any) -> None:
    """Mutate `model.generation_config` in place: do_sample=False, num_beams=1,
    and temperature/top_p/top_k set to None. Also clears them on `model.config`
    if present. Idempotent. (Spec 05 mirrors this in gen_kwargs.)"""


# A builder takes the resolved Variant/source and returns (model, tokenizer, builder_meta).
Builder = Callable[[Variant, ModelSource, dict[str, Any]], tuple[Any, Any, dict[str, Any]]]

BUILDERS: dict[str, Builder]   # keys == QUANT_METHODS, exactly


def load_variant(
    variant_id: str,
    *,
    variants: dict[str, Variant] | None = None,     # default: load_variants()
    checkpoints_root: str | Path = "checkpoints",
    device_map: Any = None,                          # default {"": 0} (whole model on GPU 0)
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
) -> LoadedVariant:
    """Resolve the variant, dispatch on quant_method to BUILDERS[...], force greedy,
    assert no CPU/disk offload, and return LoadedVariant. Raises KeyError for an unknown
    id and ValueError if the variant is disabled."""


def smoke_test_load(
    variant_id: str,
    *,
    variants: dict[str, Variant] | None = None,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    """Load the variant, run a 1-token greedy generate, unload, and return (ok, message).
    Used for the awq-official contingency: on (False, msg) the caller sets enabled=false."""
```

```python
# src/qquant/models/__init__.py  (public surface)
from qquant.models.loaders import (
    BUILDERS, LoadedVariant, ModelSource,
    force_greedy, load_variant, resolve_model_source, smoke_test_load,
)
```

### Builder behaviour (per `quant_method`)

All builders call `AutoModelForCausalLM.from_pretrained(..., revision=source.revision,
device_map=<resolved>, attn_implementation="sdpa")` and `AutoTokenizer.from_pretrained(
source.path_or_repo, revision=source.revision)`. Only the quantization knobs differ:

| `quant_method` | Variant id(s) | `from_pretrained` knobs |
|---|---|---|
| `bf16` | `bf16` | `dtype=torch.bfloat16`, no `quantization_config` |
| `bnb-int8` | `bnb-int8` | `quantization_config=BitsAndBytesConfig(load_in_8bit=True)`, `dtype=torch.bfloat16`; do **not** enable `llm_int8_enable_fp32_cpu_offload` |
| `bnb-nf4` | `bnb-nf4` | `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`, `dtype=torch.bfloat16` |
| `gptq` | `gptq-official` | `dtype="auto"` (checkpoint is fp16); GPTQ `quantization_config` auto-detected from the repo `config.json` → gptqmodel kernel via optimum. No autogptq. |
| `awq` | `awq-official` | **CONTINGENT.** `dtype="auto"`; AWQ `quantization_config` auto-detected → kernel served by **gptqmodel** (NOT autoawq). If load/forward raises on sm_89, the variant is disabled. |
| `compressed-tensors` | `gptq-selfquant`, `awq-selfquant` | local path from `resolve_model_source`; `dtype="auto"`; W4A16 `quantization_config` auto-detected → loaded natively by `compressed-tensors`. `revision=None`. |

### `metadata` dict (returned in `LoadedVariant.metadata`)

```python
{
  "variant_id":        str,            # == variant.id
  "quant_method":      str,
  "model_source":      str,            # repo id or resolved local path
  "model_revision":    str | None,     # resolved SHA (feeds Spec 05 RunConfig.model_revision)
  "is_local":          bool,
  "dtype":             str,            # requested dtype, e.g. "bfloat16" / "auto"
  "param_dtype":       str,            # observed dtype of a sample weight tensor
  "attn_implementation": "sdpa",
  "device_map":        dict,           # model.hf_device_map (asserted offload-free)
  "load_seconds":      float,
  "transformers_version": str,
  "checkpoint_fingerprint": str | None # self-quant only: content hash for provenance
}
```

## Files to create

- `src/qquant/models/__init__.py` — public re-exports listed above.
- `src/qquant/models/loaders.py` — `ModelSource`, `LoadedVariant`, `resolve_model_source`,
  `BUILDERS` + the six builder functions, `load_variant`, `smoke_test_load`. Heavy imports
  (`torch`, `transformers`, `bitsandbytes`, …) are **deferred into the builder bodies**, so
  importing the module is torch-free and the dispatch/source-resolution tests run on the
  laptop; the imports execute only when a builder runs on the GPU box.
- `src/qquant/models/greedy.py` — `force_greedy` (kept separate so it is unit-testable with a
  stub `generation_config`, no torch needed). Re-exported by `loaders`/`__init__`.
- `tests/test_models_dispatch.py` — CPU-only: `set(BUILDERS) == QUANT_METHODS`; every variant
  in `load_variants()` maps `quant_method -> BUILDERS[...]`; `resolve_model_source` strips
  `local:` and joins `checkpoints_root`, and returns the pinned SHA for official variants.
- `tests/test_models_greedy.py` — CPU-only: `force_greedy` on a stub sets `do_sample=False`,
  `num_beams=1`, and `temperature/top_p/top_k is None`; idempotent on a second call.
- `tests/test_models_gpu.py` — `@pytest.mark.gpu` (skipped without CUDA): per enabled variant,
  `load_variant` succeeds, `metadata["device_map"]` has no `"cpu"`/`"disk"` values, a 1-token
  greedy generate runs, and a re-run yields identical token ids (determinism). Run on the box.
- Add `[project.scripts]` only if a thin `qquant-load` smoke CLI is wanted; **not required** —
  `smoke_test_load` is called from Spec 02's spike / Spec 08's pre-flight. (Do not add a
  subcommand to the torch-free `qquant` umbrella.)

## Implementation notes (decision-log points that bind THIS spec, with the "why")

- **transformers v5 loading API.** `quantization_config=` is the canonical path; the
  `load_in_8bit/4bit` shortcut kwargs were removed; `torch_dtype` is renamed to **`dtype`**
  and the default is `"auto"`. Builders therefore pass `BitsAndBytesConfig(...)` via
  `quantization_config=` and use `dtype=` (never `torch_dtype=`). *Why:* the eval env is pinned
  to `transformers==5.10.1`.
- **NF4 compute dtype.** NF4 must set `bnb_4bit_compute_dtype=torch.bfloat16`; the default
  (fp32) gives an artificially unfair speed reading. *Why:* RQ#3 speed comparison must not be
  confounded by an accidental fp32 matmul path.
- **bf16 baseline dtype = bfloat16.** ~15 GB weights fit 24 GB on the single RTX 4090, so the
  baseline runs on the same GPU as every quantized variant — the speed comparison stays
  apples-to-apples.
- **Official checkpoints are fp16.** Load `gptq-official` / `awq-official` with `dtype="auto"`
  (their native fp16) — do not upcast to bfloat16. This minor compute-dtype confound vs the
  bf16 baseline is documented in the report, not "fixed" by changing dtype.
- **Official GPTQ via gptqmodel + optimum; no autogptq.** gptqmodel 7.1 JIT-compiles
  Marlin/AWQ kernels for sm_89 on first use, which is exactly why the image is a **CUDA 12.6
  *devel*** image with `nvcc` (Spec 02/03 bootstrap fails fast if `nvcc` major.minor ≠
  `torch.version.cuda`). The loader assumes that environment; it does not install kernels.
- **`awq-official` is CONTINGENT.** Loaded via gptqmodel's AWQ kernels (NEVER `autoawq` —
  archived, pins `transformers~=4.5x`, breaks the stack). `smoke_test_load("awq-official")` is
  the gate; on failure the operator/orchestrator sets `enabled=false` and the **guaranteed**
  AWQ data point falls back to `awq-selfquant`. This spec only *reports* the result; flipping
  the YAML flag is an operator/Spec-08 action.
- **Self-quant loads natively via compressed-tensors.** `gptq-selfquant` / `awq-selfquant`
  have `quant_method == "compressed-tensors"` and are loaded straight by transformers v5 +
  `compressed-tensors==0.17.1` from `checkpoints/self-quant/<id>` (`local:` model id) — NOT via
  gptqmodel's legacy GPTQ/AWQ path. Their W4A16 may take a slow dequant path on the HF
  `generate()` backend (fast Marlin targets vLLM); the loader does not try to "fix" this —
  Spec 06 labels self-quant speed **runtime-confounded** while self-quant carries the
  calibration-controlled **quality** comparison.
- **Greedy determinism at load.** Qwen ships `generation_config` with `do_sample=true`;
  `force_greedy` sets `do_sample=False` and clears `temperature/top_p/top_k` so a stray
  sampling field cannot leak. Spec 05 mirrors this in `gen_kwargs`, and a determinism smoke
  (re-run → identical output, covered by `test_models_gpu.py`) guards it.
- **HFLM interop — never `.to()` a quantized model.** bnb/gptq/awq/compressed-tensors models
  are placed by the quantizer's `device_map`; moving them post-load corrupts them. The loader
  uses `device_map={"": 0}` (whole model on GPU 0) for *all* variants, asserts
  `set(model.hf_device_map.values())` contains no `"cpu"`/`"disk"` (silent offload would wreck
  the Spec 06 timings and risks OOM-free-but-slow runs), and returns the already-placed model.
  Spec 05 must construct `HFLM(pretrained=<model>, tokenizer=<tok>)` **without** passing a
  `device=`/`.to()` that would relocate it.
- **Pinned revisions.** Builders pass `revision=variant.revision` (the verified SHAs in
  `variants.yaml` / decision log): `a09a3545…` for `Qwen/Qwen2.5-7B-Instruct` (bf16/bnb),
  `e9c932ac…` for the official GPTQ-Int4 repo, `b2503754…` for the official AWQ repo. Self-quant
  `revision` is `None` (local checkpoints); their provenance is carried by
  `metadata["checkpoint_fingerprint"]`.
- **Sequential loading / VRAM hygiene.** Spec 05 and 06 iterate variants one at a time in a
  single process; `LoadedVariant.unload()` (`del model` → `gc.collect()` →
  `torch.cuda.empty_cache()`) and the context-manager form let them reclaim VRAM between
  variants. The loader assumes the orchestrator already exported
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Spec 06/08).

## Done-when (numbered, testable)

1. `set(BUILDERS.keys()) == QUANT_METHODS` and `load_variant` dispatches each
   `Variant.quant_method` to its builder; an unknown id raises `KeyError`, a disabled variant
   raises `ValueError`. (`tests/test_models_dispatch.py`)
2. `resolve_model_source` maps `local:checkpoints/self-quant/gptq-selfquant` →
   `ModelSource(<checkpoints_root>/self-quant/gptq-selfquant, None, True)`, and the official/
   baseline variants → `(repo_id, <pinned SHA>, False)` matching `variants.yaml`.
   (`tests/test_models_dispatch.py`)
3. `force_greedy(stub)` sets `do_sample=False`, `num_beams=1`, and
   `temperature/top_p/top_k is None`; calling it twice is a no-op the second time.
   (`tests/test_models_greedy.py`)
4. Importing `qquant.models` does **not** regress the torch-free guard: the existing
   `tests/test_import_torch_free.py` (which imports only core modules) still passes, and the
   `qquant` umbrella CLI does not import `qquant.models`.
5. **GPU, on the box (`@pytest.mark.gpu`):** for every *enabled* variant, `load_variant`
   returns a `LoadedVariant` whose `metadata["device_map"]` contains no `"cpu"`/`"disk"`, and a
   1-token greedy `generate` succeeds. (`tests/test_models_gpu.py`)
6. **GPU determinism:** re-running the same greedy generate for a loaded variant yields
   identical output token ids. (`tests/test_models_gpu.py`)
7. `smoke_test_load("awq-official")` returns `(True, …)` if the gptqmodel AWQ path works on
   sm_89, else `(False, <reason>)` — without raising — so the caller can set `enabled=false`.
8. `LoadedVariant.unload()` (and `with load_variant(...) as lv:`) frees the model and empties
   the CUDA cache, allowing a second `load_variant` of a different variant in the same process
   without OOM. (`tests/test_models_gpu.py`)
9. `metadata["model_revision"]` equals the resolved revision (pinned SHA for official/baseline;
   `None` for self-quant), so Spec 05 can feed it into `RunConfig.model_revision` and the cell
   `config` block round-trips through `is_cell_done`.

## Risks & mitigations

- **AWQ on sm_89 without autoawq fails to load.** *Mitigation:* `awq-official` is contingent by
  design — `smoke_test_load` gates it, `enabled=false` removes its 60 cells, and `awq-selfquant`
  is the guaranteed AWQ data point. Never install autoawq.
- **gptqmodel JIT kernel build fails (no `nvcc` / CUDA mismatch).** *Mitigation:* the CUDA-12.6
  *devel* image ships `nvcc`; Spec 02/03 bootstrap fails fast on
  `nvcc major.minor ≠ torch.version.cuda`. The loader surfaces the build error verbatim via
  `smoke_test_load`'s message.
- **Silent CPU/disk offload** (device_map="auto" with a tight budget) would make a variant
  "work" but run pathologically slowly, corrupting Spec 06 timings. *Mitigation:* force
  `device_map={"": 0}` and assert the resolved `hf_device_map` has no `"cpu"`/`"disk"` entries.
- **Accidental `.to(...)` on a quantized model** (e.g. by HFLM). *Mitigation:* loader returns an
  already-placed model and documents the HFLM construction contract for Spec 05; the GPU test
  asserts placement is unchanged after load.
- **NF4 compute-dtype regression** (defaulting to fp32). *Mitigation:* explicit
  `bnb_4bit_compute_dtype=torch.bfloat16` plus a `metadata["param_dtype"]` record and the
  speed sanity check in Spec 06.
- **Self-quant checkpoint missing/corrupt on a fresh box.** *Mitigation:* `resolve_model_source`
  raises a clear `FileNotFoundError` when the local path is absent; Spec 07/08 own
  exfil/re-upload of `checkpoints/self-quant/<id>` so a resume never re-quantizes.
- **transformers v5 dtype kwarg drift** (`torch_dtype` vs `dtype`). *Mitigation:* builders use
  `dtype=` exclusively, matching the pinned `transformers==5.10.1`; the dispatch test pins the
  kwarg name via a fake `from_pretrained` capture if helpful.
