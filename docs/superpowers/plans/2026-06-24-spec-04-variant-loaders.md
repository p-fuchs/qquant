# Spec 04 — Unified Variant Loaders (`qquant.models`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/qquant/models/` — the single registry-driven `load_variant()` that turns any of the 7 canonical Qwen2.5-7B variants into an identical, greedy-forced, single-GPU `(model, tokenizer)` pair for Specs 05/06/07.

**Architecture:** A thin torch-free orchestration layer (`load_variant`, dispatch table, `resolve_model_source`, `force_greedy`, metadata assembly) over six per-`quant_method` builder functions. **All torch/transformers imports are deferred into the builder bodies**, so importing the package and exercising dispatch/source-resolution/greedy-forcing needs no torch — they are TDD'd on the macOS laptop. The six builders do the real `from_pretrained` loading and are verified on a rented RTX 4090 via a `@pytest.mark.gpu` suite that auto-skips off-GPU.

**Tech Stack:** Python 3.11, transformers 5.10.1, torch 2.12.1+cu126, bitsandbytes 0.49.2, gptqmodel 7.1.0 + optimum 2.2.0, compressed-tensors 0.17.1 (all Linux-only `gpu` group); pytest; ruff.

## Global Constraints

Every task's requirements implicitly include this section. Values are verbatim from `docs/specs/contracts.md`, `docs/specs/decision-log.md`, and `docs/specs/04-variant-loaders.md`.

- **Deferred imports.** `torch`/`transformers`/`bitsandbytes`/etc. are imported **inside builder bodies only** — never at module top of `qquant.models.*`. Importing `qquant.models` must stay torch-free.
- **Torch-free guard unbroken.** `tests/test_import_torch_free.py` must stay green; the umbrella `qquant` CLI (`qquant.cli`) must never import `qquant.models`.
- **transformers v5 loading API.** Use `dtype=` (NEVER `torch_dtype=`); `quantization_config=` is the canonical path; the `load_in_8bit`/`load_in_4bit` shortcut kwargs were removed. Default dtype is `"auto"`.
- **dtype per method.** `bf16`/`bnb-int8`/`bnb-nf4` → `dtype=torch.bfloat16`. Official `gptq`/`awq` checkpoints are fp16 → `dtype="auto"` (do NOT upcast). Self-quant `compressed-tensors` → `dtype="auto"`.
- **NF4 compute dtype.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`. The fp32 default would give an unfair speed reading (RQ3).
- **int8.** `BitsAndBytesConfig(load_in_8bit=True)`; do **not** enable `llm_int8_enable_fp32_cpu_offload`.
- **Never `.to()` a quantized model.** Use `device_map={"": 0}` (whole model on GPU 0) for *all* variants; return the already-placed model. Assert `model.hf_device_map` contains no `"cpu"`/`"disk"` values (silent offload corrupts Spec 06 timings).
- **NEVER install or import `autoawq`** (archived; pins `transformers~=4.5x` → breaks the stack). Official AWQ is served by **gptqmodel** kernels via optimum, never autoawq/autogptq.
- **Pinned revisions (verbatim SHAs).** `bf16`/`bnb-int8`/`bnb-nf4` → `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`; `gptq-official` → `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` @ `e9c932ac1893a49ae0fc497ad6e1e86e2e39af20`; `awq-official` → `Qwen/Qwen2.5-7B-Instruct-AWQ` @ `b25037543e9394b818fdfca67ab2a00ecc7dd641`. Self-quant `revision=None`.
- **Greedy at load.** `do_sample=False`, `num_beams=1`, clear `temperature/top_p/top_k`. Qwen ships `do_sample=true`.
- **Canonical ids only.** Hyphenated, no aliases, no `-int4` suffixes; `set(BUILDERS) == QUANT_METHODS`.
- **`awq-official` is CONTINGENT.** `smoke_test_load("awq-official")` returns `(ok, message)` **without raising**; flipping `enabled=false` is an operator/Spec-08 action — this spec only reports.
- **Self-quant GPU verification is deferred to post-Spec-07** (checkpoints don't exist yet). `resolve_model_source`'s `FileNotFoundError` path is built + unit-tested now. GPU suite verifies the 5 official-core variants only.
- **No `qquant-load` CLI** (YAGNI). No new `[project.scripts]` line; no subcommand on the torch-free umbrella.
- **Tooling.** ruff `line-length = 88`, rules `E,F,I,UP,B`; `ruff format`. Python `>=3.11,<3.12`. The `gpu` dependency group is `sys_platform == 'linux'` only; CPU tests run on macOS without it.

---

### Task 1: `force_greedy` (greedy.py) + package marker

**Files:**
- Create: `src/qquant/models/__init__.py` (empty package marker for now; finalized in Task 5)
- Create: `src/qquant/models/greedy.py`
- Test: `tests/test_models_greedy.py`

**Interfaces:**
- Consumes: nothing (pure Python; operates on duck-typed `generation_config`/`config` objects).
- Produces: `force_greedy(model: Any) -> None` — mutates `model.generation_config` in place (`do_sample=False`, `num_beams=1`, `temperature=top_p=top_k=None`) and clears `temperature/top_p/top_k` on `model.config` if present. Idempotent. Imported by `loaders.load_variant` (Task 4) and re-exported by `__init__` (Task 5).

- [ ] **Step 1: Create the empty package marker**

Create `src/qquant/models/__init__.py` with a single line so the directory is an importable package (the full public surface is added in Task 5):

```python
"""qquant.models — registry-driven variant loaders (see Spec 04). Filled in Task 5."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_models_greedy.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from qquant.models.greedy import force_greedy


def _stub_model():
    gen = SimpleNamespace(do_sample=True, num_beams=4, temperature=0.7, top_p=0.8, top_k=20)
    cfg = SimpleNamespace(temperature=0.7, top_p=0.8, top_k=20)
    return SimpleNamespace(generation_config=gen, config=cfg)


def test_force_greedy_sets_greedy_fields():
    m = _stub_model()
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
    assert m.generation_config.temperature is None
    assert m.generation_config.top_p is None
    assert m.generation_config.top_k is None
    assert m.config.temperature is None
    assert m.config.top_p is None
    assert m.config.top_k is None


def test_force_greedy_idempotent():
    m = _stub_model()
    force_greedy(m)
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
    assert m.generation_config.temperature is None


def test_force_greedy_tolerates_missing_config():
    gen = SimpleNamespace(do_sample=True, num_beams=2, temperature=1.0, top_p=1.0, top_k=50)
    m = SimpleNamespace(generation_config=gen, config=None)
    force_greedy(m)
    assert m.generation_config.do_sample is False
    assert m.generation_config.num_beams == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_models_greedy.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'qquant.models.greedy'`.

- [ ] **Step 4: Write minimal implementation**

Create `src/qquant/models/greedy.py`:

```python
"""Greedy-decoding enforcement, kept torch-free so it is unit-testable with a stub.

Qwen ships ``generation_config`` with ``do_sample=true``; ``force_greedy`` clamps the
model to deterministic greedy decoding at load time. Spec 05 mirrors this in ``gen_kwargs``.
"""

from __future__ import annotations

from typing import Any


def force_greedy(model: Any) -> None:
    """Force greedy decoding in place. Idempotent.

    Sets ``do_sample=False``, ``num_beams=1`` and clears ``temperature/top_p/top_k`` on
    ``model.generation_config``; also clears those sampling fields on ``model.config`` when
    present, so a stray sampling default cannot leak into generation.
    """
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        gen.do_sample = False
        gen.num_beams = 1
        gen.temperature = None
        gen.top_p = None
        gen.top_k = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, None)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_models_greedy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/qquant/models/__init__.py src/qquant/models/greedy.py tests/test_models_greedy.py
git commit -m "Spec 04: force_greedy + qquant.models package skeleton"
```

---

### Task 2: `ModelSource` + `resolve_model_source` (loaders.py scaffolding)

**Files:**
- Create: `src/qquant/models/loaders.py`
- Test: `tests/test_models_dispatch.py`

**Interfaces:**
- Consumes: `qquant.registry.Variant`, `load_variants` (Task 1's `force_greedy` not used yet).
- Produces:
  - `@dataclass(frozen=True) ModelSource(path_or_repo: str, revision: str | None, is_local: bool)`
  - `resolve_model_source(variant: Variant, checkpoints_root: str | Path = "checkpoints") -> ModelSource` — strips `local:`, relocates under `checkpoints_root`, raises `FileNotFoundError` for an absent local checkpoint; otherwise returns `(model_id, variant.revision, False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_dispatch.py`:

```python
from __future__ import annotations

import pytest

from qquant.models.loaders import ModelSource, resolve_model_source
from qquant.registry import load_variants


def test_resolve_official_source():
    variants = load_variants()
    src = resolve_model_source(variants["gptq-official"])
    assert isinstance(src, ModelSource)
    assert src.path_or_repo == "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    assert src.revision == "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"
    assert src.is_local is False


def test_resolve_baseline_source():
    variants = load_variants()
    src = resolve_model_source(variants["bf16"])
    assert src.path_or_repo == "Qwen/Qwen2.5-7B-Instruct"
    assert src.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert src.is_local is False


def test_resolve_selfquant_source_existing(tmp_path):
    variants = load_variants()
    ckpt = tmp_path / "self-quant" / "gptq-selfquant"
    ckpt.mkdir(parents=True)
    src = resolve_model_source(variants["gptq-selfquant"], checkpoints_root=tmp_path)
    assert src.path_or_repo == str(ckpt)
    assert src.revision is None
    assert src.is_local is True


def test_resolve_selfquant_missing_raises(tmp_path):
    variants = load_variants()
    with pytest.raises(FileNotFoundError):
        resolve_model_source(variants["awq-selfquant"], checkpoints_root=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_dispatch.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'qquant.models.loaders'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/models/loaders.py`:

```python
"""Unified variant loaders for the 7 canonical Qwen2.5-7B variants (Spec 04).

DEFERRED IMPORTS: torch/transformers/bitsandbytes are imported INSIDE the builder bodies
only — never at module top — so importing this module stays torch-free and the dispatch
table / source resolution / greedy-forcing are unit-testable on the dev laptop. The actual
loads run on the RTX 4090. This module is NOT the torch-free core; the umbrella ``qquant``
CLI must never import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qquant.registry import Variant

_LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class ModelSource:
    """Where a variant's weights come from, after resolving ``local:`` ids."""

    path_or_repo: str  # HF repo id, or an absolute/relative local path
    revision: str | None  # pinned SHA for official/baseline; None for local self-quant
    is_local: bool


def resolve_model_source(
    variant: Variant, checkpoints_root: str | Path = "checkpoints"
) -> ModelSource:
    """Resolve a variant's weight source.

    ``local:checkpoints/self-quant/<id>`` -> ``ModelSource(<checkpoints_root>/self-quant/<id>,
    None, True)`` (raising ``FileNotFoundError`` if that path is absent — Spec 07 produces it).
    Anything else -> ``ModelSource(model_id, variant.revision, False)``.
    """
    model_id = variant.model_id
    if model_id.startswith(_LOCAL_PREFIX):
        rel = model_id[len(_LOCAL_PREFIX) :].removeprefix("checkpoints/")
        path = Path(checkpoints_root) / rel
        if not path.exists():
            raise FileNotFoundError(
                f"self-quant checkpoint for {variant.id!r} not found at {path} "
                f"(produced by Spec 07; exfil/re-upload owned by Spec 07/08)"
            )
        return ModelSource(str(path), None, True)
    return ModelSource(model_id, variant.revision, False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_dispatch.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/models/loaders.py tests/test_models_dispatch.py
git commit -m "Spec 04: ModelSource + resolve_model_source (local: resolution)"
```

---

### Task 3: `BUILDERS` dispatch table + six builder functions (deferred imports)

**Files:**
- Modify: `src/qquant/models/loaders.py` (append the builders + `BUILDERS` + `Builder` alias)
- Test: `tests/test_models_dispatch.py` (append dispatch-coverage tests)

**Interfaces:**
- Consumes: `ModelSource` (Task 2), `qquant.registry.QUANT_METHODS`, `Variant`.
- Produces:
  - `Builder = Callable[[Variant, ModelSource, dict[str, Any]], tuple[Any, Any, dict[str, Any]]]`
  - Six builders `_build_bf16`, `_build_bnb_int8`, `_build_bnb_nf4`, `_build_gptq`, `_build_awq`, `_build_compressed_tensors`, each returning `(model, tokenizer, builder_meta)` where `builder_meta` has keys `dtype: str`, `transformers_version: str`, and (compressed-tensors only) `checkpoint_fingerprint: str`.
  - `BUILDERS: dict[str, Builder]` keyed **exactly** by `QUANT_METHODS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models_dispatch.py`:

```python
def test_builders_cover_quant_methods_exactly():
    from qquant.models.loaders import BUILDERS
    from qquant.registry import QUANT_METHODS

    assert set(BUILDERS.keys()) == set(QUANT_METHODS)


def test_every_registered_variant_maps_to_a_builder():
    from qquant.models.loaders import BUILDERS

    variants = load_variants()
    for v in variants.values():
        assert v.quant_method in BUILDERS, f"{v.id}: no builder for {v.quant_method!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_dispatch.py::test_builders_cover_quant_methods_exactly -v`
Expected: FAIL — `ImportError: cannot import name 'BUILDERS' from 'qquant.models.loaders'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top-of-file imports in `src/qquant/models/loaders.py` (keep `from __future__ import annotations` first):

```python
from typing import Any, Callable
```

Append to `src/qquant/models/loaders.py` (after `resolve_model_source`):

```python
# A builder takes the resolved Variant/source/opts and returns (model, tokenizer, meta).
# torch/transformers are imported INSIDE each body so this module stays import-time torch-free.
Builder = Callable[[Variant, "ModelSource", dict[str, Any]], tuple[Any, Any, dict[str, Any]]]


def _build_bf16(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype=torch.bfloat16,
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {"dtype": "bfloat16", "transformers_version": transformers.__version__}


def _build_bnb_int8(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # LLM.int8(); do NOT enable llm_int8_enable_fp32_cpu_offload (forces a CPU offload path).
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {"dtype": "bfloat16", "transformers_version": transformers.__version__}


def _build_bnb_nf4(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # NF4 MUST set bnb_4bit_compute_dtype=bfloat16 (fp32 default => unfair speed reading).
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {"dtype": "bfloat16", "transformers_version": transformers.__version__}


def _build_gptq(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Official GPTQ-Int4 (fp16): quantization_config auto-detected from the repo config.json,
    # served by gptqmodel kernels via optimum. dtype="auto" keeps the native fp16 (no upcast).
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype="auto",
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {"dtype": "auto", "transformers_version": transformers.__version__}


def _build_awq(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # CONTINGENT: official AWQ served by gptqmodel kernels (NEVER autoawq). dtype="auto" (fp16).
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype="auto",
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {"dtype": "auto", "transformers_version": transformers.__version__}


def _build_compressed_tensors(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Self-quant W4A16 checkpoint, loaded natively by transformers v5 + compressed-tensors
    # from a local path (revision is None). NOT the gptqmodel legacy GPTQ/AWQ path.
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype="auto",
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return model, tok, {
        "dtype": "auto",
        "transformers_version": transformers.__version__,
        "checkpoint_fingerprint": _fingerprint_checkpoint(source.path_or_repo),
    }


def _fingerprint_checkpoint(path: str | Path) -> str:
    """Cheap, deterministic content fingerprint of a local checkpoint dir for provenance.

    Hashes (name, size) of every config/weights file — no weight bytes read. Torch-free.
    """
    import hashlib
    import os

    digest = hashlib.sha256()
    root = str(path)
    for name in sorted(os.listdir(root)):
        if name.endswith((".json", ".safetensors", ".bin")):
            size = os.stat(os.path.join(root, name)).st_size
            digest.update(name.encode())
            digest.update(str(size).encode())
    return digest.hexdigest()[:16]


BUILDERS: dict[str, Builder] = {
    "bf16": _build_bf16,
    "bnb-int8": _build_bnb_int8,
    "bnb-nf4": _build_bnb_nf4,
    "gptq": _build_gptq,
    "awq": _build_awq,
    "compressed-tensors": _build_compressed_tensors,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_dispatch.py -v`
Expected: PASS (6 passed — the 4 from Task 2 plus the 2 new dispatch tests).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/models/loaders.py tests/test_models_dispatch.py
git commit -m "Spec 04: BUILDERS dispatch table + six deferred-import builders"
```

---

### Task 4: `LoadedVariant` + `load_variant` orchestration + `smoke_test_load`

**Files:**
- Modify: `src/qquant/models/loaders.py` (append `LoadedVariant`, `load_variant`, `smoke_test_load`, `_param_dtype`)
- Test: `tests/test_models_dispatch.py` (append orchestration tests using injected fake builders)

**Interfaces:**
- Consumes: `BUILDERS` (Task 3), `resolve_model_source`/`ModelSource` (Task 2), `force_greedy` (Task 1), `qquant.registry.Variant`/`load_variants`.
- Produces:
  - `@dataclass LoadedVariant(variant: Variant, model: Any, tokenizer: Any, metadata: dict[str, Any])` with `.unload() -> None`, `__enter__`/`__exit__`.
  - `load_variant(variant_id, *, variants=None, checkpoints_root="checkpoints", device_map=None, attn_implementation="sdpa", trust_remote_code=False) -> LoadedVariant` — raises `KeyError` (unknown id), `ValueError` (disabled), `RuntimeError` (cpu/disk offload).
  - `smoke_test_load(variant_id, *, variants=None, checkpoints_root="checkpoints") -> tuple[bool, str]` — never raises.
  - `metadata` dict keys: `variant_id, quant_method, model_source, model_revision, is_local, dtype, param_dtype, attn_implementation, device_map, load_seconds, transformers_version, checkpoint_fingerprint`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models_dispatch.py`:

```python
from types import SimpleNamespace  # add near the top imports of the file


class _FakeParam:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeModel:
    """Stand-in for a loaded transformers model — no torch needed."""

    def __init__(self, device_map):
        self.hf_device_map = device_map
        self.generation_config = SimpleNamespace(
            do_sample=True, num_beams=3, temperature=0.7, top_p=0.9, top_k=40
        )
        self.config = SimpleNamespace()

    def parameters(self):
        yield _FakeParam("torch.bfloat16")


def _fake_builder(model):
    def builder(variant, source, opts):
        return model, object(), {"dtype": "bfloat16", "transformers_version": "5.10.1"}

    return builder


def test_load_variant_dispatches_builds_and_assembles_metadata(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    lv = loaders.load_variant("bf16")
    assert lv.model is model
    assert lv.variant.id == "bf16"
    md = lv.metadata
    assert md["variant_id"] == "bf16"
    assert md["quant_method"] == "bf16"
    assert md["model_source"] == "Qwen/Qwen2.5-7B-Instruct"
    assert md["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert md["is_local"] is False
    assert md["dtype"] == "bfloat16"
    assert md["param_dtype"] == "torch.bfloat16"
    assert md["attn_implementation"] == "sdpa"
    assert md["device_map"] == {"": 0}
    assert isinstance(md["load_seconds"], float)
    assert md["transformers_version"] == "5.10.1"
    # greedy forced post-build:
    assert lv.model.generation_config.do_sample is False
    assert lv.model.generation_config.num_beams == 1


def test_load_variant_unknown_id_raises_keyerror():
    from qquant.models import loaders

    with pytest.raises(KeyError):
        loaders.load_variant("nope-not-a-variant")


def test_load_variant_disabled_raises_valueerror():
    from qquant.models import loaders
    from qquant.registry import Variant

    fake = {
        "bf16": Variant(
            id="bf16",
            quant_method="bf16",
            source="baseline",
            enabled=False,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            notes="",
        )
    }
    with pytest.raises(ValueError):
        loaders.load_variant("bf16", variants=fake)


def test_load_variant_rejects_cpu_offload(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0, "model.layers.20": "cpu"})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    with pytest.raises(RuntimeError):
        loaders.load_variant("bf16")


def test_loaded_variant_context_manager_unloads(monkeypatch):
    from qquant.models import loaders

    model = _FakeModel({"": 0})
    monkeypatch.setitem(loaders.BUILDERS, "bf16", _fake_builder(model))
    with loaders.load_variant("bf16") as lv:
        assert lv.model is model
    assert lv.model is None  # unload() ran on context exit


def test_smoke_test_load_never_raises(monkeypatch):
    from qquant.models import loaders

    def boom(*args, **kwargs):
        raise RuntimeError("kapow")

    monkeypatch.setattr(loaders, "load_variant", boom)
    ok, msg = loaders.smoke_test_load("awq-official")
    assert ok is False
    assert "kapow" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_dispatch.py::test_load_variant_dispatches_builds_and_assembles_metadata -v`
Expected: FAIL — `AttributeError: module 'qquant.models.loaders' has no attribute 'load_variant'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top-of-file imports in `src/qquant/models/loaders.py`:

```python
from time import perf_counter

from qquant.models.greedy import force_greedy
from qquant.registry import Variant, load_variants
```

(Replace the existing `from qquant.registry import Variant` line with the combined import above; keep `from dataclasses import dataclass`, `from pathlib import Path`, and `from typing import Any, Callable`.)

Append to `src/qquant/models/loaders.py`:

```python
@dataclass
class LoadedVariant:
    """A loaded, greedy-forced, GPU-resident model ready for HFLM / profiling."""

    variant: Variant
    model: Any  # transformers PreTrainedModel, already device-placed
    tokenizer: Any  # PreTrainedTokenizerBase
    metadata: dict[str, Any]

    def unload(self) -> None:
        """Free the model and empty the CUDA cache so the next variant can load."""
        import gc

        self.model = None
        self.tokenizer = None
        gc.collect()
        try:  # torch may be absent (laptop) — unload() must still drop references
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "LoadedVariant":
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()


def _param_dtype(model: Any) -> str:
    try:
        return str(next(model.parameters()).dtype)
    except Exception:
        return ""


def load_variant(
    variant_id: str,
    *,
    variants: dict[str, Variant] | None = None,
    checkpoints_root: str | Path = "checkpoints",
    device_map: Any = None,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
) -> LoadedVariant:
    """Resolve, dispatch on quant_method, force greedy, assert no offload, return LoadedVariant.

    Raises ``KeyError`` for an unknown id, ``ValueError`` for a disabled variant, and
    ``RuntimeError`` if the resolved device map offloads any shard to ``cpu``/``disk``.
    """
    variants = variants if variants is not None else load_variants()
    if variant_id not in variants:
        raise KeyError(f"unknown variant id {variant_id!r}; known: {sorted(variants)}")
    variant = variants[variant_id]
    if not variant.enabled:
        raise ValueError(f"variant {variant_id!r} is disabled in the registry")

    source = resolve_model_source(variant, checkpoints_root)
    builder = BUILDERS[variant.quant_method]
    opts = {
        "device_map": device_map if device_map is not None else {"": 0},
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }

    t0 = perf_counter()
    model, tokenizer, builder_meta = builder(variant, source, opts)
    load_seconds = perf_counter() - t0

    force_greedy(model)

    device_map_resolved = dict(getattr(model, "hf_device_map", {}) or {})
    offloaded = {str(d) for d in device_map_resolved.values()} & {"cpu", "disk"}
    if offloaded:
        raise RuntimeError(
            f"variant {variant_id!r} offloaded to {sorted(offloaded)} "
            f"(device_map={device_map_resolved}); refusing — corrupts Spec 06 timings"
        )

    metadata = {
        "variant_id": variant.id,
        "quant_method": variant.quant_method,
        "model_source": source.path_or_repo,
        "model_revision": source.revision,
        "is_local": source.is_local,
        "dtype": builder_meta.get("dtype"),
        "param_dtype": _param_dtype(model),
        "attn_implementation": attn_implementation,
        "device_map": device_map_resolved,
        "load_seconds": load_seconds,
        "transformers_version": builder_meta.get("transformers_version"),
        "checkpoint_fingerprint": builder_meta.get("checkpoint_fingerprint"),
    }
    return LoadedVariant(variant=variant, model=model, tokenizer=tokenizer, metadata=metadata)


def smoke_test_load(
    variant_id: str,
    *,
    variants: dict[str, Variant] | None = None,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    """Load + run a 1-token greedy generate + unload; return (ok, message) WITHOUT raising.

    Used for the awq-official contingency: on (False, msg) the operator sets enabled=false.
    """
    try:
        lv = load_variant(
            variant_id, variants=variants, checkpoints_root=checkpoints_root
        )
    except Exception as exc:
        return False, f"load failed for {variant_id!r}: {exc!r}"
    try:
        import torch

        device = next(lv.model.parameters()).device
        enc = lv.tokenizer("Hello", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            lv.model.generate(**enc, max_new_tokens=1, do_sample=False)
        return True, f"{variant_id!r} loaded + generated 1 token on {device}"
    except Exception as exc:
        return False, f"forward failed for {variant_id!r}: {exc!r}"
    finally:
        lv.unload()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_dispatch.py -v`
Expected: PASS (12 passed — 6 from Tasks 2-3 plus the 6 orchestration tests).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/models/loaders.py tests/test_models_dispatch.py
git commit -m "Spec 04: load_variant orchestration + LoadedVariant + smoke_test_load"
```

---

### Task 5: Public surface, GPU test suite, marker/skip wiring, full verification

**Files:**
- Modify: `src/qquant/models/__init__.py` (full public re-exports)
- Modify: `pyproject.toml` (register the `gpu` pytest marker)
- Create: `tests/conftest.py` (auto-skip `gpu`-marked tests without CUDA)
- Create: `tests/test_models_gpu.py` (`@pytest.mark.gpu`, verified on the 4090)
- Test: `tests/test_models_dispatch.py` (append the deferred-import torch-free assertion)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `qquant.models` public surface (`BUILDERS, LoadedVariant, ModelSource, force_greedy, load_variant, resolve_model_source, smoke_test_load`); a `gpu` marker auto-skipped off-CUDA.

- [ ] **Step 1: Finalize the public surface**

Replace the contents of `src/qquant/models/__init__.py` with:

```python
"""qquant.models — registry-driven variant loaders (Spec 04).

Importing this package is torch-free (heavy imports are deferred into the builder bodies),
so dispatch / source-resolution / greedy-forcing are unit-testable off-GPU. This is NOT part
of the torch-free core: the umbrella ``qquant`` CLI must never import it.
"""

from __future__ import annotations

from qquant.models.greedy import force_greedy
from qquant.models.loaders import (
    BUILDERS,
    LoadedVariant,
    ModelSource,
    load_variant,
    resolve_model_source,
    smoke_test_load,
)

__all__ = [
    "BUILDERS",
    "LoadedVariant",
    "ModelSource",
    "force_greedy",
    "load_variant",
    "resolve_model_source",
    "smoke_test_load",
]
```

- [ ] **Step 2: Register the `gpu` marker + add the conftest skip**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
    "gpu: requires a CUDA GPU; run on the vast.ai box (auto-skipped without CUDA)",
]
```

Create `tests/conftest.py`:

```python
from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip every ``@pytest.mark.gpu`` test unless a CUDA GPU is present."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU (run on the vast.ai box)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
```

- [ ] **Step 3: Write the deferred-import (torch-free) assertion test**

Append to `tests/test_models_dispatch.py`:

```python
import subprocess  # add near the top imports of the file
import sys  # add near the top imports of the file


def test_importing_qquant_models_is_torch_free():
    """Importing the package (not running a builder) must not pull torch in.

    Run in a fresh interpreter so another test's torch import can't mask a regression.
    """
    code = (
        "import importlib, sys;"
        "importlib.import_module('qquant.models');"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"qquant.models imported torch: {result.stderr}"
```

- [ ] **Step 4: Write the GPU suite (verified on the 4090, auto-skipped locally)**

Create `tests/test_models_gpu.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip the whole module where torch is absent (laptop)

from qquant.models import load_variant, smoke_test_load  # noqa: E402
from qquant.registry import enabled_variants, load_variants  # noqa: E402

pytestmark = pytest.mark.gpu


def _core_enabled_ids():
    """Enabled official-core variants. Self-quant checkpoints don't exist until Spec 07."""
    variants = load_variants()
    return [v.id for v in enabled_variants(variants) if v.source != "selfquant"]


@pytest.mark.parametrize("variant_id", _core_enabled_ids())
def test_load_variant_no_offload_and_generates(variant_id):
    with load_variant(variant_id) as lv:
        values = {str(d) for d in lv.metadata["device_map"].values()}
        assert not (values & {"cpu", "disk"}), f"offload in {lv.metadata['device_map']}"
        device = next(lv.model.parameters()).device
        enc = lv.tokenizer("The capital of France is", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = lv.model.generate(**enc, max_new_tokens=1, do_sample=False)
        assert out.shape[-1] == enc["input_ids"].shape[-1] + 1


@pytest.mark.parametrize("variant_id", _core_enabled_ids())
def test_greedy_generate_is_deterministic(variant_id):
    with load_variant(variant_id) as lv:
        device = next(lv.model.parameters()).device
        enc = lv.tokenizer("List three primary colors:", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        def gen():
            return lv.model.generate(**enc, max_new_tokens=8, do_sample=False).tolist()

        assert gen() == gen()


def test_smoke_test_load_awq_official_reports():
    ok, msg = smoke_test_load("awq-official")
    assert isinstance(ok, bool)
    assert isinstance(msg, str) and msg


def test_unload_allows_sequential_load_without_oom():
    ids = _core_enabled_ids()
    if len(ids) < 2:
        pytest.skip("need >= 2 enabled core variants for a sequential-load check")
    first = load_variant(ids[0])
    first.unload()
    second = load_variant(ids[1])
    second.unload()
```

- [ ] **Step 5: Run the full local suite + lint to verify everything is green**

Run: `uv run pytest -v`
Expected: all CPU tests PASS (existing 58 + the new greedy/dispatch tests); every `tests/test_models_gpu.py` test reports SKIPPED ("requires a CUDA GPU"); `test_importing_qquant_models_is_torch_free` and the existing `test_core_import_is_torch_free` both PASS.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: "All checks passed!" and "N files already formatted".

Run: `uv run python -c "import qquant.models; print(sorted(qquant.models.__all__))"`
Expected: `['BUILDERS', 'LoadedVariant', 'ModelSource', 'force_greedy', 'load_variant', 'resolve_model_source', 'smoke_test_load']` printed with no torch import error.

- [ ] **Step 6: Commit**

```bash
git add src/qquant/models/__init__.py pyproject.toml tests/conftest.py tests/test_models_gpu.py tests/test_models_dispatch.py
git commit -m "Spec 04: public surface + gpu marker/skip + GPU suite + torch-free guard"
```

---

## GPU verification (on the box, after this plan + Spec 03/05)

The CPU plan above is complete and shippable on the laptop. The `@pytest.mark.gpu` suite is the real proof and runs on a rented RTX 4090 during a GPU bring-up session (folds into the existing matrix-run budget — no extra spend):

```bash
# on the vast.ai box, in the repo, after `uv sync --group gpu`
uv run pytest tests/test_models_gpu.py -v   # CUDA present => tests run instead of skip
```

This satisfies Spec 04 Done-when #5 (no-offload + 1-token generate per enabled core variant), #6 (greedy determinism), #7 (`smoke_test_load("awq-official")` reports without raising), and #8 (sequential load after `unload()`). Self-quant variants (`gptq-selfquant`, `awq-selfquant`) are verified after Spec 07 writes their checkpoints.

---

## Self-Review

**1. Spec coverage** (each Spec 04 Done-when → task):
- DW1 `set(BUILDERS)==QUANT_METHODS`, dispatch, KeyError/ValueError → Task 3 (keys) + Task 4 (dispatch/errors). ✅
- DW2 `resolve_model_source` mapping (local + official SHA) → Task 2. ✅
- DW3 `force_greedy` greedy fields + idempotent → Task 1. ✅
- DW4 torch-free guard not regressed + umbrella doesn't import models → Task 5 (`test_importing_qquant_models_is_torch_free`; deferred imports; no CLI line added). ✅
- DW5 GPU no-offload + 1-token generate per enabled variant → Task 5 `test_load_variant_no_offload_and_generates`. ✅
- DW6 GPU greedy determinism → Task 5 `test_greedy_generate_is_deterministic`. ✅
- DW7 `smoke_test_load("awq-official")` (no raise) → Task 4 (CPU no-raise) + Task 5 (GPU reports). ✅
- DW8 `unload()`/context-manager frees VRAM for sequential load → Task 4 (CPU bookkeeping) + Task 5 (GPU sequential). ✅
- DW9 `metadata["model_revision"]` == resolved revision → Task 4 metadata assertions. ✅
- Builder knob table (bf16/int8/nf4/gptq/awq/compressed-tensors) → Task 3, each builder verbatim per the Global Constraints. ✅
- `metadata` dict shape → Task 4. ✅
- Files-to-create list (`__init__.py`, `loaders.py`, `greedy.py`, 3 test files) → all created; optional `qquant-load` CLI intentionally skipped per Global Constraints. ✅

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases"/"similar to Task N". Every code step shows full code; every run step shows the exact command + expected output. ✅

**3. Type consistency:** `Builder` signature `(Variant, ModelSource, dict) -> (model, tokenizer, builder_meta)` is identical in Task 3 (definitions) and Task 4 (consumed by `load_variant`). `builder_meta` keys (`dtype`, `transformers_version`, `checkpoint_fingerprint`) match between the builders (Task 3) and the metadata assembly (Task 4). `load_variant`/`smoke_test_load`/`resolve_model_source`/`force_greedy` signatures match the `__init__` re-exports (Task 5) and the test call-sites. `LoadedVariant` field names (`variant/model/tokenizer/metadata`) are consistent across definition (Task 4), tests, and the context-manager usage. ✅
