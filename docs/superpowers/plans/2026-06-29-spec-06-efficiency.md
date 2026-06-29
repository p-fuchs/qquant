# Spec 06 — Efficiency Profiler (`qquant.efficiency` + `qquant-profile`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `qquant.efficiency` — the per-variant efficiency profiler that loads one variant in a fresh process via the Spec-04 loader, measures the 8 RQ#3 cost/throughput metrics on the RTX 4090, and writes one schema-valid `results/<variant>/efficiency.json`.

**Architecture:** A torch-free `schema.py` (JSON-schema load/validate, the `efficiency_path` carve-out, the `is_efficiency_done` resume predicate mirroring `qquant.matrix.is_cell_done`) + a `profiler.py` whose torch-free core (`ProfileConfig`, `median`, disk sizing, self-quant labeling, the `build_efficiency_result` assembler) is CPU-TDD'd and whose GPU measurements live behind an injectable `engine` (a `TorchEngine` that imports torch only inside methods) so the orchestration is fully unit-testable with a fake. The `cli.py` resume/dry-run/arg paths are torch-free; the real load+measure path verifies on the 4090.

**Tech Stack:** Python 3.11, torch 2.12.1+cu126 / transformers 5.10.1 (Linux `gpu` group), `huggingface_hub` (disk sizing), `jsonschema`; pytest; ruff. Consumes Spec-01 core (`qquant.registry/paths/config`) and Spec-04 `qquant.models.load_variant`.

## Global Constraints

Every task implicitly includes these (verbatim from `docs/specs/06-efficiency.md`, `contracts.md`, `decision-log.md`).

- **Torch-free boundary.** `qquant.efficiency.schema` imports **no** torch (jsonschema only). `qquant.efficiency.__init__` imports only `.schema` eagerly and resolves `main`/`profile_variant`/`ProfileConfig`/`build_efficiency_result` via module-level `__getattr__`. `qquant.efficiency.profiler` imports torch **only inside functions/methods** (importing the module stays torch-free). `python -c "import qquant.efficiency, qquant.efficiency.schema, qquant.efficiency.profiler"` must not pull `torch` into `sys.modules`; the umbrella `qquant` CLI never imports `qquant.efficiency`.
- **Spec-04 loader contract (shipped API).** `qquant.models.load_variant(variant_id: str, *, variants=None, checkpoints_root="checkpoints", device_map=None, attn_implementation="sdpa", trust_remote_code=False) -> LoadedVariant`. It takes the **string id** (not a `Variant`); device placement is `device_map` (default `{"": 0}` = cuda:0); there is **no `device=` kwarg**. `LoadedVariant` has `.variant` (the `Variant`), `.model` (greedy-forced, device-placed), `.tokenizer`, `.metadata` (dict with `dtype`, `param_dtype`, `model_revision`, `quant_method`, `transformers_version`, …), and `.unload()`; it is a context manager. The loader already **refuses** any cpu/disk offload (`RuntimeError`), so the profiler gets a clean GPU-resident model. Source `dtype`/`model_revision`/`transformers_version` from `loaded.metadata`; introspect only `torch_version`/`cuda_version`/`gpu_name` from torch.
- **One variant per process.** The CLI profiles exactly **one** `--variant` (missing/>1 → exit 2; unknown id ∉ `VARIANT_IDS` → exit 2). Spec 08 spawns the per-variant loop.
- **Artifact carve-out (contracts §8).** `results/<variant>/efficiency.json` is the ONE sanctioned non-cell artifact: its **own** path helper (`efficiency_path`), schema (`efficiency_result.schema.json`), and resume predicate (`is_efficiency_done`), all owned by `qquant.efficiency`. `qquant.paths` is **not** edited.
- **Resume predicate shape.** `is_efficiency_done(result, active_config=None)` mirrors `is_cell_done`: structural validity ALWAYS; if `active_config` given, every overlapping `EFFICIENCY_PROVENANCE_KEYS` value must match or the artifact is stale. `EFFICIENCY_PROVENANCE_KEYS = ("torch_version","transformers_version","cuda_version","gpu_name","dtype","model_revision","prompt_buckets","decode_tokens","batch_seq_len")`.
- **CLI resume is torch-free.** The CLI compares only the torch-free subset of the provenance keys it can know pre-load (`model_revision`, `prompt_buckets`, `decode_tokens`, `batch_seq_len`); a matching, structurally-done artifact → skip (exit 0, no load) unless `--force`. Runtime-fact staleness (torch/cuda/gpu/dtype) is not detectable pre-load (documented); the full match is exercised by `is_efficiency_done` unit tests.
- **Greedy, exact token counts.** Generation forces `do_sample=False` and `min_new_tokens == max_new_tokens == decode_tokens` (EOS disabled) so decode-token counts are identical across variants/prompts. Throughput buckets are measured at **batch 1**; multi-batch capacity is the separate OOM sweep.
- **Self-quant speed is runtime-confounded.** For `variant.source == "selfquant"`: `speed_label = "runtime-confounded"` and `confounded_metrics == ["prefill_tok_s","decode_tok_s","e2e_latency_s","ttft_s"]` (the speed set only — **disk** and **memory** are never confounded). All other variants: `speed_label = "comparable"`, `confounded_metrics == []`.
- **OOM sweep runs LAST**, bounded by `--batch-ceiling`; catches `torch.cuda.OutOfMemoryError`/`RuntimeError("out of memory")`, `empty_cache()` + `reset_peak_memory_stats()` between trials, records the boundary, never crashes the earlier metrics.
- **`expandable_segments` is observed, not set.** The profiler records `PYTORCH_CUDA_ALLOC_CONF`'s observed value and warns if unset; the launcher (Spec 03/08) sets it before CUDA init.
- **Exit codes.** `0` = success or up-to-date skip; `2` = bad args (unknown/missing/>1 variant); `3` = load failure or OOM-at-load (no artifact written). `--dry-run` prints the resolved plan + path + resume decision without importing torch (exit 0).
- **Tooling.** ruff line-length 88, rules E,F,I,UP,B; `ruff format`. Python `>=3.11,<3.12`. `from __future__ import annotations` in every module. The `gpu` pytest marker already exists (Spec 04); GPU tests ride `@pytest.mark.gpu` (auto-skip off-CUDA via `tests/conftest.py`) and `pytest.importorskip("torch")`.

---

### Task 1: efficiency schema JSON + package `__init__` + `schema.py` validators

**Files:**
- Create: `src/qquant/schemas/efficiency_result.schema.json`
- Create: `src/qquant/efficiency/__init__.py`
- Create: `src/qquant/efficiency/schema.py`
- Modify: `tests/test_import_torch_free.py` (add efficiency modules to the guard)
- Test: `tests/test_efficiency_schema.py`

**Interfaces:**
- Consumes: `jsonschema`, `importlib.resources`.
- Produces:
  - `EFFICIENCY_SCHEMA_VERSION = 1`
  - `load_efficiency_schema() -> dict` (lru_cache; reads the json resource)
  - `validate_efficiency(result: dict) -> None` (raises `jsonschema.ValidationError`)
  - `is_valid_efficiency(result: dict) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_efficiency_schema.py`:

```python
from __future__ import annotations

import pytest

from qquant.efficiency.schema import (
    EFFICIENCY_SCHEMA_VERSION,
    is_valid_efficiency,
    load_efficiency_schema,
    validate_efficiency,
)


def _minimal_result() -> dict:
    return {
        "schema_version": 1,
        "variant": "bf16",
        "disk": {"weights_bytes": 1, "source": "hf-cache"},
        "memory": {
            "weights_resident_bytes": 1,
            "load_peak_bytes": 2,
            "generate_peak_bytes": 3,
        },
        "throughput": {
            "short": {
                "prompt_tokens": 128,
                "gen_tokens": 256,
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "e2e_latency_s": 1.0,
                "ttft_s": 1.0,
            }
        },
        "max_batch_size": {"value": 1, "oom_at": 2, "ceiling": 64},
        "config": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "torch_version": "2.12.1+cu126",
            "transformers_version": "5.10.1",
            "cuda_version": "12.6",
            "decode_tokens": 256,
            "prompt_buckets": {"short": 128, "medium": 1024, "long": 4096},
            "batch_seq_len": 2048,
        },
    }


def test_schema_version_constant():
    assert EFFICIENCY_SCHEMA_VERSION == 1
    assert load_efficiency_schema()["properties"]["schema_version"]["const"] == 1


def test_minimal_result_validates():
    validate_efficiency(_minimal_result())  # must not raise
    assert is_valid_efficiency(_minimal_result()) is True


def test_missing_required_top_level_key_fails():
    bad = _minimal_result()
    del bad["throughput"]
    with pytest.raises(Exception):
        validate_efficiency(bad)
    assert is_valid_efficiency(bad) is False


def test_wrong_schema_version_fails():
    bad = _minimal_result()
    bad["schema_version"] = 2
    assert is_valid_efficiency(bad) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_efficiency_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.efficiency'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/schemas/efficiency_result.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://qquant.local/schemas/efficiency_result.schema.json",
  "title": "qquant efficiency result",
  "description": "Per-variant efficiency profile. schema_version 1.",
  "type": "object",
  "additionalProperties": true,
  "required": [
    "schema_version",
    "variant",
    "disk",
    "memory",
    "throughput",
    "max_batch_size",
    "config"
  ],
  "properties": {
    "schema_version": { "const": 1 },
    "variant": { "type": "string", "minLength": 1 },
    "quant_method": { "type": "string" },
    "source": { "type": "string" },
    "model_id": { "type": "string" },
    "model_revision": { "type": ["string", "null"] },
    "speed_label": { "enum": ["comparable", "runtime-confounded"] },
    "confounded_metrics": { "type": "array", "items": { "type": "string" } },
    "disk": {
      "type": "object",
      "additionalProperties": true,
      "required": ["weights_bytes", "source"],
      "properties": {
        "weights_bytes": { "type": "integer", "exclusiveMinimum": 0 },
        "snapshot_bytes": { "type": "integer", "minimum": 0 },
        "weights_gib": { "type": "number" },
        "source": { "enum": ["hf-cache", "local"] },
        "path": { "type": "string" }
      }
    },
    "memory": {
      "type": "object",
      "additionalProperties": true,
      "required": [
        "weights_resident_bytes",
        "load_peak_bytes",
        "generate_peak_bytes"
      ],
      "properties": {
        "weights_resident_bytes": { "type": "integer", "exclusiveMinimum": 0 },
        "load_peak_bytes": { "type": "integer", "exclusiveMinimum": 0 },
        "load_peak_reserved_bytes": { "type": "integer", "minimum": 0 },
        "generate_peak_bytes": { "type": "integer", "exclusiveMinimum": 0 },
        "generate_peak_reserved_bytes": { "type": "integer", "minimum": 0 },
        "load_peak_gib": { "type": "number" },
        "generate_peak_gib": { "type": "number" }
      }
    },
    "throughput": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "type": "object",
        "additionalProperties": true,
        "required": [
          "prompt_tokens",
          "gen_tokens",
          "prefill_tok_s",
          "decode_tok_s",
          "e2e_latency_s",
          "ttft_s"
        ],
        "properties": {
          "prompt_tokens": { "type": "integer", "exclusiveMinimum": 0 },
          "gen_tokens": { "type": "integer", "exclusiveMinimum": 0 },
          "prefill_tok_s": { "type": "number", "exclusiveMinimum": 0 },
          "decode_tok_s": { "type": "number", "exclusiveMinimum": 0 },
          "e2e_latency_s": { "type": "number", "exclusiveMinimum": 0 },
          "ttft_s": { "type": "number", "exclusiveMinimum": 0 }
        }
      }
    },
    "max_batch_size": {
      "type": "object",
      "additionalProperties": true,
      "required": ["value", "ceiling"],
      "properties": {
        "value": { "type": "integer", "minimum": 1 },
        "oom_at": { "type": ["integer", "null"] },
        "ceiling": { "type": "integer", "minimum": 1 },
        "seq_len": { "type": "integer", "minimum": 1 },
        "gen_tokens": { "type": "integer", "minimum": 1 },
        "ladder": { "type": "array", "items": { "type": "integer" } }
      }
    },
    "config": {
      "type": "object",
      "additionalProperties": true,
      "required": [
        "gpu_name",
        "dtype",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "decode_tokens",
        "prompt_buckets",
        "batch_seq_len"
      ]
    },
    "samples": { "type": "object" },
    "meta": { "type": "object" }
  }
}
```

Create `src/qquant/efficiency/schema.py`:

```python
"""Efficiency-artifact schema, path, and resume predicate — TORCH-FREE (jsonschema only).

This is the per-variant efficiency carve-out (contracts.md §8): its own schema, path helper,
and resume predicate, all owned here. Mirrors qquant.schemas + qquant.matrix.is_cell_done so
the orchestrator's resume/copy model stays uniform.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

EFFICIENCY_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def load_efficiency_schema() -> dict[str, Any]:
    text = (files("qquant.schemas") / "efficiency_result.schema.json").read_text()
    return json.loads(text)


def validate_efficiency(result: dict) -> None:
    """Raise jsonschema.ValidationError if result does not match the v1 schema."""
    jsonschema.validate(instance=result, schema=load_efficiency_schema())


def is_valid_efficiency(result: dict) -> bool:
    try:
        validate_efficiency(result)
    except jsonschema.ValidationError:
        return False
    return True
```

Create `src/qquant/efficiency/__init__.py`:

```python
"""qquant.efficiency — per-variant efficiency profiler (Spec 06).

Importing this package is torch-free: only ``.schema`` is imported eagerly. The torch-touching
``profile_variant``/``ProfileConfig``/``build_efficiency_result`` and the CLI ``main`` resolve
lazily via ``__getattr__`` so ``import qquant.efficiency`` never pulls torch. NOT part of the
torch-free core — the umbrella ``qquant`` CLI must never import it.
"""

from __future__ import annotations

from typing import Any

from qquant.efficiency.schema import (
    EFFICIENCY_SCHEMA_VERSION,
    is_valid_efficiency,
    load_efficiency_schema,
    validate_efficiency,
)

__all__ = [
    "EFFICIENCY_SCHEMA_VERSION",
    "ProfileConfig",
    "build_efficiency_result",
    "is_valid_efficiency",
    "load_efficiency_schema",
    "main",
    "profile_variant",
    "validate_efficiency",
]

_LAZY = {
    "ProfileConfig": ("qquant.efficiency.profiler", "ProfileConfig"),
    "build_efficiency_result": ("qquant.efficiency.profiler", "build_efficiency_result"),
    "profile_variant": ("qquant.efficiency.profiler", "profile_variant"),
    "main": ("qquant.efficiency.cli", "main"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Modify `tests/test_import_torch_free.py` — add the efficiency modules to the fresh-interpreter module list (keep the existing tuple, append the three names):

```python
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant','qquant.cli','qquant.matrix','qquant.registry','qquant.paths',"
        "'qquant.config','qquant.eval.runner',"
        "'qquant.efficiency','qquant.efficiency.schema','qquant.efficiency.profiler',"
        "'qquant.orchestrate.vastai','qquant.orchestrate.spike')];"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_efficiency_schema.py tests/test_import_torch_free.py -v`
Expected: PASS (4 + 1). (`qquant.efficiency.profiler` does not exist yet — this step adds it to the guard string but the import will fail until Task 3. If running the guard now fails on the missing profiler module, temporarily omit `qquant.efficiency.profiler` from the string and add it back in Task 3 Step 3. Prefer: create an empty `src/qquant/efficiency/profiler.py` with only `"""placeholder — filled in Task 3."""\nfrom __future__ import annotations` so the guard passes now.)

Per that note, also create the placeholder `src/qquant/efficiency/profiler.py`:

```python
"""qquant.efficiency.profiler — torch-touching engine (filled in Tasks 3-4). Torch-free import."""

from __future__ import annotations
```

- [ ] **Step 5: Commit**

```bash
git add src/qquant/schemas/efficiency_result.schema.json src/qquant/efficiency/__init__.py \
        src/qquant/efficiency/schema.py src/qquant/efficiency/profiler.py \
        tests/test_efficiency_schema.py tests/test_import_torch_free.py
git commit -m "Spec 06: efficiency schema + package skeleton (torch-free) + guard"
```

---

### Task 2: `schema.py` — path, resume predicate, active-config

**Files:**
- Modify: `src/qquant/efficiency/schema.py`
- Test: `tests/test_efficiency_resume.py`

**Interfaces:**
- Consumes: `qquant.paths.Paths`, `qquant.registry.Variant`.
- Produces:
  - `EFFICIENCY_PROVENANCE_KEYS: tuple[str, ...]`
  - `efficiency_path(results_root, variant) -> Path` (`<root>/<variant>/efficiency.json` via `Paths`)
  - `is_efficiency_done(result: object, active_config: dict | None = None) -> bool`
  - `efficiency_active_config(cfg, variant, runtime: dict) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_efficiency_resume.py`:

```python
from __future__ import annotations

from qquant.efficiency.schema import (
    EFFICIENCY_PROVENANCE_KEYS,
    efficiency_active_config,
    efficiency_path,
    is_efficiency_done,
)
from qquant.registry import load_variants


def _runtime() -> dict:
    return {
        "torch_version": "2.12.1+cu126",
        "transformers_version": "5.10.1",
        "cuda_version": "12.6",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "dtype": "bfloat16",
    }


class _Cfg:
    prompt_buckets = {"short": 128, "medium": 1024, "long": 4096}
    decode_tokens = 256
    batch_seq_len = 2048


def test_efficiency_path_layout(tmp_path):
    p = efficiency_path(tmp_path, "bf16")
    assert p == tmp_path / "bf16" / "efficiency.json"


def test_provenance_keys_exact():
    assert EFFICIENCY_PROVENANCE_KEYS == (
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
        "dtype",
        "model_revision",
        "prompt_buckets",
        "decode_tokens",
        "batch_seq_len",
    )


def test_active_config_built_from_cfg_variant_runtime():
    variants = load_variants()
    active = efficiency_active_config(_Cfg(), variants["bf16"], _runtime())
    assert set(active) == set(EFFICIENCY_PROVENANCE_KEYS)
    assert active["model_revision"] == variants["bf16"].revision
    assert active["decode_tokens"] == 256
    assert active["gpu_name"] == "NVIDIA GeForce RTX 4090"


def test_is_efficiency_done_structural():
    assert is_efficiency_done("not a dict") is False
    assert is_efficiency_done({"schema_version": 2}) is False
    valid = {"schema_version": 1, "config": {"decode_tokens": 256}}
    # structurally incomplete (no required blocks) -> not done
    assert is_efficiency_done(valid) is False


def test_is_efficiency_done_provenance_match_and_stale():
    variants = load_variants()
    active = efficiency_active_config(_Cfg(), variants["bf16"], _runtime())
    result = {
        "schema_version": 1,
        "variant": "bf16",
        "disk": {"weights_bytes": 1, "source": "hf-cache"},
        "memory": {
            "weights_resident_bytes": 1,
            "load_peak_bytes": 2,
            "generate_peak_bytes": 3,
        },
        "throughput": {
            "short": {
                "prompt_tokens": 128,
                "gen_tokens": 256,
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "e2e_latency_s": 1.0,
                "ttft_s": 1.0,
            }
        },
        "max_batch_size": {"value": 1, "ceiling": 64},
        "config": dict(active)
        | {"device": "cuda:0", "do_sample": False, "warmup": 2, "repeats": 5},
    }
    assert is_efficiency_done(result, active) is True
    stale = dict(active) | {"decode_tokens": 512}
    assert is_efficiency_done(result, stale) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_efficiency_resume.py -v`
Expected: FAIL — `ImportError: cannot import name 'efficiency_path' ...`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/qquant/efficiency/schema.py` (after the validators; add the two imports at the top):

```python
from pathlib import Path  # add to the existing import block

from qquant.paths import Paths  # add (torch-free)

EFFICIENCY_PROVENANCE_KEYS: tuple[str, ...] = (
    "torch_version",
    "transformers_version",
    "cuda_version",
    "gpu_name",
    "dtype",
    "model_revision",
    "prompt_buckets",
    "decode_tokens",
    "batch_seq_len",
)


def efficiency_path(results_root: str | Path, variant: str) -> Path:
    """results_root/<variant>/efficiency.json — the Spec-06 carve-out (contracts §8)."""
    return Paths.from_root(results_root).results_root / variant / "efficiency.json"


def is_efficiency_done(result: object, active_config: dict | None = None) -> bool:
    """Structural validity always; if active_config given, every overlapping
    EFFICIENCY_PROVENANCE_KEYS value must match or the artifact is stale (recompute).
    Mirrors qquant.matrix.is_cell_done.
    """
    if not isinstance(result, dict):
        return False
    if result.get("schema_version") != EFFICIENCY_SCHEMA_VERSION:
        return False
    if not is_valid_efficiency(result):
        return False
    if active_config is not None:
        stored = result.get("config") or {}
        for key in EFFICIENCY_PROVENANCE_KEYS:
            if key in active_config and stored.get(key) != active_config[key]:
                return False
    return True


def efficiency_active_config(cfg: Any, variant: Any, runtime: dict) -> dict:
    """The comparable provenance block (exactly EFFICIENCY_PROVENANCE_KEYS) from the run
    config + runtime facts. ``runtime`` carries torch/transformers/cuda versions, gpu_name,
    dtype; model_revision comes from the Variant (pinned SHA; None for self-quant).
    """
    return {
        "torch_version": runtime["torch_version"],
        "transformers_version": runtime["transformers_version"],
        "cuda_version": runtime["cuda_version"],
        "gpu_name": runtime["gpu_name"],
        "dtype": runtime["dtype"],
        "model_revision": variant.revision,
        "prompt_buckets": dict(cfg.prompt_buckets),
        "decode_tokens": cfg.decode_tokens,
        "batch_seq_len": cfg.batch_seq_len,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_efficiency_resume.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/efficiency/schema.py tests/test_efficiency_resume.py
git commit -m "Spec 06: efficiency_path + is_efficiency_done + active_config (resume predicate)"
```

---

### Task 3: `profiler.py` torch-free core — `ProfileConfig`, `median`, disk size, self-quant label, `build_efficiency_result`

**Files:**
- Modify: `src/qquant/efficiency/profiler.py` (replace the placeholder)
- Test: `tests/test_efficiency_profiler_cpu.py`

**Interfaces:**
- Consumes: `qquant.efficiency.schema` (validate + active_config), `qquant.registry.Variant`.
- Produces:
  - `@dataclass(frozen=True) ProfileConfig` (device, warmup, repeats, decode_tokens, prompt_buckets [default_factory], batch_seq_len, batch_gen_tokens, batch_ceiling)
  - `median(xs: list[float]) -> float`
  - `speed_label_and_confounded(variant) -> tuple[str, list[str]]`
  - `sum_weight_bytes(snapshot_dir) -> dict` and `measure_disk_size(variant, *, snapshot_dir=None) -> dict`
  - `build_efficiency_result(*, variant, cfg, runtime, disk, memory, throughput, max_batch, env_meta=None, samples=None, meta_extra=None) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_efficiency_profiler_cpu.py`:

```python
from __future__ import annotations

import pytest

from qquant.efficiency.profiler import (
    ProfileConfig,
    build_efficiency_result,
    measure_disk_size,
    median,
    speed_label_and_confounded,
    sum_weight_bytes,
)
from qquant.efficiency.schema import is_efficiency_done, validate_efficiency
from qquant.registry import load_variants

CONFOUNDED = ["prefill_tok_s", "decode_tok_s", "e2e_latency_s", "ttft_s"]


def _runtime() -> dict:
    return {
        "torch_version": "2.12.1+cu126",
        "transformers_version": "5.10.1",
        "cuda_version": "12.6",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "dtype": "bfloat16",
        "expandable_segments": True,
        "timestamp": "2026-06-29T00:00:00Z",
        "hostname": "test",
    }


def _disk() -> dict:
    return {"weights_bytes": 15231000000, "source": "hf-cache", "weights_gib": 14.19}


def _memory() -> dict:
    return {
        "weights_resident_bytes": 15000000000,
        "load_peak_bytes": 15500000000,
        "generate_peak_bytes": 19000000000,
    }


def _throughput() -> dict:
    return {
        "short": {
            "prompt_tokens": 128,
            "gen_tokens": 256,
            "prefill_tok_s": 900.0,
            "decode_tok_s": 60.0,
            "e2e_latency_s": 4.3,
            "ttft_s": 0.14,
        }
    }


def _max_batch() -> dict:
    return {"value": 16, "oom_at": 32, "ceiling": 64, "ladder": [1, 2, 4, 8, 16]}


def test_profile_config_defaults_are_independent_dicts():
    a, b = ProfileConfig(), ProfileConfig()
    assert a.prompt_buckets == {"short": 128, "medium": 1024, "long": 4096}
    assert a.prompt_buckets is not b.prompt_buckets  # default_factory, not shared mutable
    assert a.decode_tokens == 256 and a.repeats == 5 and a.batch_ceiling == 64


def test_median_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_speed_label_selfquant_vs_comparable():
    variants = load_variants()
    label, conf = speed_label_and_confounded(variants["gptq-selfquant"])
    assert label == "runtime-confounded"
    assert conf == CONFOUNDED
    label2, conf2 = speed_label_and_confounded(variants["bf16"])
    assert label2 == "comparable"
    assert conf2 == []


def test_sum_weight_bytes_follows_symlinks(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "b1").write_bytes(b"x" * 100)
    (blobs / "b2").write_bytes(b"y" * 250)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model-00001.safetensors").symlink_to(blobs / "b1")
    (snap / "model-00002.safetensors").symlink_to(blobs / "b2")
    (snap / "config.json").write_bytes(b"{}")  # not a weight file
    out = sum_weight_bytes(snap)
    assert out["weights_bytes"] == 350
    assert out["snapshot_bytes"] >= 350


def test_measure_disk_size_with_injected_snapshot_dir(tmp_path):
    variants = load_variants()
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"z" * 500)
    out = measure_disk_size(variants["bf16"], snapshot_dir=snap)
    assert out["weights_bytes"] == 500
    assert out["source"] == "hf-cache"


def test_build_result_is_schema_valid_and_labels_selfquant():
    variants = load_variants()
    res = build_efficiency_result(
        variant=variants["gptq-selfquant"],
        cfg=ProfileConfig(),
        runtime=_runtime(),
        disk=_disk(),
        memory=_memory(),
        throughput=_throughput(),
        max_batch=_max_batch(),
    )
    validate_efficiency(res)  # must not raise
    assert res["schema_version"] == 1
    assert res["variant"] == "gptq-selfquant"
    assert res["speed_label"] == "runtime-confounded"
    assert res["confounded_metrics"] == CONFOUNDED
    assert res["config"]["decode_tokens"] == 256
    assert res["config"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert res["model_revision"] == variants["gptq-selfquant"].revision  # None for selfquant


def test_build_result_comparable_for_bf16_and_is_done_roundtrip():
    variants = load_variants()
    cfg = ProfileConfig()
    res = build_efficiency_result(
        variant=variants["bf16"],
        cfg=cfg,
        runtime=_runtime(),
        disk=_disk(),
        memory=_memory(),
        throughput=_throughput(),
        max_batch=_max_batch(),
    )
    assert res["speed_label"] == "comparable"
    assert res["confounded_metrics"] == []
    from qquant.efficiency.schema import efficiency_active_config

    active = efficiency_active_config(cfg, variants["bf16"], _runtime())
    assert is_efficiency_done(res, active) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_efficiency_profiler_cpu.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProfileConfig' from 'qquant.efficiency.profiler'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/qquant/efficiency/profiler.py` with (torch-free core only; the torch engine is added in Task 4):

```python
"""qquant.efficiency.profiler — measurement engine.

The torch-free core (ProfileConfig, median, disk sizing, self-quant labeling, the
``build_efficiency_result`` assembler) is unit-tested off-GPU. The torch-touching
measurements live behind ``TorchEngine`` (added in Task 4); torch is imported only inside
its methods, so importing this module stays torch-free.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qquant.efficiency.schema import efficiency_active_config, validate_efficiency

_SPEED_METRICS = ["prefill_tok_s", "decode_tok_s", "e2e_latency_s", "ttft_s"]
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_GIB = float(1024**3)


@dataclass(frozen=True)
class ProfileConfig:
    device: str = "cuda:0"
    warmup: int = 2
    repeats: int = 5
    decode_tokens: int = 256
    prompt_buckets: dict[str, int] = field(
        default_factory=lambda: {"short": 128, "medium": 1024, "long": 4096}
    )
    batch_seq_len: int = 2048
    batch_gen_tokens: int = 64
    batch_ceiling: int = 64


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def speed_label_and_confounded(variant: Any) -> tuple[str, list[str]]:
    """Self-quant speed is runtime-confounded (HF generate dequant path on sm_89).

    disk and memory are NEVER confounded — only the speed metrics.
    """
    if variant.source == "selfquant":
        return "runtime-confounded", list(_SPEED_METRICS)
    return "comparable", []


def sum_weight_bytes(snapshot_dir: str | Path) -> dict:
    """Sum real sizes (symlinks resolved) of *.safetensors/*.bin under snapshot_dir, plus the
    full resolved snapshot size."""
    root = Path(snapshot_dir)
    weights = 0
    snapshot = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size  # Path.stat follows symlinks
        snapshot += size
        if p.name.endswith(_WEIGHT_SUFFIXES):
            weights += size
    return {"weights_bytes": weights, "snapshot_bytes": snapshot}


def measure_disk_size(variant: Any, *, snapshot_dir: str | Path | None = None) -> dict:
    """Disk footprint for a variant. ``snapshot_dir`` injects the resolved dir (tests); else it
    is resolved from the HF cache (repo id) or the ``local:`` checkpoint dir (self-quant).
    """
    if snapshot_dir is not None:
        source = "local" if str(variant.model_id).startswith("local:") else "hf-cache"
        out = sum_weight_bytes(snapshot_dir)
        out["source"] = source
        out["path"] = str(snapshot_dir)
        out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
        return out
    if str(variant.model_id).startswith("local:"):
        local = Path(str(variant.model_id).split("local:", 1)[1])
        out = sum_weight_bytes(local)
        out["source"] = "local"
        out["path"] = str(local)
    else:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            variant.model_id, revision=variant.revision, local_files_only=True
        )
        out = sum_weight_bytes(resolved)
        out["source"] = "hf-cache"
        out["path"] = str(resolved)
    out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
    return out


def build_efficiency_result(
    *,
    variant: Any,
    cfg: ProfileConfig,
    runtime: dict,
    disk: dict,
    memory: dict,
    throughput: dict,
    max_batch: dict,
    env_meta: dict | None = None,
    samples: dict | None = None,
    meta_extra: dict | None = None,
) -> dict:
    """Assemble + validate a v1 efficiency result. ``config`` carries the comparable provenance
    block (EFFICIENCY_PROVENANCE_KEYS) plus the extra reproducibility keys. Does not persist.
    """
    speed_label, confounded = speed_label_and_confounded(variant)
    active = efficiency_active_config(cfg, variant, runtime)
    config = dict(active) | {
        "device": cfg.device,
        "expandable_segments": runtime.get("expandable_segments"),
        "do_sample": False,
        "warmup": cfg.warmup,
        "repeats": cfg.repeats,
        "batch_gen_tokens": cfg.batch_gen_tokens,
    }
    meta = {
        "timestamp": runtime.get("timestamp"),
        "hostname": runtime.get("hostname"),
    }
    if env_meta:
        meta["env"] = env_meta
    if meta_extra:
        meta.update(meta_extra)
    result = {
        "schema_version": 1,
        "variant": variant.id,
        "quant_method": variant.quant_method,
        "source": variant.source,
        "model_id": variant.model_id,
        "model_revision": variant.revision,
        "speed_label": speed_label,
        "confounded_metrics": confounded,
        "disk": disk,
        "memory": memory,
        "throughput": throughput,
        "max_batch_size": max_batch,
        "config": config,
        "meta": meta,
    }
    if samples is not None:
        result["samples"] = samples
    validate_efficiency(result)
    return result
```

Note: `import os` is used by `TorchEngine` in Task 4 (`os.environ` for `expandable_segments`); it is imported now so Task 4 only appends. If ruff flags `os` as unused at this step, add `# noqa: F401` to the `import os` line and remove the noqa in Task 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_efficiency_profiler_cpu.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/efficiency/profiler.py tests/test_efficiency_profiler_cpu.py
git commit -m "Spec 06: profiler torch-free core (ProfileConfig, disk, label, build_efficiency_result)"
```

---

### Task 4: `profiler.py` torch-touching engine + `profile_variant` orchestrator

**Files:**
- Modify: `src/qquant/efficiency/profiler.py` (append the engine + orchestrator)
- Test: `tests/test_efficiency_orchestration.py`

**Interfaces:**
- Consumes: `qquant.models.load_variant` (lazy), the Task-3 torch-free core.
- Produces:
  - `class TorchEngine` with methods `runtime_facts(loaded, cfg) -> dict`, `load_and_measure(load_model, variant, cfg) -> tuple[loaded, dict]`, `throughput_and_genpeak(loaded, cfg) -> tuple[dict, dict]`, `sweep(loaded, cfg) -> dict`, `disk(variant) -> dict`.
  - `profile_variant(variant, cfg, env_meta=None, *, load_model=None, engine=None) -> dict` — loads once, measures in a fixed order with the **OOM sweep LAST**, unloads in `finally`, returns the assembled result. Injectable `engine`/`load_model` make the orchestration CPU-testable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_efficiency_orchestration.py`:

```python
from __future__ import annotations

from qquant.efficiency.profiler import ProfileConfig, profile_variant
from qquant.efficiency.schema import validate_efficiency
from qquant.registry import load_variants


class _FakeLoaded:
    def __init__(self, variant):
        self.variant = variant
        self.model = object()
        self.tokenizer = object()
        self.metadata = {"dtype": "bfloat16", "model_revision": variant.revision}
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def disk(self, variant):
        self.calls.append("disk")
        return {"weights_bytes": 15231000000, "source": "hf-cache", "weights_gib": 14.19}

    def runtime_facts(self, loaded, cfg):
        self.calls.append("runtime")
        return {
            "torch_version": "2.12.1+cu126",
            "transformers_version": "5.10.1",
            "cuda_version": "12.6",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "dtype": "bfloat16",
            "expandable_segments": True,
            "timestamp": "2026-06-29T00:00:00Z",
            "hostname": "test",
        }

    def load_and_measure(self, load_model, variant, cfg):
        self.calls.append("load")
        loaded = load_model(variant.id, device_map={"": 0})
        return loaded, {
            "weights_resident_bytes": 15000000000,
            "load_peak_bytes": 15500000000,
        }

    def throughput_and_genpeak(self, loaded, cfg):
        self.calls.append("throughput")
        tp = {
            b: {
                "prompt_tokens": n,
                "gen_tokens": cfg.decode_tokens,
                "prefill_tok_s": 900.0,
                "decode_tok_s": 60.0,
                "e2e_latency_s": 4.3,
                "ttft_s": 0.14,
            }
            for b, n in cfg.prompt_buckets.items()
        }
        return tp, {"generate_peak_bytes": 19000000000}

    def sweep(self, loaded, cfg):
        self.calls.append("sweep")
        return {"value": 16, "oom_at": 32, "ceiling": cfg.batch_ceiling}


def test_profile_variant_loads_once_sweeps_last_unloads_and_assembles():
    variants = load_variants()
    loaded_box = {}

    def fake_load(variant_id, **kw):
        loaded_box["loaded"] = _FakeLoaded(variants[variant_id])
        return loaded_box["loaded"]

    engine = _FakeEngine()
    res = profile_variant(
        variants["bf16"], ProfileConfig(), load_model=fake_load, engine=engine
    )
    validate_efficiency(res)
    assert res["variant"] == "bf16"
    assert res["memory"]["generate_peak_bytes"] == 19000000000
    assert res["memory"]["load_peak_bytes"] == 15500000000
    assert set(res["throughput"]) == {"short", "medium", "long"}
    # sweep must run AFTER throughput/memory (last GPU measurement)
    assert engine.calls.index("sweep") > engine.calls.index("throughput")
    assert engine.calls.index("sweep") > engine.calls.index("load")
    # model freed even on the happy path
    assert loaded_box["loaded"].unloaded is True


def test_profile_variant_unloads_on_measurement_failure():
    variants = load_variants()
    loaded_box = {}

    def fake_load(variant_id, **kw):
        loaded_box["loaded"] = _FakeLoaded(variants[variant_id])
        return loaded_box["loaded"]

    class _BoomEngine(_FakeEngine):
        def throughput_and_genpeak(self, loaded, cfg):
            raise RuntimeError("boom")

    try:
        profile_variant(
            variants["bf16"], ProfileConfig(), load_model=fake_load, engine=_BoomEngine()
        )
    except RuntimeError:
        pass
    assert loaded_box["loaded"].unloaded is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_efficiency_orchestration.py -v`
Expected: FAIL — `ImportError: cannot import name 'profile_variant' from 'qquant.efficiency.profiler'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/qquant/efficiency/profiler.py` (after `build_efficiency_result`). If you added `# noqa: F401` to `import os` in Task 3, remove it now (it is used here):

```python
def _default_load_model(variant_id: str, **kwargs: Any):
    from qquant.models import load_variant

    return load_variant(variant_id, **kwargs)


class TorchEngine:
    """The real torch-touching measurements. torch is imported INSIDE each method so importing
    this module stays torch-free. All timing is greedy, sync-bounded, under inference_mode.
    """

    def disk(self, variant: Any) -> dict:
        return measure_disk_size(variant)

    def runtime_facts(self, loaded: Any, cfg: ProfileConfig) -> dict:
        import datetime
        import socket

        import torch
        import transformers

        if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            import warnings

            warnings.warn(
                "PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True "
                "(must be set before CUDA init by the launcher; recording observed value)",
                stacklevel=2,
            )
        meta = getattr(loaded, "metadata", {}) or {}
        return {
            "torch_version": torch.__version__,
            "transformers_version": meta.get("transformers_version")
            or transformers.__version__,
            "cuda_version": torch.version.cuda or "",
            "gpu_name": torch.cuda.get_device_name(cfg.device),
            "dtype": meta.get("dtype") or "",
            "expandable_segments": "expandable_segments"
            in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
        }

    def load_and_measure(self, load_model: Any, variant: Any, cfg: ProfileConfig):
        import torch

        torch.cuda.reset_peak_memory_stats(cfg.device)
        loaded = load_model(variant.id, device_map={"": int(cfg.device.split(":")[-1])})
        mem = {
            "weights_resident_bytes": int(torch.cuda.memory_allocated(cfg.device)),
            "load_peak_bytes": int(torch.cuda.max_memory_allocated(cfg.device)),
            "load_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cfg.device)),
        }
        return loaded, mem

    def _make_inputs(self, tokenizer: Any, n_tokens: int, batch: int, device: str):
        import torch

        vocab = int(getattr(tokenizer, "vocab_size", 32000))
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab, (batch, n_tokens), generator=gen).to(device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def _timed_generate(self, model: Any, inputs: dict, *, max_new: int, cfg: ProfileConfig):
        import time

        import torch

        torch.cuda.synchronize(cfg.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=max_new,
                max_new_tokens=max_new,
                use_cache=True,
            )
        torch.cuda.synchronize(cfg.device)
        total_s = time.perf_counter() - t0
        n_new = int(out.shape[-1] - inputs["input_ids"].shape[-1])
        return total_s, n_new

    def throughput_and_genpeak(self, loaded: Any, cfg: ProfileConfig):
        import torch

        model, tok = loaded.model, loaded.tokenizer
        model.eval()
        torch.cuda.reset_peak_memory_stats(cfg.device)
        out: dict = {}
        for bucket, n_prompt in cfg.prompt_buckets.items():
            inputs = self._make_inputs(tok, n_prompt, 1, cfg.device)
            for _ in range(cfg.warmup):
                self._timed_generate(model, inputs, max_new=cfg.decode_tokens, cfg=cfg)
            ttfts, totals = [], []
            for _ in range(cfg.repeats):
                ttft_s, _ = self._timed_generate(model, inputs, max_new=1, cfg=cfg)
                total_s, _ = self._timed_generate(
                    model, inputs, max_new=cfg.decode_tokens, cfg=cfg
                )
                ttfts.append(ttft_s)
                totals.append(total_s)
            ttft = median(ttfts)
            total = median(totals)
            out[bucket] = {
                "prompt_tokens": n_prompt,
                "gen_tokens": cfg.decode_tokens,
                "prefill_tok_s": n_prompt / ttft,
                "decode_tok_s": (cfg.decode_tokens - 1) / max(total - ttft, 1e-9),
                "e2e_latency_s": total,
                "ttft_s": ttft,
            }
        gen_peak = {
            "generate_peak_bytes": int(torch.cuda.max_memory_allocated(cfg.device)),
            "generate_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(cfg.device)
            ),
        }
        return out, gen_peak

    def sweep(self, loaded: Any, cfg: ProfileConfig) -> dict:
        import torch

        model, tok = loaded.model, loaded.tokenizer
        ladder: list[int] = []
        value, oom_at = 0, None
        batch = 1
        while batch <= cfg.batch_ceiling:
            try:
                inputs = self._make_inputs(tok, cfg.batch_seq_len, batch, cfg.device)
                self._timed_generate(
                    model, inputs, max_new=cfg.batch_gen_tokens, cfg=cfg
                )
                value = batch
                ladder.append(batch)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(
                    exc, torch.cuda.OutOfMemoryError
                ):
                    raise
                oom_at = batch
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(cfg.device)
                break
            batch *= 2
        return {
            "value": max(value, 1) if value else 1,
            "oom_at": oom_at,
            "ceiling": cfg.batch_ceiling,
            "seq_len": cfg.batch_seq_len,
            "gen_tokens": cfg.batch_gen_tokens,
            "ladder": ladder,
        }


def profile_variant(
    variant: Any,
    cfg: ProfileConfig,
    env_meta: dict | None = None,
    *,
    load_model: Any = None,
    engine: Any = None,
) -> dict:
    """Load the variant ONCE, measure in a fixed order (disk, runtime, load-mem, throughput,
    then the OOM sweep LAST so it cannot pollute earlier peaks), unload in finally, and return
    the assembled schema-valid result. Injectable engine/load_model for off-GPU tests.
    """
    engine = engine if engine is not None else TorchEngine()
    load_model = load_model if load_model is not None else _default_load_model

    disk = engine.disk(variant)
    loaded, load_mem = engine.load_and_measure(load_model, variant, cfg)
    try:
        runtime = engine.runtime_facts(loaded, cfg)
        throughput, gen_mem = engine.throughput_and_genpeak(loaded, cfg)
        max_batch = engine.sweep(loaded, cfg)
    finally:
        loaded.unload()
    memory = {**load_mem, **gen_mem}
    return build_efficiency_result(
        variant=variant,
        cfg=cfg,
        runtime=runtime,
        disk=disk,
        memory=memory,
        throughput=throughput,
        max_batch=max_batch,
        env_meta=env_meta,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_efficiency_orchestration.py -v`
Expected: PASS (2 passed). Then confirm the module stays torch-free: `uv run pytest tests/test_import_torch_free.py -v` (PASS).

- [ ] **Step 5: Commit**

```bash
git add src/qquant/efficiency/profiler.py tests/test_efficiency_orchestration.py
git commit -m "Spec 06: TorchEngine + profile_variant (load-once, sweep-last, unload-in-finally)"
```

---

### Task 5: `cli.py` — `qquant-profile` entry point (arg validation, dry-run, resume)

**Files:**
- Create: `src/qquant/efficiency/cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`: add `qquant-profile`)
- Test: `tests/test_efficiency_cli.py`

**Interfaces:**
- Consumes: `qquant.registry` (`load_variants`, `VARIANT_IDS`), `qquant.efficiency.schema` (`efficiency_path`, `is_efficiency_done`), `qquant.efficiency.profiler` (`ProfileConfig`, `profile_variant`, lazily).
- Produces: `build_parser() -> argparse.ArgumentParser`; `main(argv=None) -> int` (exit codes 0/2/3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_efficiency_cli.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from qquant.efficiency.cli import main


def test_import_is_torch_free():
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant.efficiency','qquant.efficiency.schema','qquant.efficiency.cli')];"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_missing_variant_is_exit_2(capsys):
    assert main([]) == 2


def test_unknown_variant_is_exit_2(capsys):
    assert main(["--variant", "nope"]) == 2


def test_dry_run_is_torch_free_and_prints_plan(tmp_path, capsys):
    rc = main(["--variant", "bf16", "--results", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bf16" in out
    assert str(tmp_path / "bf16" / "efficiency.json") in out
    assert "torch" not in sys.modules  # dry-run never imports torch


def test_resume_skip_when_done_artifact_present(tmp_path, capsys):
    # Pre-write a structurally-done artifact whose torch-free provenance matches the defaults.
    from qquant.efficiency.profiler import ProfileConfig
    from qquant.registry import load_variants

    cfg = ProfileConfig()
    variants = load_variants()
    art = {
        "schema_version": 1,
        "variant": "bf16",
        "disk": {"weights_bytes": 1, "source": "hf-cache"},
        "memory": {
            "weights_resident_bytes": 1,
            "load_peak_bytes": 2,
            "generate_peak_bytes": 3,
        },
        "throughput": {
            "short": {
                "prompt_tokens": 128,
                "gen_tokens": 256,
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "e2e_latency_s": 1.0,
                "ttft_s": 1.0,
            }
        },
        "max_batch_size": {"value": 1, "ceiling": 64},
        "config": {
            "gpu_name": "x",
            "dtype": "bfloat16",
            "torch_version": "t",
            "transformers_version": "tr",
            "cuda_version": "c",
            "decode_tokens": cfg.decode_tokens,
            "prompt_buckets": dict(cfg.prompt_buckets),
            "batch_seq_len": cfg.batch_seq_len,
            "model_revision": variants["bf16"].revision,
        },
    }
    p = tmp_path / "bf16" / "efficiency.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(art))
    rc = main(["--variant", "bf16", "--results", str(tmp_path)])  # no --force
    out = capsys.readouterr().out
    assert rc == 0
    assert "up-to-date" in out or "skip" in out.lower()
    assert "torch" not in sys.modules  # skip path never loads


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_efficiency_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qquant.efficiency.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qquant/efficiency/cli.py`:

```python
"""qquant-profile entry point. --dry-run and the resume-skip path are torch-free; only the
actual profile loads torch (lazily, inside main)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qquant.efficiency.schema import (
    efficiency_path,
    is_efficiency_done,
)
from qquant.registry import VARIANT_IDS, load_variants


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qquant-profile")
    p.add_argument("--variant", action="append", default=[], dest="variants")
    p.add_argument("--results", default="results")
    p.add_argument("--variants-file")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--prompt-lens", default="128,1024,4096")
    p.add_argument("--batch-seq-len", type=int, default=2048)
    p.add_argument("--batch-gen-tokens", type=int, default=64)
    p.add_argument("--batch-ceiling", type=int, default=64)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _make_config(args):
    from qquant.efficiency.profiler import ProfileConfig

    s, m, length = (int(x) for x in args.prompt_lens.split(","))
    return ProfileConfig(
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        decode_tokens=args.decode_tokens,
        prompt_buckets={"short": s, "medium": m, "long": length},
        batch_seq_len=args.batch_seq_len,
        batch_gen_tokens=args.batch_gen_tokens,
        batch_ceiling=args.batch_ceiling,
    )


def _preload_active(cfg, variant) -> dict:
    """The torch-free subset of EFFICIENCY_PROVENANCE_KEYS knowable before load."""
    return {
        "model_revision": variant.revision,
        "prompt_buckets": dict(cfg.prompt_buckets),
        "decode_tokens": cfg.decode_tokens,
        "batch_seq_len": cfg.batch_seq_len,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.variants) != 1:
        print("error: exactly one --variant is required", file=sys.stderr)
        return 2
    variant_id = args.variants[0]
    if variant_id not in VARIANT_IDS:
        print(
            f"error: unknown variant {variant_id!r}; known: {list(VARIANT_IDS)}",
            file=sys.stderr,
        )
        return 2

    variants = load_variants(args.variants_file)
    variant = variants[variant_id]
    cfg = _make_config(args)
    out_path = efficiency_path(args.results, variant_id)

    if args.dry_run:
        print(f"variant={variant_id} source={variant.source}")
        print(f"config={cfg}")
        print(f"output={out_path}")
        done = _resume_skip(out_path, cfg, variant) and not args.force
        print(f"resume_decision={'skip' if done else 'recompute'}")
        return 0

    if not args.force and _resume_skip(out_path, cfg, variant):
        print(f"{out_path} is up-to-date (skip); use --force to recompute")
        return 0

    from qquant.efficiency.profiler import profile_variant

    try:
        result = profile_variant(variant, cfg)
    except Exception as exc:  # noqa: BLE001 — load/OOM failure: no artifact, exit 3
        print(f"profile failed for {variant_id}: {exc!r}", file=sys.stderr)
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out_path)
    print(f"wrote {out_path}")
    return 0


def _resume_skip(out_path: Path, cfg, variant) -> bool:
    if not out_path.exists():
        return False
    try:
        loaded = json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    return is_efficiency_done(loaded, _preload_active(cfg, variant))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

In `pyproject.toml`, under `[project.scripts]`, add after the `qquant-eval` line:

```toml
qquant-profile = "qquant.efficiency.cli:main"  # Spec 06
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_efficiency_cli.py -v`
Expected: PASS (6 passed).

Run: `uv sync && uv run qquant-profile --help`
Expected: prints usage, exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/qquant/efficiency/cli.py pyproject.toml tests/test_efficiency_cli.py
git commit -m "Spec 06: qquant-profile CLI (arg validation, torch-free dry-run + resume)"
```

---

### Task 6: GPU integration test + final verification

**Files:**
- Create: `tests/test_efficiency_gpu.py` (`@pytest.mark.gpu`)
- Test: full suite + ruff + entry point + torch-free guard

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: the opt-in GPU smoke that verifies the real load+measure path on a 4090.

- [ ] **Step 1: Write the GPU test**

Create `tests/test_efficiency_gpu.py`:

```python
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from qquant.efficiency.cli import main  # noqa: E402
from qquant.efficiency.schema import efficiency_path, is_valid_efficiency  # noqa: E402

pytestmark = pytest.mark.gpu


def test_bf16_profile_writes_valid_artifact(tmp_path):
    rc = main(
        [
            "--variant",
            "bf16",
            "--results",
            str(tmp_path),
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--decode-tokens",
            "16",
            "--prompt-lens",
            "32,64,128",
            "--batch-ceiling",
            "4",
        ]
    )
    assert rc == 0
    art = json.loads(efficiency_path(tmp_path, "bf16").read_text())
    assert is_valid_efficiency(art)
    mem = art["memory"]
    assert (
        mem["generate_peak_bytes"]
        >= mem["load_peak_bytes"]
        >= mem["weights_resident_bytes"]
        > 0
    )
    for bucket in ("short", "medium", "long"):
        tp = art["throughput"][bucket]
        assert tp["gen_tokens"] == 16
        assert tp["decode_tok_s"] > 0 and tp["prefill_tok_s"] > 0
    assert art["max_batch_size"]["value"] >= 1
```

- [ ] **Step 2: Verify it skips off-CUDA**

Run: `uv run pytest tests/test_efficiency_gpu.py -v`
Expected: SKIPPED on the laptop (no CUDA) — collected, not errored.

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q`
Expected: all CPU tests PASS (existing + Tasks 1–5); `tests/test_efficiency_gpu.py` SKIPPED.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: "All checks passed!" + "N files already formatted".

Run: `uv run python -c "import qquant.efficiency; print(sorted(qquant.efficiency.__all__)); import sys; assert 'torch' not in sys.modules"`
Expected: prints the `__all__` list; no torch imported.

Run: `uv run qquant-profile --variant bf16 --results /tmp/eff --dry-run`
Expected: prints the resolved config + output path + resume decision, exit 0, torch-free.

- [ ] **Step 4: Commit**

```bash
git add tests/test_efficiency_gpu.py
git commit -m "Spec 06: GPU smoke (bf16 profile artifact + memory ordering + determinism)"
```

---

## GPU verification (on the box)

The CPU plan is complete on the laptop. The real torch path — `TorchEngine` (load-peak via `reset_peak_memory_stats`, generate-peak transient, prefill/decode separation, the OOM sweep) — verifies on a rented RTX 4090:

```bash
# on the box, after `uv sync --group gpu`
uv run pytest tests/test_efficiency_gpu.py -v          # CUDA present => runs instead of skip
uv run qquant-profile --variant bf16                   # writes results/bf16/efficiency.json
```

This satisfies done-when #1 (artifact written + valid), #5 (memory ordering), #6 (OOM sweep), and #8 (determinism). Self-quant variants (`gptq-selfquant`, `awq-selfquant`) profile after Spec 07 produces their checkpoints; their `speed_label`/`confounded_metrics` are already CPU-verified here.

---

## Self-Review

**1. Spec coverage** (done-when → task):
- DW1 artifact written + valid on 4090 → T6 GPU test (`test_bf16_profile_writes_valid_artifact`). ✅
- DW2 resume skip (no load) + `--force` + provenance staleness → T2 (`is_efficiency_done` stale), T5 (`_resume_skip`, `test_resume_skip_when_done_artifact_present`). ✅
- DW3 self-quant labelling (speed set only; disk/memory not listed) → T3 (`speed_label_and_confounded`, `test_build_result_is_schema_valid_and_labels_selfquant`). ✅
- DW4 throughput buckets median over repeats, `gen_tokens==decode_tokens` → T4 (`throughput_and_genpeak` uses `median`), T6 GPU asserts `gen_tokens`. ✅
- DW5 memory ordering generate≥load≥resident>0 → T6 GPU assertion. ✅
- DW6 OOM sweep last, catch OOM, `value>=1`, `oom_at>value` or null → T4 (`sweep`, `profile_variant` order; `test_..._sweeps_last`), T6. ✅
- DW7 one variant per process (missing/>1/unknown → exit 2) → T5 (`test_missing_variant_is_exit_2`, `test_unknown_variant_is_exit_2`). ✅
- DW8 determinism: exact decode_tokens, do_sample False, identical tokens → T4 (`_timed_generate` forces min==max, do_sample=False), T6 GPU. ✅
- DW9 torch-free boundary + `qquant-profile` registered → T1 (guard), T5 (`test_import_is_torch_free`, pyproject line). ✅
- DW10 disk size sums weight files, source ∈ {hf-cache,local} → T3 (`sum_weight_bytes`/`measure_disk_size`, `test_sum_weight_bytes_follows_symlinks`). ✅
- DW11 `--dry-run` torch-free, prints config+path+decision → T5 (`test_dry_run_is_torch_free_and_prints_plan`). ✅
- DW12 ruff + pytest green, GPU skipped → T6 Step 3. ✅

**2. Placeholder scan:** No "TBD/TODO/handle edge cases/similar to Task N". Every code step shows full code; every run step shows the command + expected output. The one cross-task note (placeholder `profiler.py` in T1, replaced in T3; `import os` carried from T3 to T4) is explicit. ✅

**3. Type consistency:** `ProfileConfig` fields match across `efficiency_active_config`/`build_efficiency_result`/`_make_config`. `runtime` dict keys (`torch_version`/`transformers_version`/`cuda_version`/`gpu_name`/`dtype` + `expandable_segments`/`timestamp`/`hostname`) match between `TorchEngine.runtime_facts`, `efficiency_active_config`, and `build_efficiency_result`. The engine method set (`disk`/`runtime_facts`/`load_and_measure`/`throughput_and_genpeak`/`sweep`) matches `profile_variant`'s call sites and the `_FakeEngine` test double. `efficiency_path(results_root, variant)` signature matches T5's `efficiency_path(args.results, variant_id)`. `is_efficiency_done(result, active_config)` matches T5's `_resume_skip`. `build_efficiency_result(...)` kwargs match `profile_variant`'s call. ✅
