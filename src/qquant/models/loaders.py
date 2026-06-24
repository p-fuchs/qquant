"""Unified variant loaders for the 7 canonical Qwen2.5-7B variants (Spec 04).

DEFERRED IMPORTS: torch/transformers/bitsandbytes are imported INSIDE the
builder bodies only — never at module top — so importing this module stays
torch-free and the dispatch table / source resolution / greedy-forcing are
unit-testable on the dev laptop. The actual loads run on the RTX 4090. This
module is NOT the torch-free core; the umbrella ``qquant`` CLI must never
import it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from qquant.models.greedy import force_greedy
from qquant.registry import Variant, load_variants

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

    ``local:checkpoints/self-quant/<id>`` -> ``ModelSource(<checkpoints_root>
    /self-quant/<id>, None, True)`` (raising ``FileNotFoundError`` if that
    path is absent — Spec 07 produces it). Anything else ->
    ``ModelSource(model_id, variant.revision, False)``.
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


# A builder takes the resolved Variant/source/opts and returns (model, tokenizer, meta).
# torch/transformers are imported INSIDE each body so this module stays torch-free.
Builder = Callable[
    [Variant, "ModelSource", dict[str, Any]], tuple[Any, Any, dict[str, Any]]
]


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
    return (
        model,
        tok,
        {
            "dtype": "bfloat16",
            "transformers_version": transformers.__version__,
        },
    )


def _build_bnb_int8(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # LLM.int8(); do NOT enable llm_int8_enable_fp32_cpu_offload (CPU offload path).
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
    return (
        model,
        tok,
        {
            "dtype": "bfloat16",
            "transformers_version": transformers.__version__,
        },
    )


def _build_bnb_nf4(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # NF4 MUST set bnb_4bit_compute_dtype=bfloat16 (fp32 default => unfair speed).
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
    return (
        model,
        tok,
        {
            "dtype": "bfloat16",
            "transformers_version": transformers.__version__,
        },
    )


def _build_gptq(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Official GPTQ-Int4 (fp16): config auto-detected from repo, served by gptqmodel
    # kernels via optimum. dtype="auto" keeps native fp16 (no upcast).
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype="auto",
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return (
        model,
        tok,
        {
            "dtype": "auto",
            "transformers_version": transformers.__version__,
        },
    )


def _build_awq(variant: Variant, source: ModelSource, opts: dict[str, Any]):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Official AWQ served by gptqmodel kernels (NEVER autoawq). dtype="auto" (fp16).
    model = AutoModelForCausalLM.from_pretrained(
        source.path_or_repo,
        revision=source.revision,
        dtype="auto",
        device_map=opts["device_map"],
        attn_implementation=opts["attn_implementation"],
        trust_remote_code=opts["trust_remote_code"],
    )
    tok = AutoTokenizer.from_pretrained(source.path_or_repo, revision=source.revision)
    return (
        model,
        tok,
        {
            "dtype": "auto",
            "transformers_version": transformers.__version__,
        },
    )


def _build_compressed_tensors(
    variant: Variant, source: ModelSource, opts: dict[str, Any]
):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Self-quant W4A16 checkpoint, loaded natively by transformers v5.
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
    return (
        model,
        tok,
        {
            "dtype": "auto",
            "transformers_version": transformers.__version__,
            "checkpoint_fingerprint": _fingerprint_checkpoint(source.path_or_repo),
        },
    )


def _fingerprint_checkpoint(path: str | Path) -> str:
    """Cheap, deterministic content fingerprint of a local checkpoint dir.

    Hashes (name, size) of every config/weights file — no weight bytes read.
    Torch-free.
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

    def __enter__(self) -> LoadedVariant:
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
    """Resolve source, dispatch on quant_method, force greedy, assert no offload.

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
    return LoadedVariant(
        variant=variant, model=model, tokenizer=tokenizer, metadata=metadata
    )


def smoke_test_load(
    variant_id: str,
    *,
    variants: dict[str, Variant] | None = None,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    """Load, generate 1 token, unload; return (ok, message) WITHOUT raising.

    Used for the awq-official contingency: on (False, msg), operator sets enabled=false.
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
