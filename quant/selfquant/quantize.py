"""Self-quant driver: build the C4 calibration ONCE, then run llm-compressor oneshot for
each variant into an idempotent compressed-tensors checkpoint. Box-only (lazy imports).

Canonical invocation: ``cd quant && uv run python -m selfquant.quantize --id all``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

from selfquant.calibration import CalibrationSpec, build_calibration, input_ids_sha256
from selfquant.manifest import QuantManifest, selfquant_checkpoint_done, write_manifest
from selfquant.recipes import (
    GROUP_SIZE,
    SCHEME,
    SCHEME_W8A8,
    assert_schemes_match,
    awq_recipe,
    gptq_recipe,
    smoothquant_w8a8_recipe,
)

SELF_QUANT_IDS = ("gptq-selfquant", "awq-selfquant")  # the W4A16 fairness pair ("all")
# EXT-3 (Spec 12): W8A8 is opt-in (`--id w8a8-selfquant`), NOT part of "all", so the v1
# fairness-pair behaviour is unchanged. It is a different bit-width axis.
EXT_SELF_QUANT_IDS = ("w8a8-selfquant",)
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"  # decision-log baseline SHA

_ALGORITHM = {
    "gptq-selfquant": "gptq",
    "awq-selfquant": "awq",
    "w8a8-selfquant": "smoothquant-w8a8",
}
# Per-variant scheme + group_size (W8A8 is not group-quantised → group_size None).
_SCHEME = {
    "gptq-selfquant": (SCHEME, GROUP_SIZE),
    "awq-selfquant": (SCHEME, GROUP_SIZE),
    "w8a8-selfquant": (SCHEME_W8A8, None),
}
_DEFAULT_CHECKPOINTS_ROOT = "checkpoints/self-quant"


def _print_observed_weight_args(variant_id: str, out_dir: Path) -> None:
    """Best-effort done-when #3 confirmation: print the produced config.json weight
    quant args so the operator can verify both checkpoints match the controlled target.
    Never raises (the authoritative check compares the two manifests/configs)."""
    import json

    try:
        config = json.loads((out_dir / "config.json").read_text())
        groups = (config.get("quantization_config") or {}).get("config_groups") or {}
        first = next(iter(groups.values()), {})
        weights = first.get("weights") if isinstance(first, dict) else None
        print(f"[{variant_id}] config.json weight quant args: {weights}")
    except Exception as exc:  # noqa: BLE001 — diagnostic print only
        print(f"[{variant_id}] (could not read config.json weight args: {exc!r})")


def recipe_for(variant_id: str) -> list:
    if variant_id == "gptq-selfquant":
        return gptq_recipe()
    if variant_id == "awq-selfquant":
        return awq_recipe()
    if variant_id == "w8a8-selfquant":
        return smoothquant_w8a8_recipe()
    known = (*SELF_QUANT_IDS, *EXT_SELF_QUANT_IDS)
    raise ValueError(f"unknown self-quant id {variant_id!r}; known: {known}")


def quantize_variant(
    variant_id: str,
    calib_ds: Any,
    out_dir: str | Path,
    *,
    base_model_id: str = BASE_MODEL_ID,
    base_revision: str = BASE_REVISION,
    calib_sha256: str,
    calib_id: str,
    force: bool = False,
) -> Path:
    """oneshot-quantize one variant into a compressed-tensors checkpoint + manifest.

    Skips (returns out_dir) when a matching checkpoint exists and not force. Loads the
    base model at the pinned SHA, runs oneshot with the per-algorithm recipe and the
    SHARED calibration set, saves compressed, writes the manifest, frees the model.
    """
    import datetime
    import importlib.metadata as md

    out_dir = Path(out_dir)
    algorithm = _ALGORITHM[variant_id]
    scheme, group_size = _SCHEME[variant_id]
    if not force and selfquant_checkpoint_done(
        out_dir,
        algorithm=algorithm,
        base_revision=base_revision,
        scheme=scheme,
        group_size=group_size,
        calib_sha256=calib_sha256,
    ):
        print(f"[{variant_id}] up-to-date checkpoint at {out_dir} (skip)")
        return out_dir

    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM

    print(f"[{variant_id}] loading base {base_model_id}@{base_revision[:12]}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, revision=base_revision, dtype="auto"
    )
    print(f"[{variant_id}] oneshot ({algorithm}, {scheme}, g{group_size})")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe_for(variant_id),
        max_seq_length=2048,
        num_calibration_samples=128,
        shuffle_calibration_samples=False,  # we pre-select; keep the set fixed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    _print_observed_weight_args(variant_id, out_dir)

    write_manifest(
        out_dir,
        QuantManifest(
            variant_id=variant_id,
            algorithm=algorithm,
            scheme=scheme,
            group_size=group_size,
            base_model_id=base_model_id,
            base_revision=base_revision,
            calib_id=calib_id,
            calib_sha256=calib_sha256,
            llmcompressor_version=md.version("llmcompressor"),
            transformers_version=md.version("transformers"),
            compressed_tensors_version=md.version("compressed-tensors"),
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        ),
    )
    print(f"[{variant_id}] wrote checkpoint + manifest to {out_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selfquant.quantize")
    # "all" = the W4A16 fairness pair only (v1 behaviour). EXT-3 W8A8 is opt-in by id.
    p.add_argument(
        "--id", choices=[*SELF_QUANT_IDS, *EXT_SELF_QUANT_IDS, "all"], default="all"
    )
    p.add_argument("--checkpoints-root", default=_DEFAULT_CHECKPOINTS_ROOT)
    p.add_argument("--base-revision", default=BASE_REVISION)
    p.add_argument("--calib-only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Fail fast: the fairness guarantee before any GPU work.
    assert_schemes_match(gptq_recipe(), awq_recipe())

    from transformers import AutoTokenizer

    root = Path(args.checkpoints_root)
    spec = CalibrationSpec()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=args.base_revision
    )
    print(f"building calibration {spec.calib_id} ...")
    calib_ds = build_calibration(tokenizer, spec, root / "_calib")
    calib_sha = input_ids_sha256(calib_ds["input_ids"])
    print(f"calibration ready: {spec.calib_id} sha256={calib_sha[:12]}")
    if args.calib_only:
        return 0

    ids = list(SELF_QUANT_IDS) if args.id == "all" else [args.id]
    failures: list[str] = []
    for vid in ids:
        try:
            quantize_variant(
                vid,
                calib_ds,
                root / vid,
                base_revision=args.base_revision,
                calib_sha256=calib_sha,
                calib_id=spec.calib_id,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 — record + continue (resume-friendly)
            failures.append(f"{vid}: {exc!r}")
            print(f"[{vid}] FAILED: {exc!r}", file=sys.stderr)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
