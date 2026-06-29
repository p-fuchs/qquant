"""qquant-profile entry point.

--dry-run and the resume-skip path are torch-free; only the actual profile
loads torch (lazily, inside main).
"""

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


def _resume_skip(out_path: Path, cfg, variant) -> bool:
    if not out_path.exists():
        return False
    try:
        loaded = json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    return is_efficiency_done(loaded, _preload_active(cfg, variant))


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
