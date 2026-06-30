"""qquant-ext entry point — the gated runner for the Spec 12 extensions.

Subcommands (all rooted at ``results-ext/`` by default, never ``results/``):

* ``plan``    — torch-free: list missing ext cells for the enabled ext matrix.
* ``eval``    — lm-eval quality sweep over enabled ext variants (EXT-1 Mistral, EXT-3
                W8A8, EXT-4 kvq) × the v1 tasks, via ``EvalRunner`` + the ext loader.
* ``judge``   — EXT-2 JudgeBench over the chosen registry (v1 core or ext variants).
* ``profile`` — EXT-4 efficiency profile of one ext variant (e.g. ``bf16-kvq4``).

EXT-5 QLoRA *training* runs separately in the isolated quant env
(``cd quant && uv run python -m selfquant.train_qlora ...``); this CLI only *evaluates*
the already-trained ``bnb-nf4-qlora`` adapter (the loader attaches it).

Nothing runs unless the relevant registry rows are ``enabled: true``. Lazy: ``plan`` is
torch-free; ``eval``/``judge``/``profile`` import torch only inside their handlers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

from qquant.config import RunConfig, Seeds
from qquant.ext.registry import (
    enabled_ext_variants,
    load_ext_tasks,
    load_ext_variants,
)

_DEFAULT_RESULTS = "results-ext"


def _resolve_lm_eval_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _run_config(args) -> RunConfig:
    return RunConfig(
        lm_eval_version=_resolve_lm_eval_version(),
        model_revision=None,
        max_length=getattr(args, "max_length", None),
        seeds=Seeds(),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qquant-ext")
    p.add_argument("--results", default=_DEFAULT_RESULTS)
    p.add_argument("--variants-yaml")
    p.add_argument("--tasks-yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="lm-eval quality sweep over enabled ext variants")
    pe.add_argument("--variant", action="append", default=[], dest="variants")
    pe.add_argument("--task", action="append", default=[], dest="tasks")
    pe.add_argument("--max-length", type=int, default=None)
    pe.add_argument("-v", "--verbose", action="store_true")

    pp = sub.add_parser("plan", help="torch-free: list missing ext cells")
    pp.add_argument("--variant", action="append", default=[], dest="variants")
    pp.add_argument("--task", action="append", default=[], dest="tasks")
    pp.add_argument("--max-length", type=int, default=None)

    pj = sub.add_parser("judge", help="EXT-2 JudgeBench meta-eval")
    pj.add_argument(
        "--registry",
        choices=["v1", "ext"],
        default="v1",
        help="which variants act as judges (v1 core quant variants, or ext variants)",
    )
    pj.add_argument("--variant", action="append", default=[], dest="variants")
    pj.add_argument("--max-length", type=int, default=None)
    pj.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap total JudgeBench pairs (taken evenly across splits) to bound cost",
    )

    pr = sub.add_parser("profile", help="EXT-4 efficiency profile of one ext variant")
    pr.add_argument("--variant", required=True)
    pr.add_argument("--device", default="cuda:0")
    return p


def _eval_manifest(args, ext_variants, tasks):
    vids = args.variants or [v.id for v in enabled_ext_variants(ext_variants)]
    tids = args.tasks or list(tasks)
    return [(v, t) for v in vids for t in tids]


def _cmd_eval(args, *, plan_only: bool) -> int:
    from qquant.eval.runner import EvalRunner
    from qquant.ext.loaders import ext_load_variant
    from qquant.registry import load_tasks

    ext_variants = load_ext_variants(args.variants_yaml)
    tasks = load_tasks(args.tasks_yaml)  # ext variants are evaluated on the v1 tasks
    run = _run_config(args)

    try:
        manifest = _eval_manifest(args, ext_variants, tasks)
    except (ValueError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    runner = EvalRunner(
        args.results,
        run,
        load_model=lambda vid, **kw: ext_load_variant(vid, variants=ext_variants, **kw),
    )

    if plan_only:
        for cell in runner.plan(manifest, ext_variants, tasks):
            print(cell.cell_id)
        return 0

    summary = runner.run_manifest(manifest, ext_variants, tasks)
    if getattr(args, "verbose", False):
        print(
            f"written={summary.cells_written} skipped={summary.cells_skipped} "
            f"loaded={summary.variants_loaded}"
        )
    for f in summary.failures:
        print(f"FAILED {f}", file=sys.stderr)
    return 1 if summary.failures else 0


def _cmd_judge(args) -> int:
    from qquant.ext.judge import JudgeRunner

    tasks = load_ext_tasks(args.tasks_yaml)
    task = tasks["judgebench"]
    run = _run_config(args)

    if args.registry == "ext":
        ext_variants = load_ext_variants(args.variants_yaml)
        from qquant.ext.loaders import ext_load_variant

        chosen = args.variants or [v.id for v in enabled_ext_variants(ext_variants)]
        variants = {vid: ext_variants[vid] for vid in chosen}
        runner = JudgeRunner(
            args.results,
            run,
            load_model=lambda vid: ext_load_variant(vid, variants=ext_variants),
            limit=args.limit,
        )
    else:  # v1 core variants act as judges (the headline RQ)
        from qquant.models import load_variant
        from qquant.registry import enabled_variants, load_variants

        v1 = load_variants(args.variants_yaml)
        chosen = args.variants or [v.id for v in enabled_variants(v1)]
        variants = {vid: v1[vid] for vid in chosen}
        runner = JudgeRunner(
            args.results,
            run,
            load_model=lambda vid: load_variant(vid, variants=v1),
            limit=args.limit,
        )

    failures: list[str] = []
    for vid in variants:
        try:
            runner.run_variant(variants[vid], task)
        except Exception as exc:  # noqa: BLE001 — record + continue, keep run resumable
            failures.append(f"{vid}: {exc!r}")
            print(f"FAILED {vid}: {exc!r}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_profile(args) -> int:
    import json

    from qquant.efficiency.profiler import ProfileConfig, profile_variant
    from qquant.efficiency.schema import efficiency_path
    from qquant.ext.loaders import ext_load_variant

    ext_variants = load_ext_variants(args.variants_yaml)
    if args.variant not in ext_variants:
        print(f"error: unknown ext variant {args.variant!r}", file=sys.stderr)
        return 2
    variant = ext_variants[args.variant]
    cfg = ProfileConfig(device=args.device)
    out_path = efficiency_path(args.results, args.variant)
    try:
        result = profile_variant(
            variant,
            cfg,
            load_model=lambda vid, **kw: ext_load_variant(
                vid, variants=ext_variants, **kw
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"profile failed for {args.variant}: {exc!r}", file=sys.stderr)
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out_path)
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "eval":
        return _cmd_eval(args, plan_only=False)
    if args.cmd == "plan":
        return _cmd_eval(args, plan_only=True)
    if args.cmd == "judge":
        return _cmd_judge(args)
    if args.cmd == "profile":
        return _cmd_profile(args)
    print(f"unknown command {args.cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
