"""qquant-eval entry point. Lazy: only --plan/--list-missing run torch-free."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

from qquant.config import RunConfig, Seeds
from qquant.eval.manifest import default_manifest, load_manifest
from qquant.eval.runner import EvalRunner
from qquant.registry import enabled_variants, load_tasks, load_variants


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qquant-eval")
    p.add_argument("--results", default="results")
    p.add_argument("--manifest")
    p.add_argument("--variant", action="append", default=[], dest="variants")
    p.add_argument("--task", action="append", default=[], dest="tasks")
    p.add_argument("--variants-yaml")
    p.add_argument("--tasks-yaml")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed-random", type=int, default=0)
    p.add_argument("--seed-numpy", type=int, default=1234)
    p.add_argument("--seed-torch", type=int, default=1234)
    p.add_argument("--seed-fewshot", type=int, default=1234)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--list-missing", action="store_true")
    p.add_argument("--determinism-check", nargs=2, metavar=("VARIANT", "TASK"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_lm_eval_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _manifest_from_args(args, variants, tasks) -> list[tuple[str, str]]:
    if args.manifest:
        return load_manifest(args.manifest)
    if args.variants or args.tasks:
        vids = args.variants or [v.id for v in enabled_variants(variants)]
        tids = args.tasks or list(tasks)
        return [(v, t) for v in vids for t in tids]
    return default_manifest(enabled_variants(variants), list(tasks.values()))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    variants = load_variants(args.variants_yaml)
    tasks = load_tasks(args.tasks_yaml)
    run = RunConfig(
        lm_eval_version=_resolve_lm_eval_version(),
        model_revision=None,
        max_length=args.max_length,
        seeds=Seeds(
            random=args.seed_random,
            numpy=args.seed_numpy,
            torch=args.seed_torch,
            fewshot=args.seed_fewshot,
        ),
    )
    try:
        manifest = _manifest_from_args(args, variants, tasks)
    except (ValueError, KeyError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    runner = EvalRunner(args.results, run)

    if args.plan or args.list_missing:
        for cell in runner.plan(manifest, variants, tasks):
            print(cell.cell_id)
        return 0

    if args.determinism_check:
        vid, tid = args.determinism_check
        limit = args.limit if args.limit is not None else 8
        first = runner.run_smoke(variants[vid], tasks[tid], limit=limit)
        second = runner.run_smoke(variants[vid], tasks[tid], limit=limit)
        ok = first == second
        print("determinism: OK" if ok else "determinism: MISMATCH")
        return 0 if ok else 1

    if args.limit is not None:  # smoke: build + validate in memory, never persist
        for vid, tid in manifest:
            runner.run_smoke(variants[vid], tasks[tid], limit=args.limit)
        return 0

    summary = runner.run_manifest(manifest, variants, tasks)
    if args.verbose:
        print(
            f"written={summary.cells_written} skipped={summary.cells_skipped} "
            f"loaded={summary.variants_loaded}"
        )
    for f in summary.failures:
        print(f"FAILED {f}", file=sys.stderr)
    return 1 if summary.failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
