"""Torch-free umbrella CLI: inspect the registries and audit the result matrix.

Subcommands: ``variants``, ``tasks``, ``matrix``, ``audit``, ``version``. This CLI never
imports torch; GPU-touching CLIs (eval/profile/orchestrate) are separate entry points.
"""

from __future__ import annotations

import argparse

from qquant import __version__
from qquant.matrix import expand_matrix, missing_cells
from qquant.registry import (
    TASK_IDS,
    enabled_variants,
    load_tasks,
    load_variants,
)


def _cmd_variants(args: argparse.Namespace) -> int:
    variants = load_variants()
    for vid, v in variants.items():
        flag = "on " if v.enabled else "off"
        rev = (v.revision or "-")[:12]
        cols = f"{vid:<14} {v.quant_method:<15} {v.source:<10} {rev:<12}"
        print(f"[{flag}] {cols} {v.model_id}")
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    for tid, t in tasks.items():
        kind = "per-subject" if t.per_subject else "single-cell"
        print(
            f"{tid:<10} {t.lm_eval_task:<20} metric={t.primary_metric:<22} "
            f"{kind} fewshot={t.num_fewshot} chat_template={t.apply_chat_template}"
        )
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    variants = enabled_variants(load_variants())
    tasks = list(load_tasks().values())
    cells = expand_matrix(variants, tasks)
    print(f"enabled variants : {len(variants)}")
    print(f"tasks            : {len(tasks)}")
    print(f"total cells      : {len(cells)}")
    by_task: dict[str, int] = {}
    for c in cells:
        by_task[c.task] = by_task.get(c.task, 0) + 1
    for tid in TASK_IDS:
        if tid in by_task:
            print(f"  {tid:<10} {by_task[tid]} cells")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    variants = enabled_variants(load_variants())
    tasks = list(load_tasks().values())
    cells = expand_matrix(variants, tasks)
    missing = missing_cells(cells, args.results)
    done = len(cells) - len(missing)
    print(f"results root : {args.results}")
    print(f"done         : {done}/{len(cells)}")
    print(f"missing      : {len(missing)}")
    if args.list:
        for c in missing:
            print(f"  {c.cell_id}")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qquant", description="qquant torch-free core CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("variants", help="list the variant registry").set_defaults(
        func=_cmd_variants
    )
    sub.add_parser("tasks", help="list the task registry").set_defaults(func=_cmd_tasks)
    sub.add_parser("matrix", help="show matrix cell counts").set_defaults(
        func=_cmd_matrix
    )

    p_audit = sub.add_parser("audit", help="report missing result cells")
    p_audit.add_argument("--results", default="results", help="results root directory")
    p_audit.add_argument("--list", action="store_true", help="list missing cell ids")
    p_audit.set_defaults(func=_cmd_audit)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
