"""Manifest-driven, resumable eval runner.

lm_eval / qquant.models / datasets / langdetect are imported lazily inside
methods so importing this module stays torch-free.

When a fake ``evaluate_fn`` is injected (test mode), the runner skips
``build_hflm`` (requires lm_eval) and dataset overrides (requires datasets),
so the test suite runs on the dev laptop without any GPU dependencies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

from qquant.config import RunConfig, cell_provenance
from qquant.eval.policy import simple_evaluate_kwargs
from qquant.eval.results import build_cell, cell_inputs_from_results, write_cell
from qquant.matrix import Cell, expand_matrix, missing_cells
from qquant.paths import Paths
from qquant.registry import Task, Variant

log = logging.getLogger("qquant.eval")


@dataclass
class RunSummary:
    cells_written: int
    cells_skipped: int
    variants_loaded: int
    failures: list[str]


def code_exec_allowed() -> bool:
    """HumanEval double-env gate: both switches must be exactly '1'."""
    return (
        os.environ.get("HF_ALLOW_CODE_EVAL") == "1"
        and os.environ.get("QQUANT_ALLOW_CODE_EXEC") == "1"
    )


class EvalRunner:
    def __init__(
        self,
        results_root,
        run: RunConfig,
        load_model=None,
        evaluate_fn=None,
    ):
        self.results_root = Path(results_root)
        self.run = run
        self._load_model = load_model
        self._evaluate_fn = evaluate_fn

    # --- lazy defaults -------------------------------------------------------------

    def _load(self):
        if self._load_model is not None:
            return self._load_model
        from qquant.models import load_variant

        return load_variant

    def _evaluate(self):
        if self._evaluate_fn is not None:
            return self._evaluate_fn
        import lm_eval

        return lm_eval.simple_evaluate

    def _run_for(self, variant: Variant) -> RunConfig:
        """Per-variant provenance: model_revision = pinned SHA (None for self-quant)."""
        return replace(self.run, model_revision=variant.revision)

    # --- planning (torch-free) -----------------------------------------------------

    def plan(
        self,
        manifest,
        variants: dict[str, Variant],
        tasks: dict[str, Task],
    ) -> list[Cell]:
        out: list[Cell] = []
        for vid, tid in manifest:
            variant, task = variants[vid], tasks[tid]
            active = cell_provenance(self._run_for(variant), task, vid)
            expanded = expand_matrix([variant], [task])
            out.extend(missing_cells(expanded, self.results_root, active_config=active))
        return out

    # --- execution -----------------------------------------------------------------

    def run_variant(self, variant: Variant, tasks: list[Task]) -> list[Cell]:
        run = self._run_for(variant)
        per_task_missing: dict[str, list[Cell]] = {}
        for task in tasks:
            active = cell_provenance(run, task, variant.id)
            expanded = expand_matrix([variant], [task])
            per_task_missing[task.id] = missing_cells(
                expanded, self.results_root, active_config=active
            )
        if not any(per_task_missing.values()):
            return []  # never load a fully-present variant

        loaded = self._load()(variant.id)
        written: list[Cell] = []
        try:
            for task in tasks:
                miss = per_task_missing[task.id]
                if not miss:
                    continue
                if task.code_exec and not code_exec_allowed():
                    log.warning(
                        "skipping %s/%s: code-exec gate not set "
                        "(need HF_ALLOW_CODE_EVAL=1 and QQUANT_ALLOW_CODE_EXEC=1)",
                        variant.id,
                        task.id,
                    )
                    continue
                if task.id == "ifeval":
                    import langdetect

                    langdetect.DetectorFactory.seed = 0
                # Apply the gsm8k dataset-id rewrite only in production (when lm_eval
                # is the real evaluator); the datasets package is not installed in the
                # torch-free dev environment.
                if task.id == "gsm8k" and self._evaluate_fn is None:
                    from qquant.eval.datasets import apply_dataset_overrides

                    apply_dataset_overrides()
                written.extend(self._eval_task(variant, task, miss, run, loaded))
        finally:
            loaded.unload()
        return written

    def _eval_task(
        self,
        variant: Variant,
        task: Task,
        miss: list[Cell],
        run: RunConfig,
        loaded,
    ) -> list[Cell]:
        # In test mode (fake evaluate_fn), skip build_hflm so lm_eval/HFLM is not
        # imported — lm_eval is a gpu-only dependency unavailable on the dev laptop.
        if self._evaluate_fn is not None:
            hflm = loaded.model
        else:
            from qquant.eval.hflm import build_hflm

            hflm = build_hflm(
                loaded.model,
                loaded.tokenizer,
                batch_size=task.batch_size_for(variant.id),
                max_length=run.max_length,
            )

        lm_eval_tasks = [c.lm_eval_task for c in miss]
        kwargs = simple_evaluate_kwargs(
            task, variant.id, run, lm_eval_tasks, limit=None
        )
        results = self._evaluate()(model=hflm, **kwargs)

        written: list[Cell] = []
        for cell in miss:
            inp = cell_inputs_from_results(results, cell.lm_eval_task, task)
            meta_extra = self._meta_extra(variant, task, inp)
            doc = build_cell(
                cell=cell,
                task=task,
                variant=variant,
                run=run,
                metric_value=inp["metric_value"],
                metric_stderr=inp["metric_stderr"],
                n_samples=inp["n_samples"],
                extra_metrics=inp["extra_metrics"],
                item_correct=inp["item_correct"],
                item_ids=inp["item_ids"],
                meta_extra=meta_extra,
            )
            write_cell(doc, cell.path(self.results_root))
            written.append(cell)
        if task.id == "mmlu":
            self._write_mmlu_group(variant, results)
        return written

    def _meta_extra(self, variant: Variant, task: Task, inp: dict) -> dict:
        meta: dict = {
            "lm_eval_version": self.run.lm_eval_version,
            "model_revision": variant.revision,
            "quant_method": variant.quant_method,
            "source": variant.source,
        }
        if task.primary_metric == "pass@1":
            from qquant.eval.metrics import wilson_interval

            k, n = sum(inp["item_correct"]), inp["n_samples"]
            meta["wilson_ci"] = list(wilson_interval(k, n)) if n else None
            if variant.id == "bf16" and inp["metric_value"] <= 0.5:
                meta["sanity_failed"] = True
                log.warning(
                    "bf16 %s pass@1=%.3f <= 0.5 (sanity gate)",
                    task.id,
                    inp["metric_value"],
                )
        return meta

    def _write_mmlu_group(self, variant: Variant, results: dict) -> None:
        import json as _json

        group = results.get("results", {}).get("mmlu", {})
        if not group:
            return
        meta_dir = Paths.from_root(self.results_root).meta_dir
        meta_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": variant.id,
            "acc": group.get("acc,none", group.get("acc")),
            "acc_stderr": group.get("acc_stderr,none", group.get("acc_stderr")),
            "n_subjects": 57,
            "lm_eval_version": self.run.lm_eval_version,
        }
        (meta_dir / f"mmlu_group_{variant.id}.json").write_text(
            _json.dumps(payload, indent=2, sort_keys=True)
        )

    def run_manifest(
        self,
        manifest,
        variants: dict[str, Variant],
        tasks: dict[str, Task],
    ) -> RunSummary:
        written = skipped = loaded = 0
        failures: list[str] = []
        by_variant: dict[str, list[Task]] = {}
        for vid, tid in manifest:
            by_variant.setdefault(vid, []).append(tasks[tid])
        for vid, task_list in by_variant.items():
            try:
                before = self.plan([(vid, t.id) for t in task_list], variants, tasks)
                if not before:
                    continue
                loaded += 1
                cells = self.run_variant(variants[vid], task_list)
                written += len(cells)
                skipped += max(0, len(before) - len(cells))
            except Exception as exc:  # noqa: BLE001 — record + continue, keep run resumable
                failures.append(f"{vid}: {exc!r}")
                log.exception("variant %s failed", vid)
        return RunSummary(written, skipped, loaded, failures)

    def run_smoke(self, variant: Variant, task: Task, *, limit: int) -> dict:
        """Eval (variant, task) at ``limit``, build+validate cells IN MEMORY.

        Nothing is written. Backs ``--limit`` (smoke) and ``--determinism-check``
        (run twice, compare). Loads the model once.
        """
        run = self._run_for(variant)
        if task.code_exec and not code_exec_allowed():
            raise RuntimeError(f"{task.id}: code-exec env gate not set")
        if task.id == "ifeval":
            import langdetect

            langdetect.DetectorFactory.seed = 0
        if task.id == "gsm8k" and self._evaluate_fn is None:
            from qquant.eval.datasets import apply_dataset_overrides

            apply_dataset_overrides()
        cells = expand_matrix([variant], [task])
        loaded = self._load()(variant.id)
        try:
            if self._evaluate_fn is not None:
                hflm = loaded.model
            else:
                from qquant.eval.hflm import build_hflm

                hflm = build_hflm(
                    loaded.model,
                    loaded.tokenizer,
                    batch_size=task.batch_size_for(variant.id),
                    max_length=run.max_length,
                )
            lm_eval_tasks = [c.lm_eval_task for c in cells]
            kwargs = simple_evaluate_kwargs(
                task, variant.id, run, lm_eval_tasks, limit=limit
            )
            results = self._evaluate()(model=hflm, **kwargs)
            out = []
            for cell in cells:
                inp = cell_inputs_from_results(results, cell.lm_eval_task, task)
                build_cell(  # validates only — no write
                    cell=cell,
                    task=task,
                    variant=variant,
                    run=run,
                    metric_value=inp["metric_value"],
                    metric_stderr=inp["metric_stderr"],
                    n_samples=inp["n_samples"],
                    extra_metrics=inp["extra_metrics"],
                    item_correct=inp["item_correct"],
                    item_ids=inp["item_ids"],
                    meta_extra=self._meta_extra(variant, task, inp),
                )
                out.append(
                    {
                        "cell_id": cell.cell_id,
                        "metric_value": inp["metric_value"],
                        "item_correct": inp["item_correct"],
                    }
                )
            return {"cells": out}
        finally:
            loaded.unload()
