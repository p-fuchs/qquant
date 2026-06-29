from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from types import SimpleNamespace

from qquant.config import RunConfig, Seeds, cell_provenance
from qquant.eval.runner import EvalRunner, RunSummary
from qquant.matrix import expand_matrix
from qquant.paths import cell_path
from qquant.registry import MMLU_SUBJECTS, load_tasks, load_variants


def _fake_datasets_module():
    """Minimal fake datasets module accepted by apply_dataset_overrides."""
    calls = []

    def load_dataset(path, *args, **kwargs):
        calls.append(("load_dataset", path, args, kwargs))
        return object()

    def load_dataset_builder(path, *args, **kwargs):
        calls.append(("load_dataset_builder", path, args, kwargs))
        return object()

    mod = SimpleNamespace(
        load_dataset=load_dataset, load_dataset_builder=load_dataset_builder
    )
    mod._calls = calls
    return mod


def _run():
    return RunConfig(
        lm_eval_version="lmv", model_revision=None, max_length=4096, seeds=Seeds()
    )


def _results_for(lm_eval_tasks, primary, value=0.5):
    """Fake lm-eval output covering the given subtasks with 2 samples each.

    Uses the bare primary key (no filter suffix) so extract_metric hits the
    bare-name fallback that every task's metric_keys list includes.
    """
    return {
        "results": {
            t: {f"{primary}": value, f"{primary}_stderr,none": 0.01}
            for t in lm_eval_tasks
        },
        "samples": {
            t: [{"doc_id": 0, primary: 1}, {"doc_id": 1, primary: 0}]
            for t in lm_eval_tasks
        },
    }


def test_plan_returns_missing_cells_only(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    runner = EvalRunner(tmp_path, _run(), load_model=None, evaluate_fn=None)
    cells = runner.plan([("bf16", "gsm8k")], variants, tasks)
    assert [c.cell_id for c in cells] == ["bf16/gsm8k"]


def test_run_variant_skips_load_when_all_present(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    run = _run()
    # Pre-write a done gsm8k cell with matching provenance.
    # Provenance must use the actual bf16 revision (EvalRunner._run_for sets it).
    cell = expand_matrix([variants["bf16"]], [tasks["gsm8k"]])[0]
    run_for_bf16 = dc_replace(run, model_revision=variants["bf16"].revision)
    prov = cell_provenance(run_for_bf16, tasks["gsm8k"], "bf16")
    doc = {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "variant": "bf16",
        "task": "gsm8k",
        "primary_metric": "exact_match",
        "metric_value": 0.5,
        "n_samples": 2,
        "config": prov,
        "meta": {"item_correct": [1, 0], "item_ids": [0, 1], "n_correct": 1},
    }
    p = cell.path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))

    calls = {"load": 0}

    def fake_load(vid):
        calls["load"] += 1
        raise AssertionError("must not load when nothing is missing")

    runner = EvalRunner(tmp_path, run, load_model=fake_load, evaluate_fn=None)
    written = runner.run_variant(variants["bf16"], [tasks["gsm8k"]])
    assert written == []
    assert calls["load"] == 0


def test_run_manifest_loads_once_and_writes_cells(tmp_path):
    variants = load_variants()
    tasks = load_tasks()
    load_calls = []

    class FakeLoaded:
        def __init__(self, vid):
            self.model = object()
            self.tokenizer = object()
            self.metadata = {"model_revision": variants[vid].revision}

        def unload(self):
            pass

    def fake_load(vid):
        load_calls.append(vid)
        return FakeLoaded(vid)

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "exact_match")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=fake_load,
        evaluate_fn=fake_eval,
        datasets_module=_fake_datasets_module(),
    )
    summary = runner.run_manifest([("bf16", "gsm8k")], variants, tasks)
    assert isinstance(summary, RunSummary)
    assert load_calls == ["bf16"]  # loaded exactly once
    cell_doc = json.loads(cell_path(tmp_path, "bf16", "gsm8k").read_text())
    assert cell_doc["metric_value"] == 0.5
    assert cell_doc["meta"]["item_correct"] == [1, 0]


def test_mmlu_writes_57_cells_with_item_vectors(tmp_path):
    variants = load_variants()
    tasks = load_tasks()

    class FakeLoaded:
        model = object()
        tokenizer = object()
        metadata = {"model_revision": variants["bf16"].revision}

        def unload(self):
            pass

    def fake_eval(model=None, tasks=None, **kw):
        res = _results_for(tasks, "acc")
        # Add the MMLU group aggregate so _write_mmlu_group is exercised.
        res["results"]["mmlu"] = {"acc,none": 0.7, "acc_stderr,none": 0.01}
        return res

    runner = EvalRunner(
        tmp_path, _run(), load_model=lambda vid: FakeLoaded(), evaluate_fn=fake_eval
    )
    written = runner.run_variant(variants["bf16"], [tasks["mmlu"]])
    subjects = {c.subject for c in written}
    assert subjects == set(MMLU_SUBJECTS)
    assert len(written) == 57
    one = json.loads(cell_path(tmp_path, "bf16", "mmlu", "anatomy").read_text())
    assert one["task_lm_eval"] == "mmlu_anatomy"
    assert one["meta"]["item_correct"] == [1, 0]
    # Verify the group aggregate file was written.
    group_file = tmp_path / "_meta" / "mmlu_group_bf16.json"
    assert group_file.exists(), "_meta/mmlu_group_bf16.json was not written"
    group_doc = json.loads(group_file.read_text())
    assert group_doc["variant"] == "bf16"
    assert group_doc["acc"] == 0.7
    assert group_doc["acc_stderr"] == 0.01
    assert group_doc["n_subjects"] == 57
    assert group_doc["lm_eval_version"] == _run().lm_eval_version


def test_code_exec_skipped_unless_both_env_set(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    monkeypatch.delenv("QQUANT_ALLOW_CODE_EXEC", raising=False)

    def fake_eval(model=None, tasks=None, **kw):
        raise AssertionError("must not eval a gated code-exec task")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=lambda vid: _FakeLoaded(variants, vid),
        evaluate_fn=fake_eval,
    )
    written = runner.run_variant(variants["bf16"], [tasks["humaneval"]])
    assert written == []  # left missing, not errored


def test_code_exec_runs_and_passes_confirm_when_gated(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    monkeypatch.setenv("QQUANT_ALLOW_CODE_EXEC", "1")
    seen = {}

    def fake_eval(model=None, tasks=None, **kw):
        seen.update(kw)
        return _results_for(tasks, "pass@1")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=lambda vid: _FakeLoaded(variants, vid),
        evaluate_fn=fake_eval,
    )
    runner.run_variant(variants["bf16"], [tasks["humaneval"]])
    assert seen.get("confirm_run_unsafe_code") is True


class _FakeLoaded:
    def __init__(self, variants, vid):
        self.model = object()
        self.tokenizer = object()
        self.metadata = {"model_revision": variants[vid].revision}

    def unload(self):
        pass


def test_ifeval_seeds_langdetect(tmp_path, monkeypatch):
    variants = load_variants()
    tasks = load_tasks()
    import types

    fake_ld = types.SimpleNamespace(DetectorFactory=types.SimpleNamespace(seed=None))
    monkeypatch.setitem(__import__("sys").modules, "langdetect", fake_ld)

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "prompt_level_strict_acc")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=lambda vid: _FakeLoaded(variants, vid),
        evaluate_fn=fake_eval,
    )
    runner.run_variant(variants["bf16"], [tasks["ifeval"]])
    assert fake_ld.DetectorFactory.seed == 0


def test_run_smoke_builds_in_memory_without_writing(tmp_path):
    variants = load_variants()
    tasks = load_tasks()

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "exact_match")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=lambda vid: _FakeLoaded(variants, vid),
        evaluate_fn=fake_eval,
        datasets_module=_fake_datasets_module(),
    )
    out = runner.run_smoke(variants["bf16"], tasks["gsm8k"], limit=8)
    assert out["cells"][0]["metric_value"] == 0.5
    assert out["cells"][0]["item_correct"] == [1, 0]
    # nothing persisted to the results tree
    assert not (tmp_path / "bf16").exists()


def test_gsm8k_runner_applies_dataset_overrides(tmp_path):
    """Runner must call apply_dataset_overrides() before any gsm8k eval.

    Injects a fake datasets module and proves the patch flag was set and the
    gsm8k->openai/gsm8k rewrite is active after run_variant completes.
    """
    from qquant.eval.datasets import _PATCH_FLAG

    variants = load_variants()
    tasks = load_tasks()
    fake_ds = _fake_datasets_module()

    def fake_eval(model=None, tasks=None, **kw):
        return _results_for(tasks, "exact_match")

    runner = EvalRunner(
        tmp_path,
        _run(),
        load_model=lambda vid: _FakeLoaded(variants, vid),
        evaluate_fn=fake_eval,
        datasets_module=fake_ds,
    )
    runner.run_variant(variants["bf16"], [tasks["gsm8k"]])

    # The patch flag must be set — proves apply_dataset_overrides was invoked.
    assert getattr(fake_ds, _PATCH_FLAG, False), (
        "apply_dataset_overrides was not called"
    )

    # Calling the patched loader with the bare id must rewrite to openai/gsm8k.
    fake_ds.load_dataset("gsm8k", "main")
    last_call = fake_ds._calls[-1]
    assert last_call[1] == "openai/gsm8k", (
        f"gsm8k was not rewritten; got path={last_call[1]!r}"
    )
