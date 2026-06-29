from __future__ import annotations

from types import SimpleNamespace

from qquant.eval.datasets import DATASET_OVERRIDES, apply_dataset_overrides


def _fake_datasets():
    calls = []

    def load_dataset(path, *args, **kwargs):
        calls.append(("load_dataset", path, args, kwargs))
        return ("ds", path, args, kwargs)

    def load_dataset_builder(path, *args, **kwargs):
        calls.append(("load_dataset_builder", path, args, kwargs))
        return ("builder", path, args, kwargs)

    return SimpleNamespace(
        load_dataset=load_dataset, load_dataset_builder=load_dataset_builder
    ), calls


def test_rewrites_bare_gsm8k_and_drops_revision():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    mod.load_dataset("gsm8k", "main", revision="deadbeef")
    name, path, args, kwargs = calls[0]
    assert path == "openai/gsm8k"
    assert args == ("main",)
    assert "revision" not in kwargs


def test_passes_through_non_overridden_ids_unchanged():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    mod.load_dataset_builder("openai/gsm8k", revision="keep")
    name, path, args, kwargs = calls[0]
    assert path == "openai/gsm8k"
    assert kwargs == {"revision": "keep"}


def test_is_idempotent():
    mod, calls = _fake_datasets()
    apply_dataset_overrides(mod)
    first = mod.load_dataset
    apply_dataset_overrides(mod)
    assert mod.load_dataset is first  # not double-wrapped


def test_overrides_table_maps_gsm8k():
    assert DATASET_OVERRIDES["gsm8k"] == "openai/gsm8k"
