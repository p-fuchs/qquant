from __future__ import annotations

from pathlib import Path

from qquant.paths import Paths, cell_path


def test_cell_path_subject():
    p = cell_path("results", "bf16", "mmlu", "anatomy")
    assert p == Path("results/bf16/mmlu/anatomy.json")


def test_cell_path_single_cell():
    p = cell_path("results", "gptq-official", "gsm8k")
    assert p == Path("results/gptq-official/gsm8k/result.json")


def test_paths_meta_and_helpers():
    paths = Paths.from_root("out")
    assert paths.env_file == Path("out/_meta/env.json")
    assert paths.done_manifest == Path("out/_meta/done_manifest.json")
    assert paths.cell("bf16", "ifeval") == Path("out/bf16/ifeval/result.json")
