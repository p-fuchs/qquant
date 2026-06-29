from __future__ import annotations

import json

import pytest

from qquant.eval.manifest import default_manifest, load_manifest
from qquant.registry import enabled_variants, load_tasks, load_variants


def test_load_manifest_parses_pairs(tmp_path):
    p = tmp_path / "m.json"
    data = {
        "cells": [
            {"variant": "bf16", "task": "mmlu"},
            {"variant": "gptq-official", "task": "gsm8k"},
        ]
    }
    p.write_text(json.dumps(data))
    assert load_manifest(p) == [("bf16", "mmlu"), ("gptq-official", "gsm8k")]


def test_load_manifest_rejects_unknown_ids(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"cells": [{"variant": "nope", "task": "mmlu"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)
    p.write_text(json.dumps({"cells": [{"variant": "bf16", "task": "nope"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)


def test_load_manifest_rejects_duplicate_pairs(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            {
                "cells": [
                    {"variant": "bf16", "task": "mmlu"},
                    {"variant": "bf16", "task": "mmlu"},
                ]
            }
        )
    )
    with pytest.raises(ValueError):
        load_manifest(p)


def test_default_manifest_is_enabled_matrix():
    variants = load_variants()
    tasks = load_tasks()
    enabled = enabled_variants(variants)
    task_list = list(tasks.values())
    pairs = default_manifest(enabled, task_list)
    assert len(pairs) == len(enabled) * len(task_list)
    assert ("bf16", "mmlu") in pairs
