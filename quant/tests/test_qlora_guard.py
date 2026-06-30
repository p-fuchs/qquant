"""EXT-5 QLoRA: contamination guard + resume gate (torch-free; peft/trl box-only)."""

from __future__ import annotations

import json

import pytest
from selfquant.train_qlora import (
    QLoRAConfig,
    adapter_done,
    assert_corpus_disjoint,
)


def test_corpus_disjoint_passes_for_general_data():
    assert_corpus_disjoint("HuggingFaceH4/ultrachat_200k")  # no raise


@pytest.mark.parametrize(
    "bad",
    ["openai/gsm8k", "cais/mmlu", "openai_humaneval", "google/IFEval"],
)
def test_corpus_disjoint_blocks_eval_task_data(bad):
    with pytest.raises(ValueError, match="contamination"):
        assert_corpus_disjoint(bad)


def test_adapter_done_false_when_missing(tmp_path):
    cfg = QLoRAConfig(dataset_id="general/data")
    assert adapter_done(tmp_path, cfg) is False


def test_adapter_done_true_on_match(tmp_path):
    cfg = QLoRAConfig(dataset_id="general/data", seed=42, lora_r=16)
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "qlora_manifest.json").write_text(
        json.dumps({"dataset_id": "general/data", "seed": 42, "lora_r": 16})
    )
    assert adapter_done(tmp_path, cfg) is True


def test_adapter_done_false_on_dataset_mismatch(tmp_path):
    cfg = QLoRAConfig(dataset_id="general/data")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "qlora_manifest.json").write_text(
        json.dumps({"dataset_id": "OTHER", "seed": 42, "lora_r": 16})
    )
    assert adapter_done(tmp_path, cfg) is False
