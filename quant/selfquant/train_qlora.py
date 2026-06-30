"""EXT-5 (Spec 12) — QLoRA recovery trainer. Box-only; runs in the isolated quant env.

Trains a LoRA adapter on the FROZEN NF4 base (the canonical QLoRA recipe) on GENERAL
instruction data, persists it to ``checkpoints/qlora/bnb-nf4-qlora`` + a manifest, and
is idempotent/resumable (skips a present, matching adapter). The eval env later attaches
the adapter via ``qquant.ext.loaders.ext_load_variant`` (peft) over the same NF4 base.

CONTAMINATION IS THE HEADLINE RISK: the training corpus MUST be disjoint from the eval
tasks (MMLU / GSM8K / HumanEval / IFEval). ``assert_corpus_disjoint`` is a fail-fast
guard; the report must also state the corpus + the disjointness argument. Determinism
extends to a SEEDED training run.

Canonical invocation (box, quant env with peft+trl+datasets installed):
``cd quant && uv run python -m selfquant.train_qlora --dataset <general-instruct-id>``.
The exact peft/trl pins + dataset id are resolved + VERIFIED on the box at promotion
(catalog EXT-5); see docs/specs/13-ext-runbook.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Eval-task dataset ids/markers the QLoRA corpus must NEVER overlap (contamination).
FORBIDDEN_CORPUS_MARKERS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "openai/gsm8k",
    "humaneval",
    "openai_humaneval",
    "ifeval",
    "google/ifeval",
)

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
_DEFAULT_ADAPTER_DIR = "checkpoints/qlora/bnb-nf4-qlora"
_VARIANT_ID = "bnb-nf4-qlora"


@dataclass(frozen=True)
class QLoRAConfig:
    """Canonical QLoRA hyperparameters (LoRA over the frozen NF4 base)."""

    dataset_id: str
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: float = 1.0
    max_seq_length: int = 2048
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class QLoRAManifest:
    variant_id: str
    base_model_id: str
    base_revision: str
    dataset_id: str
    seed: int
    lora_r: int
    lora_alpha: int
    peft_version: str
    transformers_version: str
    created_at: str


def assert_corpus_disjoint(dataset_id: str) -> None:
    """Fail fast if the training dataset id collides with any eval-task marker."""
    lowered = dataset_id.lower()
    hit = [m for m in FORBIDDEN_CORPUS_MARKERS if m in lowered]
    if hit:
        raise ValueError(
            f"QLoRA training corpus {dataset_id!r} overlaps eval-task markers {hit}: "
            f"this would be contamination, not recovery. Use GENERAL instruction data."
        )


def adapter_done(adapter_dir: str | Path, cfg: QLoRAConfig) -> bool:
    """Resume gate: a present adapter + a manifest whose (dataset, seed, r) match."""
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        return False
    mpath = adapter_dir / "qlora_manifest.json"
    if not mpath.exists():
        return False
    try:
        m = json.loads(mpath.read_text())
    except (OSError, ValueError):
        return False
    return (
        m.get("dataset_id") == cfg.dataset_id
        and m.get("seed") == cfg.seed
        and m.get("lora_r") == cfg.lora_r
    )


def train_qlora(
    cfg: QLoRAConfig, adapter_dir: str | Path = _DEFAULT_ADAPTER_DIR
) -> Path:
    """Train + persist the LoRA adapter. Box-only (lazy torch/peft/trl/datasets)."""
    import datetime
    import importlib.metadata as md

    adapter_dir = Path(adapter_dir)
    assert_corpus_disjoint(cfg.dataset_id)
    if adapter_done(adapter_dir, cfg):
        print(f"[{_VARIANT_ID}] up-to-date adapter at {adapter_dir} (skip)")
        return adapter_dir

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(cfg.seed)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(
        f"[{_VARIANT_ID}] loading frozen NF4 base {BASE_MODEL_ID}@{BASE_REVISION[:12]}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_REVISION,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    print(f"[{_VARIANT_ID}] loading general instruction corpus {cfg.dataset_id}")
    train_ds = load_dataset(cfg.dataset_id, split="train")

    sft_cfg = SFTConfig(
        output_dir=str(adapter_dir / "_trainer"),
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_length,
        seed=cfg.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    _write_manifest(
        adapter_dir,
        QLoRAManifest(
            variant_id=_VARIANT_ID,
            base_model_id=BASE_MODEL_ID,
            base_revision=BASE_REVISION,
            dataset_id=cfg.dataset_id,
            seed=cfg.seed,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            peft_version=md.version("peft"),
            transformers_version=md.version("transformers"),
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        ),
    )
    print(f"[{_VARIANT_ID}] wrote adapter + manifest to {adapter_dir}")
    return adapter_dir


def _write_manifest(adapter_dir: Path, m: QLoRAManifest) -> None:
    path = adapter_dir / "qlora_manifest.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(m), indent=2, sort_keys=True))
    tmp.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selfquant.train_qlora")
    p.add_argument(
        "--dataset",
        required=True,
        help="HF id of a GENERAL instruction dataset (NOT any eval-task data)",
    )
    p.add_argument("--adapter-dir", default=_DEFAULT_ADAPTER_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = QLoRAConfig(
        dataset_id=args.dataset,
        seed=args.seed,
        lora_r=args.lora_r,
        num_epochs=args.epochs,
    )
    # Fail fast on contamination BEFORE any GPU work.
    assert_corpus_disjoint(cfg.dataset_id)
    train_qlora(cfg, args.adapter_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
