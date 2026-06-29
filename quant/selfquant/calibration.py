"""Deterministic C4 calibration set for the controlled self-quant comparison.

The calibration set is the controlled variable: it must be reconstructible and IDENTICAL
for both algorithms. The deterministic selection + sha256 logic (``select_indices``,
``input_ids_sha256``) is pure stdlib and unit-tested off-box on a fake corpus;
``build_calibration`` imports ``datasets``/``transformers`` lazily and runs on the box.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

C4_DATASET = "allenai/c4"  # datasets 4.x: parquet config (loading scripts removed)
C4_CONFIG = "en"
C4_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 2048
CALIB_SEED = 42


@dataclass(frozen=True)
class CalibrationSpec:
    dataset: str = C4_DATASET
    config: str = C4_CONFIG
    split: str = C4_SPLIT
    num_samples: int = NUM_CALIBRATION_SAMPLES
    max_seq_len: int = MAX_SEQ_LEN
    seed: int = CALIB_SEED

    @property
    def calib_id(self) -> str:
        return f"c4-{self.num_samples}x{self.max_seq_len}-s{self.seed}"


def select_indices(doc_token_counts: Sequence[int], spec: CalibrationSpec) -> list[int]:
    """Deterministically pick spec.num_samples source-doc indices whose token count is
    >= spec.max_seq_len, in seeded-shuffle order. Pure: no datasets/torch. Raises
    ValueError if too few candidates qualify."""
    candidates = [i for i, c in enumerate(doc_token_counts) if c >= spec.max_seq_len]
    random.Random(spec.seed).shuffle(candidates)
    if len(candidates) < spec.num_samples:
        raise ValueError(
            f"only {len(candidates)} docs have >= {spec.max_seq_len} tokens; "
            f"need {spec.num_samples}"
        )
    return candidates[: spec.num_samples]


def input_ids_sha256(rows: Sequence[Sequence[int]]) -> str:
    """sha256 over concatenated input_ids (comma-joined, row-separated stream). Pure."""
    h = hashlib.sha256()
    for row in rows:
        h.update(",".join(map(str, row)).encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


def load_calibration(out_dir: str | Path) -> Any:
    """Load a previously-built tokenized calibration Dataset (load_from_disk)."""
    import datasets

    return datasets.load_from_disk(str(out_dir))


def build_calibration(
    tokenizer: Any, spec: CalibrationSpec, out_dir: str | Path
) -> Any:
    """Build (or load, if present) the deterministic tokenized C4 calibration Dataset.

    Streams C4, tokenizes a candidate POOL, selects via select_indices, truncates the
    selected docs to EXACTLY spec.max_seq_len tokens, saves the Dataset (input_ids +
    attention_mask) to out_dir/<calib_id> and writes the calib manifest beside it.
    Idempotent: a valid existing artifact is loaded and returned. Box-only.
    """
    import datasets

    out_dir = Path(out_dir)
    ds_dir = out_dir / spec.calib_id
    manifest_path = out_dir / f"{spec.calib_id}.manifest.json"
    if ds_dir.exists() and manifest_path.exists():
        return datasets.load_from_disk(str(ds_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    pool_target = spec.num_samples * 8
    stream = datasets.load_dataset(
        spec.dataset, spec.config, split=spec.split, streaming=True
    )
    texts: list[str] = []
    token_counts: list[int] = []
    for row in stream:
        text = row.get("text", "")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        texts.append(text)
        token_counts.append(len(ids))
        if (
            len(texts) >= pool_target
            and sum(1 for c in token_counts if c >= spec.max_seq_len)
            >= spec.num_samples
        ):
            break

    chosen = select_indices(token_counts, spec)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    for i in chosen:
        ids = tokenizer(texts[i], add_special_tokens=False)["input_ids"][
            : spec.max_seq_len
        ]
        input_ids.append(ids)
        attention.append([1] * len(ids))

    ds = datasets.Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention}
    )
    ds.save_to_disk(str(ds_dir))

    resolved_revision = None
    try:  # best-effort dataset fingerprint
        builder = datasets.load_dataset_builder(spec.dataset, spec.config)
        resolved_revision = getattr(builder.info, "version", None)
        resolved_revision = str(resolved_revision) if resolved_revision else None
    except Exception:
        resolved_revision = None

    manifest = {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "revision": resolved_revision,
        "num_samples": spec.num_samples,
        "max_seq_len": spec.max_seq_len,
        "seed": spec.seed,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "tokenizer_revision": getattr(tokenizer, "revision", None),
        "selected_indices": chosen,
        "calib_sha256": input_ids_sha256(input_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return ds
