"""EXT-2 JudgeBench runner — the quantized variant judges response pairs.

lm-eval ships no JudgeBench task, so this is a custom runner (NOT
``lm_eval.simple_evaluate``). The variant is shown a question and two candidate
responses and must pick the better one; we score **agreement with the gold label**
(pairwise judge accuracy), with position-bias control (run both A/B orders and require
consistency). Each variant's result is written as a STANDARD v1 cell via
``build_cell`` / ``write_cell`` — ``meta.item_correct`` / ``meta.item_ids`` make
paired McNemar (vs ``bf16``) work with zero stats changes.

The pure functions (prompt render, verdict parse, pair scoring) are torch-free and unit
tested on the laptop; only generation + dataset load touch the box (lazy imports).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qquant.config import RunConfig, cell_provenance
from qquant.eval.results import build_cell, write_cell
from qquant.ext.registry import ExtTask, ExtVariant
from qquant.matrix import Cell, expand_matrix

_JUDGE_INSTRUCTION = (
    "You are an impartial judge. Read the question and the two candidate answers, "
    "then decide which answer is better (more correct and helpful). "
    "Respond with exactly one character: 'A' or 'B'. Do not explain."
)


@dataclass(frozen=True)
class JudgeItem:
    """One JudgeBench pair: ``label`` is the correct response in the ORIGINAL order."""

    id: Any
    question: str
    response_a: str
    response_b: str
    label: str  # "A" or "B"


def render_judge_prompt(question: str, response_a: str, response_b: str) -> str:
    """Render the judge prompt for a single (question, A, B) triple (deterministic)."""
    return (
        f"{_JUDGE_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer A:\n{response_a}\n\n"
        f"Answer B:\n{response_b}\n\n"
        f"Better answer (A or B):"
    )


_VERDICT_RE = re.compile(r"\b([AB])\b")


def parse_verdict(text: str) -> str | None:
    """Extract 'A' or 'B' from a model verdict; None if neither is found.

    Takes the FIRST standalone A/B token (case-insensitive). Robust to a short leading
    rationale (instruct models sometimes add a word), but greedy decoding +
    ``max_gen_toks`` keep verdicts short.
    """
    if not text:
        return None
    m = _VERDICT_RE.search(text.upper())
    return m.group(1) if m else None


def _swapped_to_original(verdict_swapped: str | None) -> str | None:
    """Map a verdict given in (B,A) order back to the original A/B labelling."""
    if verdict_swapped == "A":
        return "B"
    if verdict_swapped == "B":
        return "A"
    return None


@dataclass(frozen=True)
class PairScore:
    correct: int  # 1 if the judged-better response matches gold, else 0
    consistent: bool  # both orders agreed on the same original response
    decided: str | None  # the original-order response the judge settled on (A/B/None)


def score_pair(
    verdict_first: str | None,
    verdict_swapped: str | None,
    gold: str,
    *,
    position_bias_control: bool,
) -> PairScore:
    """Score one pair.

    ``verdict_first`` is the A/B verdict in the original order; ``verdict_swapped``
    is the A/B verdict when responses were shown in (B,A) order (mapped back here).
    With position-bias control the pair counts correct ONLY if both orders agree AND the
    agreed response equals ``gold`` — a position-biased (inconsistent) judge scores 0.
    Without control, only ``verdict_first`` is used.
    """
    if not position_bias_control:
        decided = verdict_first
        return PairScore(
            correct=int(decided == gold),
            consistent=True,
            decided=decided,
        )
    mapped = _swapped_to_original(verdict_swapped)
    consistent = verdict_first is not None and verdict_first == mapped
    decided = verdict_first if consistent else None
    return PairScore(
        correct=int(consistent and decided == gold),
        consistent=consistent,
        decided=decided,
    )


# A generate function maps a prompt string to the model's decoded completion.
GenerateFn = Callable[[str], str]


class JudgeRunner:
    """Resumable JudgeBench runner. Generation/dataset injected or lazily imported."""

    def __init__(
        self,
        results_root: str | Path,
        run: RunConfig,
        *,
        load_model: Callable[[str], Any] | None = None,
        generate_fn: Callable[[Any, Any], GenerateFn] | None = None,
        dataset_fn: Callable[[ExtTask], list[JudgeItem]] | None = None,
    ):
        self.results_root = Path(results_root)
        self.run = run
        self._load_model = load_model
        self._generate_fn = generate_fn
        self._dataset_fn = dataset_fn

    # --- injectable defaults (box-only) --------------------------------------------

    def _load(self):
        if self._load_model is not None:
            return self._load_model
        from qquant.ext.loaders import ext_load_variant

        return ext_load_variant

    def _dataset(self, task: ExtTask) -> list[JudgeItem]:
        if self._dataset_fn is not None:
            return self._dataset_fn(task)
        return load_judgebench(task)

    def _greedy_generate(self, loaded: Any, task: ExtTask) -> GenerateFn:
        """Return a prompt->text greedy generator over the loaded model (box-only)."""
        if self._generate_fn is not None:
            return self._generate_fn(loaded, task)
        import torch

        model, tokenizer = loaded.model, loaded.tokenizer
        device = next(model.parameters()).device
        max_new = task.max_gen_toks or 64

        def _gen(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model.generate(enc, max_new_tokens=max_new, do_sample=False)
            return tokenizer.decode(out[0][enc.shape[-1] :], skip_special_tokens=True)

        return _gen

    # --- scoring -------------------------------------------------------------------

    def score_items(
        self, items: list[JudgeItem], gen: GenerateFn, task: ExtTask
    ) -> dict:
        """Run the judge over all items; return build_cell inputs."""
        item_correct: list[int] = []
        item_ids: list[Any] = []
        consistent_count = 0
        for item in items:
            v_first = parse_verdict(
                gen(
                    render_judge_prompt(item.question, item.response_a, item.response_b)
                )
            )
            v_swapped = None
            if task.position_bias_control:
                v_swapped = parse_verdict(
                    gen(
                        render_judge_prompt(
                            item.question, item.response_b, item.response_a
                        )
                    )
                )
            ps = score_pair(
                v_first,
                v_swapped,
                item.label,
                position_bias_control=task.position_bias_control,
            )
            item_correct.append(ps.correct)
            item_ids.append(item.id)
            consistent_count += int(ps.consistent)
        n = len(items)
        return {
            "metric_value": (sum(item_correct) / n) if n else 0.0,
            "n_samples": n,
            "item_correct": item_correct,
            "item_ids": item_ids,
            "consistency_rate": (consistent_count / n) if n else None,
        }

    def run_variant(self, variant: ExtVariant, task: ExtTask) -> Cell | None:
        """Evaluate one variant on JudgeBench; write its cell. Returns the Cell or None.

        Resumable: skips when the cell already exists with matching provenance.
        """
        from dataclasses import replace

        run = replace(self.run, model_revision=variant.revision)
        (cell,) = expand_matrix([variant], [task])
        path = cell.path(self.results_root)
        active = cell_provenance(run, task, variant.id)
        if path.exists():
            import json

            try:
                loaded_doc = json.loads(path.read_text())
                from qquant.matrix import is_cell_done

                if is_cell_done(loaded_doc, active):
                    return None
            except (OSError, ValueError):
                pass

        loaded = self._load()(variant.id)
        try:
            items = self._dataset(task)
            gen = self._greedy_generate(loaded, task)
            inp = self.score_items(items, gen, task)
        finally:
            loaded.unload()

        doc = build_cell(
            cell=cell,
            task=task,
            variant=variant,
            run=run,
            metric_value=inp["metric_value"],
            metric_stderr=None,
            n_samples=inp["n_samples"],
            extra_metrics={"consistency_rate": inp["consistency_rate"]},
            item_correct=inp["item_correct"],
            item_ids=inp["item_ids"],
            meta_extra={
                "quant_method": variant.quant_method,
                "source": variant.source,
                "position_bias_control": task.position_bias_control,
                "consistency_rate": inp["consistency_rate"],
            },
        )
        write_cell(doc, path)
        return cell


def load_judgebench(task: ExtTask) -> list[JudgeItem]:
    """Load JudgeBench from HF into ``JudgeItem``s (box-only; lazy ``datasets`` import).

    The exact dataset id, split, and field names are VERIFIED ON THE BOX at promotion
    (catalog EXT-2). This maps the dataset's pairwise records into the project's
    ``JudgeItem`` shape; adjust the field accessors once the real schema is confirmed.
    """
    from datasets import load_dataset

    dataset_id = task.hf_dataset_id
    if not dataset_id:
        raise ValueError("judgebench task has no hf_dataset_id set")
    ds = load_dataset(dataset_id, split="train")
    items: list[JudgeItem] = []
    for i, row in enumerate(ds):
        label = _normalize_label(row)
        if label is None:  # skip ties / unlabelled
            continue
        items.append(
            JudgeItem(
                id=row.get("id", i),
                question=row.get("question") or row.get("prompt") or "",
                response_a=row.get("response_A")
                or row.get("response_a")
                or row.get("answer_A")
                or "",
                response_b=row.get("response_B")
                or row.get("response_b")
                or row.get("answer_B")
                or "",
                label=label,
            )
        )
    return items


def _normalize_label(row: dict) -> str | None:
    """Map a JudgeBench gold field to 'A'/'B'; None for tie/unknown (box-verify)."""
    raw = row.get("label") or row.get("winner") or row.get("gold")
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if s in ("A", "RESPONSE_A", "0"):
        return "A"
    if s in ("B", "RESPONSE_B", "1"):
        return "B"
    return None
