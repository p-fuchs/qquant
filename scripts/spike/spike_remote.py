#!/usr/bin/env python
"""On-box GPU checks for the go/no-go spike (Spec 02).

Standalone: the ``qquant`` package never imports this; it is copied to the rented 4090
and run there. Heavy imports (torch / transformers / lm_eval) are **lazy inside each
check** so ``--help`` works on a plain machine with no GPU stack. Writes verdict JSON
(matching ``qquant.orchestrate.spike.SpikeVerdict.raw``) to ``--out`` and prints it to
stdout; exits 0 iff the verdict is GO.

``verdict == GO`` iff every *gating* check passes; ``awq_official_loads`` is reported
but non-gating (it flips ``variants.yaml`` ``awq-official.enabled``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Gating checks (all must pass for GO) + the single non-gating contingency check.
GATING_CHECKS = (
    "nvcc_matches_torch",
    "mmlu_subjects_match",
    "chat_template_renders",
    "mmlu_subtask_scored",
    "generate_until_ok",
    "bnb_loads",
    "gptq_loads",
)
NONGATING_CHECKS = ("awq_official_loads",)
ALL_CHECKS = GATING_CHECKS + NONGATING_CHECKS

# Authoritative SSOT: ``qquant.registry.MMLU_SUBJECTS`` (a CI test enforces equality).
# Embedded here because the spike box has only this standalone script, not the package.
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

LM_EVAL_REF_FILE = "/workspace/lm_eval_ref.txt"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _find_nvcc() -> str | None:
    """Resolve nvcc even when the ssh session's PATH lacks /usr/local/cuda/bin."""
    found = shutil.which("nvcc")
    if found:
        return found
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    for cand in (f"{cuda_home}/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if Path(cand).exists():
            return cand
    return None


def _nvcc_cuda() -> str:
    nvcc = _find_nvcc()
    if not nvcc:
        return ""
    proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    match = re.search(r"release (\d+\.\d+)", proc.stdout)
    return match.group(1) if match else ""


def _resolve_ref(arg_ref: str) -> str:
    """Resolved ref recorded by the bootstrap (a SHA for ``main``); else the arg."""
    path = Path(LM_EVAL_REF_FILE)
    if path.exists():
        text = path.read_text().strip()
        if text:
            return text
    return arg_ref


def _gen_one_token(model, tokenizer) -> bool:
    import torch

    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    return int(out.shape[-1]) > int(inputs["input_ids"].shape[-1])


def _known_tasks(task_manager) -> set:
    for attr in ("all_tasks",):
        value = getattr(task_manager, attr, None)
        if value:
            return set(value)
    index = getattr(task_manager, "task_index", None) or getattr(
        task_manager, "_task_index", None
    )
    return set(index or [])


# --------------------------------------------------------------------------------------
# checks (each returns bool; exceptions are caught by the runner and recorded as a note)
# --------------------------------------------------------------------------------------


def check_nvcc_matches_torch(state: dict) -> bool:
    import torch

    state["torch_version"] = torch.__version__
    torch_cuda = torch.version.cuda or ""
    state["torch_cuda"] = torch_cuda
    nvcc = _nvcc_cuda()
    state["nvcc_cuda"] = nvcc
    return (
        bool(nvcc)
        and bool(torch_cuda)
        and nvcc.split(".")[:2] == torch_cuda.split(".")[:2]
    )


def check_mmlu_subjects(args) -> bool:
    from lm_eval.tasks import TaskManager

    known = _known_tasks(TaskManager())
    expected = {f"mmlu_{s}" for s in MMLU_SUBJECTS}
    return len(expected) == 57 and expected.issubset(known)


def check_chat_template(args) -> bool:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return isinstance(prompt, str) and bool(prompt.strip())


def check_mmlu_subtask(args) -> bool:
    from lm_eval import simple_evaluate

    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},revision={args.revision},dtype=bfloat16",
        tasks=["mmlu_anatomy"],
        limit=4,
        batch_size=1,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
    )
    row = res["results"]["mmlu_anatomy"]
    acc = row.get("acc,none", row.get("acc"))
    return acc is not None and math.isfinite(float(acc))


def _patch_gsm8k_dataset_id() -> None:
    """lm-eval's gsm8k task uses the bare ``gsm8k`` id, renamed to ``openai/gsm8k``.

    datasets>=4 rejects the bare id, so rewrite it (and drop the stale revision, which
    belongs to the old repo) at the datasets layer. The real eval (Spec 05/06) must
    apply the same fix to the gsm8k task config.
    """
    import datasets

    for name in ("load_dataset", "load_dataset_builder"):
        orig = getattr(datasets, name)

        def patched(path, *a, _orig=orig, **k):
            if path == "gsm8k":
                path = "openai/gsm8k"
                k.pop("revision", None)
            return _orig(path, *a, **k)

        setattr(datasets, name, patched)


def check_generate_until(args) -> bool:
    from lm_eval import simple_evaluate

    _patch_gsm8k_dataset_id()
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},revision={args.revision},dtype=bfloat16",
        tasks=["gsm8k"],
        limit=2,
        batch_size=1,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        gen_kwargs="do_sample=False,temperature=0.0",
    )
    row = res["results"]["gsm8k"]
    em = row.get(
        "exact_match,strict-match",
        row.get("exact_match,flexible-extract", row.get("exact_match")),
    )
    return em is not None


def check_bnb_loads(args) -> bool:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    int8 = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    ok8 = _gen_one_token(int8, tok)
    del int8
    torch.cuda.empty_cache()

    nf4_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    nf4 = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        device_map="cuda:0",
        quantization_config=nf4_cfg,
    )
    ok4 = _gen_one_token(nf4, tok)
    del nf4
    torch.cuda.empty_cache()
    return ok8 and ok4


def check_gptq_loads(args) -> bool:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.gptq_model, revision=args.gptq_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.gptq_model, revision=args.gptq_revision, device_map="cuda:0"
    )
    ok = _gen_one_token(model, tok)
    del model
    torch.cuda.empty_cache()
    return ok


def check_awq_loads(args) -> bool:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.awq_model, revision=args.awq_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.awq_model, revision=args.awq_revision, device_map="cuda:0"
    )
    ok = _gen_one_token(model, tok)
    del model
    torch.cuda.empty_cache()
    return ok


def _safe(name: str, fn, checks: dict, notes: list) -> bool:
    try:
        result = bool(fn())
    except Exception as exc:  # noqa: BLE001 — a failed check is data, not a crash.
        checks[name] = False
        notes.append(f"{name}: {type(exc).__name__}: {exc}")
        return False
    checks[name] = result
    if not result:
        notes.append(f"{name}: returned False")
    return result


def _record_versions(state: dict, notes: list) -> None:
    try:
        import transformers

        state["transformers_version"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        notes.append(f"transformers import: {exc}")
    try:
        import lm_eval

        state["lm_eval_version"] = getattr(lm_eval, "__version__", "")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"lm_eval import: {exc}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spike_remote.py",
        description="On-box go/no-go GPU checks for the qquant spike.",
    )
    parser.add_argument("--lm-eval-ref", default="main")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--revision", default="a09a35458c702b33eeacc393d103063234e8bc28"
    )
    parser.add_argument("--gptq-model", default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    parser.add_argument(
        "--gptq-revision", default="e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"
    )
    parser.add_argument("--awq-model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--awq-revision", default="b25037543e9394b818fdfca67ab2a00ecc7dd641"
    )
    parser.add_argument("--out", default="/workspace/spike_verdict.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    checks: dict[str, bool] = {}
    notes: list[str] = []
    state = {
        "lm_eval_ref": _resolve_ref(args.lm_eval_ref),
        "lm_eval_version": "",
        "transformers_version": "",
        "torch_version": "",
        "torch_cuda": "",
        "nvcc_cuda": "",
    }

    # Fail fast on the toolchain: a wrong image is the cheapest thing to catch.
    if not _safe(
        "nvcc_matches_torch", lambda: check_nvcc_matches_torch(state), checks, notes
    ):
        notes.append("nvcc != torch CUDA; skipping downstream GPU checks (wrong image)")
        return _finalize(args, checks, state, notes)

    _record_versions(state, notes)
    _safe("mmlu_subjects_match", lambda: check_mmlu_subjects(args), checks, notes)
    _safe("chat_template_renders", lambda: check_chat_template(args), checks, notes)
    _safe("mmlu_subtask_scored", lambda: check_mmlu_subtask(args), checks, notes)
    _safe("generate_until_ok", lambda: check_generate_until(args), checks, notes)
    _safe("bnb_loads", lambda: check_bnb_loads(args), checks, notes)
    _safe("gptq_loads", lambda: check_gptq_loads(args), checks, notes)
    _safe("awq_official_loads", lambda: check_awq_loads(args), checks, notes)
    return _finalize(args, checks, state, notes)


def _finalize(args, checks: dict, state: dict, notes: list) -> int:
    for key in ALL_CHECKS:
        checks.setdefault(key, False)
    is_go = all(checks[k] for k in GATING_CHECKS)
    payload = {
        "verdict": "GO" if is_go else "NO-GO",
        **state,
        "checks": checks,
        "awq_official_loads": checks["awq_official_loads"],
        "notes": notes,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(args.out).write_text(text)
    print(text)
    return 0 if is_go else 1


if __name__ == "__main__":
    sys.exit(main())
