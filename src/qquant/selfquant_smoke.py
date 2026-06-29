"""Kernel smoke for a self-quant checkpoint (root eval env, GPU-touching).

DIAGNOSTIC ONLY: records which W4A16 linear kernel/module class the compressed-tensors
HF backend uses on sm_89 (Marlin vs a dequant fallback), the compressed-tensors format,
a rough greedy decode_tok_s, and peak_vram_gb — to substantiate the "self-quant speed is
runtime-confounded" caveat. The AUTHORITATIVE speed number is Spec 06 (qquant-profile).

torch/qquant.models are imported lazily INSIDE functions; this module MUST NOT be
imported by the torch-free umbrella qquant.cli (run it via python -m).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _quantized_linear_class(model: Any) -> str | None:
    """Class name of a representative quantized Linear in the transformer body (e.g. a
    ``q_proj``), so the report can state which W4A16 module/kernel path fired."""
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")):
            return type(module).__name__
    return None


def run_kernel_smoke(
    variant_id: str,
    ckpt_dir: str | Path,
    *,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    out_path: str | Path | None = None,
) -> dict:
    """Load the checkpoint via the Spec-04 loader, record the W4A16 kernel diagnostic.

    Writes JSON to out_path (default ckpt_dir/kernel_smoke.json) so it travels with the
    checkpoint. Greedy, deterministic (do_sample=False); cross-checks identical output
    on two runs. Returns the recorded dict.
    """
    import time

    import torch

    from qquant.models import load_variant

    ckpt_dir = Path(ckpt_dir)
    checkpoints_root = ckpt_dir.parent.parent  # <root>/self-quant/<id> -> <root>
    out_path = (
        Path(out_path) if out_path is not None else ckpt_dir / "kernel_smoke.json"
    )

    loaded = load_variant(variant_id, checkpoints_root=checkpoints_root)
    try:
        model, tokenizer = loaded.model, loaded.tokenizer
        model.eval()
        device = next(model.parameters()).device

        gen = torch.Generator().manual_seed(0)
        vocab = int(getattr(tokenizer, "vocab_size", 32000))
        input_ids = torch.randint(0, vocab, (1, prompt_tokens), generator=gen).to(
            device
        )
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            do_sample=False,
            min_new_tokens=gen_tokens,
            max_new_tokens=gen_tokens,
            use_cache=True,
        )

        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            out1 = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out2 = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

        n_new = int(out2.shape[-1] - input_ids.shape[-1])
        qconfig = getattr(model.config, "quantization_config", None)
        ct_format = None
        if qconfig is not None:
            ct_format = (
                qconfig.get("format")
                if isinstance(qconfig, dict)
                else getattr(qconfig, "format", None)
            )

        record = {
            "variant_id": variant_id,
            "checkpoint": str(ckpt_dir),
            "module_class": _quantized_linear_class(model),
            "compressed_tensors_format": ct_format,
            "decode_tok_s": (n_new / elapsed) if elapsed > 0 else None,
            "gen_tokens": n_new,
            "prompt_tokens": prompt_tokens,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "dtype": loaded.metadata.get("dtype")
            if isinstance(loaded.metadata, dict)
            else None,
            "deterministic": bool(torch.equal(out1, out2)),
            "note": "diagnostic only; authoritative speed = Spec 06 qquant-profile. "
            "W4A16_ASYM disables Marlin on the HF generate backend (sm_89).",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return record
    finally:
        loaded.unload()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="qquant.selfquant_smoke")
    p.add_argument("--variant", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    args = p.parse_args(argv)

    record = run_kernel_smoke(
        args.variant,
        args.ckpt,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        out_path=args.out,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
