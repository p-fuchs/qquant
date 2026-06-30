"""qquant.efficiency.profiler — measurement engine.

The torch-free core (ProfileConfig, median, disk sizing, self-quant labeling, the
``build_efficiency_result`` assembler) is unit-tested off-GPU. The torch-touching
measurements live behind ``TorchEngine`` (added in Task 4); torch is imported only
inside its methods, so importing this module stays torch-free.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qquant.efficiency.schema import efficiency_active_config, validate_efficiency

_SPEED_METRICS = ["prefill_tok_s", "decode_tok_s", "e2e_latency_s", "ttft_s"]
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_GIB = float(1024**3)


@dataclass(frozen=True)
class ProfileConfig:
    device: str = "cuda:0"
    warmup: int = 2
    repeats: int = 5
    decode_tokens: int = 256
    prompt_buckets: dict[str, int] = field(
        default_factory=lambda: {"short": 128, "medium": 1024, "long": 4096}
    )
    batch_seq_len: int = 2048
    batch_gen_tokens: int = 64
    batch_ceiling: int = 64


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def speed_label_and_confounded(variant: Any) -> tuple[str, list[str]]:
    """Self-quant speed is runtime-confounded (HF generate dequant path on sm_89).

    disk and memory are NEVER confounded — only the speed metrics.
    """
    if variant.source == "selfquant":
        return "runtime-confounded", list(_SPEED_METRICS)
    return "comparable", []


def sum_weight_bytes(snapshot_dir: str | Path) -> dict:
    """Sum real sizes (symlinks resolved) of *.safetensors/*.bin under snapshot_dir.

    Also returns the full resolved snapshot size.
    """
    root = Path(snapshot_dir)
    weights = 0
    snapshot = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size  # Path.stat follows symlinks
        snapshot += size
        if p.name.endswith(_WEIGHT_SUFFIXES):
            weights += size
    return {"weights_bytes": weights, "snapshot_bytes": snapshot}


def measure_disk_size(variant: Any, *, snapshot_dir: str | Path | None = None) -> dict:
    """Disk footprint for a variant.

    ``snapshot_dir`` injects the resolved dir (tests); else it is resolved from the
    HF cache (repo id) or the ``local:`` checkpoint dir (self-quant).
    """
    if snapshot_dir is not None:
        source = "local" if str(variant.model_id).startswith("local:") else "hf-cache"
        out = sum_weight_bytes(snapshot_dir)
        out["source"] = source
        out["path"] = str(snapshot_dir)
        out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
        return out
    if str(variant.model_id).startswith("local:"):
        local = Path(str(variant.model_id).split("local:", 1)[1])
        out = sum_weight_bytes(local)
        out["source"] = "local"
        out["path"] = str(local)
    else:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            variant.model_id, revision=variant.revision, local_files_only=True
        )
        out = sum_weight_bytes(resolved)
        out["source"] = "hf-cache"
        out["path"] = str(resolved)
    out["weights_gib"] = round(out["weights_bytes"] / _GIB, 2)
    return out


def build_efficiency_result(
    *,
    variant: Any,
    cfg: ProfileConfig,
    runtime: dict,
    disk: dict,
    memory: dict,
    throughput: dict,
    max_batch: dict,
    env_meta: dict | None = None,
    samples: dict | None = None,
    meta_extra: dict | None = None,
) -> dict:
    """Assemble + validate a v1 efficiency result.

    ``config`` carries the comparable provenance block (EFFICIENCY_PROVENANCE_KEYS) plus
    the extra reproducibility keys. Does not persist.
    """
    speed_label, confounded = speed_label_and_confounded(variant)
    active = efficiency_active_config(cfg, variant, runtime)
    config = dict(active) | {
        "device": cfg.device,
        "expandable_segments": runtime.get("expandable_segments"),
        "do_sample": False,
        "warmup": cfg.warmup,
        "repeats": cfg.repeats,
        "batch_gen_tokens": cfg.batch_gen_tokens,
    }
    meta = {
        "timestamp": runtime.get("timestamp"),
        "hostname": runtime.get("hostname"),
    }
    if env_meta:
        meta["env"] = env_meta
    if meta_extra:
        meta.update(meta_extra)
    result = {
        "schema_version": 1,
        "variant": variant.id,
        "quant_method": variant.quant_method,
        "source": variant.source,
        "model_id": variant.model_id,
        "model_revision": variant.revision,
        "speed_label": speed_label,
        "confounded_metrics": confounded,
        "disk": disk,
        "memory": memory,
        "throughput": throughput,
        "max_batch_size": max_batch,
        "config": config,
        "meta": meta,
    }
    if samples is not None:
        result["samples"] = samples
    validate_efficiency(result)
    return result


def _default_load_model(variant_id: str, **kwargs: Any):
    from qquant.models import load_variant

    return load_variant(variant_id, **kwargs)


class TorchEngine:
    """Real torch-touching measurements.

    torch is imported INSIDE each method so importing this module stays torch-free.
    All timing is greedy, sync-bounded, under inference_mode.
    """

    def disk(self, variant: Any) -> dict:
        return measure_disk_size(variant)

    def runtime_facts(self, loaded: Any, cfg: ProfileConfig) -> dict:
        import datetime
        import socket

        import torch
        import transformers

        if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            import warnings

            warnings.warn(
                "PYTORCH_CUDA_ALLOC_CONF lacks expandable_segments:True "
                "(must be set before CUDA init by the launcher; "
                "recording observed value)",
                stacklevel=2,
            )
        meta = getattr(loaded, "metadata", {}) or {}
        return {
            "torch_version": torch.__version__,
            "transformers_version": meta.get("transformers_version")
            or transformers.__version__,
            "cuda_version": torch.version.cuda or "",
            "gpu_name": torch.cuda.get_device_name(cfg.device),
            "dtype": meta.get("dtype") or "",
            "expandable_segments": "expandable_segments"
            in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "hostname": socket.gethostname(),
        }

    def load_and_measure(self, load_model: Any, variant: Any, cfg: ProfileConfig):
        import torch

        # Initialize the CUDA context first: on a fresh process, reset_peak_memory_stats
        # as the very first CUDA call raises "Invalid device argument" (the device's
        # context/stats tracking isn't set up until CUDA is initialized).
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cfg.device)
        loaded = load_model(variant.id, device_map={"": int(cfg.device.split(":")[-1])})
        mem = {
            "weights_resident_bytes": int(torch.cuda.memory_allocated(cfg.device)),
            "load_peak_bytes": int(torch.cuda.max_memory_allocated(cfg.device)),
            "load_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cfg.device)),
        }
        return loaded, mem

    def _make_inputs(self, tokenizer: Any, n_tokens: int, batch: int, device: str):
        import torch

        vocab = int(getattr(tokenizer, "vocab_size", 32000))
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab, (batch, n_tokens), generator=gen).to(device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def _timed_generate(
        self, model: Any, inputs: dict, *, max_new: int, cfg: ProfileConfig
    ):
        import time

        import torch

        torch.cuda.synchronize(cfg.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=max_new,
                max_new_tokens=max_new,
                use_cache=True,
            )
        torch.cuda.synchronize(cfg.device)
        total_s = time.perf_counter() - t0
        n_new = int(out.shape[-1] - inputs["input_ids"].shape[-1])
        return total_s, n_new

    def throughput_and_genpeak(self, loaded: Any, cfg: ProfileConfig):
        import torch

        model, tok = loaded.model, loaded.tokenizer
        model.eval()
        torch.cuda.reset_peak_memory_stats(cfg.device)
        out: dict = {}
        for bucket, n_prompt in cfg.prompt_buckets.items():
            inputs = self._make_inputs(tok, n_prompt, 1, cfg.device)
            for _ in range(cfg.warmup):
                self._timed_generate(model, inputs, max_new=cfg.decode_tokens, cfg=cfg)
            ttfts, totals = [], []
            for _ in range(cfg.repeats):
                ttft_s, _ = self._timed_generate(model, inputs, max_new=1, cfg=cfg)
                total_s, _ = self._timed_generate(
                    model, inputs, max_new=cfg.decode_tokens, cfg=cfg
                )
                ttfts.append(ttft_s)
                totals.append(total_s)
            ttft = median(ttfts)
            total = median(totals)
            out[bucket] = {
                "prompt_tokens": n_prompt,
                "gen_tokens": cfg.decode_tokens,
                "prefill_tok_s": n_prompt / ttft,
                "decode_tok_s": (cfg.decode_tokens - 1) / max(total - ttft, 1e-9),
                "e2e_latency_s": total,
                "ttft_s": ttft,
            }
        gen_peak = {
            "generate_peak_bytes": int(torch.cuda.max_memory_allocated(cfg.device)),
            "generate_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(cfg.device)
            ),
        }
        return out, gen_peak

    def sweep(self, loaded: Any, cfg: ProfileConfig) -> dict:
        import torch

        model, tok = loaded.model, loaded.tokenizer
        ladder: list[int] = []
        value, oom_at = 0, None
        batch = 1
        while batch <= cfg.batch_ceiling:
            try:
                inputs = self._make_inputs(tok, cfg.batch_seq_len, batch, cfg.device)
                self._timed_generate(
                    model, inputs, max_new=cfg.batch_gen_tokens, cfg=cfg
                )
                value = batch
                ladder.append(batch)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(
                    exc, torch.cuda.OutOfMemoryError
                ):
                    raise
                oom_at = batch
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(cfg.device)
                break
            batch *= 2
        return {
            "value": max(value, 1) if value else 1,
            "oom_at": oom_at,
            "ceiling": cfg.batch_ceiling,
            "seq_len": cfg.batch_seq_len,
            "gen_tokens": cfg.batch_gen_tokens,
            "ladder": ladder,
        }


def profile_variant(
    variant: Any,
    cfg: ProfileConfig,
    env_meta: dict | None = None,
    *,
    load_model: Any = None,
    engine: Any = None,
) -> dict:
    """Load the variant ONCE, measure in a fixed order, unload in finally.

    Order: disk → runtime → load-mem → throughput → sweep (OOM sweep LAST so it
    cannot pollute earlier peaks). Returns the assembled schema-valid result.
    Injectable engine/load_model make this fully CPU-testable with a fake.
    """
    engine = engine if engine is not None else TorchEngine()
    load_model = load_model if load_model is not None else _default_load_model

    disk = engine.disk(variant)
    loaded, load_mem = engine.load_and_measure(load_model, variant, cfg)
    try:
        runtime = engine.runtime_facts(loaded, cfg)
        throughput, gen_mem = engine.throughput_and_genpeak(loaded, cfg)
        max_batch = engine.sweep(loaded, cfg)
    finally:
        loaded.unload()
    memory = {**load_mem, **gen_mem}
    return build_efficiency_result(
        variant=variant,
        cfg=cfg,
        runtime=runtime,
        disk=disk,
        memory=memory,
        throughput=throughput,
        max_batch=max_batch,
        env_meta=env_meta,
    )
