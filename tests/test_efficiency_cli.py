from __future__ import annotations

import json
import subprocess
import sys

import pytest

from qquant.efficiency.cli import main


def test_import_is_torch_free():
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant.efficiency','qquant.efficiency.schema','qquant.efficiency.cli')];"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_missing_variant_is_exit_2(capsys):
    assert main([]) == 2


def test_unknown_variant_is_exit_2(capsys):
    assert main(["--variant", "nope"]) == 2


def test_dry_run_is_torch_free_and_prints_plan(tmp_path, capsys):
    rc = main(["--variant", "bf16", "--results", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bf16" in out
    assert str(tmp_path / "bf16" / "efficiency.json") in out
    assert "torch" not in sys.modules  # dry-run never imports torch


def test_resume_skip_when_done_artifact_present(tmp_path, capsys):
    # Pre-write a structurally-done artifact whose torch-free provenance
    # matches the defaults.
    from qquant.efficiency.profiler import ProfileConfig
    from qquant.registry import load_variants

    cfg = ProfileConfig()
    variants = load_variants()
    art = {
        "schema_version": 1,
        "variant": "bf16",
        "disk": {"weights_bytes": 1, "source": "hf-cache"},
        "memory": {
            "weights_resident_bytes": 1,
            "load_peak_bytes": 2,
            "generate_peak_bytes": 3,
        },
        "throughput": {
            "short": {
                "prompt_tokens": 128,
                "gen_tokens": 256,
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "e2e_latency_s": 1.0,
                "ttft_s": 1.0,
            }
        },
        "max_batch_size": {"value": 1, "ceiling": 64},
        "config": {
            "gpu_name": "x",
            "dtype": "bfloat16",
            "torch_version": "t",
            "transformers_version": "tr",
            "cuda_version": "c",
            "decode_tokens": cfg.decode_tokens,
            "prompt_buckets": dict(cfg.prompt_buckets),
            "batch_seq_len": cfg.batch_seq_len,
            "model_revision": variants["bf16"].revision,
        },
    }
    p = tmp_path / "bf16" / "efficiency.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(art))
    rc = main(["--variant", "bf16", "--results", str(tmp_path)])  # no --force
    out = capsys.readouterr().out
    assert rc == 0
    assert "up-to-date" in out or "skip" in out.lower()
    assert "torch" not in sys.modules  # skip path never loads


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
