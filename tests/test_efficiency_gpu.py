from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from qquant.efficiency.cli import main  # noqa: E402
from qquant.efficiency.schema import efficiency_path, is_valid_efficiency  # noqa: E402
from qquant.models import load_variant  # noqa: E402

pytestmark = pytest.mark.gpu

_GIB_24 = 24 * 1024**3


def test_bf16_profile_writes_valid_artifact(tmp_path):
    rc = main(
        [
            "--variant",
            "bf16",
            "--results",
            str(tmp_path),
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--decode-tokens",
            "16",
            "--prompt-lens",
            "32,64,128",
            "--batch-ceiling",
            "4",
        ]
    )
    assert rc == 0
    art = json.loads(efficiency_path(tmp_path, "bf16").read_text())
    assert is_valid_efficiency(art)
    mem = art["memory"]
    # Physically-guaranteed memory ordering (generate_peak >= load_peak is NOT
    # asserted — load-time scratch on a ~15 GB bf16 load can exceed a small-prompt
    # generate transient).
    assert mem["weights_resident_bytes"] > 0
    assert mem["load_peak_bytes"] >= mem["weights_resident_bytes"]
    assert mem["generate_peak_bytes"] >= mem["weights_resident_bytes"]
    assert mem["weights_resident_bytes"] <= _GIB_24
    assert mem["load_peak_bytes"] <= _GIB_24
    assert mem["generate_peak_bytes"] <= _GIB_24
    for bucket in ("short", "medium", "long"):
        tp = art["throughput"][bucket]
        assert tp["gen_tokens"] == 16
        assert tp["decode_tok_s"] > 0 and tp["prefill_tok_s"] > 0
    assert art["max_batch_size"]["value"] >= 1


def test_bf16_greedy_determinism():
    """Verify done-when #8: consecutive greedy generations yield identical tokens."""
    import torch

    loaded = load_variant("bf16")
    try:
        device = next(loaded.model.parameters()).device
        vocab = int(getattr(loaded.tokenizer, "vocab_size", 32000))
        gen = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, vocab, (1, 16), generator=gen).to(device)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        gen_kwargs = {"do_sample": False, "min_new_tokens": 16, "max_new_tokens": 16}
        with torch.inference_mode():
            out1 = loaded.model.generate(**inputs, **gen_kwargs)
            out2 = loaded.model.generate(**inputs, **gen_kwargs)
        assert torch.equal(out1, out2), (
            "greedy generation is non-deterministic: consecutive outputs differ"
        )
    finally:
        loaded.unload()
