from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from qquant.efficiency.cli import main  # noqa: E402
from qquant.efficiency.schema import efficiency_path, is_valid_efficiency  # noqa: E402

pytestmark = pytest.mark.gpu


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
    assert (
        mem["generate_peak_bytes"]
        >= mem["load_peak_bytes"]
        >= mem["weights_resident_bytes"]
        > 0
    )
    for bucket in ("short", "medium", "long"):
        tp = art["throughput"][bucket]
        assert tp["gen_tokens"] == 16
        assert tp["decode_tok_s"] > 0 and tp["prefill_tok_s"] > 0
    assert art["max_batch_size"]["value"] >= 1
