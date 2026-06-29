from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from qquant.eval.cli import main  # noqa: E402

pytestmark = pytest.mark.gpu


def test_gsm8k_smoke_and_determinism(tmp_path):
    rc = main(
        [
            "--results",
            str(tmp_path),
            "--variant",
            "bf16",
            "--task",
            "gsm8k",
            "--limit",
            "8",
            "--determinism-check",
            "bf16",
            "gsm8k",
        ]
    )
    assert rc == 0
