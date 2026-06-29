from __future__ import annotations

import subprocess
import sys


def test_importing_qquant_eval_is_torch_and_lmeval_free():
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant.eval','qquant.eval.policy','qquant.eval.metrics','qquant.eval.results',"
        "'qquant.eval.manifest','qquant.eval.datasets','qquant.aggregate.stats')];"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.') "
        "or m=='lm_eval' or m.startswith('lm_eval.')];"
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_plan_is_torch_free_and_lists_cells(tmp_path, capsys):
    from qquant.eval.cli import main

    rc = main(
        ["--results", str(tmp_path), "--variant", "bf16", "--task", "gsm8k", "--plan"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "bf16/gsm8k" in out


def test_help_exits_zero():
    import pytest as _pytest

    from qquant.eval.cli import main

    with _pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
