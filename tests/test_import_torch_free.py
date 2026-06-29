from __future__ import annotations

import subprocess
import sys


def test_core_import_is_torch_free():
    """`import qquant` (+ submodules + CLI) must not pull torch into sys.modules.

    Run in a fresh interpreter so another test's torch import can't mask a regression.
    """
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in "
        "('qquant','qquant.cli','qquant.matrix','qquant.registry','qquant.paths',"
        "'qquant.config','qquant.eval.runner',"
        "'qquant.efficiency','qquant.efficiency.schema','qquant.efficiency.profiler',"
        "'qquant.orchestrate.vastai','qquant.orchestrate.spike')];"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, f"torch was imported by the core: {result.stderr}"
