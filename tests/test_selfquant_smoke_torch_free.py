from __future__ import annotations

import subprocess
import sys


def test_umbrella_does_not_import_selfquant_smoke():
    """Importing qquant / qquant.cli must NOT pull in qquant.selfquant_smoke (GPU) nor
    torch — fresh interpreter so another test's import can't mask a regression.
    """
    code = (
        "import importlib, sys;"
        "[importlib.import_module(m) for m in ('qquant', 'qquant.cli')];"
        "bad=[m for m in sys.modules "
        "if m == 'qquant.selfquant_smoke' or m == 'torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_selfquant_smoke_module_is_importable():
    """The module itself imports torch-free at top (heavy imports are deferred)."""
    code = (
        "import importlib, sys;"
        "importlib.import_module('qquant.selfquant_smoke');"
        "bad=[m for m in sys.modules if m == 'torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
