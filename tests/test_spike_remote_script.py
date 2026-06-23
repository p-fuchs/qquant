"""Guards for the standalone on-box spike script.

It must run ``--help`` on a plain machine (no GPU stack) and must not pull torch in at
import time — the heavy imports are lazy inside each check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike" / "spike_remote.py"


def test_spike_remote_help_runs_on_plain_machine():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_spike_remote_import_is_torch_free():
    # Importing the script (defs only; main not invoked) must not import torch.
    code = (
        "import importlib.util, sys;"
        f"spec=importlib.util.spec_from_file_location('spike_remote', r'{_SCRIPT}');"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "bad=[k for k in sys.modules if k=='torch' or k.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, f"torch imported at module load: {result.stderr}"
