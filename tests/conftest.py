from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip every ``@pytest.mark.gpu`` test unless a CUDA GPU is present."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU (run on the vast.ai box)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
