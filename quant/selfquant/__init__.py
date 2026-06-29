"""selfquant — isolated self-quantization package (Spec 07).

Produces the two calibration-controlled compressed-tensors checkpoints
(``gptq-selfquant``/``awq-selfquant``) under one protocol (W4A16_ASYM, group_size 128,
identical C4 calibration, same base SHA — only the algorithm differs). Runs in the isolated
``quant/`` uv env (``llmcompressor==0.12.0``) on the CUDA box; the root eval env only loads the
checkpoints it writes. ``manifest`` is pure stdlib (importable anywhere); ``calibration``,
``recipes``, and ``quantize`` import the heavy deps lazily inside functions.
"""

from __future__ import annotations
