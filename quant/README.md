# quant/ — self-quantization environment (Spec 08)

Isolated uv project that **produces** the `gptq-selfquant` and `awq-selfquant`
checkpoints with `llm-compressor`, from a single shared C4 calibration set, so the
GPTQ-vs-AWQ comparison is calibration-controlled.

It is separate from the root eval env because `llm-compressor 0.12.0` caps
`transformers<=5.10.1` with tight transitive pins that cannot co-resolve with the eval
stack (`gptqmodel` + `lm-eval`). The eval env only needs `compressed-tensors` to *load*
the checkpoints this env writes.

Run on Linux + CUDA (the vast.ai box):

```bash
cd quant && uv sync && uv run python -m selfquant.quantize --help
```

Output checkpoints (compressed-tensors format) are written under `../checkpoints/self-quant/`
and exfiltrated alongside results. See `docs/specs/08-*.md`.
