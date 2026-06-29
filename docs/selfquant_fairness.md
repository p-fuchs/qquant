# Self-quant fairness — the calibration-controlled GPTQ-vs-AWQ comparison

The two self-quantized variants `gptq-selfquant` and `awq-selfquant` are produced under a
single controlled protocol so the GPTQ-vs-AWQ **quality** comparison isolates exactly one
variable: the quantization **algorithm**.

## Controlled axes (held identical for both)

| Axis | Value | Where it is enforced |
|------|-------|----------------------|
| Scheme | **W4A16_ASYM** (4-bit weights, fp16 activations, asymmetric) | `selfquant.recipes.SCHEME` |
| Group size | **128** (per-group) | `selfquant.recipes.GROUP_SIZE` |
| Ignored layers | `["lm_head"]` | both recipes |
| Calibration set | **`c4-128x2048-s42`** — the SAME tokenized object passed to both `oneshot` calls (128 docs × 2048 tokens, seed 42, from `allenai/c4`/`en`/`train`) | `selfquant.calibration` + `quantize.main` builds it once |
| Base weights | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | `selfquant.quantize.BASE_REVISION` |

The **only intentional difference** is the algorithm: GPTQ error-compensation
(`GPTQModifier`) vs AWQ activation-aware scaling (`AWQModifier` + `QuantizationModifier`).

### Machine-checked guarantee

`selfquant.recipes.assert_schemes_match(gptq_recipe(), awq_recipe())` runs as a pre-flight in
`quantize.main` and fails fast unless **both** recipes resolve to identical weight quant args
— `num_bits=4, symmetric=False, strategy="group", group_size=128`. A typo (e.g. one recipe
symmetric) aborts before any GPU work.

### Reproducibility provenance

- The calibration manifest `checkpoints/self-quant/_calib/c4-128x2048-s42.manifest.json`
  records the selected source indices, the resolved dataset revision, the tokenizer, the seed,
  and the `input_ids` **sha256** — so the calibration set is bit-reconstructible.
- Each checkpoint's `quant_manifest.json` carries `algorithm`, `base_revision`, `scheme`,
  `group_size`, and `calib_sha256`. Both manifests share every field **except `algorithm`**.
  This fingerprint also feeds Spec 05's self-quant `RunConfig.model_revision`, so a
  re-quantization (new base / calibration / algorithm) marks the affected eval cells stale.

## Speed is runtime-confounded (do not read self-quant speed as algorithmic)

W4A16_ASYM disables Marlin, and the HF `generate()` backend may take a slow dequant path on
sm_89 (fast Marlin targets vLLM). Therefore self-quant **speed** is **runtime-confounded** and
must not be reported as an algorithmic speed result. The diagnostic
`checkpoints/self-quant/<id>/kernel_smoke.json` (from `qquant.selfquant_smoke`) records the
W4A16 module/kernel class actually used, the compressed-tensors `format`, a rough
`decode_tok_s`, and `peak_vram_gb` **to substantiate this caveat** — it is not the speed number
in the report. The **authoritative** efficiency numbers come from Spec 06 (`qquant-profile`),
which labels the self-quant speed metrics `runtime-confounded`. Self-quant **disk** and
**memory** are real and fair for every variant.

## How they are produced

```bash
cd quant && uv sync && uv run python -m selfquant.quantize --id all
python -m qquant.selfquant_smoke --variant gptq-selfquant --ckpt checkpoints/self-quant/gptq-selfquant
python -m qquant.selfquant_smoke --variant awq-selfquant  --ckpt checkpoints/self-quant/awq-selfquant
```

Re-running skips done checkpoints (manifest match); `--force` recomputes. The eval env loads
these `compressed-tensors` checkpoints natively (Spec 04, `quant_method == "compressed-tensors"`).
