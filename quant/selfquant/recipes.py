"""W4A16_ASYM recipes for GPTQ and AWQ + the machine-checked fairness guarantee.

The controlled comparison requires BOTH algorithms to resolve to IDENTICAL weight quant
args (num_bits=4, symmetric=False, strategy="group", group_size=128); the only intended
difference is the algorithm. ``assert_schemes_match`` fails fast otherwise. Box-only:
imports llmcompressor/compressed_tensors lazily inside functions (the quant env).
"""

from __future__ import annotations

from typing import Any

GROUP_SIZE = 128
SCHEME = "W4A16_ASYM"  # 4-bit weights, fp16 activations, ASYMMETRIC, per-group
IGNORE = ["lm_head"]

_EXPECTED_WEIGHT_ARGS = {
    "num_bits": 4,
    "symmetric": False,
    "strategy": "group",
    "group_size": GROUP_SIZE,
}


def gptq_recipe(group_size: int = GROUP_SIZE) -> list:
    """GPTQ error-compensation at W4A16_ASYM. GPTQModifier carries the quant scheme."""
    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError:  # older layout
        from llmcompressor.modifiers.gptq import GPTQModifier

    return [GPTQModifier(targets="Linear", scheme=SCHEME, ignore=list(IGNORE))]


def awq_recipe(group_size: int = GROUP_SIZE) -> list:
    """AWQ activation-aware smoothing, then QuantizationModifier applies W4A16_ASYM."""
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return [
        AWQModifier(),
        QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=list(IGNORE)),
    ]


def _scheme_name(modifier: Any) -> str | None:
    return getattr(modifier, "scheme", None)


def resolved_quant_args(recipe: list) -> dict:
    """Resolve a recipe's scheme to its comparable weight quant args.

    Reads the scheme string off the quant-carrying modifier (GPTQModifier or
    QuantizationModifier). When compressed-tensors can expand that preset, returns the
    concrete weight args (num_bits, symmetric, strategy, group_size); otherwise it
    degrades to just the scheme name (the authoritative concrete-args check is the
    produced config.json, done-when #3). Same scheme -> equal compare either way.
    """
    scheme_name = next(
        (s for s in (_scheme_name(m) for m in recipe) if isinstance(s, str)), None
    )
    if scheme_name is None:
        raise ValueError("no modifier in the recipe carries a string `scheme`")
    try:
        from compressed_tensors.quantization import preset_name_to_scheme

        w = preset_name_to_scheme(scheme_name, ["Linear"]).weights
        return {
            "num_bits": w.num_bits,
            "symmetric": w.symmetric,
            "strategy": str(getattr(w.strategy, "value", w.strategy)),
            "group_size": w.group_size,
        }
    except Exception:
        return {"scheme": scheme_name}


def assert_schemes_match(a: list, b: list) -> None:
    """Fail fast unless both recipes resolve to IDENTICAL weight quant args. When the
    concrete args are available, also assert they equal the controlled target
    (num_bits=4, symmetric=False, strategy='group', group_size=128). The machine-checked
    guarantee that the only difference is the algorithm.
    """
    ra, rb = resolved_quant_args(a), resolved_quant_args(b)
    if ra != rb:
        raise AssertionError(f"recipe weight quant args differ: {ra} != {rb}")
    if "num_bits" in ra and ra != _EXPECTED_WEIGHT_ARGS:
        raise AssertionError(
            f"weight quant args {ra} != controlled target {_EXPECTED_WEIGHT_ARGS}"
        )
