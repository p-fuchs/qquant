"""EXT-3 W8A8: maps + dispatch are correct and DON'T disturb the W4A16 fairness pair.

llm-compressor is box-only (linux marker), so we never call the recipe builders here —
only the pure dispatch/constants that are import-time torch-free.
"""

from __future__ import annotations

import pytest
from selfquant import quantize
from selfquant.recipes import SCHEME, SCHEME_W8A8


def test_all_excludes_w8a8():
    """`--id all` must stay the W4A16 fairness pair only (v1 behaviour unchanged)."""
    assert quantize.SELF_QUANT_IDS == ("gptq-selfquant", "awq-selfquant")
    assert "w8a8-selfquant" not in quantize.SELF_QUANT_IDS
    assert "w8a8-selfquant" in quantize.EXT_SELF_QUANT_IDS


def test_scheme_and_algorithm_maps():
    assert quantize._SCHEME["w8a8-selfquant"] == (SCHEME_W8A8, None)
    assert quantize._SCHEME["gptq-selfquant"][0] == SCHEME
    assert quantize._ALGORITHM["w8a8-selfquant"] == "smoothquant-w8a8"


def test_recipe_for_unknown_lists_known_ids():
    with pytest.raises(ValueError, match="w8a8-selfquant"):
        quantize.recipe_for("bogus")


def test_parser_accepts_w8a8_and_defaults_to_all():
    parser = quantize._build_parser()
    assert parser.parse_args([]).id == "all"
    assert parser.parse_args(["--id", "w8a8-selfquant"]).id == "w8a8-selfquant"
