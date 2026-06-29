from __future__ import annotations

import pytest

from qquant.aggregate.stats import wilson_interval


def test_wilson_known_reference():
    lo, hi = wilson_interval(140, 164)
    assert 0.79 < lo < 0.81
    assert 0.89 < hi < 0.91


def test_wilson_zero_successes_clamps_low_at_zero():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 0.35


def test_wilson_all_successes_clamps_high_at_one():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert 0.65 < lo < 1.0


def test_wilson_rejects_bad_n_and_k():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
