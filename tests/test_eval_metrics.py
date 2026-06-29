from __future__ import annotations

import pytest

from qquant.eval.metrics import extract_metric, extract_stderr, wilson_interval


def test_extract_metric_first_key_in_order_wins():
    res = {"acc,none": 0.81, "acc": 0.79}
    assert extract_metric(res, ("acc,none", "acc")) == ("acc,none", 0.81)


def test_extract_metric_falls_through_to_later_key():
    res = {"pass@1,none": 0.42}
    key, value = extract_metric(res, ("pass@1,create_test", "pass@1,none", "pass@1"))
    assert (key, value) == ("pass@1,none", 0.42)


def test_extract_metric_raises_listing_keys_when_none_match():
    with pytest.raises(KeyError) as exc:
        extract_metric({"foo": 1.0}, ("acc,none", "acc"))
    assert "foo" in str(exc.value)


def test_extract_stderr_finds_suffixed_key():
    res = {"acc,none": 0.81, "acc_stderr,none": 0.012}
    assert extract_stderr(res, "acc") == pytest.approx(0.012)


def test_extract_stderr_returns_none_when_absent_or_na():
    assert extract_stderr({"pass@1,none": 0.4}, "pass@1") is None
    assert extract_stderr({"acc_stderr,none": "N/A"}, "acc") is None


def test_wilson_interval_is_reexported_here():
    lo, hi = wilson_interval(140, 164)
    assert 0.79 < lo < 0.81 and 0.89 < hi < 0.92
