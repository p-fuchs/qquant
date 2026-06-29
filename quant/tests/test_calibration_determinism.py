from __future__ import annotations

import pytest
from selfquant.calibration import (
    CalibrationSpec,
    input_ids_sha256,
    select_indices,
)


def test_calib_id_format():
    assert CalibrationSpec().calib_id == "c4-128x2048-s42"
    assert CalibrationSpec(num_samples=4, max_seq_len=8, seed=1).calib_id == "c4-4x8-s1"


def test_select_indices_picks_only_long_enough_docs():
    spec = CalibrationSpec(num_samples=4, max_seq_len=8, seed=42)
    counts = [2, 10, 8, 3, 9, 100, 7, 8, 8, 50]  # qualifying: idx 1,2,4,5,7,8,9
    chosen = select_indices(counts, spec)
    assert len(chosen) == 4
    assert all(counts[i] >= 8 for i in chosen)
    assert len(set(chosen)) == 4  # no duplicates


def test_select_indices_is_deterministic():
    spec = CalibrationSpec(num_samples=4, max_seq_len=8, seed=42)
    counts = [2, 10, 8, 3, 9, 100, 7, 8, 8, 50]
    assert select_indices(counts, spec) == select_indices(counts, spec)


def test_select_indices_raises_when_too_few_candidates():
    spec = CalibrationSpec(num_samples=4, max_seq_len=8, seed=42)
    with pytest.raises(ValueError):
        select_indices([1, 2, 3, 8], spec)  # only one qualifies, need 4


def test_input_ids_sha256_stable_and_sensitive():
    a = input_ids_sha256([[1, 2, 3], [4, 5]])
    assert a == input_ids_sha256([[1, 2, 3], [4, 5]])  # stable
    assert a != input_ids_sha256([[1, 2, 3], [4, 6]])  # one token changed
    assert a != input_ids_sha256([[1, 2], [3, 4, 5]])  # boundary matters (separator)
