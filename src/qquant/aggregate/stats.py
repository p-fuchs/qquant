"""Statistics for the qquant analysis layer. Torch-free, pure stdlib.

``wilson_interval`` is the SINGLE implementation of the Wilson score interval;
Spec 05's eval-time HumanEval pass@1 CI re-exports it from here, and Spec 09
builds the rest of the aggregation on top of this module.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Returns (lo, hi) clamped to [0, 1]. Raises ValueError for n <= 0 or k not in [0, n].
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}], got {k}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))
