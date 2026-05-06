"""Statistical utilities for the Wearable Assistant Context Bench.

Provides Wilson score confidence intervals for binary proportions and
non-parametric percentile bootstrap CIs for arbitrary statistics. The
Wilson method is well-calibrated for small samples and proportions
near 0 or 1; the bootstrap covers cases (means of proportions, derived
statistics) where Wilson is awkward.

Reference: Wilson, E. B. (1927). Probable inference, the law of
succession, and statistical inference. JASA, 22(158), 209-212.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class WilsonCI:
    """Wilson score confidence interval for a binary proportion.

    Attributes:
        proportion: Point estimate (successes / total).
        lower: Lower bound of the CI.
        upper: Upper bound of the CI.
        n: Total observations.
        k: Number of successes.
        confidence: Confidence level (default 0.95).
    """

    proportion: float
    lower: float
    upper: float
    n: int
    k: int
    confidence: float = 0.95


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> WilsonCI:
    """Compute a Wilson score confidence interval.

    Wilson is well-calibrated for small samples and proportions near 0
    or 1, unlike the normal approximation (Wald) which can produce
    negative bounds or overshoot 1.0.

    Args:
        k: Number of successes.
        n: Total observations. Must be > 0.
        confidence: Confidence level in (0, 1). Default 0.95.

    Returns:
        A WilsonCI with proportion, lower, upper, n, k.
    """
    if n <= 0:
        return WilsonCI(
            proportion=0.0, lower=0.0, upper=0.0, n=0, k=0,
            confidence=confidence,
        )

    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z2 = z * z

    p_hat = k / n
    denom = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)

    lower = (center - spread) / denom
    upper = (center + spread) / denom

    return WilsonCI(
        proportion=p_hat,
        lower=max(0.0, lower),
        upper=min(1.0, upper),
        n=n,
        k=k,
        confidence=confidence,
    )


@dataclass
class BootstrapCI:
    """Non-parametric percentile bootstrap confidence interval.

    Attributes:
        point: Point estimate (statistic of the original sample).
        lower: Lower percentile bound.
        upper: Upper percentile bound.
        n_iter: Number of bootstrap resamples.
        confidence: Confidence level.
    """

    point: float
    lower: float
    upper: float
    n_iter: int
    confidence: float = 0.95


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_iter: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI | None:
    """Compute a percentile bootstrap CI for an arbitrary statistic.

    Resamples ``values`` with replacement ``n_iter`` times, computes
    the chosen statistic on each resample, and returns the percentile
    bounds. ``statistic`` defaults to the mean.

    Returns ``None`` when ``values`` is empty.
    """
    if not values:
        return None
    if statistic is None:
        def statistic(xs: Sequence[float]) -> float:
            return sum(xs) / len(xs)
    rng = random.Random(seed)
    n = len(values)
    bootstraps: list[float] = []
    seq = list(values)
    for _ in range(n_iter):
        sample = [seq[rng.randrange(n)] for _ in range(n)]
        bootstraps.append(statistic(sample))
    bootstraps.sort()
    alpha = 1 - confidence
    lo_idx = max(0, int(round(alpha / 2 * n_iter)) - 1)
    hi_idx = min(n_iter - 1, int(round((1 - alpha / 2) * n_iter)) - 1)
    return BootstrapCI(
        point=statistic(seq),
        lower=bootstraps[lo_idx],
        upper=bootstraps[hi_idx],
        n_iter=n_iter,
        confidence=confidence,
    )
