"""Theoretical and sample moments for distribution validation.

Pure functions, no I/O, no shared state. Importable on its own.
"""

from __future__ import annotations

import numpy as np

from distributions.base import Distribution


def sample_mean(samples: np.ndarray) -> float:
    """Return the sample mean of the drawn values.

    Parameters:
        samples: Array of samples from a distribution.

    Returns:
        Arithmetic mean of the samples.
    """
    return float(np.mean(samples))


def sample_variance(samples: np.ndarray) -> float:
    """Return the population sample variance (ddof=0).

    This matches the convention in tests/conftest.py:24, keeping the
    framework and the test suite numerically consistent.

    Parameters:
        samples: Array of samples from a distribution.

    Returns:
        Population variance of the samples.
    """
    return float(np.var(samples, ddof=0))


def theoretical_mean(distribution: Distribution) -> float:
    """Return the theoretical mean of a distribution.

    Parameters:
        distribution: Any object implementing the Distribution protocol.

    Returns:
        The distribution's theoretical expected value.
    """
    return float(distribution.mean())


def theoretical_variance(distribution: Distribution) -> float:
    """Return the theoretical variance of a distribution.

    Parameters:
        distribution: Any object implementing the Distribution protocol.

    Returns:
        The distribution's theoretical variance.
    """
    return float(distribution.variance())


def mean_relative_error(theoretical: float, sample: float) -> float:
    """Return the relative error |sample - theoretical| / |theoretical|.

    Returns 0.0 when theoretical is exactly zero to avoid division by zero
    (e.g., a degenerate distribution whose mean is 0).

    Parameters:
        theoretical: Theoretical mean.
        sample: Sample mean.

    Returns:
        Non-negative relative error.
    """
    if theoretical == 0.0:
        return 0.0 if sample == 0.0 else float("inf")
    return abs(sample - theoretical) / abs(theoretical)


def variance_relative_error(theoretical: float, sample: float) -> float:
    """Return the relative error |sample - theoretical| / |theoretical|.

    Returns 0.0 when theoretical is exactly zero.

    Parameters:
        theoretical: Theoretical variance.
        sample: Sample variance.

    Returns:
        Non-negative relative error.
    """
    if theoretical == 0.0:
        return 0.0 if sample == 0.0 else float("inf")
    return abs(sample - theoretical) / abs(theoretical)
