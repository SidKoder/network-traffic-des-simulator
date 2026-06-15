"""Shared pytest fixtures and statistical helpers."""

import numpy as np
import pytest


def assert_moments_close(
    samples: np.ndarray,
    expected_mean: float,
    expected_variance: float,
    mean_rtol: float = 0.05,
    var_rtol: float = 0.10,
) -> None:
    """Assert sample mean and variance are close to theoretical values.

    Parameters:
        samples: Drawn random samples.
        expected_mean: Theoretical mean.
        expected_variance: Theoretical variance.
        mean_rtol: Relative tolerance for mean comparison.
        var_rtol: Relative tolerance for variance comparison.
    """
    sample_mean = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=0))

    assert sample_mean == pytest.approx(expected_mean, rel=mean_rtol), (
        f"Sample mean {sample_mean} != expected {expected_mean}"
    )
    assert sample_var == pytest.approx(expected_variance, rel=var_rtol), (
        f"Sample variance {sample_var} != expected {expected_variance}"
    )
