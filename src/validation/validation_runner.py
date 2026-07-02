"""Validation runner: compute sample and theoretical moments for a suite of
distributions and produce a structured, printable report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from distributions.base import Distribution
from distributions.continuous import (
    ExponentialDistribution,
    GammaDistribution,
    NormalDistribution,
)
from distributions.discrete import (
    BernoulliDistribution,
    GeometricDistribution,
    WeightedDiscreteDistribution,
)
from validation.theoretical_metrics import (
    mean_relative_error as _mean_rel_err,
    sample_mean,
    sample_variance,
    theoretical_mean,
    theoretical_variance,
    variance_relative_error as _var_rel_err,
)

DEFAULT_SAMPLE_SIZE: int = 50_000
DEFAULT_SEED: int = 42


# Module-level suite: one instance of each concrete Distribution, with
# parameters chosen to match the existing test suite in
# tests/test_distributions.py so the validation framework and the pytest
# tests exercise the same distributions in the same regimes.
DEFAULT_SUITE: list[Distribution] = [
    ExponentialDistribution(rate=3.0),
    NormalDistribution(mean=10.0, std=2.5),
    GammaDistribution(shape=2.0, scale=3.0),
    BernoulliDistribution(probability=0.3),
    GeometricDistribution(probability=0.4),
    WeightedDiscreteDistribution(
        values=[1.0, 2.0, 5.0],
        weights=[1.0, 2.0, 1.0],
    ),
]


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome for a single distribution.

    Attributes:
        name: Human-readable name of the distribution.
        theoretical_mean: Mean declared by the distribution.
        sample_mean: Mean computed from drawn samples.
        theoretical_variance: Variance declared by the distribution.
        sample_variance: Variance computed from drawn samples.
        mean_relative_error: |sample_mean - theoretical_mean| / |theoretical_mean|.
        variance_relative_error: |sample_var - theoretical_var| / |theoretical_var|.
    """

    name: str
    theoretical_mean: float
    sample_mean: float
    theoretical_variance: float
    sample_variance: float
    mean_relative_error: float
    variance_relative_error: float

    def __str__(self) -> str:
        """Format this result as a single fixed-width table row."""
        m, v = self.mean_relative_error, self.variance_relative_error
        return (
            f"{self.name:<40}"
            f"{self.theoretical_mean:>10.4f} / {self.sample_mean:<10.4f}"
            f"{self.theoretical_variance:>10.4f} / {self.sample_variance:<10.4f}"
            f"{m * 100:>7.3f}% / {v * 100:.3f}%"
        )


def _name(distribution: Distribution) -> str:
    """Return a short, parameterised name for a distribution."""
    cls = type(distribution).__name__
    if isinstance(distribution, ExponentialDistribution):
        return f"{cls}(rate={distribution.rate})"
    if isinstance(distribution, NormalDistribution):
        return f"{cls}(mean={distribution.mean()}, std={distribution.std():.4g})"
    if isinstance(distribution, GammaDistribution):
        return f"{cls}(shape={distribution._shape}, scale={distribution._scale})"
    if isinstance(distribution, BernoulliDistribution):
        return f"{cls}(p={distribution._probability})"
    if isinstance(distribution, GeometricDistribution):
        return f"{cls}(p={distribution._probability})"
    if isinstance(distribution, WeightedDiscreteDistribution):
        return f"{cls}(values={distribution._values.tolist()}, weights={distribution._weights.tolist()})"
    return cls


def run_suite(
    distributions: Iterable[Distribution] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[ValidationResult]:
    """Run validation on each distribution and return a list of results.

    A single seeded RNG is created and shared with every distribution so
    that the run is fully reproducible and a regression in one
    distribution's sampling does not perturb the rest.

    Parameters:
        distributions: Iterable of distributions to validate. Defaults to
            :data:`DEFAULT_SUITE` when None.
        sample_size: Number of samples to draw from each distribution.
        seed: Seed for the shared ``numpy.random.Generator``.

    Returns:
        List of ``ValidationResult``, one per distribution, in input order.
    """
    if distributions is None:
        distributions = DEFAULT_SUITE

    rng = np.random.default_rng(seed)
    results: list[ValidationResult] = []

    for dist in distributions:
        # Inject the shared RNG so the run is reproducible end to end.
        dist._rng = rng  # type: ignore[attr-defined]
        samples = dist.sample(sample_size)

        th_mean = theoretical_mean(dist)
        th_var = theoretical_variance(dist)
        sa_mean = sample_mean(samples)
        sa_var = sample_variance(samples)

        results.append(
            ValidationResult(
                name=_name(dist),
                theoretical_mean=th_mean,
                sample_mean=sa_mean,
                theoretical_variance=th_var,
                sample_variance=sa_var,
                mean_relative_error=_mean_rel_err(th_mean, sa_mean),
                variance_relative_error=_var_rel_err(th_var, sa_var),
            )
        )

    return results


def print_results(results: Iterable[ValidationResult]) -> None:
    """Print a fixed-width table of validation results.

    Parameters:
        results: Iterable of ``ValidationResult`` (typically from
            :func:`run_suite`).
    """
    header = (
        f"{'Distribution':<40}"
        f"{'Mean (th / sample)':>22}"
        f"{'Variance (th / sample)':>25}"
        f"{'Rel. Err (M / V)':>17}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(result)
