"""Statistical validation framework for probability distributions.

Public surface:
    - :mod:`validation.theoretical_metrics` — pure moment and relative-error math
    - :mod:`validation.validation_runner`  — suite definition, runner, table printer
"""

from validation.theoretical_metrics import (
    mean_relative_error,
    sample_mean,
    sample_variance,
    theoretical_mean,
    theoretical_variance,
    variance_relative_error,
)
from validation.validation_runner import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUITE,
    ValidationResult,
    print_results,
    run_suite,
)

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_SUITE",
    "ValidationResult",
    "mean_relative_error",
    "print_results",
    "run_suite",
    "sample_mean",
    "sample_variance",
    "theoretical_mean",
    "theoretical_variance",
    "variance_relative_error",
]
