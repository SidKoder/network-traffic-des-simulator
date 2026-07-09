"""Statistical validation framework for probability distributions.

Public surface:
    - :mod:`validation.theoretical_metrics`      — pure moment and relative-error math
    - :mod:`validation.validation_runner`         — suite definition, runner, table printer
    - :mod:`validation.mm1_validation`            — M/M/1 theoretical vs observed validation
    - :mod:`validation.convergence_experiment`    — Monte Carlo convergence experiments
    - :mod:`validation.parameter_sweep`           — performance exploration under varying load
    - :mod:`validation.plots`                     — parameter sweep visualization layer
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
from validation.mm1_validation import (
    MM1MetricComparison,
    MM1ObservedMetrics,
    MM1TheoreticalMetrics,
    MM1ValidationResult,
    format_mm1_validation_report,
    mm1_theoretical_metrics,
    run_mm1_validation,
)
from validation.convergence_experiment import (
    ConvergencePoint,
    ConvergenceMetricSeries,
    ConvergenceResult,
    run_convergence_experiment,
    format_convergence_report,
)
from validation.parameter_sweep import (
    SweepPoint,
    ParameterSweepResult,
    run_parameter_sweep,
    format_parameter_sweep_report,
)
from validation.plots import (
    plot_parameter_sweep,
)

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_SUITE",
    "ValidationResult",
    "mean_relative_error",
    "print_results",
    "run_suite",
    "MM1MetricComparison",
    "MM1ObservedMetrics",
    "MM1TheoreticalMetrics",
    "MM1ValidationResult",
    "format_mm1_validation_report",
    "mm1_theoretical_metrics",
    "run_mm1_validation",
    "ConvergencePoint",
    "ConvergenceMetricSeries",
    "ConvergenceResult",
    "run_convergence_experiment",
    "format_convergence_report",
    "SweepPoint",
    "ParameterSweepResult",
    "run_parameter_sweep",
    "format_parameter_sweep_report",
    "plot_parameter_sweep",
    "sample_mean",
    "sample_variance",
    "theoretical_mean",
    "theoretical_variance",
    "variance_relative_error",
]

