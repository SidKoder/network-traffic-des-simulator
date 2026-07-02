# Distribution Architecture

This document describes the design and structure of the `src/distributions/` module — the probability-distribution engine that underlies packet arrivals, service times, and any other stochastic behavior in the DES simulator.

## Module Layout

```
src/distributions/
├── __init__.py            # Public API re-exports
├── base.py                # Abstract base class: Distribution
├── continuous.py          # ExponentialDistribution, NormalDistribution, GammaDistribution
├── discrete.py            # BernoulliDistribution, GeometricDistribution, WeightedDiscreteDistribution
└── poisson.py             # HomogeneousPoissonProcess (composes ExponentialDistribution)
```

The module is intentionally **network-agnostic** — distributions depend only on NumPy and have no knowledge of queues, events, or the simulator. This keeps the math reusable, testable in isolation, and easy to extend.

## Design Principles

1. **One abstract base, two specializations.** All univariate distributions inherit from a single `Distribution` ABC. Continuous and discrete variants live in separate files purely for organization — both implement the same contract.
2. **Poisson process is composition, not inheritance.** `HomogeneousPoissonProcess` is *not* a `Distribution` subclass; it composes an `ExponentialDistribution` internally to model inter-arrival times. This keeps the class focused on a different abstraction (a stream of arrival timestamps) rather than a probability law.
3. **Reproducible by injection.** Every constructor accepts an optional `np.random.Generator`. If omitted, a default generator is created. Seeded runs are therefore a matter of passing the same `Generator` instance to every consumer.
4. **Validation up front.** Parameters are checked in `__init__` and a `ValueError` is raised with a descriptive message on invalid input. The object is never constructed in a half-initialized state.
5. **Theoretical moments, not estimates.** `mean()`, `variance()`, and `std()` return the *theoretical* values derived from the parameters. This makes it trivial to compare simulated output to analytic expectations (e.g., for chi-square or KS tests in the test suite).

## The Base Class — `Distribution`

`src/distributions/base.py` defines the contract every distribution must satisfy.

```python
class Distribution(ABC):
    def __init__(self, rng: Generator | None = None) -> None
    @property
    def rng(self) -> Generator           # exposed NumPy RNG

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray

    @abstractmethod
    def mean(self) -> float

    @abstractmethod
    def variance(self) -> float

    def std(self) -> float               # concrete: sqrt(variance())
```

### Contract

| Member | Kind | Required? | Purpose |
|---|---|---|---|
| `rng` | property | yes | Returns the underlying `numpy.random.Generator`. Subclasses should sample from `self._rng`, never from the global NumPy state. |
| `sample(size)` | abstract method | yes | Draw `size` i.i.d. samples and return them as a `np.ndarray`. The output dtype may be float (continuous) or float-cast integer (discrete). |
| `mean()` | abstract method | yes | Return the theoretical expected value. |
| `variance()` | abstract method | yes | Return the theoretical variance. |
| `std()` | concrete method | no | Default implementation: `float(np.sqrt(self.variance()))`. Subclasses almost never need to override it. |

### Why abstract?

Forcing subclasses to implement `sample`, `mean`, and `variance` guarantees a uniform interface for the simulator. A queue model, a traffic generator, or a test harness can hold a `Distribution` reference and call any of those three methods without knowing the concrete type.

## Continuous Distributions — `continuous.py`

All three implement `Distribution` and follow the same pattern: validate parameters, store them privately, then delegate sampling to `self._rng`.

| Class | Parameters | Domain | Mean | Variance |
|---|---|---|---|---|
| `ExponentialDistribution` | `rate` (λ) | (0, ∞) | `1/rate` | `1/rate²` |
| `NormalDistribution` | `mean`, `std` | (-∞, ∞) | `mean` | `std²` |
| `GammaDistribution` | `shape` (k), `scale` (θ) | (0, ∞) | `k·θ` | `k·θ²` |

Validation rules:

- `ExponentialDistribution`: `rate > 0`
- `NormalDistribution`: `std > 0`
- `GammaDistribution`: `shape > 0` and `scale > 0`

Sampling is a thin wrapper over the corresponding `Generator` method (`exponential`, `normal`, `gamma`) — no custom inverse-CDF code is needed.

## Discrete Distributions — `discrete.py`

| Class | Parameters | Support | Mean | Variance |
|---|---|---|---|---|
| `BernoulliDistribution` | `probability` p | {0, 1} | `p` | `p(1 − p)` |
| `GeometricDistribution` | `probability` p | {1, 2, …} | `1/p` | `(1 − p)/p²` |
| `WeightedDiscreteDistribution` | `values`, `weights` | user-supplied list | `Σ vᵢ pᵢ` | `Σ pᵢ (vᵢ − μ)²` |

Notes:

- `BernoulliDistribution` is implemented via `Generator.binomial(n=1, …)` cast to float for a uniform ndarray return type.
- `GeometricDistribution` uses NumPy's "number of trials until first success" convention, so the support starts at 1. Validation: `p ∈ (0, 1]`.
- `WeightedDiscreteDistribution` accepts any finite set of values paired with non-negative weights; weights are normalized internally to probabilities. The constructor checks length parity, non-negativity, and a positive total weight. Sampling uses `Generator.choice` with the precomputed probability vector.

## Poisson Process — `poisson.py`

`HomogeneousPoissonProcess` models a constant-rate stream of arrivals (e.g., packet generation). It is **not** a `Distribution` — it is a process.

```python
class HomogeneousPoissonProcess:
    def __init__(self, arrival_rate: float, rng: Generator | None = None)
    @property
    def arrival_rate(self) -> float
    def sample_inter_arrival_time(self) -> float
    def generate_arrival_times(self, start_time: float, end_time: float) -> list[float]
    def expected_arrivals(self, duration: float) -> float
```

Internally it holds an `ExponentialDistribution(rate=arrival_rate, rng=rng)` and steps through time by adding successive inter-arrival samples until the window is exhausted. The returned list is sorted by construction (events accumulate monotonically). `expected_arrivals` returns the analytic mean `λ · duration`.

This is the only piece of the module that produces *time* rather than *values*; everything else is a probability law on a support set.

## Public API — `__init__.py`

```python
from distributions.base import Distribution
from distributions.continuous import (
    ExponentialDistribution, GammaDistribution, NormalDistribution,
)
from distributions.discrete import (
    BernoulliDistribution, GeometricDistribution, WeightedDiscreteDistribution,
)
from distributions.poisson import HomogeneousPoissonProcess
```

`__all__` exposes the seven concrete classes plus the base. External code should import from `distributions` (not from the submodules) so the public surface stays stable as internals are reorganized.

## Class Hierarchy

```
                       ┌──────────────────────────┐
                       │  Distribution (ABC)      │
                       │  + rng                   │
                       │  + sample()   [abstract] │
                       │  + mean()     [abstract] │
                       │  + variance() [abstract] │
                       │  + std()                  │
                       └────────────┬─────────────┘
                                    │
                ┌───────────────────┼─────────────────────────┐
                │                   │                         │
        continuous.py         discrete.py                 poisson.py
                │                   │                         │
   ExponentialDistribution  BernoulliDistribution   HomogeneousPoissonProcess
   NormalDistribution       GeometricDistribution    (composes
   GammaDistribution        WeightedDiscreteDistribution   ExponentialDistribution)
```

## Extending the Module

To add a new distribution, e.g. a `UniformDistribution`:

1. Create or pick a file (`continuous.py` is appropriate for continuous supports).
2. Subclass `Distribution`.
3. Validate parameters in `__init__` and call `super().__init__(rng)`.
4. Implement `sample`, `mean`, and `variance` using `self._rng` for sampling.
5. Re-export the class from `__init__.py` and add it to `__all__`.
6. Add unit tests in `tests/distributions/` that compare sample moments to theoretical values.

The `std()` method is inherited unchanged; no override is needed.
