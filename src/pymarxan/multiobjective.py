"""Exact multi-objective planning on one ``ConservationProblem`` (Stage 3, prioritizr 9 parity).

Objectives are specs (``MinCost``, ``MinBoundary``, ``MinShortfall``, ``MinLargestShortfall``,
``MaxFeatures``, ``MaxWeightedFeatures``, ``MaxWtdSum``) combined by an approach:

- ``Hierarchical(rel_tol, abs_tol=1e-9)``: lexicographic, in list order. Each stage caps the
  objective just solved at ``v + r·|v| + abs_tol`` (minimise) or ``v − (r·|v| + abs_tol)``
  (maximise) and solves the next. At ``v = 0`` only ``abs_tol`` is allowed.
- ``WeightedSum(weights)``: one solve of ``Σ (w/Σw)·s·f`` per weight row (s = −1 for a
  maximised objective). Objectives are **not** scaled: objectives in different units need
  weights scaled to their magnitudes (Das & Dennis 1997,
  https://doi.org/10.1007/BF01197559).

Targets are soft (only representation objectives use them) and ``COSTBUDGET`` is optional.
``weights_matrix`` / ``rel_tol_matrix`` build prioritizr's parameter sweeps and
``nondominated`` filters the resulting solutions. Deviations from prioritizr 9.0.1 are listed
in the CHANGELOG for v0.41.0 (Hanson et al. 2024, https://doi.org/10.1111/cobi.14376;
Williams & Kendall 2017, https://doi.org/10.1016/j.ecolmodel.2016.10.010).
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar

from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    ObjectiveTerm,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution

__all__ = [
    "Hierarchical",
    "MaxFeatures",
    "MaxWeightedFeatures",
    "MaxWtdSum",
    "MinBoundary",
    "MinCost",
    "MinLargestShortfall",
    "MinShortfall",
    "ObjectiveSpec",
    "WeightedSum",
    "nondominated",
    "rel_tol_matrix",
    "weights_matrix",
]


# --- objective specs -------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective. ``key`` (``label or name``) names its reported value and owns its rows.

    ``label`` must be a Python identifier: PuLP rewrites other characters in names, so two
    distinct labels could collide.
    """

    name: ClassVar[str] = ""
    sense: ClassVar[str] = "min"
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.label is not None and not (
            isinstance(self.label, str) and self.label.isidentifier()
        ):
            raise ValueError(f"label must be a Python identifier, got {self.label!r}")

    @property
    def key(self) -> str:
        return self.label or self.name

    def _shortfall(self) -> str:
        return "relative"

    def _params(self) -> dict[str, Any]:
        return {}

    def build(self, ctx: ObjectiveContext) -> ObjectiveTerm:
        """The builder's term; owned names are prefixed ``f"{key}__"``. Call once per ctx."""
        return BUILDERS[self.name](
            ctx, self._shortfall(), owner=f"{self.key}__", **self._params(),
        )


@dataclass(frozen=True)
class MinCost(ObjectiveSpec):
    """cost + BLM·boundary + connectivity + probability penalty (what ``min_set`` minimises).

    With BLM > 0 it already counts boundary, so a weighted sum with ``MinBoundary`` counts
    boundary twice.
    """

    name: ClassVar[str] = "min_cost"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinBoundary(ObjectiveSpec):
    """The boundary length alone, unweighted by BLM. Needs a boundary table."""

    name: ClassVar[str] = "min_boundary"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinShortfall(ObjectiveSpec):
    """Σ SPF·shortfall, relative (÷ target × MISSLEVEL, prioritizr) or absolute."""

    name: ClassVar[str] = "min_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MinLargestShortfall(ObjectiveSpec):
    """The largest (relative or absolute) shortfall over targets, unweighted."""

    name: ClassVar[str] = "min_largest_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MaxFeatures(ObjectiveSpec):
    """Count of targets met (features with target > 0)."""

    name: ClassVar[str] = "max_features"
    sense: ClassVar[str] = "max"


@dataclass(frozen=True)
class MaxWeightedFeatures(ObjectiveSpec):
    """SPF-weighted count of targets met."""

    name: ClassVar[str] = "max_weighted_features"
    sense: ClassVar[str] = "max"


@dataclass(frozen=True)
class MaxWtdSum(ObjectiveSpec):
    """Σ w_j · amount held_j (prioritizr ``max_wtd_sum``).

    ``weights``: ``None`` (every feature 1) or ``{feature_id: w}`` covering every feature in
    the problem (finite, any sign); coverage is checked at ``solve``.

    Caution (as prioritizr notes for its weighted-sum objectives): a weighted sum of amounts
    ignores complementarity. More of an already well-held feature scores as much as the
    same amount of a missing one, so the plan can leave features unrepresented; pair it
    with a shortfall objective when representation matters.
    """

    name: ClassVar[str] = "max_wtd_sum"
    sense: ClassVar[str] = "max"
    weights: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weights is not None:
            bad = sorted(k for k, w in self.weights.items() if not _finite(w))
            if bad:
                raise ValueError(f"weights must be finite numbers; bad feature ids {bad}")
            object.__setattr__(
                self, "weights", {int(k): float(w) for k, w in self.weights.items()},
            )

    def _params(self) -> dict[str, Any]:
        return {"weights": self.weights}


# --- approaches -----------------------------------------------------------------


def _finite(v: object) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _nonneg(name: str, v: object) -> float:
    if not _finite(v) or float(v) < 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} values must be finite and >= 0, got {v!r}")
    return float(v)  # type: ignore[arg-type]


def _as_rows(name: str, value: object) -> list[list[float]]:
    """A float -> one row of one value; a 1-D sequence -> one row; a 2-D -> its rows."""
    if hasattr(value, "tolist"):  # numpy array or scalar
        value = value.tolist()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return [[_nonneg(name, value)]]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a number, a sequence or a 2-D sequence")
    items = list(value)
    if all(isinstance(v, Sequence) and not isinstance(v, str) for v in items) and items:
        return [[_nonneg(name, v) for v in row] for row in items]
    return [[_nonneg(name, v) for v in items]]


@dataclass(frozen=True)
class Hierarchical:
    """Lexicographic optimisation in list order (spec §4.1).

    ``rel_tol``: a float (every stage), n−1 floats, or a 2-D sequence (one row of n−1 per
    run); finite, >= 0, values > 1 allowed. The k-th tolerance caps the k-th stage.
    """

    rel_tol: float | Sequence[float] | Sequence[Sequence[float]]
    abs_tol: float = 1e-9

    def __post_init__(self) -> None:
        _as_rows("rel_tol", self.rel_tol)
        _nonneg("abs_tol", self.abs_tol)

    def rows(self, n: int) -> list[list[float]]:
        """One rel_tol row of n−1 values per run."""
        if isinstance(self.rel_tol, numbers.Real):
            return [[float(self.rel_tol)] * (n - 1)]
        rows = _as_rows("rel_tol", self.rel_tol)
        for i, row in enumerate(rows):
            if len(row) != n - 1:
                raise ValueError(
                    f"rel_tol row {i} has {len(row)} values; {n} objectives need {n - 1}"
                )
        return rows


@dataclass(frozen=True)
class WeightedSum:
    """One solve of ``Σ (w/Σw)·s·f`` per row (spec §4.2); no objective scaling.

    ``weights``: n floats, or a 2-D sequence (one row of n per run); finite, >= 0, and no
    row may sum to 0.
    """

    weights: Sequence[float] | Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        for i, row in enumerate(_as_rows("weights", self.weights)):
            if sum(row) == 0:
                raise ValueError(f"weights row {i} sums to 0")

    def rows(self, n: int) -> list[list[float]]:
        rows = _as_rows("weights", self.weights)
        for i, row in enumerate(rows):
            if len(row) != n:
                raise ValueError(f"weights row {i} has {len(row)} values; expected {n}")
        return rows


# --- prioritizr parameter matrices ------------------------------------------------------


def _check_matrix_args(n_objectives: int, n_values: int) -> tuple[int, int]:
    """Both sizes as ``int`` (numpy integers accepted, ``bool`` rejected)."""
    for name, v in (("n_objectives", n_objectives), ("n_values", n_values)):
        if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < 2:
            raise ValueError(f"{name} must be an integer >= 2, got {v!r}")
    return int(n_objectives), int(n_values)


def _grid(values: list[float], n_cols: int) -> list[list[float]]:
    """R ``expand.grid`` row order: the first column varies fastest."""
    return [list(reversed(combo)) for combo in product(values, repeat=n_cols)]


def _points(top: float, n_values: int, include_zeros: bool) -> list[float]:
    if include_zeros:
        return [top * i / (n_values - 1) for i in range(n_values)]
    return [top * i / n_values for i in range(1, n_values + 1)]


def weights_matrix(
    n_objectives: int,
    n_values: int,
    *,
    include_zeros: bool = True,
    include_extremes: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_weights_matrix`` (9.0.1), row for row.

    A grid of ``n_values`` points in [0, 1] (with ``include_zeros=False``, the
    ``n_values`` non-zero points of an ``n_values + 1`` grid); rows whose values are all
    equal, or with fewer than two entries > 1e-6, are dropped; the identity rows are
    prepended if ``include_extremes`` and a row of ones always comes first. Despite
    prioritizr's documentation, proportional duplicates such as (0.5, 0.25) and (1, 0.5)
    are kept.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    rows = [
        row for row in _grid(_points(1.0, n_values, include_zeros), n_objectives)
        if len(set(row)) > 1 and sum(v > 1e-6 for v in row) >= 2
    ]
    if include_extremes:
        identity = [
            [1.0 if i == j else 0.0 for j in range(n_objectives)] for i in range(n_objectives)
        ]
        rows = identity + rows
    return [[1.0] * n_objectives] + rows


def rel_tol_matrix(
    n_objectives: int,
    n_values: int,
    max: float,  # prioritizr's argument name
    *,
    include_zeros: bool = True,
    order: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_rel_tol_matrix`` (9.0.1), row for row.

    A grid over n−1 columns of ``n_values`` points in [0, max] (with
    ``include_zeros=False``, the ``n_values`` non-zero points of an ``n_values + 1`` grid).
    With ``order``, only rows that are non-increasing across columns are kept, so an
    earlier stage gets at least the tolerance of a later one.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    if not _finite(max) or max < 0:
        raise ValueError(f"max must be finite and >= 0, got {max!r}")
    rows = _grid(_points(float(max), n_values, include_zeros), n_objectives - 1)
    if order:
        rows = [r for r in rows if all(r[i - 1] >= r[i] for i in range(1, len(r)))]
    return rows


# --- nondominated -------------------------------------------------------------------


def nondominated(solutions: Sequence[Solution]) -> list[Solution]:
    """The solutions no other solution dominates, in input order.

    Uses ``metadata["objective_values"]`` and ``["objective_senses"]``. A solution is
    dominated if another is at least as good on every key and strictly better on one.
    Values are compared exactly. Of identical value vectors only the first is kept.
    """
    if not solutions:
        return []
    keys: list[str] | None = None
    senses: dict[str, str] = {}
    vectors: list[list[float]] = []
    for i, sol in enumerate(solutions):
        values = sol.metadata.get("objective_values")
        sol_senses = sol.metadata.get("objective_senses")
        if not isinstance(values, Mapping) or not isinstance(sol_senses, Mapping):
            raise ValueError(
                f"solution {i} lacks metadata 'objective_values' / 'objective_senses'"
            )
        if keys is None:
            keys, senses = sorted(values), dict(sol_senses)
        if sorted(values) != keys or dict(sol_senses) != senses or set(senses) != set(keys):
            raise ValueError(f"solution {i} has different objective keys or senses")
        # Orient every key as a minimisation.
        vectors.append([
            float(values[k]) if senses[k] == "min" else -float(values[k]) for k in keys
        ])

    def dominates(a: list[float], b: list[float]) -> bool:
        return all(p <= q for p, q in zip(a, b, strict=True)) and a != b

    kept: list[Solution] = []
    for i, vec in enumerate(vectors):
        if any(dominates(other, vec) for other in vectors):
            continue
        if any(vectors[j] == vec for j in range(i)):
            continue
        kept.append(solutions[i])
    return kept
