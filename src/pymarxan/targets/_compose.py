"""Simple target methods and combinators.

See the Stage 2 design spec §7.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    as_feature_id,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    per_feature,
    select_features,
)


def absolute_targets(
    problem: ConservationProblem,
    *,
    amount: PerFeatureFloat,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Explicit targets in native units (scalar or ``{feature_id: amount}``).

    Not clipped to the feature total: an explicit target is the caller's
    statement (``presolve_check`` flags targets above totals). Mirrors
    prioritizr 9.0.1 ``spec_absolute_targets``.
    """
    fids = select_features(problem, features)
    return numeric_per_feature("amount", amount, fids, lower=0.0)


def area_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    area_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Fixed-area targets: ``t = min(area_km2, cap_km2, range)``, in native units.

    Mirrors prioritizr 9.0.1 ``spec_area_targets``.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    a = numeric_per_feature("area_km2", area_km2, fids, lower=0.0)
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(a[fid], totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }


def rule_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    baseline: float,
    rules: Mapping[str, float],
    traits: Mapping[int, Collection[str]] | pd.Series,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Rule-based targets (Harris & Holness 2023).

    ``proportion = clamp(baseline + Σ rules[τ] for τ in traits[feature], 0, 1)``;
    ``t = min(proportion·range, cap_km2, range)``.

    Args:
        baseline: Baseline proportion in [0, 1].
        rules: Per-trait adjustment, each in [-1, 1], e.g.
            ``{"endemic": 0.2, "harvested": -0.1}``.
        traits: ``{feature_id: {"endemic", ...}}`` (or a Series indexed by
            feature id) for every selected feature; an empty collection is
            allowed. A bare string is rejected (it would be read as one trait
            per character): wrap it, ``{"endemic"}``. A trait missing from
            ``rules`` raises ``ValueError``.

    Harris & Holness 2023, *Biol. Conserv.* 285:110218,
    https://doi.org/10.1016/j.biocon.2023.110218. Mirrors prioritizr 9.0.1
    ``spec_rule_targets``, joined by feature id.
    """
    base = check_scalar("baseline", baseline, lower=0.0, upper=1.0)
    for name, adj in rules.items():
        check_scalar(f"rules[{name!r}]", adj, lower=-1.0, upper=1.0)
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    if not isinstance(traits, (Mapping, pd.Series)):
        raise ValueError("traits must map feature id to a collection of trait names")
    feature_traits = per_feature("traits", traits, fids)
    bare = [fid for fid, ts in feature_traits.items() if isinstance(ts, str)]
    if bare:
        raise ValueError(
            f"traits for feature(s) {bare} must be a collection of trait names, "
            'not a string; wrap it, e.g. {"endemic"}'
        )
    malformed = [
        fid
        for fid, ts in feature_traits.items()
        if not isinstance(ts, Collection) or any(not isinstance(t, str) for t in ts)
    ]
    if malformed:
        raise ValueError(
            f"traits for feature(s) {malformed} must be a collection of trait names (str)"
        )
    unknown = sorted({t for ts in feature_traits.values() for t in ts}.difference(rules))
    if unknown:
        raise ValueError(f"traits not in rules: {unknown}")
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        prop = base + sum(float(rules[t]) for t in feature_traits[fid])
        prop = min(max(prop, 0.0), 1.0)
        f = totals[fid] * factors[fid]
        out[fid] = finish_area(prop * f, f, factors[fid], cap_v)
    return out


def _combine(
    name: str,
    results: tuple[Mapping[int, float], ...],
    pick: Callable[[float, float], float],
) -> dict[int, float]:
    if len(results) < 2:
        raise ValueError(f"{name} needs at least 2 target mappings, got {len(results)}")
    first = [as_feature_id(f"{name} keys", k) for k in results[0]]
    keys = set(first)
    for i, res in enumerate(results[1:], start=1):
        other = {as_feature_id(f"{name} keys", k) for k in res}
        if other != keys:
            raise ValueError(
                f"{name}: input {i} has different features from input 0 "
                f"(only in input 0: {sorted(keys - other)}, "
                f"only in input {i}: {sorted(other - keys)})"
            )
    norm = [
        {as_feature_id(f"{name} keys", k): float(v) for k, v in res.items()}
        for res in results
    ]
    out: dict[int, float] = {}
    for fid in first:
        value = norm[0][fid]
        for res in norm[1:]:
            value = pick(value, res[fid])
        out[fid] = value
    return out


def min_targets(*results: Mapping[int, float]) -> dict[int, float]:
    """Feature-wise minimum of two or more target mappings (same feature ids)."""
    return _combine("min_targets", results, min)


def max_targets(*results: Mapping[int, float]) -> dict[int, float]:
    """Feature-wise maximum of two or more target mappings (same feature ids).

    Example: ``max_targets(area_targets(p, km2_per_unit=k, area_km2=1000),
    relative_targets(p, 0.3))``.
    """
    return _combine("max_targets", results, max)
