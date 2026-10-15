"""Population-based target methods (population size, Wilson, Durán).

See the Stage 2 design spec §6. Per-feature inputs are scalars or
``{feature_id: value}`` mappings; densities are individuals per km².
"""
from __future__ import annotations

import math
import sys
from collections.abc import Iterable

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    check_cap,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    select_features,
)

_LN_FLOAT_MAX = math.log(sys.float_info.max)
_WILSON_RTOL = 0.01


def _exp_or_inf(ln_x: float) -> float:
    return math.inf if ln_x > _LN_FLOAT_MAX else math.exp(ln_x)


def _lande_exact_x(base: float) -> float:
    """Solve ``expm1(x) - x = base`` for ``x > 0`` (Newton; convex, increasing).

    Lande's (1993) exact mean time to extinction, ``T·σ²·β²/2 = e^x − 1 − x``
    with ``x = β·ln K``; the large-population approximation drops ``− 1 − x``.
    """
    if not math.isfinite(base):
        return math.inf
    if base <= 0.0:  # reachable only via float underflow of 1e5·σ²·β²/2
        return 0.0
    if base < 1e-200:  # x*x would underflow; expm1(x) - x = x²/2 to double precision
        return math.sqrt(2.0 * base)
    x = max(math.log(base), 1.0) if base > 1.0 else math.sqrt(2.0 * base)
    for _ in range(200):
        if x < 1e-4:  # series for expm1(x) - x avoids cancellation
            g = x * x / 2.0 * (1.0 + x / 3.0 + x * x / 12.0)
            gp = x * (1.0 + x / 2.0 + x * x / 6.0)
        else:
            g = math.expm1(x) - x
            gp = math.expm1(x)
        step = (g - base) / gp
        x -= step
        if abs(step) <= 1e-15 * x:
            break
    return x


def pop_size_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    pop_size: PerFeatureFloat,
    density_per_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Area needed to hold a target population: ``t = min(n / d, cap, range)``.

    Args:
        pop_size: Target number of individuals ``n > 0`` (scalar or per feature).
        density_per_km2: Individuals per km² ``d > 0`` (scalar or per feature).
            Convert other densities first (per ha × 100 = per km²).

    prioritizr's docs write ``n·d``; its code and tests divide, as here.
    Mirrors prioritizr 9.0.1 ``spec_pop_size_targets``.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    n = numeric_per_feature("pop_size", pop_size, fids, lower=0.0, lower_inclusive=False)
    d = numeric_per_feature(
        "density_per_km2", density_per_km2, fids, lower=0.0, lower_inclusive=False
    )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(n[fid] / d[fid], totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }


def wilson_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    mean_growth_rate: PerFeatureFloat,
    var_growth_rate: PerFeatureFloat,
    density_per_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Wilson et al. (2010) persistence-based targets.

    ``β = 2r/σ² − 1``; the population needed to persist 100,000 years is
    ``k = (1e5·σ²·β²/2)^(1/β)``; ``t = min(k / d, cap, range)``. This is the
    large-population approximation of Lande's (1993) mean time to extinction
    under environmental stochasticity, solved for ``k``
    (https://doi.org/10.1086/285580).

    That approximation is accurate only when ``1e5·σ²·β²/2 ≫ 1``; near and
    below 1 it can give a near-zero target where the exact model needs an
    enormous population. So the target is also computed from Lande's exact
    model, and ``ValueError`` (naming the features) is raised wherever the two
    targets differ by more than 1% after the cap and clip; ``β <= 0``
    (``r <= σ²/2``) also raises. Where no error is raised the result is
    prioritizr's. For refused features, set a minimum viable population and
    use :func:`pop_size_targets`. ``k`` is computed in log space; an
    overflowing ``k`` is treated as infinite (target ``min(cap, range)``).

    Wilson et al. 2010, *Ecol. Appl.* 20:1721-1732,
    https://doi.org/10.1890/09-1051.1. The formula follows prioritizr 9.0.1
    ``spec_wilson_targets``; it could not be checked against the (closed
    access) paper text.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    r = numeric_per_feature(
        "mean_growth_rate", mean_growth_rate, fids, lower=0.0, lower_inclusive=False
    )
    s2 = numeric_per_feature(
        "var_growth_rate", var_growth_rate, fids, lower=0.0, lower_inclusive=False
    )
    d = numeric_per_feature(
        "density_per_km2", density_per_km2, fids, lower=0.0, lower_inclusive=False
    )
    beta = {fid: 2.0 * r[fid] / s2[fid] - 1.0 for fid in fids}
    bad = [fid for fid in fids if beta[fid] <= 0.0]
    if bad:
        raise ValueError(
            "wilson_targets needs mean_growth_rate > var_growth_rate / 2 "
            f"(beta > 0); invalid for feature(s) {bad}"
        )
    base = {fid: 1e5 * s2[fid] * beta[fid] ** 2 / 2.0 for fid in fids}
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    inaccurate: list[int] = []
    for fid in fids:
        k_f = factors[fid]
        f = totals[fid] * k_f
        ln_base = math.log(base[fid]) if base[fid] > 0.0 else -math.inf
        k_approx = _exp_or_inf(ln_base / beta[fid])
        k_exact = _exp_or_inf(_lande_exact_x(base[fid]) / beta[fid])
        t_approx = finish_area(k_approx / d[fid], f, k_f, cap_v)
        t_exact = finish_area(k_exact / d[fid], f, k_f, cap_v)
        if abs(t_approx - t_exact) > _WILSON_RTOL * t_exact:
            inaccurate.append(fid)
        out[fid] = t_approx
    if inaccurate:
        raise ValueError(
            "wilson_targets' large-population approximation differs from Lande's exact "
            f"model by more than {_WILSON_RTOL:.0%} for feature(s) {inaccurate} "
            "(1e5·σ²·β²/2 is not much larger than 1: r near σ²/2, or σ² small); set a "
            "minimum viable population and use pop_size_targets instead"
        )
    return out


def duran_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    probability_target: PerFeatureFloat,
    historical_area_km2: PerFeatureFloat,
    cap_km2: float | None = None,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Durán et al. (2020) persistence-probability targets.

    Persistence ``P = (A / A_hist)^0.25`` solved for area: ``t = min(h·p⁴, range)``
    (and ``cap_km2`` if given; no cap by default, as in prioritizr).

    Args:
        probability_target: ``p`` in [0, 1] (scalar or per feature). Published
            per-category values: Davis et al. 2018 (LC 0.9983, NT 0.9859,
            VU 0.9, EN 0.3277, CR 0.001); Gumbs et al. 2023 (LC 0.939375,
            NT 0.87875, VU 0.7575, EN 0.515, CR 0.03).
        historical_area_km2: Historical range ``h > 0`` in km².

    Durán et al. 2020, *Methods Ecol. Evol.* 11:910-921,
    https://doi.org/10.1111/2041-210X.13427 (z = 0.25).
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    p = numeric_per_feature("probability_target", probability_target, fids, lower=0.0, upper=1.0)
    h = numeric_per_feature(
        "historical_area_km2", historical_area_km2, fids, lower=0.0, lower_inclusive=False
    )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(h[fid] * p[fid] ** 4, totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }
