"""Range-size interpolation target methods.

``interp_area_targets`` / ``interp_absolute_targets`` implement the general
rule from prioritizr 9.0.1 (``spec_interp_area_targets`` /
``spec_interp_absolute_targets``); the presets fix its parameters to the
published values. See ``docs/plans/2026-10-15-stage2-target-methods-design.md`` §4.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Literal

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    select_features,
)

Interp = Literal["log10", "linear"]
Method = Literal["min", "max"]

_METHODS: dict[str, Callable[[float, float], float]] = {"min": min, "max": max}
_INTERPS = ("log10", "linear")


def _opt_float(x: float | None) -> float | None:
    return None if x is None else float(x)


def _validate_rule(
    *,
    unit: str,
    a: float,
    b: float,
    c: float | None,
    rare_method: Method | None,
    e: float,
    g: float,
    h: float | None,
    common_method: Method | None,
    interp: Interp,
) -> None:
    check_scalar(f"rare_{unit}_threshold", a, lower=0.0)
    check_scalar(f"common_{unit}_threshold", e, lower=0.0)
    if a > e:
        raise ValueError(
            f"rare_{unit}_threshold ({a}) must not exceed common_{unit}_threshold ({e})"
        )
    if interp not in _INTERPS:
        raise ValueError(f"interp must be one of {_INTERPS}, got {interp!r}")
    if interp == "log10" and a <= 0:
        raise ValueError(f"interp='log10' requires rare_{unit}_threshold > 0, got {a}")
    check_scalar("rare_relative_target", b, lower=0.0, upper=1.0)
    check_scalar("common_relative_target", g, lower=0.0, upper=1.0)
    for side, target, method in (("rare", c, rare_method), ("common", h, common_method)):
        if (target is None) != (method is None):
            raise ValueError(
                f"{side}_{unit}_target and {side}_method must be given together"
            )
        if target is not None:
            check_scalar(f"{side}_{unit}_target", target, lower=0.0)
            if method not in _METHODS:
                raise ValueError(f"{side}_method must be 'min' or 'max', got {method!r}")


def _rule(
    f: float,
    *,
    a: float,
    b: float,
    c: float | None,
    rare_method: Method | None,
    e: float,
    g: float,
    h: float | None,
    common_method: Method | None,
    interp: Interp,
) -> float:
    """Un-capped, un-clipped target for total ``f`` (same unit as thresholds)."""
    if f < a or (a == e and f == a):
        t = b * f
        if c is not None and rare_method is not None:
            t = _METHODS[rare_method](c, t)
        return t
    if f > e:
        t = g * f
        if h is not None and common_method is not None:
            t = _METHODS[common_method](h, t)
        return t
    if interp == "log10":
        x, x1, x2 = math.log10(f), math.log10(a), math.log10(e)
    else:
        x, x1, x2 = f, a, e
    return f * (b + (g - b) * (x - x1) / (x2 - x1))


def interp_absolute_targets(
    problem: ConservationProblem,
    *,
    rare_absolute_threshold: float,
    rare_relative_target: float,
    common_absolute_threshold: float,
    common_relative_target: float,
    interp: Interp,
    rare_absolute_target: float | None = None,
    rare_method: Method | None = None,
    common_absolute_target: float | None = None,
    common_method: Method | None = None,
    cap: float | None = None,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Interpolated targets on feature totals in **native units**.

    Features whose total is below ``rare_absolute_threshold`` get
    ``rare_relative_target`` of their total (optionally combined with
    ``rare_absolute_target`` via ``rare_method``); above
    ``common_absolute_threshold`` they get ``common_relative_target``
    (optionally combined with ``common_absolute_target`` via
    ``common_method``); in between the proportion is interpolated on
    ``log10`` of the total (``interp="log10"``) or linearly
    (``interp="linear"``). The result is capped at ``cap`` (if given) and
    clipped to the feature total. When both thresholds are equal, a total
    exactly at the threshold takes the rare branch.

    Replaces the deprecated :func:`loglinear_targets`. Mirrors prioritizr
    9.0.1 ``spec_interp_absolute_targets``.

    Returns:
        Mapping of feature id to target amount (native units).
    """
    _validate_rule(
        unit="absolute",
        a=rare_absolute_threshold,
        b=rare_relative_target,
        c=rare_absolute_target,
        rare_method=rare_method,
        e=common_absolute_threshold,
        g=common_relative_target,
        h=common_absolute_target,
        common_method=common_method,
        interp=interp,
    )
    cap_v = check_cap("cap", cap)
    fids = select_features(problem, features)
    totals = feature_totals(problem, fids)
    out: dict[int, float] = {}
    for fid in fids:
        f = totals[fid]
        if f == 0.0:
            out[fid] = 0.0
            continue
        t = _rule(
            f,
            a=float(rare_absolute_threshold),
            b=float(rare_relative_target),
            c=_opt_float(rare_absolute_target),
            rare_method=rare_method,
            e=float(common_absolute_threshold),
            g=float(common_relative_target),
            h=_opt_float(common_absolute_target),
            common_method=common_method,
            interp=interp,
        )
        out[fid] = finish_area(t, f, 1.0, cap_v)
    return out


def interp_area_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float,
    rare_relative_target: float,
    common_area_threshold: float,
    common_relative_target: float,
    interp: Interp,
    rare_area_target: float | None = None,
    rare_method: Method | None = None,
    common_area_target: float | None = None,
    common_method: Method | None = None,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Interpolated range-size targets, with thresholds and targets in km².

    Same rule as :func:`interp_absolute_targets`, applied to each feature's
    total converted to km² with ``km2_per_unit`` (a factor or a
    ``{feature_id: factor}`` mapping; required, no default). Targets are
    capped at ``cap_km2`` (default 1,000,000 km², Butchart et al. 2015,
    https://doi.org/10.1111/conl.12158; ``None`` disables), clipped to the
    feature total, and returned in native units. Mirrors prioritizr 9.0.1
    ``spec_interp_area_targets``.

    Returns:
        Mapping of feature id to target amount (native units).
    """
    _validate_rule(
        unit="area",
        a=rare_area_threshold,
        b=rare_relative_target,
        c=rare_area_target,
        rare_method=rare_method,
        e=common_area_threshold,
        g=common_relative_target,
        h=common_area_target,
        common_method=common_method,
        interp=interp,
    )
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        k = factors[fid]
        f = totals[fid] * k
        if f == 0.0:
            out[fid] = 0.0
            continue
        t = _rule(
            f,
            a=float(rare_area_threshold),
            b=float(rare_relative_target),
            c=_opt_float(rare_area_target),
            rare_method=rare_method,
            e=float(common_area_threshold),
            g=float(common_relative_target),
            h=_opt_float(common_area_target),
            common_method=common_method,
            interp=interp,
        )
        out[fid] = finish_area(t, f, k, cap_v)
    return out


def rodrigues_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 1_000.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 250_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Rodrigues et al. (2004) range-size targets.

    100% of range below 1,000 km², 10% above 250,000 km², interpolated on
    log10 of range size in between, capped at 1,000,000 km² (Butchart et al.
    2015). Rodrigues et al. 2004, *BioScience* 54:1092-1100,
    https://doi.org/10.1641/0006-3568(2004)054[1092:GGAPRF]2.0.CO;2 — rule and
    cap as restated in Butchart et al. 2015's methods. Mirrors prioritizr 9.0.1
    ``spec_rodrigues_targets``.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="log10",
        cap_km2=cap_km2,
        features=features,
    )


def sreekar_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 200.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 250_000.0,
    common_relative_target: float = 0.3,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Sreekar & Watson (2026) updated range-size targets.

    100% below 200 km², 30% above 250,000 km², log10 interpolation, capped at
    1,000,000 km². Sreekar & Watson 2026, *Nature Reviews Biodiversity*
    2:419-420, https://doi.org/10.1038/s44358-026-00169-7. The numeric values
    follow prioritizr 9.0.1 ``spec_sreekar_targets``; they could not be
    confirmed from accessible paper text.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="log10",
        cap_km2=cap_km2,
        features=features,
    )


def polak_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 1_000.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 10_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Polak et al. (2015) range-size targets.

    100% below 1,000 km², 10% above 10,000 km², **linear** interpolation in
    between, capped at 1,000,000 km². The absolute target is non-monotone in
    range size (9,000 km² → 1,800 km² but 10,000 km² → 1,000 km²); that is
    inherent to the rule. Polak et al. 2015, *R. Soc. Open Sci.* 2:150107,
    https://doi.org/10.1098/rsos.150107 — the breakpoints and linear
    interpolation are stated in its Methods. Polak et al. also set 100% for
    Critically Endangered species regardless of range; this preset (like
    prioritizr's) does not: target CR features separately, e.g.
    ``apply_targets(problem, group_targets(problem, {f: "CR" for f in cr_ids},
    {"CR": 1.0}))``.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="linear",
        cap_km2=cap_km2,
        features=features,
    )


def watson_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 10_000.0,
    rare_relative_target: float = 1.0,
    rare_area_target: float = 1_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Watson et al. (2010) range-size targets.

    Below 10,000 km² the target is ``min(range, 1,000 km²)``; above it, 10% of
    range; capped at 1,000,000 km². The common threshold always equals
    ``rare_area_threshold``; a range exactly at it takes the rare branch
    (prioritizr errors there). Watson et al. 2010, *Conservation Biology*
    25:324-332, https://doi.org/10.1111/j.1523-1739.2010.01587.x. Values follow
    prioritizr 9.0.1 ``spec_watson_targets``; they could not be confirmed from
    the paper text. Polak et al. (2015), which says it follows Watson et al.,
    describes a different shape (linear 100% → 10% between 1,000 and
    10,000 km², see :func:`polak_targets`); check the Watson et al. paper
    before relying on this preset's exact shape.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        rare_area_target=rare_area_target,
        rare_method="min",
        common_area_threshold=rare_area_threshold,
        common_relative_target=common_relative_target,
        interp="linear",
        cap_km2=cap_km2,
        features=features,
    )
