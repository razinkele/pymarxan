"""Shared validation and unit helpers for the target-setting methods.

Private module: every public method in :mod:`pymarxan.targets` funnels its
inputs through these helpers so the contract (feature selection, per-feature
inputs keyed by feature id, explicit km² conversion, cap, clip to the
feature total) is enforced in one place.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem

# Input types accepted by the public methods (exported from pymarxan.targets):
# one value for every feature, a ``{feature_id: value}`` mapping, or a pandas
# Series indexed by feature id.
PerFeatureFloat: TypeAlias = float | Mapping[int, float] | pd.Series
PerFeatureStr: TypeAlias = str | Mapping[int, str] | pd.Series


def as_feature_id(name: str, key: object) -> int:
    """Return ``key`` as an int feature id; only integral numbers are accepted.

    Accepts ``int``, numpy integers and integral floats (``3.0``); rejects
    booleans, non-integral floats and strings with a ``ValueError`` naming
    ``name``, instead of silently truncating or misreading them.
    """
    if isinstance(key, (bool, np.bool_)):
        raise ValueError(f"{name} must be feature ids (int), got a boolean {key!r}")
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, (float, np.floating)) and float(key).is_integer():
        return int(key)
    raise ValueError(f"{name} must be feature ids (int), got {key!r}")


def select_features(
    problem: ConservationProblem, features: Iterable[int] | None
) -> list[int]:
    """Feature ids to compute, in ``problem.features`` order.

    A boolean mask (e.g. ``df["threatened"]``) is rejected: pass the ids,
    e.g. ``df.loc[df["threatened"], "id"]``.
    """
    ids = [int(f) for f in problem.features["id"]]
    if features is None:
        return ids
    if isinstance(features, pd.Series) and features.dtype == bool:
        raise ValueError(
            "features must be feature ids, not a boolean mask; "
            "use e.g. df.loc[mask, 'id']"
        )
    try:
        items = list(features)
    except TypeError:
        raise ValueError(
            f"features must be an iterable of feature ids, got {features!r}"
        ) from None
    wanted = {as_feature_id("features", f) for f in items}
    unknown = sorted(wanted.difference(ids))
    if unknown:
        raise ValueError(f"features not in problem: {unknown}")
    return [f for f in ids if f in wanted]


def per_feature(name: str, value: Any, fids: list[int]) -> dict[int, Any]:
    """Expand a scalar or a ``{feature_id: value}`` mapping to every id in ``fids``.

    A :class:`pandas.Series` is read as a mapping from its index (a duplicated
    index raises, and so does a default ``RangeIndex``, which holds row
    positions rather than ids: index it by feature id first). Keys must be
    feature ids. Extra keys are ignored; a missing
    key raises ``ValueError`` naming the ids.
    """
    if isinstance(value, pd.Series):
        if isinstance(value.index, pd.RangeIndex) and value.index.equals(
            pd.RangeIndex(len(value))
        ):
            raise ValueError(
                f"{name} is a pandas Series with a default RangeIndex (row positions, "
                "not feature ids); index it by feature id, e.g. "
                "df.set_index('id')[column], or pass .to_dict() if the ids really "
                "are 0..n-1"
            )
        dup = value.index[value.index.duplicated()]
        if len(dup):
            raise ValueError(
                f"{name} has duplicate entries for feature(s) {sorted(set(dup.tolist()))}"
            )
        value = value.to_dict()
    if isinstance(value, Mapping):
        by_id = {as_feature_id(f"{name} keys", k): v for k, v in value.items()}
        missing = [f for f in fids if f not in by_id]
        if missing:
            raise ValueError(f"{name} has no value for feature(s) {missing}")
        return {f: by_id[f] for f in fids}
    return {f: value for f in fids}


def numeric_per_feature(
    name: str,
    value: Any,
    fids: list[int],
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> dict[int, float]:
    """Like :func:`per_feature`, but every value must be a finite float in range."""
    raw = per_feature(name, value, fids)
    out: dict[int, float] = {}
    bad: list[int] = []
    for f, v in raw.items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            bad.append(f)
            continue
        if not math.isfinite(x):
            bad.append(f)
        elif lower is not None and (x < lower or (x == lower and not lower_inclusive)):
            bad.append(f)
        elif upper is not None and x > upper:
            bad.append(f)
        else:
            out[f] = x
    if bad:
        op = ">=" if lower_inclusive else ">"
        bounds = ", ".join(
            b
            for b in (
                None if lower is None else f"{op} {lower}",
                None if upper is None else f"<= {upper}",
            )
            if b
        )
        raise ValueError(
            f"{name} must be finite{' and ' + bounds if bounds else ''}; "
            f"invalid for feature(s) {bad}"
        )
    return out


def check_scalar(
    name: str,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> float:
    """Validate one scalar argument: finite and (optionally) within bounds."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    ok = math.isfinite(x)
    if ok and lower is not None:
        ok = x > lower or (lower_inclusive and x == lower)
    if ok and upper is not None:
        ok = x <= upper
    if not ok:
        raise ValueError(f"{name} out of range: {value!r}")
    return x


def check_cap(name: str, cap: float | None) -> float | None:
    """``None`` (no cap) or a finite value >= 0."""
    if cap is None:
        return None
    return check_scalar(name, cap, lower=0.0)


def feature_totals(problem: ConservationProblem, fids: list[int]) -> dict[int, float]:
    """Native-unit total amount per feature (0.0 for features with no rows).

    Totals include every planning unit, locked-out (status 3) ones too, as
    prioritizr does. Raises ``ValueError`` naming the features when any of
    their ``pu_vs_features`` amounts is negative or non-finite (pandas would
    otherwise skip NaN rows silently).
    """
    pvf = problem.pu_vs_features
    rows = pvf[pvf["species"].isin(fids)]
    amounts = rows["amount"].astype(float)
    bad_rows = rows.loc[~np.isfinite(amounts) | (amounts < 0), "species"]
    if len(bad_rows):
        raise ValueError(
            "pu_vs_features amounts must be finite and >= 0; invalid for feature(s) "
            f"{sorted({int(f) for f in bad_rows})}"
        )
    totals = problem.feature_amounts()
    return {f: float(totals.get(f, 0.0)) for f in fids}


def km2_factors(km2_per_unit: Any, fids: list[int]) -> dict[int, float]:
    """Resolve ``km2_per_unit`` (scalar or mapping) to a factor > 0 per feature."""
    return numeric_per_feature(
        "km2_per_unit", km2_per_unit, fids, lower=0.0, lower_inclusive=False
    )


def finish_area(t_km2: float, f_km2: float, k: float, cap_km2: float | None) -> float:
    """Clip a km² target to the feature total and the cap, return native units."""
    if f_km2 == 0.0:
        return 0.0
    t = min(t_km2, f_km2)
    if cap_km2 is not None:
        t = min(t, cap_km2)
    return t / k
