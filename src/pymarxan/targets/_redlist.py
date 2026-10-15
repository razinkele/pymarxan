"""IUCN Red List target methods (species and ecosystems).

The target is the area that would leave a feature qualifying at no worse
than ``status`` if only the protected part survived:
``t = min(m(B·(1+p), f·(1−A)·(1+p)), cap, f)`` with ``A`` the criterion-A
reduction threshold and ``B`` the criterion-B range threshold (km²).
Mirrors prioritizr 9.0.1 ``spec_rl_species_targets`` /
``spec_rl_ecosystem_targets``. See the Stage 2 design spec §5.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    PerFeatureStr,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    per_feature,
    select_features,
)

_Table = dict[str, dict[str, float]]

_STATUSES = ("CR", "EN", "VU")
_A2_SPECIES = {"CR": 0.8, "EN": 0.5, "VU": 0.3}

# IUCN (2012) Red List Categories and Criteria v3.1: criterion A reduction thresholds and
# criterion B range thresholds (km²; B1 = extent of occurrence, B2 = area of
# occupancy). Species EN-B2 is 500 km² (prioritizr's docs say 50; its code
# and IUCN say 500).
_SPECIES_A: _Table = {
    "A1": {"CR": 0.9, "EN": 0.7, "VU": 0.5},
    "A2": _A2_SPECIES,
    "A3": _A2_SPECIES,
    "A4": _A2_SPECIES,
}
_SPECIES_B: _Table = {
    "B1": {"CR": 100.0, "EN": 5_000.0, "VU": 20_000.0},
    "B2": {"CR": 10.0, "EN": 500.0, "VU": 2_000.0},
}

_A1_ECO = {"CR": 0.8, "EN": 0.5, "VU": 0.3}
# IUCN Red List of Ecosystems guidelines v2.0. A3 VU is 0.5 (IUCN thresholds
# 90/70/50%); prioritizr 9.0.1 has 0.3 — a deliberate deviation.
_ECOSYSTEM_A: _Table = {
    "A1": _A1_ECO,
    "A2A": _A1_ECO,
    "A2B": _A1_ECO,
    "A3": {"CR": 0.9, "EN": 0.7, "VU": 0.5},
}
_ECOSYSTEM_B: _Table = {
    "B1": {"CR": 2_000.0, "EN": 20_000.0, "VU": 50_000.0},
    "B2": {"CR": 200.0, "EN": 2_000.0, "VU": 5_000.0},
}


def _norm(value: object) -> str:
    return str(value).strip().upper()


def _rl_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float,
    method: Literal["max", "min"],
    cap_km2: float | None,
    features: Iterable[int] | None,
    a_table: _Table,
    b_table: _Table,
) -> dict[int, float]:
    crit_a = _norm(criterion_a)
    crit_b = _norm(criterion_b)
    if crit_a not in a_table:
        raise ValueError(f"criterion_a must be one of {sorted(a_table)}, got {criterion_a!r}")
    if crit_b not in b_table:
        raise ValueError(f"criterion_b must be one of {sorted(b_table)}, got {criterion_b!r}")
    p = check_scalar("prop_uplift", prop_uplift, lower=0.0)
    if method not in ("max", "min"):
        raise ValueError(f"method must be 'max' or 'min', got {method!r}")
    combine = max if method == "max" else min
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    statuses = {f: _norm(s) for f, s in per_feature("status", status, fids).items()}
    bad = [f for f, s in statuses.items() if s not in _STATUSES]
    if bad:
        raise ValueError(
            f"status must be one of {list(_STATUSES)}; invalid for feature(s) {bad}"
        )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        k = factors[fid]
        f = totals[fid] * k
        reduction = a_table[crit_a][statuses[fid]]
        range_km2 = b_table[crit_b][statuses[fid]]
        t = combine(range_km2 * (1.0 + p), f * (1.0 - reduction) * (1.0 + p))
        out[fid] = finish_area(t, f, k, cap_v)
    return out


def rl_species_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float = 0.0,
    method: Literal["max", "min"] = "max",
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """IUCN Red List of Threatened Species targets.

    ``t = min(m(B·(1+p), f·(1−A)·(1+p)), cap_km2, f)`` where ``A`` is the
    ``criterion_a`` reduction threshold (A1-A4) and ``B`` the ``criterion_b``
    range threshold (B1 EOO / B2 AOO, km²) for each feature's ``status``.

    Args:
        status: ``"CR"``, ``"EN"`` or ``"VU"`` for every feature, or a
            ``{feature_id: status}`` mapping (case-insensitive). Other
            categories raise ``ValueError``; select threatened features with
            ``features=`` and target the rest with another method.
        criterion_a: ``"A1"``-``"A4"`` (case-insensitive).
        criterion_b: ``"B1"`` or ``"B2"`` (case-insensitive).
        prop_uplift: Proportional uplift ``p >= 0``.
        method: ``"max"`` (default) or ``"min"`` combination of the two terms.

    Thresholds from IUCN (2012) Red List Categories and Criteria, Version 3.1,
    2nd edition (Gland & Cambridge: IUCN); cap from Butchart et al. 2015.
    Mirrors prioritizr 9.0.1 ``spec_rl_species_targets`` but accepts a
    per-feature status.
    """
    return _rl_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a=criterion_a,
        criterion_b=criterion_b,
        prop_uplift=prop_uplift,
        method=method,
        cap_km2=cap_km2,
        features=features,
        a_table=_SPECIES_A,
        b_table=_SPECIES_B,
    )


def rl_ecosystem_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float = 0.0,
    method: Literal["max", "min"] = "max",
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """IUCN Red List of Ecosystems targets.

    Same formula as :func:`rl_species_targets` with the ecosystem thresholds:
    ``criterion_a`` in A1, A2a, A2b, A3 and ``criterion_b`` in B1, B2
    (case-insensitive). A3 VU uses 0.5 per the IUCN (2024) Red List of
    Ecosystems guidelines v2.0 (thresholds 90/70/50%; criteria defined in
    Keith et al. 2013, *PLoS ONE* 8:e62111,
    https://doi.org/10.1371/journal.pone.0062111); prioritizr 9.0.1 uses 0.3.
    """
    return _rl_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a=criterion_a,
        criterion_b=criterion_b,
        prop_uplift=prop_uplift,
        method=method,
        cap_km2=cap_km2,
        features=features,
        a_table=_ECOSYSTEM_A,
        b_table=_ECOSYSTEM_B,
    )


def jung_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr = "VU",
    prop_uplift: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Jung et al. (2021) targets: Red List species rule at VU, A2/B2, +10%.

    With the defaults: ``min(max(2,200 km², 0.77·range), 1,000,000 km², range)``
    (prioritizr's docs say 80%; its code and tests give 77%). Jung et al.
    2021, *Nat. Ecol. Evol.* 5:1499-1509,
    https://doi.org/10.1038/s41559-021-01528-7 (correction:
    https://doi.org/10.1038/s41559-021-01569-y). The 2,200 km² floor and the
    100% clip for small ranges are stated in its Methods ("Target setting");
    the 77% proportional term follows prioritizr 9.0.1 ``spec_jung_targets``
    (the equation is not recoverable from accessible text).
    """
    return rl_species_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a="A2",
        criterion_b="B2",
        prop_uplift=prop_uplift,
        method="max",
        cap_km2=cap_km2,
        features=features,
    )


def ward_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr = "CR",
    prop_uplift: float = 0.0,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Ward et al. (2025) targets: Red List species rule at CR, A2/B1.

    With the defaults: ``min(max(100 km², 0.2·range), 1,000,000 km², range)``
    (prioritizr's docs say 100,000 km²; its code uses the CR-B1 threshold
    100 km²). Ward et al. 2025, *PNAS* 122:e2414985122,
    https://doi.org/10.1073/pnas.2414985122. Parameters follow prioritizr
    9.0.1 ``spec_ward_targets``; not confirmable from accessible paper text.
    """
    return rl_species_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a="A2",
        criterion_b="B1",
        prop_uplift=prop_uplift,
        method="max",
        cap_km2=cap_km2,
        features=features,
    )
