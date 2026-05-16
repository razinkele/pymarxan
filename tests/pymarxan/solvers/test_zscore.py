"""Tests for the Marxan-faithful Z-score chance-constraint math.

Phase 18 Batch 2: pure-functional layer for PROBMODE 3 (Z-score chance
constraints). Lives in ``pymarxan.solvers.probability``.

Sign convention: Z = (T - E) / sqrt(V), positive = shortfall side
(matches Marxan v4 ``computation.hpp::computeProbMeasures``).
Upper-tail probability P_j = norm.sf(Z_j) = 1 - Φ(Z_j).
Penalty normalisation: SPF_j * (ptarget_j - P_j) / ptarget_j, summed
and scaled by PROBABILITYWEIGHTING.
"""
from __future__ import annotations

import pytest
from scipy.stats import norm

from pymarxan.solvers.probability import (
    compute_zscore_penalty,
    compute_zscore_per_feature,
)

# --- Task 4: compute_zscore_per_feature ----------------------------------


class TestZscorePerFeature:
    """Marxan sign: Z = (T - E)/sqrt(V); positive Z = shortfall."""

    def test_zero_variance_returns_marxan_sentinel(self):
        """Marxan classic sets rZ = 4 when variance == 0 (probZUT(4) ≈ 0)."""
        z = compute_zscore_per_feature(
            achieved_mean={1: 20.0},
            achieved_variance={1: 0.0},
            targets={1: 10.0},
        )
        assert z[1] == 4.0

    def test_zero_variance_target_missed_also_returns_sentinel(self):
        """Marxan returns the sentinel regardless of whether the
        deterministic target was met — variance-zero short-circuits."""
        z = compute_zscore_per_feature(
            achieved_mean={1: 5.0},
            achieved_variance={1: 0.0},
            targets={1: 10.0},
        )
        assert z[1] == 4.0

    def test_normal_case_exceedance(self):
        """E=12, V=4, T=10 -> Z = (10-12)/2 = -1.0 (exceedance side)."""
        z = compute_zscore_per_feature(
            achieved_mean={1: 12.0},
            achieved_variance={1: 4.0},
            targets={1: 10.0},
        )
        assert z[1] == pytest.approx(-1.0)

    def test_normal_case_shortfall(self):
        """E=8, V=4, T=10 -> Z = (10-8)/2 = 1.0 (shortfall side)."""
        z = compute_zscore_per_feature(
            achieved_mean={1: 8.0},
            achieved_variance={1: 4.0},
            targets={1: 10.0},
        )
        assert z[1] == pytest.approx(1.0)

    def test_multi_feature(self):
        z = compute_zscore_per_feature(
            achieved_mean={1: 12.0, 2: 5.0},
            achieved_variance={1: 4.0, 2: 1.0},
            targets={1: 10.0, 2: 10.0},
        )
        assert z[1] == pytest.approx(-1.0)
        assert z[2] == pytest.approx(5.0)  # (10-5)/1

    def test_missing_keys_default_to_zero(self):
        """Feature missing from variance / target dicts treated as 0."""
        z = compute_zscore_per_feature(
            achieved_mean={1: 5.0},
            achieved_variance={},
            targets={},
        )
        # variance=0 -> sentinel
        assert z[1] == 4.0


# --- Task 5: compute_zscore_penalty --------------------------------------


class TestZscorePenalty:
    """Marxan-normalised: SPF_j * (ptarget_j - P_j) / ptarget_j."""

    def test_zero_when_prob_target_met(self):
        # Z = -2.0 -> P = norm.sf(-2.0) ≈ 0.977 > ptarget 0.5
        p = compute_zscore_penalty(
            zscore_per_feature={1: -2.0},
            prob_targets={1: 0.5},
            spf={1: 1.0},
            weight=1.0,
        )
        assert p == pytest.approx(0.0)

    def test_normalised_by_ptarget(self):
        """Verifies the (ptarget - P)/ptarget normalisation against the raw
        subtraction the v1 design would have produced."""
        # Z = 1.0 -> P = norm.sf(1.0) ≈ 0.1587
        # penalty = SPF · (ptarget - P) / ptarget = 2.0 · (0.95 - 0.1587) / 0.95
        p = compute_zscore_penalty(
            zscore_per_feature={1: 1.0},
            prob_targets={1: 0.95},
            spf={1: 2.0},
            weight=1.0,
        )
        expected_raw = 2.0 * (0.95 - norm.sf(1.0))
        expected_norm = 2.0 * (0.95 - norm.sf(1.0)) / 0.95
        # Marxan normalises by ptarget; this is the distinguishing test.
        assert p == pytest.approx(expected_norm, abs=1e-9)
        assert p != pytest.approx(expected_raw, abs=1e-4)

    def test_disabled_when_ptarget_negative(self):
        """ptarget == -1 (Marxan disabled sentinel) -> feature contributes 0
        regardless of how bad the Z is."""
        p = compute_zscore_penalty(
            zscore_per_feature={1: 5.0},  # huge shortfall
            prob_targets={1: -1.0},
            spf={1: 1.0},
            weight=1.0,
        )
        assert p == 0.0

    def test_disabled_when_ptarget_zero(self):
        """Zero ptarget also disabled (avoids divide-by-zero)."""
        p = compute_zscore_penalty(
            zscore_per_feature={1: 5.0},
            prob_targets={1: 0.0},
            spf={1: 1.0},
            weight=1.0,
        )
        assert p == 0.0

    def test_weight_scales_linearly(self):
        """PROBABILITYWEIGHTING scales the whole sum."""
        base = compute_zscore_penalty(
            zscore_per_feature={1: 1.0},
            prob_targets={1: 0.95},
            spf={1: 1.0},
            weight=1.0,
        )
        scaled = compute_zscore_penalty(
            zscore_per_feature={1: 1.0},
            prob_targets={1: 0.95},
            spf={1: 1.0},
            weight=3.0,
        )
        assert scaled == pytest.approx(3.0 * base)

    def test_sums_over_features(self):
        """Multi-feature penalty is the SPF-weighted sum."""
        p = compute_zscore_penalty(
            zscore_per_feature={1: 1.0, 2: -1.0},
            prob_targets={1: 0.95, 2: 0.95},
            spf={1: 1.0, 2: 3.0},
        )
        # feature 1: shortfall side. feature 2: exceedance side.
        e1 = max(0.0, (0.95 - norm.sf(1.0)) / 0.95)
        e2 = max(0.0, (0.95 - norm.sf(-1.0)) / 0.95)
        expected = 1.0 * e1 + 3.0 * e2
        assert p == pytest.approx(expected, abs=1e-9)

    def test_sentinel_variance_means_deterministic_no_penalty(self):
        """Z=4 (Marxan zero-variance sentinel) -> no chance-constraint
        penalty. The deterministic-target shortfall path handles any
        deterministic miss; the probability penalty is vacuously
        satisfied (P=1) when there's no uncertainty.

        Matches Marxan's intent (probability.cpp comment: "degenerate
        case → P=1") even though norm.sf(4) ≈ 3e-5 would give a
        non-trivial penalty if we passed it through naïvely.
        """
        p = compute_zscore_penalty(
            zscore_per_feature={1: 4.0},
            prob_targets={1: 0.95},
            spf={1: 1.0},
            weight=1.0,
        )
        assert p == 0.0
