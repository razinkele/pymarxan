"""Greedy heuristic solver for conservation planning.

Selects planning units one-by-one based on a configurable scoring
strategy (HEURTYPE 0-7, matching Marxan). Fast baseline for comparison
with SA and MIP solvers.
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution

_VALID_HEURTYPES = range(8)


class HeuristicSolver(Solver):
    """Greedy heuristic solver supporting all 8 Marxan HEURTYPE modes.

    Parameters
    ----------
    heurtype : int
        Scoring strategy (0-7). Default is 2 (Max Rarity), matching Marxan.
        If the problem's ``parameters`` dict contains ``HEURTYPE``, that
        value overrides the constructor argument at solve time.

    Scoring strategies
    ------------------
    0 - Richness: count of unmet features the PU contributes to.
    1 - Greedy cheapest: negative cost (prefer low cost).
    2 - Max Rarity: highest rarity among contributed unmet features.
    3 - Best Rarity: best (rarity / cost) ratio.
    4 - Average Rarity: average rarity across contributed unmet features.
    5 - Sum Rarity: sum of rarity scores.
    6 - Product Irreplaceability: product of 1/(1-irreplaceability).
    7 - Summation Irreplaceability: sum of irreplaceability scores.
    """

    def __init__(self, heurtype: int = 2) -> None:
        heurtype = int(heurtype)  # coerce, may raise for non-numeric
        if heurtype not in _VALID_HEURTYPES:
            msg = f"heurtype must be 0-7, got {heurtype}"
            raise ValueError(msg)
        self.heurtype = heurtype

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig(num_solutions=1)

        rng = np.random.default_rng(config.seed)
        solutions = []

        for _ in range(config.num_solutions):
            sol = self._solve_once(problem, rng)
            solutions.append(sol)

        return solutions

    # ------------------------------------------------------------------
    # Scoring function
    # ------------------------------------------------------------------

    @staticmethod
    def _score_pu(
        idx: int,
        heurtype: int,
        costs: np.ndarray,
        contributions: dict[int, dict[int, float]],
        remaining: dict[int, float],
        total_available: dict[int, float],
        noise: float,
    ) -> float | None:
        """Return a score for candidate planning unit *idx*.

        Higher scores are preferred. A small random *noise* term is added
        for tie-breaking diversity. Returns ``None`` if the PU contributes
        nothing to any unmet feature.
        """
        pu_contribs = contributions.get(int(idx), {})
        # Only consider features that still have unmet targets
        unmet = {
            fid: amt
            for fid, amt in pu_contribs.items()
            if remaining.get(fid, 0.0) > 0
        }

        if not unmet:
            return None

        cost = max(float(costs[idx]), 1e-10)
        # Small cost-inverse tiebreaker: when primary scores are equal,
        # prefer cheaper PUs (matching Marxan behaviour).
        cost_tiebreaker = 1e-6 / cost

        if heurtype == 0:
            # Richness: count of unmet features this PU contributes to
            return float(len(unmet)) + cost_tiebreaker + noise

        if heurtype == 1:
            # Greedy cheapest: negative cost (prefer low cost)
            return -cost + noise

        # Rarity for each unmet feature: target / total_available
        rarities: list[float] = []
        for fid in unmet:
            avail = total_available.get(fid, 1.0)
            target = remaining.get(fid, 0.0)
            rarity = target / max(avail, 1e-10)
            rarities.append(rarity)

        if heurtype == 2:
            # Max Rarity: contribute to the rarest unmet feature
            return max(rarities) + cost_tiebreaker + noise

        if heurtype == 3:
            # Best Rarity: best (rarity / cost) ratio
            return max(rarities) / cost + noise

        if heurtype == 4:
            # Average Rarity
            return (sum(rarities) / len(rarities)) + cost_tiebreaker + noise

        if heurtype == 5:
            # Sum Rarity
            return sum(rarities) + cost_tiebreaker + noise

        # Irreplaceability for each feature: amount / total_available
        irreplaceabilities: list[float] = []
        for fid, amt in unmet.items():
            avail = total_available.get(fid, 1.0)
            irrep = min(amt / max(avail, 1e-10), 1.0 - 1e-10)
            irreplaceabilities.append(irrep)

        if heurtype == 6:
            # Product Irreplaceability: product of 1/(1-irrep)
            product = 1.0
            for irrep in irreplaceabilities:
                product *= 1.0 / (1.0 - irrep)
            return product + cost_tiebreaker + noise

        # heurtype == 7
        # Summation Irreplaceability: sum of irreplaceability scores
        return sum(irreplaceabilities) + cost_tiebreaker + noise

    # ------------------------------------------------------------------
    # Core greedy loop
    # ------------------------------------------------------------------

    def _solve_once(
        self,
        problem: ConservationProblem,
        rng: np.random.Generator,
    ) -> Solution:
        # Resolve effective heurtype: problem.parameters overrides constructor
        effective_heurtype = int(
            problem.parameters.get("HEURTYPE", self.heurtype)
        )

        n = problem.n_planning_units
        pu_ids = problem.planning_units["id"].values
        costs = np.asarray(problem.planning_units["cost"].values, dtype=float)
        statuses = problem.planning_units["status"].values.astype(int)

        selected = np.zeros(n, dtype=bool)

        # Lock-in (status 2) and lock-out (status 3)
        locked_in = statuses == 2
        locked_out = statuses == 3
        selected[locked_in] = True

        # Status 1: initial include — start selected but swappable
        initial_include = statuses == 1
        selected[initial_include] = True

        # Build feature contribution lookup: pu_index -> {fid: amount}
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        contributions: dict[int, dict[int, float]] = {}
        pv_pu = problem.pu_vs_features["pu"].values
        pv_sp = problem.pu_vs_features["species"].values
        pv_am = problem.pu_vs_features["amount"].values

        # Probability support
        prob_mode = int(problem.parameters.get("PROBMODE", 1))
        prob_weight = float(
            problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
        )
        prob_map: dict[int, float] = {}
        if problem.probability is not None:
            p_pu = problem.probability["pu"].values
            p_val = problem.probability["probability"].values
            for pk in range(len(p_pu)):
                prob_map[int(p_pu[pk])] = float(p_val[pk])

        for k in range(len(pv_pu)):
            idx = pu_id_to_idx.get(int(pv_pu[k]))
            if idx is not None:
                raw_amt = float(pv_am[k])
                if prob_map and prob_mode == 2:
                    pid = int(pv_pu[k])
                    raw_amt *= 1.0 - prob_map.get(pid, 0.0)
                contributions.setdefault(idx, {})[int(pv_sp[k])] = raw_amt

        # Total available amount per feature (for rarity / irreplaceability)
        # Exclude locked-out PUs from availability calculation
        total_available: dict[int, float] = {}
        for idx_val, amt_val in contributions.items():
            if locked_out[idx_val]:
                continue
            for fid, amt in amt_val.items():
                total_available[fid] = total_available.get(fid, 0.0) + amt

        # Track remaining need per feature (with MISSLEVEL)
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        feat_ids = problem.features["id"].values
        feat_targets = problem.features["target"].values.astype(float)
        feat_spf = problem.features["spf"].values.astype(float) if "spf" in problem.features.columns else np.ones(len(feat_ids))
        remaining: dict[int, float] = {}
        for k in range(len(feat_ids)):
            remaining[int(feat_ids[k])] = float(feat_targets[k]) * misslevel

        # Subtract locked-in contributions
        for idx in np.where(locked_in)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount

        # Subtract initial-include contributions
        for idx in np.where(initial_include & ~locked_in)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount

        # Greedy loop: select highest-scoring PU until all targets met
        available = np.where(~selected & ~locked_out)[0]
        # Add small noise for diversity across runs (kept tiny so
        # it does not override meaningful tiebreakers like cost).
        noise = rng.uniform(0.0, 1e-8, size=n)

        # Mode 1: inflate effective costs by probability risk premium
        effective_costs = costs.copy()
        if prob_map and prob_mode == 1 and prob_weight > 0:
            for i, pid in enumerate(pu_ids):
                prob = prob_map.get(int(pid), 0.0)
                effective_costs[i] += prob_weight * prob * costs[i]

        while any(r > 0 for r in remaining.values()) and len(available) > 0:
            best_idx = -1
            best_score: float | None = None

            for idx in available:
                score = self._score_pu(
                    idx=idx,
                    heurtype=effective_heurtype,
                    costs=effective_costs,
                    contributions=contributions,
                    remaining=remaining,
                    total_available=total_available,
                    noise=noise[idx],
                )
                if score is not None and (
                    best_score is None or score > best_score
                ):
                    best_score = score
                    best_idx = idx

            if best_idx < 0 or best_score is None:
                break

            selected[best_idx] = True
            for fid, amount in contributions.get(int(best_idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount
            available = available[available != best_idx]

        # Build solution using shared utility (handles boundary, penalty,
        # targets, shortfall, cost-threshold consistently with all solvers)
        blm = float(problem.parameters.get("BLM", 0.0))

        return build_solution(
            problem,
            selected,
            blm,
            metadata={"solver": "greedy", "heurtype": effective_heurtype},
        )

    def name(self) -> str:
        return "greedy"

    def supports_zones(self) -> bool:
        return False
