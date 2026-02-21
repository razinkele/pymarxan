"""Conservation problem domain model."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models import boundary as boundary_mod


@dataclass
class ConservationProblem:
    """Central container for a Marxan conservation planning problem.

    Parameters
    ----------
    planning_units : pd.DataFrame
        DataFrame with columns ``id``, ``cost``, ``status``.
    features : pd.DataFrame
        DataFrame with columns ``id``, ``name``, ``target``, ``spf``.
    pu_vs_features : pd.DataFrame
        DataFrame with columns ``species``, ``pu``, ``amount``.
    boundary : pd.DataFrame | None
        Optional DataFrame with columns ``id1``, ``id2``, ``boundary``.
    parameters : dict
        Marxan parameters (e.g. ``{"BLM": 1.0}``).
    """

    planning_units: pd.DataFrame
    features: pd.DataFrame
    pu_vs_features: pd.DataFrame
    boundary: pd.DataFrame | None = None
    parameters: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_planning_units(self) -> int:
        """Number of planning units."""
        return len(self.planning_units)

    @property
    def n_features(self) -> int:
        """Number of conservation features."""
        return len(self.features)

    @property
    def pu_ids(self) -> set:
        """Set of planning unit IDs."""
        return set(self.planning_units["id"])

    @property
    def feature_ids(self) -> set:
        """Set of feature IDs."""
        return set(self.features["id"])

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def feature_amounts(self) -> dict[int, float]:
        """Total amount of each feature across all planning units.

        Returns
        -------
        dict[int, float]
            Mapping from feature id to total amount.
        """
        totals = self.pu_vs_features.groupby("species")["amount"].sum()
        return dict(totals.to_dict())  # type: ignore[arg-type]

    def targets_achievable(self) -> bool:
        """Check whether all feature targets can be met.

        Compares each feature's target against the total amount available
        across all planning units.

        Returns
        -------
        bool
            ``True`` if every feature's target is achievable.
        """
        amounts = self.feature_amounts()
        for _, row in self.features.iterrows():
            fid = row["id"]
            target = row["target"]
            available = amounts.get(fid, 0.0)
            if available < target:
                return False
        return True

    def validate(self) -> list[str]:
        """Validate the problem data and return a list of error messages.

        Checks performed:
        - Required columns in ``planning_units`` (id, cost, status).
        - Required columns in ``features`` (id, name, target, spf).
        - Required columns in ``pu_vs_features`` (species, pu, amount).
        - Planning unit IDs referenced in ``pu_vs_features`` exist in
          ``planning_units``.
        - Feature (species) IDs referenced in ``pu_vs_features`` exist in
          ``features``.
        - If ``boundary`` is provided, validate its required columns.

        Returns
        -------
        list[str]
            A list of human-readable error strings. Empty when valid.
        """
        errors: list[str] = []

        # --- Required columns ---
        pu_required = {"id", "cost", "status"}
        missing_pu = pu_required - set(self.planning_units.columns)
        if missing_pu:
            errors.append(
                f"planning_units missing columns: {sorted(missing_pu)}"
            )

        feat_required = {"id", "name", "target", "spf"}
        missing_feat = feat_required - set(self.features.columns)
        if missing_feat:
            errors.append(
                f"features missing columns: {sorted(missing_feat)}"
            )

        puvspr_required = {"species", "pu", "amount"}
        missing_puvspr = puvspr_required - set(self.pu_vs_features.columns)
        if missing_puvspr:
            errors.append(
                f"pu_vs_features missing columns: {sorted(missing_puvspr)}"
            )

        # --- Cross-reference IDs ---
        puvspr_pu_ids = set(self.pu_vs_features["pu"])
        unknown_pus = puvspr_pu_ids - self.pu_ids
        if unknown_pus:
            errors.append(
                f"pu_vs_features references planning unit IDs not in "
                f"planning_units: {sorted(unknown_pus)}"
            )

        puvspr_species_ids = set(self.pu_vs_features["species"])
        unknown_features = puvspr_species_ids - self.feature_ids
        if unknown_features:
            errors.append(
                f"pu_vs_features references feature IDs not in "
                f"features: {sorted(unknown_features)}"
            )

        # --- Boundary columns ---
        if self.boundary is not None:
            missing_bnd = boundary_mod.REQUIRED_COLUMNS - set(
                self.boundary.columns
            )
            if missing_bnd:
                errors.append(
                    f"boundary missing columns: {sorted(missing_bnd)}"
                )

        return errors

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the problem.

        Returns
        -------
        str
            Summary string.
        """
        lines = [
            "ConservationProblem Summary",
            "-" * 30,
            f"  {self.n_planning_units} planning units",
            f"  {self.n_features} features",
            f"  {len(self.pu_vs_features)} planning-unit vs feature records",
        ]
        if self.boundary is not None:
            lines.append(f"  {len(self.boundary)} boundary records")
        else:
            lines.append("  No boundary data")
        if self.parameters:
            lines.append(f"  Parameters: {self.parameters}")
        return "\n".join(lines)
