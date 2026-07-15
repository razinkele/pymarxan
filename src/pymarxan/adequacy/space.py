"""compute_space_held — raptr's 1 - WSS/TSS proportion of attribute-space variation captured."""
from __future__ import annotations

import numpy as np

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.models.problem import ConservationProblem


def compute_space_held(
    problem: ConservationProblem,
    selected: np.ndarray,
    spec: SpaceSpec | None = None,
) -> dict[int, float]:
    """Per-feature space held = ``1 - WSS/TSS`` (raptr, Hanson et al. 2018,
    doi:10.1111/2041-210x.12862).

    Demand points = the occupied PUs (amount>0), positioned in ``spec``'s (z-scored) attribute
    space, weight = feature amount. ``WSS = Σ_d w_d·min_{selected occupied}‖p_d − pos‖²``;
    ``TSS = Σ_d w_d·‖p_d − c‖²``, ``c`` = unweighted mean of demand-point coords. Empty
    selection → 0; ``TSS == 0`` → 1. (Occupied-PU demand points are a documented deviation from
    raptr's KDE-sample.) Returns ``{feature_id: space_held}``, each clipped to ``[0, 1]``.
    """
    spec = spec or SpaceSpec()
    pos = pu_attribute_space(problem, spec)  # (n_pu, n_dim)
    idx = problem.pu_id_to_index
    pv = problem.pu_vs_features
    species = np.asarray(pv["species"].to_numpy(), dtype=np.int64)
    pu_all = np.asarray(pv["pu"].to_numpy(), dtype=np.int64)
    amt_all = np.asarray(pv["amount"].to_numpy(), dtype=float)
    held: dict[int, float] = {}
    for fid in np.unique(species):
        fmask = (species == fid) & (amt_all > 0)
        # Keep-mask so weights stay aligned to demand points even when pu_vs_features
        # references PU ids absent from planning_units (design-review BUG-B; the defensive
        # pattern from separation.py). A positional ``[:len(occ)]`` slice misaligns.
        pu_ids = pu_all[fmask]
        keep = np.array([int(p) in idx for p in pu_ids], dtype=bool)
        occ = np.array([idx[int(p)] for p in pu_ids[keep]], dtype=int)
        w = amt_all[fmask][keep]
        if len(occ) == 0:
            held[int(fid)] = 0.0
            continue
        p_d = pos[occ]                        # (n_dp, n_dim) demand-point coords
        c = p_d.mean(axis=0)                  # unweighted centroid
        tss = float(np.sum(w * np.sum((p_d - c) ** 2, axis=1)))
        sel_occ = occ[selected[occ]]
        if len(sel_occ) == 0:
            held[int(fid)] = 0.0
            continue
        if tss == 0.0:
            held[int(fid)] = 1.0
            continue
        d2 = ((p_d[:, None, :] - pos[sel_occ][None, :, :]) ** 2).sum(axis=2)  # (n_dp, n_sel)
        wss = float(np.sum(w * d2.min(axis=1)))
        held[int(fid)] = float(np.clip(1.0 - wss / tss, 0.0, 1.0))
    return held


def evaluate_solution_space(
    problem: ConservationProblem,
    selected: np.ndarray,
    spec: SpaceSpec | None = None,
) -> tuple[dict[int, float], float]:
    """Post-hoc space reporting for a solution: ``(space_held, total_space_penalty)``.

    Over features with ``space_target > 0``: ``space_held`` per feature (via
    :func:`compute_space_held`) and the soft penalty ``Σ_f space_spf_f · max(0, space_target_f −
    space_held_f)`` (``space_spf`` defaults to the amount ``spf``). DataFrame-level — called once
    per solution, so it need not be as fast as :class:`~pymarxan.solvers.space_state.SpaceState`.
    """
    spec = spec or SpaceSpec()
    feats = problem.features
    spf_col = "space_spf" if "space_spf" in feats.columns else "spf"
    held_all = compute_space_held(problem, selected, spec)
    held: dict[int, float] = {}
    penalty = 0.0
    for _, row in feats.iterrows():
        tgt = float(row.get("space_target", 0.0) or 0.0)
        if tgt <= 0.0:
            continue
        fid = int(row["id"])
        h = float(held_all.get(fid, 0.0))
        held[fid] = h
        penalty += float(row.get(spf_col, 1.0)) * max(0.0, tgt - h)
    return held, penalty
