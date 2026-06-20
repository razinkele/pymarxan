"""Dendritic Connectivity Index — DCIp / DCId (Phase A, read-only metrics).

Formulas (Côté et al. 2009), with ``L = Σ l_i`` and weights ``w_i = l_i / L``:

- **Potamodromous:** ``DCIp = 100 · Σ_i Σ_j w_i w_j · c_ij``,
  ``c_ij = ∏_{b ∈ path(i, j)} p_b``, ``c_ii = 1``.
- **Diadromous:** ``DCId = 100 · Σ_i w_i · c_i``,
  ``c_i = ∏_{b ∈ path(i, mouth)} p_b``.

``c_ij`` is computed as a **direct product over the barriers on the path**
(``RiverNetwork.path_barriers``), never via the closed form
``root_prod(i)·root_prod(j) / root_prod(lca)²`` — that algebra divides ``0/0``
→ NaN whenever a barrier below the confluence is impassable (``p = 0``), which
the binary case routinely produces. See design §4 (review finding H1).

``passabilities`` is an optional ``{barrier_id: p}`` override so callers can
score a candidate decision (e.g. a removed barrier → ``p = 1.0``) without
mutating the network.

References
----------
- Côté, D., Kehler, D. G., Bourne, C., & Wiersma, Y. F. (2009). A new measure
  of longitudinal connectivity for stream networks. *Landscape Ecology, 24*(1),
  101–113. https://doi.org/10.1007/s10980-008-9283-y
"""
from __future__ import annotations

from pymarxan.rivers.network import RiverNetwork


def _prod(passmap: dict[int, float], barrier_ids: list[int]) -> float:
    p = 1.0
    for bid in barrier_ids:
        p *= passmap[bid]
    return p


def dci_diadromous(
    network: RiverNetwork,
    passabilities: dict[int, float] | None = None,
    *,
    direction: str = "single_pass",
) -> float:
    """Diadromous DCI (sea ↔ each segment). Default ``direction="single_pass"``
    matches the R ``dci`` package. ``direction="round_trip"`` is deferred until
    validated (raises ``NotImplementedError``).
    """
    if direction == "round_trip":
        raise NotImplementedError("round_trip DCI is not yet validated")
    if direction != "single_pass":
        raise ValueError(f"unknown direction {direction!r}")
    w = network.weights()
    c = network.root_products(passabilities)
    return 100.0 * sum(w[s] * c[s] for s in w)


def dci_potamodromous(
    network: RiverNetwork,
    passabilities: dict[int, float] | None = None,
) -> float:
    """Potamodromous DCI (all within-network segment pairs)."""
    w = network.weights()
    passmap = network.barrier_passabilities(passabilities)
    seg_ids = list(w)
    total = 0.0
    for i in seg_ids:
        wi = w[i]
        for j in seg_ids:
            c_ij = 1.0 if i == j else _prod(passmap, network.path_barriers(i, j))
            total += wi * w[j] * c_ij
    return 100.0 * total


def segment_connectivity(
    network: RiverNetwork,
    passabilities: dict[int, float] | None = None,
    form: str = "diadromous",
) -> dict[int, float]:
    """Per-segment connectivity.

    ``form="diadromous"`` → each segment's ``c_i`` (path-to-mouth product).
    ``form="potamodromous"`` → each segment's marginal
    ``m_i = Σ_j w_j c_ij`` so that ``Σ_i w_i m_i = DCIp / 100`` (the pairwise
    index reduced to a per-segment scalar).
    """
    if form == "diadromous":
        return network.root_products(passabilities)
    if form == "potamodromous":
        w = network.weights()
        passmap = network.barrier_passabilities(passabilities)
        seg_ids = list(w)
        out: dict[int, float] = {}
        for i in seg_ids:
            m = 0.0
            for j in seg_ids:
                c_ij = 1.0 if i == j else _prod(passmap, network.path_barriers(i, j))
                m += w[j] * c_ij
            out[i] = m
        return out
    raise ValueError(f"unknown form {form!r}")


__all__ = [
    "dci_diadromous",
    "dci_potamodromous",
    "segment_connectivity",
]
