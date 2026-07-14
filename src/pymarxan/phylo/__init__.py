"""Phylogenetic-diversity objectives for pymarxan.

Faith (1992) phylogenetic diversity via a branch-as-feature decomposition
(Rodrigues & Gaston 2002): each branch of a phylogeny becomes a synthetic
conservation feature weighted by its length, so the existing solvers maximize
PD directly. See ``docs/plans/2026-07-14-phylo-diversity-design.md``.
"""
from __future__ import annotations

from pymarxan.phylo.diversity import PDResult, compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree

__all__ = [
    "PhylogeneticTree",
    "PDResult",
    "compute_phylogenetic_diversity",
]
