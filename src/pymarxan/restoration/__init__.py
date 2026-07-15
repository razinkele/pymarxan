"""restoptr-style ecological restoration planning — landscape-pattern indices.

Hosts landscape *pattern / fragmentation* indices used for restoration planning: MESH (effective
mesh size, Jaeger 2000) now; IIC / PC patch-graph connectivity indices (Pascual-Hortal & Saura
2006; Saura & Pascual-Hortal 2007) as a future addition, each with its own ``compute_*`` + result
type. This is distinct from :mod:`pymarxan.connectivity`, which handles circuit / graph *flow*
connectivity (Omniscape, climate velocity, dispersal smoothing).
"""
from __future__ import annotations

from pymarxan.restoration.mesh import MeshResult, compute_mesh

__all__ = ["MeshResult", "compute_mesh"]
