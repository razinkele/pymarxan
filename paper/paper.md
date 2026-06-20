---
title: 'pymarxan: A pure-Python library for systematic conservation planning'
tags:
  - Python
  - conservation planning
  - spatial prioritization
  - Marxan
  - reserve design
  - optimization
authors:
  - name: Artūras Razinkovas-Baziukas
    orcid: 0000-0000-0000-0000  # TODO: insert ORCID before submission
    affiliation: 1
affiliations:
  - name: Marine Research Institute, Klaipėda University, Lithuania
    index: 1
date: 12 June 2026
bibliography: paper.bib
---

# Summary

Systematic conservation planning [@margules2000] asks a precise question:
which set of places, selected together, meets a set of biodiversity
representation targets at the lowest cost? `pymarxan` is a pure-Python
library that answers it. It reproduces the algorithms of Marxan
[@possingham2000; @watts2009] — the most widely used conservation
planning software — adds the exact-optimization capabilities of the
`prioritizr` R package [@hanson2024], and exposes the whole toolkit
through a documented Python API and an optional web interface.

A planning problem is expressed as planning units (sites with a cost and
a status), conservation features (species or habitats with a target),
and the amount of each feature in each unit. `pymarxan` reads and writes
these in the native Marxan file format (`input.dat`, `pu.dat`,
`spec.dat`, `puvspr.dat`, `bound.dat`, and the `ssoln`/`mvbest`/`sum`
output files), so existing Marxan projects run unchanged. It then solves
the problem with a choice of engines: a native simulated-annealing solver
[@kirkpatrick1983], a greedy heuristic, and exact mixed-integer
programming via the open-source CBC and HiGHS solvers or commercial
Gurobi. It also covers the wider Marxan family — multi-zone planning
(as in Marxan with Zones [@watts2009]), boundary-length and connectivity
penalties, probabilistic targets, feature clumping, and minimum
separation distance — together with post-hoc analyses such as
irreplaceability, selection frequency, gap analysis, solution
portfolios, and graph-theoretic connectivity metrics. Beyond site
selection, `pymarxan` adds native **river-network connectivity** — the
Dendritic Connectivity Index [@cote2009] — and **barrier-removal
optimization**: choosing which dams, weirs, or culverts to remove under a
budget to maximise reconnected riverine habitat, via greedy,
simulated-annealing, and exact integer-programming engines (the
connected-habitat formulation of @ohanley2011).

The package is built on the scientific Python stack (`numpy`, `pandas`,
`geopandas`, `networkx`, `pulp`) and is distributed as an installable
wheel. Its accompanying Shiny-for-Python application provides an
interactive GUI for users who prefer not to write code.

# Statement of need

Conservation planning software has bifurcated. Marxan, written in C++ and
the de facto standard for two decades, is effectively dormant: its last
release was in 2021, and the surrounding GUI ecosystem (Zonae Cogito,
MarxanConnect) is unmaintained. The living, actively developed
alternative, `prioritizr` [@hanson2024], is excellent but is written in
R. There is no comparably complete, maintained implementation in Python —
the language in which a large and growing share of ecologists, students,
and spatial data scientists now work, and the native environment of
`geopandas`, `rasterio`, and Jupyter.

This gap matters for three reasons. First, **reproducibility and
integration**: a Python library lets a planning workflow live in one
notebook alongside data acquisition, modelling, and figures, rather than
shelling out to a separate binary with bespoke file formats. Second,
**solver quality**: Schuster et al. [@schuster2020] showed that exact
integer-programming solvers produce reserve plans 12–30 % cheaper than
simulated annealing and require no penalty-factor calibration; Marxan
offers only simulated annealing, whereas `pymarxan` offers both, so users
can prototype with the heuristic and finish with a provably optimal
solution. Third, **continuity**: by reading the native Marxan format,
`pymarxan` lets the large body of existing Marxan projects and teaching
material carry forward into a maintained, modern toolchain.

`pymarxan` also spans a breadth no single maintained tool currently
covers: the Marxan family for area-based site selection **and** native
riverine barrier-removal optimization in one library. `prioritizr` has no
dendritic-connectivity or barrier model; `restoptr` addresses
terrestrial-2D restoration through a Java backend; and the dedicated
river-barrier tools are typically narrow or closed. Bringing both
paradigms into one inspectable Python API lets marine, terrestrial, and
freshwater planning share data structures, solvers, and workflows.

`pymarxan` targets researchers and practitioners doing reserve design,
marine and terrestrial spatial planning, and the "30×30" protected-area
expansion agenda, as well as instructors who want a dependency-light,
inspectable implementation of the classic algorithms for teaching. The
library is organized in three layers — a solver/analysis core, a set of
reusable Shiny UI modules, and a runnable application — so that it can be
used as a programmatic dependency or as an end-user tool. A worked,
self-contained example (a south-eastern Baltic marine planning scenario)
and an end-to-end tutorial ship with the source.

# Acknowledgements

We thank the developers of Marxan and `prioritizr`, whose published
methods and open documentation made an independent, compatible
reimplementation possible.

# References
