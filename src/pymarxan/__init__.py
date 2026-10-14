"""pymarxan: Modular Python toolkit for Marxan conservation planning.

The version is read from installed package metadata so we only edit
``pyproject.toml`` on each release. When running from an uninstalled
source checkout the metadata isn't available and we fall back to a
PEP 440 local-version string so callers (Shiny footer, logging) get a
parseable value rather than crashing.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymarxan")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def __getattr__(name: str) -> object:
    # Lazy: ``import pymarxan`` must not import pandas / pulp / zones.
    if name in ("presolve_check", "PresolveWarning"):
        from pymarxan.solvers import presolve

        return getattr(presolve, name)
    raise AttributeError(f"module 'pymarxan' has no attribute {name!r}")
