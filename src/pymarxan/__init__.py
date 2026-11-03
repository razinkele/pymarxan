"""pymarxan: Modular Python toolkit for Marxan conservation planning.

The version lives only in ``pyproject.toml``. Run from a source checkout (``src/pymarxan``
beside the project's ``pyproject.toml``, e.g. the Shiny Server deployment, which imports the
checkout without installing it) the version is read from that file, so a missing or stale
install record cannot misreport it. An installed package reads its metadata; with neither,
a PEP 440 local version keeps callers (Shiny About dialog, logging) parseable.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _installed_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pymarxan")
    except PackageNotFoundError:
        return None


def _resolve_version(
    package_dir: Path, metadata: Callable[[], str | None] = _installed_version
) -> str:
    """The checkout's ``pyproject.toml`` version, else ``metadata()``, else ``0.0.0+unknown``."""
    import tomllib

    try:
        project = tomllib.loads((package_dir.parents[1] / "pyproject.toml").read_text())
        project = project.get("project", {})
        if project.get("name") == "pymarxan" and isinstance(project.get("version"), str):
            return str(project["version"])
    except (OSError, IndexError, tomllib.TOMLDecodeError):
        pass
    return metadata() or "0.0.0+unknown"


__version__ = _resolve_version(Path(__file__).resolve().parent)


def __getattr__(name: str) -> object:
    # Lazy: ``import pymarxan`` must not import pandas / pulp / zones.
    if name in ("presolve_check", "PresolveWarning"):
        from pymarxan.solvers import presolve

        return getattr(presolve, name)
    raise AttributeError(f"module 'pymarxan' has no attribute {name!r}")
