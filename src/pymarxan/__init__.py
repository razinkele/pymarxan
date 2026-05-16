"""pymarxan: Modular Python toolkit for Marxan conservation planning.

The version is read from installed package metadata so we only edit
``pyproject.toml`` on each release. When running from an uninstalled
source checkout the metadata isn't available and we fall back to a
PEP 440 local-version string so callers (Shiny footer, logging) get a
parseable value rather than crashing.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymarxan")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
