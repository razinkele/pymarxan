"""pymarxan.__version__: the checkout's pyproject.toml, else package metadata."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pymarxan
from pymarxan import _resolve_version

REPO = Path(__file__).resolve().parents[2]


def _checkout(tmp_path: Path, pyproject: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(pyproject)
    pkg = tmp_path / "src" / "pymarxan"
    pkg.mkdir(parents=True)
    return pkg


def test_version_matches_this_checkouts_pyproject():
    expected = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert pymarxan.__version__ == expected


def test_source_checkout_wins_over_stale_metadata(tmp_path):
    pkg = _checkout(tmp_path, '[project]\nname = "pymarxan"\nversion = "9.8.7"\n')
    assert _resolve_version(pkg, metadata=lambda: "0.1.0") == "9.8.7"


def test_installed_package_uses_metadata(tmp_path):
    pkg = tmp_path / "site-packages" / "pymarxan"
    pkg.mkdir(parents=True)
    assert _resolve_version(pkg, metadata=lambda: "1.2.3") == "1.2.3"


def test_other_projects_pyproject_is_ignored(tmp_path):
    pkg = _checkout(tmp_path, '[project]\nname = "something-else"\nversion = "5.0"\n')
    assert _resolve_version(pkg, metadata=lambda: "1.2.3") == "1.2.3"


def test_unreadable_pyproject_falls_back_to_metadata(tmp_path):
    pkg = _checkout(tmp_path, "this is [not toml")
    assert _resolve_version(pkg, metadata=lambda: "1.2.3") == "1.2.3"


def test_no_checkout_and_no_metadata_gives_local_version(tmp_path):
    pkg = tmp_path / "site-packages" / "pymarxan"
    pkg.mkdir(parents=True)
    assert _resolve_version(pkg, metadata=lambda: None) == "0.0.0+unknown"
