"""pymarxan.objectives is deprecated in v0.39 (spec §5, Testing 13)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
PROBE = (
    "import warnings\n"
    "with warnings.catch_warnings(record=True) as w:\n"
    "    warnings.simplefilter('always')\n"
    "    import {mod}\n"
    "print(sum(issubclass(x.category, DeprecationWarning) and 'objectives' in str(x.message)"
    " for x in w))\n"
)


def _count(mod: str) -> int:
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(mod=mod)], cwd=ROOT, check=True,
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": ""},
    ).stdout
    return int(out.strip())


def test_objectives_import_warns_once() -> None:
    assert _count("pymarxan.objectives") == 1


def test_plain_import_does_not_warn() -> None:
    assert _count("pymarxan") == 0
    assert _count("pymarxan.solvers") == 0
