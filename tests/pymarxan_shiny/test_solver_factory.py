"""The shared solver factory used by both apps (v0.49.1: laguna's Run crashed on a
non-existent ``pymarxan.solvers.get_solver`` import)."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.zones.solver import ZoneSASolver
from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_from_config,
    solver_picker_ui,
)

_SRC = Path(__file__).resolve().parents[2] / "src"
_APPS = {
    "laguna": _SRC / "pymarxan_shiny" / "app.py",
    "full": _SRC / "pymarxan_app" / "app.py",
}

EXPECTED = {
    "mip": MIPSolver,
    "sa": SimulatedAnnealingSolver,
    "binary": MarxanBinarySolver,
    "zone_sa": ZoneSASolver,
    "greedy": HeuristicSolver,
    "iterative_improvement": IterativeImprovementSolver,
    "pipeline": RunModePipeline,
}


@pytest.mark.parametrize("solver_type", sorted(EXPECTED))
def test_factory_builds_each_picker_choice(solver_type):
    solver, warning = solver_from_config({"solver_type": solver_type})
    assert isinstance(solver, EXPECTED[solver_type])
    assert warning is None


def test_factory_builds_zonation():
    solver, warning = solver_from_config({"solver_type": "zonation"})
    assert solver.name().lower().startswith("zonation")
    assert warning is None


def test_factory_covers_every_choice_the_picker_offers():
    html = str(solver_picker_ui("sp"))
    select = re.search(r'id="sp-solver_type".*?</select>', html, re.S)
    assert select is not None
    offered = set(re.findall(r'value="([^"]+)"', select.group(0)))
    assert offered, "no solver choices found in the picker UI"
    for solver_type in offered:
        solver, _ = solver_from_config({"solver_type": solver_type})
        assert solver is not None


def test_factory_defaults_to_mip_and_passes_on_mip_warning():
    assert isinstance(solver_from_config({})[0], MIPSolver)
    solver, warning = solver_from_config(
        {"solver_type": "mip", "mip_decisions": "semicontinuous", "mip_upper": 0.2,
         "mip_lower": 0.8}
    )
    assert isinstance(solver, MIPSolver)
    assert warning is not None and "binary decisions" in warning


@pytest.mark.parametrize("app", sorted(_APPS))
def test_both_apps_use_the_shared_factory(app):
    text = _APPS[app].read_text()
    assert "get_solver" not in text
    assert re.search(r"(?<![\w.])solver_from_config\(", text)


def test_laguna_app_imports(monkeypatch):
    """The deployed app had no app-level test, which is how the bad import survived."""
    # Shiny Server runs it from its own directory, where ``modules.*`` resolves.
    monkeypatch.syspath_prepend(str(_APPS["laguna"].parent))
    spec = importlib.util.spec_from_file_location("pymarxan_laguna_app_under_test",
                                                  _APPS["laguna"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "app_ui")
    assert callable(module.server)
