"""Smoke test: the assembled app wires in the Rivers panel + demo network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parents[1] / "src" / "pymarxan_app" / "app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("pymarxan_app_under_test", _APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_exposes_ui_and_server():
    app = _load_app()
    assert hasattr(app, "app_ui")
    assert callable(app.server)


def test_demo_river_network_is_valid():
    app = _load_app()
    net = app._demo_river_network()
    assert net.n_segments == 5
    assert net.n_barriers == 4
    # the demo network is solvable end-to-end via the rivers API
    from pymarxan.rivers import BarrierProblem, optimize_barriers_greedy

    sol = optimize_barriers_greedy(BarrierProblem(net, budget=2.0))
    assert sol.dci_after >= sol.dci_before
