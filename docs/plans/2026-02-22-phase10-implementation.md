# Phase 10: Finish Design Doc — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the gap to 100% of the original design doc by adding MIP parameter controls, connectivity matrix upload, and all 7 solver types in the picker.

**Architecture:** Three changes: (1) MIP solver reads time_limit/gap/verbose from problem.parameters, (2) new Shiny module for connectivity CSV upload, (3) solver_picker exposes all 7 solver types with mode selectors. All changes build on existing backends.

**Tech Stack:** PuLP (MIP), Shiny for Python, pandas (CSV parse), numpy (connectivity matrix)

---

### Task 1: MIP Solver Parameter Support

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py:126`
- Test: `tests/pymarxan/solvers/test_mip_params.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_mip_params.py`:

```python
"""Tests for MIP solver parameter controls."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def _tiny_problem(**params) -> ConservationProblem:
    """3 PU, 1 feature minimal problem with custom parameters."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "target": [1.0], "spf": [1.0], "name": ["f1"]})
    puvsf = pd.DataFrame({"pu": [1, 2, 3], "species": [1, 1, 1], "amount": [1.0, 1.0, 1.0]})
    base = {"BLM": "0"}
    base.update(params)
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvsf, parameters=base,
    )


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_default_params(mock_cbc):
    """Default: timeLimit=300, gapRel=0.0, msg=0."""
    mock_cbc.return_value = mock_cbc  # so model.solve(solver) works
    mock_cbc.solve = lambda self: 1  # mock solve
    p = _tiny_problem()
    # We just check the constructor call args
    solver = MIPSolver()
    try:
        solver.solve(p, SolverConfig(verbose=False))
    except Exception:
        pass  # solve may fail with mock, that's fine
    mock_cbc.assert_called_once_with(msg=0, timeLimit=300, gapRel=0.0)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_custom_time_limit(mock_cbc):
    """MIP_TIME_LIMIT in problem.parameters is passed through."""
    mock_cbc.return_value = mock_cbc
    p = _tiny_problem(MIP_TIME_LIMIT="60")
    solver = MIPSolver()
    try:
        solver.solve(p, SolverConfig(verbose=False))
    except Exception:
        pass
    mock_cbc.assert_called_once_with(msg=0, timeLimit=60, gapRel=0.0)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_custom_gap(mock_cbc):
    """MIP_GAP in problem.parameters is passed through."""
    mock_cbc.return_value = mock_cbc
    p = _tiny_problem(MIP_GAP="0.05")
    solver = MIPSolver()
    try:
        solver.solve(p, SolverConfig(verbose=False))
    except Exception:
        pass
    mock_cbc.assert_called_once_with(msg=0, timeLimit=300, gapRel=0.05)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_verbose_flag(mock_cbc):
    """config.verbose=True sets msg=1."""
    mock_cbc.return_value = mock_cbc
    p = _tiny_problem()
    solver = MIPSolver()
    try:
        solver.solve(p, SolverConfig(verbose=True))
    except Exception:
        pass
    mock_cbc.assert_called_once_with(msg=1, timeLimit=300, gapRel=0.0)
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/pymarxan/solvers/test_mip_params.py -v`
Expected: FAIL — `PULP_CBC_CMD` is called with just `msg=0`, not the 3-arg form.

**Step 3: Implement MIP parameter reading**

In `src/pymarxan/solvers/mip_solver.py`, replace line 126:

```python
        solver = pulp.PULP_CBC_CMD(msg=0)
```

with:

```python
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        verbose = config.verbose
        solver = pulp.PULP_CBC_CMD(
            msg=int(verbose), timeLimit=time_limit, gapRel=gap,
        )
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/pymarxan/solvers/test_mip_params.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add tests/pymarxan/solvers/test_mip_params.py src/pymarxan/solvers/mip_solver.py
git commit -m "feat: MIP solver reads time_limit, gap, verbose from config"
```

---

### Task 2: MIP Parameters in Solver Picker UI

**Files:**
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`
- Test: `tests/pymarxan_shiny/test_solver_picker_mip.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_solver_picker_mip.py`:

```python
"""Tests for MIP parameter controls in solver picker."""
from __future__ import annotations

from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui


def test_solver_picker_ui_returns_tag():
    elem = solver_picker_ui("test_sp")
    assert elem is not None


def test_solver_picker_has_mip_panel():
    """UI string should reference mip_time_limit input."""
    elem = solver_picker_ui("test_sp")
    html = str(elem)
    assert "mip_time_limit" in html


def test_solver_picker_has_mip_gap():
    """UI string should reference mip_gap input."""
    elem = solver_picker_ui("test_sp")
    html = str(elem)
    assert "mip_gap" in html


def test_solver_picker_has_mip_verbose():
    """UI string should reference mip_verbose input."""
    elem = solver_picker_ui("test_sp")
    html = str(elem)
    assert "mip_verbose" in html
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solver_picker_mip.py -v`
Expected: 3 FAIL (mip_time_limit, mip_gap, mip_verbose not in HTML)

**Step 3: Add MIP panel_conditional to solver_picker**

In `src/pymarxan_shiny/modules/solver_config/solver_picker.py`, after the `zone_sa` panel_conditional block (line 74), add:

```python
                ui.panel_conditional(
                    "input.solver_type === 'mip'",
                    ui.input_numeric(
                        "mip_time_limit", "Time Limit (seconds)",
                        value=300, min=10, step=30,
                    ),
                    ui.input_numeric(
                        "mip_gap", "Optimality Gap",
                        value=0.0, min=0.0, max=1.0, step=0.01,
                    ),
                    ui.input_checkbox(
                        "mip_verbose", "Verbose output", value=False,
                    ),
                ),
```

In the `_update_config` effect, add to `@reactive.event` args: `input.mip_time_limit, input.mip_gap, input.mip_verbose`.

In the `_update_config` body, add after line 106:

```python
        if input.solver_type() == "mip":
            config["mip_time_limit"] = int(input.mip_time_limit() or 300)
            config["mip_gap"] = float(input.mip_gap() if input.mip_gap() is not None else 0.0)
            config["mip_verbose"] = bool(input.mip_verbose())
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solver_picker_mip.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/solver_config/solver_picker.py tests/pymarxan_shiny/test_solver_picker_mip.py
git commit -m "feat: add MIP time limit, gap, verbose controls to solver picker"
```

---

### Task 3: Add Missing Solver Types to Picker

**Files:**
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`
- Test: `tests/pymarxan_shiny/test_solver_picker_types.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_solver_picker_types.py`:

```python
"""Tests for all 7 solver types in solver picker."""
from __future__ import annotations

from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui


def test_picker_has_greedy():
    html = str(solver_picker_ui("t"))
    assert "greedy" in html


def test_picker_has_iterative_improvement():
    html = str(solver_picker_ui("t"))
    assert "iterative_improvement" in html


def test_picker_has_pipeline():
    html = str(solver_picker_ui("t"))
    assert "pipeline" in html


def test_picker_has_heurtype_selector():
    """Greedy panel should have heurtype dropdown."""
    html = str(solver_picker_ui("t"))
    assert "heurtype" in html


def test_picker_has_itimptype_selector():
    """Iterative improvement panel should have itimptype dropdown."""
    html = str(solver_picker_ui("t"))
    assert "itimptype" in html


def test_picker_has_runmode_selector():
    """Pipeline panel should have runmode dropdown."""
    html = str(solver_picker_ui("t"))
    assert "runmode" in html
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solver_picker_types.py -v`
Expected: 6 FAIL (greedy, iterative_improvement, pipeline, heurtype, itimptype, runmode not in HTML)

**Step 3: Add 3 solver types + mode panels**

In `solver_picker_ui()`, add to `solver_choices` dict (after `zone_sa` line):

```python
    solver_choices["greedy"] = "Greedy Heuristic"
    solver_choices["iterative_improvement"] = "Iterative Improvement"
    solver_choices["pipeline"] = "Run Mode Pipeline"
```

Add three `panel_conditional` blocks (after the MIP panel from Task 2):

```python
                ui.panel_conditional(
                    "input.solver_type === 'greedy'",
                    ui.input_select(
                        "heurtype", "Scoring Mode (HEURTYPE)",
                        choices={
                            "0": "0 - Richness",
                            "1": "1 - Greedy (cheapest)",
                            "2": "2 - Max Rarity (default)",
                            "3": "3 - Best Rarity/Cost",
                            "4": "4 - Average Rarity",
                            "5": "5 - Sum Rarity",
                            "6": "6 - Product Irreplaceability",
                            "7": "7 - Summation Irreplaceability",
                        },
                        selected="2",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'iterative_improvement'",
                    ui.input_select(
                        "itimptype", "Improvement Mode (ITIMPTYPE)",
                        choices={
                            "0": "0 - No improvement",
                            "1": "1 - Removal pass",
                            "2": "2 - Two-step (remove + add)",
                            "3": "3 - Swap",
                        },
                        selected="0",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'pipeline'",
                    ui.input_select(
                        "runmode", "Pipeline Mode (RUNMODE)",
                        choices={
                            "0": "0 - SA only (default)",
                            "1": "1 - Heuristic only",
                            "2": "2 - SA + iterative improvement",
                            "3": "3 - Heuristic + iterative improvement",
                            "4": "4 - Heuristic + SA (pick best)",
                            "5": "5 - Heur + SA + improvement",
                            "6": "6 - Iterative improvement only",
                        },
                        selected="0",
                    ),
                ),
```

In the `_update_config` effect, add to `@reactive.event` args: `input.heurtype, input.itimptype, input.runmode`.

In the `_update_config` body, add:

```python
        if input.solver_type() == "greedy":
            config["heurtype"] = int(input.heurtype() or 2)
        if input.solver_type() == "iterative_improvement":
            config["itimptype"] = int(input.itimptype() or 0)
        if input.solver_type() == "pipeline":
            config["runmode"] = int(input.runmode() or 0)
```

Update `solver_info` render to include descriptions for the 3 new types:

```python
        elif st == "greedy":
            return ("Greedy Heuristic\n----------------\n"
                    "Selects planning units one-by-one based on a scoring\n"
                    "strategy (HEURTYPE 0-7). Fast baseline for comparison.")
        elif st == "iterative_improvement":
            return ("Iterative Improvement\n---------------------\n"
                    "Refines an existing solution by trying removals,\n"
                    "additions, or swaps (ITIMPTYPE 0-3).")
        elif st == "pipeline":
            return ("Run Mode Pipeline\n-----------------\n"
                    "Chains heuristic, SA, and iterative improvement\n"
                    "in sequences matching Marxan RUNMODE 0-6.")
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solver_picker_types.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/solver_config/solver_picker.py tests/pymarxan_shiny/test_solver_picker_types.py
git commit -m "feat: add greedy, iterative_improvement, pipeline to solver picker"
```

---

### Task 4: Connectivity Matrix Input Module

**Files:**
- Create: `src/pymarxan_shiny/modules/connectivity/matrix_input.py`
- Test: `tests/pymarxan_shiny/test_matrix_input.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_matrix_input.py`:

```python
"""Tests for connectivity matrix input Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.connectivity.matrix_input import (
    matrix_input_server,
    matrix_input_ui,
    parse_format_label,
)


def test_matrix_input_ui_returns_tag():
    elem = matrix_input_ui("test_conn")
    assert elem is not None


def test_matrix_input_server_callable():
    assert callable(matrix_input_server)


def test_parse_format_label_edge_list():
    assert parse_format_label("edge_list") == "Edge List (id1, id2, value)"


def test_parse_format_label_full_matrix():
    assert parse_format_label("full_matrix") == "Full Matrix (NxN)"


def test_parse_format_label_unknown():
    assert parse_format_label("unknown") == "Unknown"
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_matrix_input.py -v`
Expected: FAIL — module does not exist

**Step 3: Create the connectivity matrix input module**

Create `src/pymarxan_shiny/modules/connectivity/matrix_input.py`:

```python
"""Connectivity matrix upload Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

FORMAT_LABELS = {
    "edge_list": "Edge List (id1, id2, value)",
    "full_matrix": "Full Matrix (NxN)",
}


def parse_format_label(fmt: str) -> str:
    """Return human-readable format label."""
    return FORMAT_LABELS.get(fmt, "Unknown")


@module.ui
def matrix_input_ui():
    return ui.card(
        ui.card_header("Connectivity Matrix"),
        ui.input_file("conn_file", "Upload CSV", accept=[".csv"]),
        ui.input_radio_buttons(
            "conn_format", "Format",
            choices={
                "edge_list": "Edge List (id1, id2, value)",
                "full_matrix": "Full Matrix (NxN)",
            },
            selected="edge_list",
        ),
        ui.output_text_verbatim("conn_preview"),
    )


@module.server
def matrix_input_server(
    input, output, session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    @reactive.effect
    @reactive.event(input.conn_file)
    def _on_upload():
        file_info = input.conn_file()
        if file_info is None or len(file_info) == 0:
            return

        path = file_info[0]["datapath"]
        fmt = input.conn_format()
        p = problem()

        try:
            if fmt == "edge_list":
                from pymarxan.connectivity.io import read_connectivity_edgelist

                if p is None:
                    ui.notification_show(
                        "Load planning units first.", type="warning",
                    )
                    return
                pu_ids = p.planning_units["id"].tolist()
                matrix = read_connectivity_edgelist(path, pu_ids)
                connectivity_pu_ids.set(pu_ids)
            else:
                from pymarxan.connectivity.io import read_connectivity_matrix

                matrix = read_connectivity_matrix(path)
                connectivity_pu_ids.set(list(range(matrix.shape[0])))

            connectivity_matrix.set(matrix)
            ui.notification_show(
                f"Loaded {matrix.shape[0]}x{matrix.shape[1]} matrix.",
                type="message",
            )
        except Exception as exc:
            ui.notification_show(f"Error: {exc}", type="error")

    @render.text
    def conn_preview():
        m = connectivity_matrix()
        if m is None:
            return "No connectivity matrix loaded."
        import numpy as np

        nonzero = int(np.count_nonzero(m))
        total = m.shape[0] * m.shape[1]
        density = 100.0 * nonzero / total if total > 0 else 0.0
        return (
            f"Shape: {m.shape[0]} x {m.shape[1]}\n"
            f"Non-zero: {nonzero}\n"
            f"Density: {density:.1f}%"
        )
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_matrix_input.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/connectivity/matrix_input.py tests/pymarxan_shiny/test_matrix_input.py
git commit -m "feat: add connectivity matrix CSV upload module"
```

---

### Task 5: Wire MIP Params and Solver Modes into app.py

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Test: (covered by Task 6 integration tests)

**Step 1: Add matrix_input import to app.py**

Add to imports section:

```python
from pymarxan_shiny.modules.connectivity.matrix_input import (
    matrix_input_server,
    matrix_input_ui,
)
```

**Step 2: Add matrix_input_ui to Connectivity tab**

In the Connectivity `ui.nav_panel`, add `matrix_input_ui("matrix_upload")` before `metrics_viz_ui`:

```python
    ui.nav_panel(
        "Connectivity",
        ui.layout_columns(
            matrix_input_ui("matrix_upload"),
            metrics_viz_ui("connectivity"),
            network_view_ui("network"),
            col_widths=[12, 12, 12],
        ),
    ),
```

**Step 3: Add matrix_input_server call**

In the `server` function, add before `metrics_viz_server`:

```python
    matrix_input_server(
        "matrix_upload",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
```

**Step 4: Pass solver mode params to problem.parameters before solving**

In the `active_solver()` reactive calc, update to pass mode params from `solver_config` into problem.parameters. The run_panel_server already has access to `problem` and `solver_config`, and each solver reads from `problem.parameters` at solve time. So add an effect that syncs config → problem.parameters:

Add a new effect after `active_solver()`:

```python
    @reactive.effect
    def _sync_solver_params():
        """Sync UI solver config values into problem.parameters."""
        p = problem()
        cfg = solver_config()
        if p is None:
            return
        st = cfg.get("solver_type", "mip")
        if st == "mip":
            p.parameters["MIP_TIME_LIMIT"] = str(cfg.get("mip_time_limit", 300))
            p.parameters["MIP_GAP"] = str(cfg.get("mip_gap", 0.0))
        elif st == "greedy":
            p.parameters["HEURTYPE"] = str(cfg.get("heurtype", 2))
        elif st == "iterative_improvement":
            p.parameters["ITIMPTYPE"] = str(cfg.get("itimptype", 0))
        elif st == "pipeline":
            p.parameters["RUNMODE"] = str(cfg.get("runmode", 0))
```

**Step 5: Commit**

```bash
git add src/pymarxan_app/app.py
git commit -m "feat: wire matrix_input into app, sync solver params to problem"
```

---

### Task 6: Integration Tests

**Files:**
- Create: `tests/test_integration_phase10.py`

**Step 1: Write the integration tests**

Create `tests/test_integration_phase10.py`:

```python
"""Phase 10 integration tests: MIP params, connectivity input, all solver types."""
from __future__ import annotations


def test_app_imports_phase10():
    """Verify the app still imports cleanly after phase 10 changes."""
    from pymarxan_app import app
    assert app.app is not None


def test_matrix_input_importable():
    """Connectivity matrix input module is importable."""
    from pymarxan_shiny.modules.connectivity.matrix_input import (
        matrix_input_server,
        matrix_input_ui,
    )
    assert callable(matrix_input_ui)
    assert callable(matrix_input_server)


def test_matrix_input_ui_renders():
    """Matrix input UI renders without error."""
    from pymarxan_shiny.modules.connectivity.matrix_input import matrix_input_ui
    elem = matrix_input_ui("test")
    assert elem is not None


def test_solver_picker_all_seven_types():
    """Solver picker shows all 7 solver types."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    for solver_type in ["mip", "sa", "zone_sa", "greedy", "iterative_improvement", "pipeline"]:
        assert solver_type in html, f"Missing solver type: {solver_type}"


def test_solver_picker_mip_params():
    """Solver picker has MIP parameter controls."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    for param in ["mip_time_limit", "mip_gap", "mip_verbose"]:
        assert param in html, f"Missing MIP param: {param}"


def test_solver_picker_mode_selectors():
    """Solver picker has mode selectors for greedy, iterative, pipeline."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    assert "heurtype" in html
    assert "itimptype" in html
    assert "runmode" in html


def test_mip_solver_accepts_params():
    """MIP solver reads time_limit and gap from problem.parameters."""
    from unittest.mock import patch
    import pandas as pd
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "target": [1.0], "spf": [1.0], "name": ["f"]})
    puvsf = pd.DataFrame({"pu": [1, 2], "species": [1, 1], "amount": [1.0, 1.0]})
    p = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvsf,
        parameters={"BLM": "0", "MIP_TIME_LIMIT": "120", "MIP_GAP": "0.01"},
    )
    with patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD") as mock_cbc:
        mock_cbc.return_value = mock_cbc
        try:
            MIPSolver().solve(p, SolverConfig(verbose=True))
        except Exception:
            pass
        mock_cbc.assert_called_once_with(msg=1, timeLimit=120, gapRel=0.01)
```

**Step 2: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_integration_phase10.py -v`
Expected: 7 PASSED

**Step 3: Commit**

```bash
git add tests/test_integration_phase10.py
git commit -m "test: add phase 10 integration tests"
```

---

### Task 7: Full Regression

**Files:** None (test-only)

**Step 1: Run full test suite**

Run: `source .venv/bin/activate && pytest tests/ -v`

Expected: All existing 463 tests + ~22 new tests pass (except 7 pre-existing networkx failures).

**Step 2: Run linter**

Run: `source .venv/bin/activate && ruff check src/ tests/ --fix`

Expected: No errors.

**Step 3: Verify app imports**

Run: `source .venv/bin/activate && python -c "from pymarxan_app.app import app; print('OK')" `

Expected: OK

**Step 4: Final commit if any lint fixes**

```bash
git add -u
git commit -m "chore: fix lint issues from phase 10"
```
