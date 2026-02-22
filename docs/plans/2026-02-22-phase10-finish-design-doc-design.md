# Phase 10: Finish Design Doc — MIP Params, Connectivity Input, All Solver Types

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Close the gap to 100% of the original design doc by adding MIP parameter controls, connectivity matrix upload, and exposing all solver types in the picker.

---

## 1. Scope

| # | Item | Type | Purpose |
|---|------|------|---------|
| 1 | MIP parameter controls | Modify | Time limit, gap tolerance, verbose in MIP solver + solver_picker |
| 2 | Connectivity matrix input | New module | CSV upload for connectivity matrix, populates reactive values |
| 3 | Missing solver types in picker | Modify | Add greedy, iterative_improvement, pipeline to radio buttons |

After this phase: **22/22 design doc modules complete** (100%).

## 2. MIP Parameter Controls

### Backend: `src/pymarxan/solvers/mip_solver.py`

Current line 126 hardcodes: `solver = pulp.PULP_CBC_CMD(msg=0)`

Change to read from `problem.parameters`:

```python
time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
gap = float(problem.parameters.get("MIP_GAP", 0.0))
verbose = config.verbose
solver = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=time_limit, gapRel=gap)
```

Parameters:
- `MIP_TIME_LIMIT`: seconds, default 300 (5 min), min 10
- `MIP_GAP`: relative optimality gap, default 0.0 (exact), range 0.0-1.0
- `config.verbose`: already exists on SolverConfig

### UI: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`

Add a `panel_conditional` block for MIP (matching the existing SA pattern):

```python
ui.panel_conditional(
    "input.solver_type === 'mip'",
    ui.input_numeric("mip_time_limit", "Time Limit (seconds)", value=300, min=10, step=30),
    ui.input_numeric("mip_gap", "Optimality Gap", value=0.0, min=0.0, max=1.0, step=0.01),
    ui.input_checkbox("mip_verbose", "Verbose output", value=False),
),
```

The server `_update_config` effect passes these through `solver_config` dict, and `app.py` writes them to `problem.parameters` before solving.

## 3. Connectivity Matrix Input

### New module: `src/pymarxan_shiny/modules/connectivity/matrix_input.py`

**UI:**
- File upload input (accept `.csv`)
- Format selector: "Edge List (id1, id2, value)" or "Full Matrix (NxN)"
- Preview: matrix shape, non-zero count, density percentage

**Server:**
1. User uploads CSV
2. Parse with existing `connectivity/io.py`:
   - Edge list format → `read_connectivity_edgelist(path, pu_ids)`
   - Full matrix format → `read_connectivity_matrix(path)`
3. Set `connectivity_matrix` and `connectivity_pu_ids` reactive values
4. Show notification on success/error

**Server signature:**
```python
def matrix_input_server(
    input, output, session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
)
```

**Dependencies:** Uses existing `pymarxan.connectivity.io` functions — no new core code needed.

**Integration:** Add to app.py Connectivity tab before metrics_viz. The `connectivity_matrix` reactive value already feeds into `metrics_viz_server` and `network_view_server`.

## 4. Missing Solver Types in Picker

### Current: 4 solver types (mip, sa, zone_sa, binary)
### Add: 3 more (greedy, iterative_improvement, pipeline)

**Changes to `solver_picker.py`:**

Add to `solver_choices` dict:
```python
solver_choices["greedy"] = "Greedy Heuristic"
solver_choices["iterative_improvement"] = "Iterative Improvement"
solver_choices["pipeline"] = "Run Mode Pipeline"
```

Add `panel_conditional` blocks:
- **Greedy:** dropdown for HEURTYPE (0-7 scoring modes)
- **Iterative Improvement:** dropdown for ITIMPTYPE (0-3 improvement modes)
- **Pipeline:** dropdown for RUNMODE (0-6 pipeline modes)

**Changes to `app.py:active_solver()`:**

Already handles all 7 types (lines 117-134). Just need to pass mode params from `solver_config` dict to `problem.parameters` before solving.

## 5. File Changes Summary

### New Files (2)

| File | Purpose |
|---|---|
| `src/pymarxan_shiny/modules/connectivity/matrix_input.py` | Connectivity CSV upload module |
| `tests/test_integration_phase10.py` | Phase 10 integration tests |

### Modified Files (3)

| File | Change |
|---|---|
| `src/pymarxan/solvers/mip_solver.py` | Read MIP_TIME_LIMIT, MIP_GAP, verbose from config |
| `src/pymarxan_shiny/modules/solver_config/solver_picker.py` | Add MIP params panel, add 3 solver types + mode selectors |
| `src/pymarxan_app/app.py` | Wire matrix_input, pass solver mode params to problem.parameters |

### Unchanged

- All 8 solvers' core logic (except MIP's solver instantiation line)
- All existing 21 Shiny modules
- All 463 tests
- Core models, I/O, calibration, analysis backends

## 6. Testing Strategy

1. All 463 existing tests must continue to pass
2. MIP tests: verify time_limit and gap are passed to CBC, verify default behavior unchanged
3. Matrix input tests: UI returns tag, server callable, parse functions dispatch correctly
4. Solver picker tests: all 7 solver types produce valid config dicts, mode params included
5. Integration tests: app imports cleanly, all modules wire correctly
