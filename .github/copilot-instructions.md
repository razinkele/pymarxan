# Copilot Instructions for pymarxan

## Build, Test, and Lint

```bash
make test            # Full suite (489 tests) with coverage
make test-fast       # Skip slow SA tests (~479 tests, ~15s)
make lint            # Ruff linter
make types           # mypy type checker
make check           # All of the above

# Single test file
pytest tests/pymarxan/solvers/test_mip.py -v

# Single test by name
pytest tests/ -k "test_blm_affects_solution" -v

# Tests by marker
pytest tests/ -m slow -v
pytest tests/ -m "not slow and not spatial" -v
```

Coverage threshold is 75% (`pyproject.toml`). The `src/pymarxan_app/` package is excluded from coverage.

## Architecture

pymarxan is a three-layer monorepo:

- **`src/pymarxan/`** — Pure Python core library (no UI dependencies). Models, solvers, I/O, calibration, analysis, spatial, connectivity, zones.
- **`src/pymarxan_shiny/`** — Reusable Shiny for Python UI modules (~22 modules across 9 groups).
- **`src/pymarxan_app/`** — Assembled 8-tab Shiny web application that wires the modules together.

### Data flow

```
File (*.dat) → load_project() → ConservationProblem → Solver.solve(problem, config) → list[Solution]
                                                                                         ↓
                                                                          Analysis / save_project()
```

### Core model (`models/problem.py`)

`ConservationProblem` is a dataclass with these DataFrames:

| Field | Required columns |
|-------|-----------------|
| `planning_units` | `id`, `cost`, `status` |
| `features` | `id`, `name`, `target`, `spf` |
| `pu_vs_features` | `species`, `pu`, `amount` |
| `boundary` (optional) | `id1`, `id2`, `boundary` |

Plus a `parameters: dict` for Marxan settings (e.g., `{"BLM": 1.0}`).

Planning unit status values: `0` = normal, `1` = initial include, `2` = locked in, `3` = locked out.

Validation uses `validate() → list[str]` (returns error strings, not exceptions).

### Solver interface (`solvers/base.py`)

All solvers inherit the `Solver` ABC:

```python
class Solver(ABC):
    def solve(self, problem: ConservationProblem, config: SolverConfig | None = None) -> list[Solution]: ...
    def name(self) -> str: ...
    def supports_zones(self) -> bool: ...
    def available(self) -> bool: return True
```

`SolverConfig` is a dataclass (`num_solutions`, `seed`, `verbose`, `metadata`). `Solution` is a dataclass with `selected: np.ndarray`, `cost`, `boundary`, `objective`, `targets_met`, and optional `zone_assignment`.

To add a new solver: inherit `Solver`, implement `solve()`, `name()`, `supports_zones()`.

### I/O pattern (`io/readers.py`, `io/writers.py`)

Symmetric read/write function pairs per file type: `read_pu`/`write_pu`, `read_spec`/`write_spec`, etc. The main entry points are `load_project(path)` and `save_project(problem, path)`. The reader auto-detects tab vs comma delimiters.

### Shiny module pattern (`pymarxan_shiny/modules/`)

Each module follows this structure:

```python
@module.ui
def module_name_ui():
    return ui.card(...)

@module.server
def module_name_server(input, output, session, problem: reactive.Value, solver: reactive.Calc):
    help_server_setup(input, "module_name")
    ...
```

Conventions: file named `{module_name}.py`, functions named `{module_name}_ui()` and `{module_name}_server()`, reactive dependencies passed as arguments, help integration via `help_server_setup()` + `help_card_header()`.

## Key Conventions

- **Python 3.11+** with `from __future__ import annotations` in all files.
- **Ruff** for linting (rules: E, F, I, UP; line length 99).
- **mypy** with `warn_return_any = true`.
- **Full type hints** throughout — maintain this when adding code.
- **Dataclasses** for domain models (`ConservationProblem`, `Solution`, `SolverConfig`).
- **Pytest markers**: `@pytest.mark.slow` for SA-heavy/zone tests, `@pytest.mark.integration` for roundtrip scenarios, `@pytest.mark.spatial` for geopandas-dependent tests.
- **Test fixture**: `tiny_problem` (6 PUs, 3 features) loaded from `tests/data/simple/` via `conftest.py`.
- **Test structure** mirrors `src/`: unit tests in `tests/pymarxan/{subpackage}/`, integration tests at `tests/test_integration*.py`.
