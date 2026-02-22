# pymarxan

Modular Python toolkit for Marxan conservation planning.

![CI](https://github.com/pymarxan/pymarxan/actions/workflows/ci.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What is this?

[Marxan](https://marxansolutions.org/) is the world's most widely used conservation planning software, helping prioritize areas for biodiversity protection. **pymarxan** is a complete Python reimplementation that combines the classic Marxan algorithms with modern exact solvers, an interactive web UI, and a modular architecture for programmatic use.

It provides a pure Python core library for headless optimization, reusable Shiny UI components, and an assembled web application — all in one package.

## Quick Start

```bash
# Clone and install
git clone https://github.com/pymarxan/pymarxan.git
cd pymarxan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Launch the web app
make app
```

Then open http://localhost:8000 in your browser.

## Architecture

```
pymarxan (three-layer monorepo)
├── src/pymarxan/          Core library: models, solvers, I/O, calibration, analysis
├── src/pymarxan_shiny/    Reusable Shiny UI modules (22 modules)
└── src/pymarxan_app/      Assembled 8-tab Shiny web application
```

- **pymarxan** — Pure Python, no UI dependencies. Use it in scripts, notebooks, or pipelines.
- **pymarxan_shiny** — Shiny for Python modules: maps, calibration, solver config, results.
- **pymarxan_app** — Wires all modules into a complete conservation planning application.

## Solvers

| Solver | Type | Description |
|--------|------|-------------|
| MIP (PuLP/CBC) | Exact | Mixed Integer Programming — guaranteed optimal solution |
| Simulated Annealing | Heuristic | Native Python SA with adaptive cooling |
| Marxan C++ Binary | Heuristic | Wraps the original Marxan executable |
| Zone SA | Heuristic | Multi-zone simulated annealing |
| Greedy Heuristic | Heuristic | 8 scoring modes (HEURTYPE 0-7) |
| Iterative Improvement | Heuristic | 4 refinement modes (ITIMPTYPE 0-3) |
| Run Mode Pipeline | Pipeline | Chains solvers per Marxan RUNMODE 0-6 |

## Development

```bash
make test        # Full test suite (489 tests) with coverage
make test-fast   # Skip slow SA tests (~479 tests, ~15s)
make lint        # Ruff linter
make types       # mypy type checker
make check       # All of the above
make docs        # Generate API docs with pdoc
```

## Docker

```bash
make docker
# or
docker compose up --build
```

The app will be available at http://localhost:8000.

## License

MIT
