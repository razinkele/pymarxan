---
name: marxan-testing
description: >
  Run, debug, or interpret the pymarxan test suite correctly. Use whenever you
  are about to run tests, are told a test failed, need to add tests, or are
  about to claim work passes — even casually ("run the tests", "did that break
  anything?", "make sure it still works"). The suite has a hard environment
  footgun (it must run under the `shiny` micromamba env, NOT `.venv`) and a
  known flaky test; reach for this skill before invoking pytest so you don't
  burn a cycle on a spurious ImportError or misread a flake as a real failure.
---

# Testing pymarxan

The single most common way to waste a cycle here is running pytest under the
wrong Python. This skill exists to prevent that and to interpret results
correctly.

## The environment footgun (read first)

`.venv` does **not** have `rasterio` or `ipyleaflet`. Roughly the whole
`pymarxan.spatial` subpackage and the Shiny map modules import them, so running
the suite under `.venv` fails with a misleading `ModuleNotFoundError: No module
named 'rasterio'` that looks like a code bug but is just the wrong interpreter.

Tests must run under the `shiny` micromamba env, which has both:

```bash
/opt/micromamba/envs/shiny/bin/pytest tests/ -x -q
```

The `Makefile` calls bare `pytest`, so `make test` / `make check` only work
when that env is already on `PATH`. If you're not sure it is, prefer the
explicit interpreter path above, or activate the env first:

```bash
source /opt/micromamba/etc/profile.d/micromamba.sh && micromamba activate shiny
```

Verify the env is right before a long run: `python -c "import rasterio,
ipyleaflet"` should succeed silently.

## Which command to run

- **Fast inner-loop while iterating** — fail fast on the first error:
  `/opt/micromamba/envs/shiny/bin/pytest tests/ -x -q`
- **A single file or test** — append the node id:
  `... tests/test_circuit.py -q` or `... tests/test_circuit.py::TestName::test_case`
- **Skip the slow SA / zone / pipeline tests** while iterating: add `-m "not slow"`.
- **Full gate before claiming done / before a PR or release**: `make check`
  (runs `lint` → `types` → `test`). This is the bar that must be green.

Pytest markers (defined in `pyproject.toml`): `slow`, `integration`,
`spatial`, `bench`. **`bench` is excluded from `make test` and CI on purpose**
— those are perf-budget gates sized for the dev machine and will flake on
slower runners. Only run them deliberately with `make bench` when you've
touched an SA inner loop and want to check the per-flip cost budget.

## Known flake — don't chase it

`test_solutions_are_different` is a stochastic SA test that occasionally fails
and **passes on rerun**. If it's the only failure, rerun just that test before
treating it as real:

```bash
/opt/micromamba/envs/shiny/bin/pytest tests/ -q -k test_solutions_are_different
```

A failure that reproduces on rerun, or any *other* test failing, is a real
signal — debug it (see `superpowers:systematic-debugging`), don't rerun-and-hope.

## Adding tests

This project is strongly TDD (see `superpowers:test-driven-development`): write
the failing test first. Two recurring gotchas when adding tests here:

- Anything that instantiates an `ipyleaflet.Map` outside a Shiny session needs
  the `_allow_widget_outside_session` fixture — `shinywidgets` hooks
  `ipywidgets` globally and the bare construction otherwise errors.
- Marxan-parity assertions should be checked against the hand-verified ground
  truth, not just "it ran" — see the `marxan-parity-check` skill.

## Before you say it passes

Don't claim green from memory. Run the relevant command, read the actual
summary line, and report the real count (the suite is ~1459 tests as of the
last full run). If you skipped slow tests or only ran a subset, say so rather
than implying full coverage (see `superpowers:verification-before-completion`).
