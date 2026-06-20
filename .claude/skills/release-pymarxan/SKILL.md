---
name: release-pymarxan
description: >
  Cut a pymarxan release the way this project does it. Use whenever the user
  asks to release, ship, tag, bump the version, publish a wheel, push to PyPI,
  or "cut v0.x.y" — and when deciding the next version number or writing the
  CHANGELOG for a release. Wraps `scripts/release.sh` and captures the
  surrounding conventions (version lives only in pyproject.toml, CHANGELOG
  [Unreleased] must be non-empty, PyPI needs a token) so a release doesn't get
  half-done (pushed tag, failed upload) or skip a required step.
---

# Releasing pymarxan

There is a single source of truth for the mechanics: `scripts/release.sh`. It
encodes the manual workflow used for v0.2.0 → v0.4.1. **Use it; don't hand-run
the steps.** This skill is the operator's guide around it.

## Always dry-run first

The script changes irreversible things (commit, tag, push, GitHub Release,
PyPI upload). Preview the full plan before committing to it:

```bash
scripts/release.sh 0.5.0 --dry-run
```

Read the output, confirm the version and the CHANGELOG section it will extract,
then run for real.

## Invocation forms

```bash
scripts/release.sh 0.5.0              # full release + publish to PyPI
scripts/release.sh 0.5.0 --no-pypi    # tag + GitHub Release only, skip PyPI
scripts/release.sh 0.5.0rc1 --test-pypi   # publish to TestPyPI
```

Pass the **bare** PEP 440 version (`0.5.0`), not `v0.5.0` — the script adds the
`v` for the tag and rejects a leading `v`.

## What the script enforces (its pre-flight)

Don't re-implement these; just know it will refuse to start unless:

- working tree is clean, on `main`, and in sync with `origin/main`;
- tag `vVERSION` does not already exist;
- the `## [Unreleased]` section in `CHANGELOG.md` is **non-empty** (it refuses
  to ship an empty release);
- for a PyPI publish: `twine` is importable **and** a credential exists
  (`TWINE_PASSWORD` env var = a PyPI API token, or `~/.pypirc`) — checked up
  front so you never get a pushed tag whose upload then fails;
- `make check` passes (lint + types + full test suite).

Then it bumps the version, promotes CHANGELOG, commits, builds wheel+sdist,
`twine check`s them, tags, pushes, creates the GitHub Release with the promoted
CHANGELOG section as the body, and (unless `--no-pypi`) uploads to PyPI.

## Your job before running it

1. **Make sure `make check` is green** under the right env — see the
   `marxan-testing` skill (bare `pytest` needs the `shiny` micromamba env on
   `PATH`, not `.venv`). The script runs `make check` itself, but failing there
   wastes the run.
2. **Fill `## [Unreleased]` in `CHANGELOG.md`.** This becomes the GitHub
   Release notes verbatim, so write it for humans. Keep the Keep-a-Changelog
   structure already in the file.
3. **Pick the version deliberately.** The version is read at runtime via
   `importlib.metadata.version("pymarxan")` and lives in **`pyproject.toml`
   only** — never hardcode it elsewhere. Bundled feature themes have a planned
   version (e.g. the "modern conservation planning" Tier-A/B work is slated for
   v0.5.0); check `MEMORY.md` / the realignment roadmap before numbering.
4. **For a real PyPI push**, confirm the token is available: this repo has
   historically shipped to GitHub Releases but **not yet to PyPI**, so the PyPI
   step is the one most likely to be new. If the user hasn't provided a token,
   either get one or use `--no-pypi`.

## After release

Update `MEMORY.md` with the shipped version, the artifact sizes, the release
URL, and whether it reached PyPI — that running log is how the next session
knows the state. (Use the `remember` skill / handoff if mid-session.)
