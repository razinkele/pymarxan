"""Smoke tests for ``scripts/release.sh``.

The release script automates the manual workflow run for v0.2.0,
v0.3.0, v0.4.0, and v0.4.1 (bump → promote → build → tag → push →
gh release). These tests use ``--dry-run`` to verify the safety
guards trip and the dry-run output reaches the final step on a
hypothetical version. They do NOT exercise the actual side-effects
(git commit, tag, push, gh release) — that's what manual invocation
is for.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    """Invoke the release script and capture output."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TERM": "dumb"},
    )


def test_release_script_exists_and_executable():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"


def test_release_script_shows_usage_without_args():
    result = _run()
    assert result.returncode != 0
    assert "usage:" in result.stderr.lower()


def test_release_script_rejects_leading_v():
    """User must pass the bare PEP 440 version string."""
    result = _run("v0.4.2", "--dry-run")
    assert result.returncode != 0
    assert "pass the bare version" in result.stderr


def test_release_script_rejects_existing_tag():
    """v0.4.1 already exists in the repo — guard must trip."""
    result = _run("0.4.1", "--dry-run")
    assert result.returncode != 0
    assert "already exists" in result.stderr


@pytest.fixture(scope="module")
def dry_run_output() -> subprocess.CompletedProcess:
    """One dry-run invocation shared across every step-presence test.
    Each subprocess call costs ~1s of Python import overhead; sharing
    keeps the test file fast."""
    return _run("9.9.9", "--dry-run")


def test_release_script_dry_run_reaches_done(dry_run_output):
    """A hypothetical future version passes every guard and reaches the
    final 'Done' step under --dry-run (no side effects)."""
    assert dry_run_output.returncode == 0, (
        f"dry-run failed:\n"
        f"stdout:\n{dry_run_output.stdout}\n"
        f"stderr:\n{dry_run_output.stderr}"
    )
    assert "Done — v9.9.9 released" in dry_run_output.stdout


@pytest.mark.parametrize("step", [
    "Pre-flight",
    "make check",
    "Bump pyproject.toml",
    "Promote CHANGELOG",
    "Commit:",
    "Build wheel",
    "Tag v9.9.9",
    "Push main + tag",
    "GitHub Release",
])
def test_release_script_dry_run_announces_every_step(dry_run_output, step):
    """Each named step appears in the dry-run output — keeps the
    script's workflow steps in lockstep with this test list when
    edited."""
    assert dry_run_output.returncode == 0
    assert step in dry_run_output.stdout, (
        f"step '{step}' missing from dry-run output:\n"
        f"{dry_run_output.stdout}"
    )
