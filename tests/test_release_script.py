"""Smoke tests for ``scripts/release.sh``.

The release script automates the manual workflow run for v0.2.0,
v0.3.0, v0.4.0, and v0.4.1 (bump → promote → build → tag → push →
gh release). These tests use ``--dry-run`` to verify the safety
guards trip and the dry-run output reaches the final step on a
hypothetical version. They do NOT exercise the actual side-effects
(git commit, tag, push, gh release) — that's what manual invocation
is for.

The git-aware tests run against a **throwaway fixture repo** built in a
temp dir, not the live checkout. The script's pre-flight insists on a
clean tree, a ``main`` branch in sync with ``origin/main``, and a
non-empty CHANGELOG ``[Unreleased]`` section — coupling the tests to the
real repo's state meant they went red on any unpushed commit or right
after a release (when ``[Unreleased]`` is empty). The fixture gives each
run a hermetic repo that satisfies those guards, with a local *bare*
remote so ``git fetch`` works offline. ``--dry-run`` never invokes
``make check`` (it's gated on a real run), so the fixture needs no
Makefile or source tree.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release.sh"

# A minimal CHANGELOG whose [Unreleased] section is non-empty, so the
# script's "nothing to ship" guard passes in the fixture.
_CHANGELOG_TEMPLATE = """# Changelog

## [Unreleased]

### Added

- Placeholder entry so the release script's non-empty [Unreleased]
  guard passes under test.

## [0.0.1] — 2020-01-01

### Added

- Initial release.

[Unreleased]: https://example.com/compare/v0.0.1...HEAD
[0.0.1]: https://example.com/releases/tag/v0.0.1
"""


def _git(repo: Path, *args: str) -> None:
    """Run a git command inside ``repo``, raising on failure."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _build_release_repo(root: Path) -> Path:
    """Create a hermetic git repo (with a local bare 'origin') that
    satisfies the release script's git + CHANGELOG pre-flight checks."""
    remote = root / "remote.git"
    repo = root / "repo"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "init", str(repo)],
        check=True, capture_output=True, text=True,
    )
    # Force a `main` branch regardless of the host's init.defaultBranch.
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.email", "release-test@example.com")
    _git(repo, "config", "user.name", "Release Test")
    (repo / "CHANGELOG.md").write_text(_CHANGELOG_TEMPLATE)
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.0.1"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    return repo


def _run(*args: str) -> subprocess.CompletedProcess:
    """Invoke the release script for arg-parsing guards that exit before
    any git operation, so they don't need a fixture repo."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TERM": "dumb"},
    )


def _run_in(repo: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """Invoke the release script with ``cwd`` set to a fixture repo, so it
    reads/operates on that repo's git state and CHANGELOG."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(repo),
        env={**os.environ, "TERM": "dumb", **env},
    )


@pytest.fixture(scope="module")
def release_repo(tmp_path_factory) -> Path:
    """A throwaway repo shared by the read-only (dry-run) tests."""
    return _build_release_repo(tmp_path_factory.mktemp("release_fixture"))


@pytest.fixture(scope="module")
def dry_run_output(release_repo: Path) -> subprocess.CompletedProcess:
    """One dry-run invocation shared across every step-presence test.
    Sharing keeps the file fast (each subprocess costs ~1s of import
    overhead, plus the one-time repo build)."""
    return _run_in(release_repo, "9.9.9", "--dry-run")


# --- arg-parsing guards (no git, no fixture needed) -------------------


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


def test_release_script_rejects_unknown_option():
    """An unrecognised flag is a usage error, not a silent no-op."""
    result = _run("9.9.9", "--bogus")
    assert result.returncode != 0
    assert "unknown option" in result.stderr


# --- git-aware guards (run against the fixture repo) ------------------


def test_release_script_rejects_existing_tag(tmp_path):
    """An already-existing tag must trip the guard."""
    repo = _build_release_repo(tmp_path)
    _git(repo, "tag", "v9.9.9")
    result = _run_in(repo, "9.9.9", "--dry-run")
    assert result.returncode != 0
    assert "already exists" in result.stderr


def test_release_script_rejects_empty_unreleased(tmp_path):
    """An empty [Unreleased] section means nothing to ship — abort."""
    repo = _build_release_repo(tmp_path)
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.0.1] — 2020-01-01\n\n- x\n"
    )
    _git(repo, "commit", "-am", "empty unreleased")
    _git(repo, "push", "origin", "main")
    result = _run_in(repo, "9.9.9", "--dry-run")
    assert result.returncode != 0
    assert "[Unreleased] section in CHANGELOG.md is empty" in result.stderr


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
    "release toolchain",
    "make check",
    "Bump pyproject.toml",
    "Promote CHANGELOG",
    "Commit:",
    "Build wheel",
    "twine check",
    "Tag v9.9.9",
    "Push main + tag",
    "GitHub Release",
    "Publish to PyPI",
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


def test_release_script_dry_run_checks_build_toolchain(dry_run_output):
    """Pre-flight announces a release-toolchain check (ruff, mypy, pytest,
    build). Its whole point is to fail in pre-flight when a tool is
    missing — rather than mid-release after the version-bump commit has
    already landed, as happened on the v0.5.0 run when ``python -m build``
    was unavailable."""
    assert dry_run_output.returncode == 0
    assert "release toolchain" in dry_run_output.stdout


def test_release_script_toolchain_check_precedes_any_mutation(dry_run_output):
    """The toolchain verification must run before the first irreversible
    step (the version bump / build), so a missing tool aborts with
    nothing committed."""
    out = dry_run_output.stdout
    assert out.index("release toolchain") < out.index("Bump pyproject.toml")
    assert out.index("release toolchain") < out.index("Build wheel")


def test_release_script_warns_on_missing_build_tool_in_dry_run(release_repo):
    """A real run aborts on a missing tool, but --dry-run only warns so it
    stays runnable in CI without the full toolchain — mirroring the twine
    credential pre-flight. Force a missing interpreter via PYTHON and
    confirm the dry-run still reaches Done with a note rather than dying."""
    result = _run_in(
        release_repo, "9.9.9", "--dry-run", "--no-pypi",
        PYTHON="definitely-not-a-real-python",
    )
    assert result.returncode == 0, result.stderr
    assert "Done — v9.9.9 released" in result.stdout
    assert "a real release would fail here" in result.stdout


def test_release_script_default_publishes_to_pypi(dry_run_output):
    """Default (no flag) targets the real PyPI index, not TestPyPI."""
    assert dry_run_output.returncode == 0
    assert "--repository pypi" in dry_run_output.stdout
    assert "testpypi" not in dry_run_output.stdout


def test_release_script_no_pypi_skips_upload(release_repo):
    """--no-pypi reaches Done but never announces a PyPI publish step."""
    result = _run_in(release_repo, "9.9.9", "--dry-run", "--no-pypi")
    assert result.returncode == 0, result.stderr
    assert "Done — v9.9.9 released" in result.stdout
    assert "Publish to PyPI" not in result.stdout
    assert "twine" not in result.stdout.lower()


def test_release_script_test_pypi_targets_testpypi(release_repo):
    """--test-pypi routes the upload to the testpypi repository."""
    result = _run_in(release_repo, "9.9.9", "--dry-run", "--test-pypi")
    assert result.returncode == 0, result.stderr
    assert "--repository testpypi" in result.stdout
