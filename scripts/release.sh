#!/usr/bin/env bash
# Release automation for pymarxan.
#
# Captures the manual release workflow run for v0.2.0, v0.3.0, v0.4.0,
# v0.4.1: bump version → promote [Unreleased] in CHANGELOG → run
# `make check` → commit → build wheel/sdist → tag → push → create
# GitHub Release with the just-promoted CHANGELOG section as the body.
#
# Usage:
#   scripts/release.sh VERSION                 # full release + publish to PyPI
#   scripts/release.sh VERSION --dry-run       # show every step, change nothing
#   scripts/release.sh VERSION --test-pypi     # publish to TestPyPI instead
#   scripts/release.sh VERSION --no-pypi       # GitHub Release only, skip PyPI
#
# Examples:
#   scripts/release.sh 0.4.2
#   scripts/release.sh 0.5.0 --dry-run
#   scripts/release.sh 0.4.2rc1 --test-pypi
#
# PyPI credentials: set the TWINE_PASSWORD env var to a PyPI API token
# (TWINE_USERNAME defaults to __token__), or configure ~/.pypirc. The
# pre-flight check refuses to start a publishing release without one, so
# you never get a pushed tag + GitHub Release that then fails to upload.
#
# Pre-flight safety:
#   - clean working tree (no uncommitted changes / untracked files in src/)
#   - on `main` branch
#   - up to date with `origin/main`
#   - tag `vVERSION` does not already exist
#   - `[Unreleased]` section in CHANGELOG.md has at least one non-empty entry
#     (refuses to ship an empty release)
#   - `make check` passes
#
# The CHANGELOG body for the GitHub Release is extracted automatically
# from the just-promoted ## [VERSION] section, so the release notes
# stay in sync with the changelog — no separate copy to maintain.

set -euo pipefail

# --- arg parsing ------------------------------------------------------

if [[ $# -lt 1 ]]; then
    cat <<EOF >&2
usage: $0 VERSION [--dry-run]

Bumps version, promotes CHANGELOG, builds, tags, and pushes a pymarxan
release. See script header for the full workflow.
EOF
    exit 2
fi

VERSION="$1"
shift
DRY_RUN=0
PUBLISH_PYPI=1
PYPI_REPO="pypi"
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=1 ;;
        --test-pypi) PYPI_REPO="testpypi" ;;
        --no-pypi)   PUBLISH_PYPI=0 ;;
        *)
            echo "error: unknown option '$arg'" >&2
            exit 2
            ;;
    esac
done

# Refuse to take a leading `v` — the script adds it for the tag and
# users should pass the bare PEP 440 version string for pyproject.toml.
if [[ "$VERSION" == v* ]]; then
    echo "error: pass the bare version (e.g. '0.4.2'), not '$VERSION'" >&2
    exit 2
fi

TAG="v${VERSION}"
DATE="$(date -u +%Y-%m-%d)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

step() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $*"
    else
        echo "==> $*"
    fi
}

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  + $*"
    else
        "$@"
    fi
}

# --- pre-flight -------------------------------------------------------

step "Pre-flight: clean working tree, on main, up to date with origin"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: working tree has uncommitted changes — commit or stash first" >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    echo "error: not on main branch (currently on '$BRANCH')" >&2
    exit 1
fi

git fetch --tags origin >/dev/null

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "error: local main not in sync with origin/main" >&2
    echo "       local:  $LOCAL" >&2
    echo "       remote: $REMOTE" >&2
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "error: tag '$TAG' already exists" >&2
    exit 1
fi

# Verify [Unreleased] has content. Awk: print everything between the
# first `## [Unreleased]` line and the next `## [`. Strip the
# `## [Unreleased]` header itself; if anything non-blank remains, the
# section is non-empty.
UNRELEASED_BODY=$(awk '
    /^## \[Unreleased\]/ { in_section = 1; next }
    in_section && /^## \[/ { exit }
    in_section { print }
' CHANGELOG.md | grep -v '^[[:space:]]*$' || true)

if [[ -z "$UNRELEASED_BODY" ]]; then
    echo "error: [Unreleased] section in CHANGELOG.md is empty —" >&2
    echo "       nothing to ship in $TAG" >&2
    exit 1
fi

# If we're going to publish to PyPI, verify the tooling and credentials
# are present NOW — before any irreversible commit/tag/push. Otherwise a
# release could push a tag + GitHub Release and then fail at the very
# last step with no way to retry the upload cleanly.
# Default so `set -u` is safe even when PyPI publishing is off or the
# real tooling check is skipped under --dry-run.
TWINE=(python3 -m twine)
if [[ $PUBLISH_PYPI -eq 1 ]]; then
    step "Pre-flight: twine available + PyPI credentials present (target: $PYPI_REPO)"
    if command -v twine >/dev/null 2>&1; then
        TWINE=(twine)
    elif python3 -m twine --version >/dev/null 2>&1; then
        TWINE=(python3 -m twine)
    elif [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] note: twine not installed — a real release would fail here"
    else
        echo "error: twine not found — install it (pip install twine) or pass --no-pypi" >&2
        exit 1
    fi
    # Credentials come from TWINE_PASSWORD (an API token) or ~/.pypirc.
    # We can't validate the token without uploading, but we can refuse to
    # start if neither source exists. Dry-run only warns — it must stay
    # runnable in CI without secrets.
    if [[ -z "${TWINE_PASSWORD:-}" && ! -f "$HOME/.pypirc" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [dry-run] note: no PyPI credentials — a real release would fail here"
        else
            echo "error: no PyPI credentials found." >&2
            echo "       set TWINE_PASSWORD to a PyPI API token (TWINE_USERNAME" >&2
            echo "       defaults to __token__), configure ~/.pypirc, or pass --no-pypi" >&2
            exit 1
        fi
    fi
fi

# --- run checks BEFORE making any changes -----------------------------

step "Run \`make check\` before any version bump"
if [[ $DRY_RUN -eq 0 ]]; then
    if ! make check; then
        echo "error: \`make check\` failed — refusing to release" >&2
        exit 1
    fi
fi

# --- bump pyproject.toml version --------------------------------------

step "Bump pyproject.toml version → $VERSION"
if [[ $DRY_RUN -eq 0 ]]; then
    # macOS sed needs '' after -i; GNU sed does not. We use a portable
    # in-place replace via python.
    python3 - <<EOF
import re
from pathlib import Path
p = Path("pyproject.toml")
text = p.read_text()
new_text, n = re.subn(
    r'^version = "[^"]+"',
    f'version = "$VERSION"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    raise SystemExit("error: failed to bump pyproject.toml version line")
p.write_text(new_text)
EOF
fi

# --- promote CHANGELOG [Unreleased] → [VERSION] ----------------------

step "Promote CHANGELOG.md [Unreleased] → [$VERSION] — $DATE"
if [[ $DRY_RUN -eq 0 ]]; then
    python3 - <<EOF
import re
from pathlib import Path

p = Path("CHANGELOG.md")
text = p.read_text()

# Insert a fresh [Unreleased] header above the existing one's body, then
# rename the existing one to [$VERSION] with today's date.
header = "## [Unreleased]"
target = f"## [$VERSION] — $DATE"

# Replace the first [Unreleased] header with: fresh empty [Unreleased]
# followed by [$VERSION] — $DATE.
new_text, n = re.subn(
    re.escape(header),
    header + "\n\n" + target,
    text,
    count=1,
)
if n != 1:
    raise SystemExit("error: couldn't find [Unreleased] header in CHANGELOG.md")

# Append the compare-link line at the bottom.
# Existing link table convention:
#   [Unreleased]: https://github.com/razinkele/pymarxan/compare/vPREV...HEAD
#   [0.4.1]: https://github.com/razinkele/pymarxan/releases/tag/v0.4.1
# Update the [Unreleased] compare to the new tag and add a new link
# entry for [$VERSION].
new_text = re.sub(
    r'^\[Unreleased\]:\s*https://github\.com/[^/]+/[^/]+/compare/v[^.]+\.\d+\.\d+\S*\.\.\.HEAD',
    "[Unreleased]: https://github.com/razinkele/pymarxan/compare/v$VERSION...HEAD",
    new_text,
    count=1,
    flags=re.MULTILINE,
)
# Insert the new tag link immediately after the [Unreleased] line.
new_text = re.sub(
    r'^(\[Unreleased\]:.*$)',
    r'\1\n[$VERSION]: https://github.com/razinkele/pymarxan/releases/tag/v$VERSION',
    new_text,
    count=1,
    flags=re.MULTILINE,
)

p.write_text(new_text)
EOF
fi

# --- commit + build + tag --------------------------------------------

step "Commit: chore(release): v$VERSION"
run git add pyproject.toml CHANGELOG.md
run git commit -m "chore(release): v$VERSION"

step "Build wheel + sdist"
run rm -rf dist/
run python -m build

# Validate the artifacts BEFORE any irreversible push. `twine check`
# catches malformed metadata / unrenderable README that PyPI would
# reject — better to find out here than after the tag is public.
if [[ $PUBLISH_PYPI -eq 1 ]]; then
    step "Validate artifacts with \`twine check\`"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  + ${TWINE[*]} check dist/pymarxan-${VERSION}-py3-none-any.whl dist/pymarxan-${VERSION}.tar.gz"
    else
        "${TWINE[@]}" check \
            "dist/pymarxan-${VERSION}-py3-none-any.whl" \
            "dist/pymarxan-${VERSION}.tar.gz"
    fi
fi

step "Tag $TAG"
run git tag -a "$TAG" -m "$TAG"

step "Push main + tag to origin"
run git push origin main
run git push origin "$TAG"

# --- GitHub Release with extracted CHANGELOG body --------------------

step "Create GitHub Release $TAG with extracted CHANGELOG body"

# Extract the just-promoted ## [VERSION] section. Awk: print everything
# from `## [$VERSION]` to (but not including) the next `## [`.
BODY_FILE=$(mktemp)
trap 'rm -f "$BODY_FILE"' EXIT
if [[ $DRY_RUN -eq 0 ]]; then
    awk -v target="## [$VERSION] — $DATE" '
        $0 == target { in_section = 1; next }
        in_section && /^## \[/ { exit }
        in_section { print }
    ' CHANGELOG.md > "$BODY_FILE"

    if [[ ! -s "$BODY_FILE" ]]; then
        echo "warning: extracted release body is empty — using fallback title only"
        echo "$TAG" > "$BODY_FILE"
    fi

    gh release create "$TAG" \
        "dist/pymarxan-${VERSION}-py3-none-any.whl" \
        "dist/pymarxan-${VERSION}.tar.gz" \
        --title "$TAG" \
        --notes-file "$BODY_FILE"
else
    echo "  + gh release create $TAG dist/pymarxan-${VERSION}-py3-none-any.whl dist/pymarxan-${VERSION}.tar.gz \\"
    echo "      --title '$TAG' --notes-file <extracted CHANGELOG section>"
fi

# --- publish to PyPI --------------------------------------------------

if [[ $PUBLISH_PYPI -eq 1 ]]; then
    step "Publish to PyPI ($PYPI_REPO)"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  + ${TWINE[*]} upload --repository $PYPI_REPO dist/pymarxan-${VERSION}-py3-none-any.whl dist/pymarxan-${VERSION}.tar.gz"
    else
        "${TWINE[@]}" upload --repository "$PYPI_REPO" \
            "dist/pymarxan-${VERSION}-py3-none-any.whl" \
            "dist/pymarxan-${VERSION}.tar.gz"
    fi
fi

step "Done — $TAG released"
if [[ $DRY_RUN -eq 0 ]]; then
    echo "    https://github.com/razinkele/pymarxan/releases/tag/$TAG"
    if [[ $PUBLISH_PYPI -eq 1 ]]; then
        if [[ "$PYPI_REPO" == "testpypi" ]]; then
            echo "    https://test.pypi.org/project/pymarxan/$VERSION/"
        else
            echo "    https://pypi.org/project/pymarxan/$VERSION/"
        fi
    fi
fi
