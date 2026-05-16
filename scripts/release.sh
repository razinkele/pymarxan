#!/usr/bin/env bash
# Release automation for pymarxan.
#
# Captures the manual release workflow run for v0.2.0, v0.3.0, v0.4.0,
# v0.4.1: bump version → promote [Unreleased] in CHANGELOG → run
# `make check` → commit → build wheel/sdist → tag → push → create
# GitHub Release with the just-promoted CHANGELOG section as the body.
#
# Usage:
#   scripts/release.sh VERSION              # full release
#   scripts/release.sh VERSION --dry-run    # show every step, change nothing
#
# Examples:
#   scripts/release.sh 0.4.2
#   scripts/release.sh 0.5.0 --dry-run
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
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

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

step "Done — $TAG released"
if [[ $DRY_RUN -eq 0 ]]; then
    echo "    https://github.com/razinkele/pymarxan/releases/tag/$TAG"
fi
