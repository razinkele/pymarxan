"""Emit a prototype-first implementation plan from prototype commits.

Usage: python plangen.py REPO SPEC.json OUT.md

SPEC.json: {"header": "<markdown before the first task>",
            "tasks": [{"commit": "<sha>", "title": "...", "interfaces": "<md>",
                       "red_cmd": "...", "red_expected": "...",
                       "green_cmd": "...", "green_expected": "...",
                       "gate_expected": "...", "notes": "<md, optional>"}],
            "gate_cmd": "...", "footer": "<markdown, optional>"}

Code blocks are git output written verbatim (no stripping), so `git apply` replays them.
New files are given in full; modified files as a diff against the previous task's tree.
"""
from __future__ import annotations

import json
import subprocess
import sys


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout


def lang(path: str) -> str:
    return {"py": "python", "md": "markdown", "toml": "toml", "json": "json"}.get(
        path.rsplit(".", 1)[-1], ""
    )


def fence(body: str, info: str) -> str:
    ticks = "````" if "```" in body else "```"
    if not body.endswith("\n"):
        body += "\n"
    return f"{ticks}{info}\n{body}{ticks}\n"


def file_block(repo: str, sha: str, status: str, path: str) -> str:
    if status == "A":
        body = git(repo, "show", f"{sha}:{path}")
        return f"Create `{path}`:\n\n" + fence(body, lang(path))
    if status == "D":
        return f"Delete `{path}`:\n\n" + fence(f"git rm {path}\n", "bash")
    body = git(repo, "diff", f"{sha}^", sha, "--", path)
    return f"Modify `{path}` (apply this diff):\n\n" + fence(body, "diff")


def task_md(repo: str, n: int, t: dict, gate_cmd: str) -> str:
    sha = t["commit"]
    status = git(repo, "diff", "--name-status", f"{sha}^", sha)
    changes = [line.split("\t") for line in status.splitlines()]
    tests = [(s[0], s[-1]) for s in changes if s[-1].startswith("tests/")]
    impl = [(s[0], s[-1]) for s in changes if not s[-1].startswith("tests/")]
    verb = {"A": "Create", "M": "Modify", "D": "Delete"}
    files = "\n".join(f"- {verb[st]}: `{pa}`" for st, pa in impl + tests)
    msg = git(repo, "log", "-1", "--format=%B", sha).rstrip("\n")
    out = [f"### Task {n}: {t['title']}\n", f"**Files:**\n{files}\n",
           f"**Interfaces:**\n{t['interfaces']}\n"]
    if t.get("notes"):
        out.append(t["notes"] + "\n")
    out.append("- [ ] **Step 1: Write the failing tests**\n")
    out += [file_block(repo, sha, s, p) for s, p in tests]
    out.append("- [ ] **Step 2: Run the tests to verify they fail**\n")
    out.append(f"Run: `{t['red_cmd']}`\nExpected: {t['red_expected']}\n")
    out.append("- [ ] **Step 3: Implement**\n")
    out += [file_block(repo, sha, s, p) for s, p in impl]
    out.append("- [ ] **Step 4: Run the tests to verify they pass**\n")
    out.append(f"Run: `{t['green_cmd']}`\nExpected: {t['green_expected']}\n")
    out.append("- [ ] **Step 5: Gate**\n")
    out.append(f"Run: `{gate_cmd}`\nExpected: {t['gate_expected']}\n")
    out.append("- [ ] **Step 6: Commit**\n")
    paths = " ".join(p for _, p in impl + tests)
    out.append(fence(f"git add -A {paths}\ngit commit -F - <<'MSG'\n{msg}\nMSG\n", "bash"))
    return "\n".join(out)


def main() -> None:
    repo, spec_path, out_path = sys.argv[1:4]
    spec = json.load(open(spec_path))
    parts = [spec["header"].rstrip("\n") + "\n"]
    for n, t in enumerate(spec["tasks"], 1):
        parts.append("---\n")
        parts.append(task_md(repo, n, t, spec["gate_cmd"]))
    if spec.get("footer"):
        parts.append("---\n")
        parts.append(spec["footer"])
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
