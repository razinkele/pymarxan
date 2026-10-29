"""Replay a prototype-first plan on a fresh worktree and compare with the prototype commits.

Usage: python replay.py PLAN.md WORKTREE SHA1 SHA2 ...   (one sha per task, in order)

For each task: every "Create `path`:" block is written verbatim; every "Modify `path` (apply
this diff):" block goes through `git apply --whitespace=nowarn`; every "Delete `path`:" removes
the file. Then the worktree must match the task's prototype commit exactly (`git diff SHA` empty),
and the changes are committed so the next task starts from the same tree.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HDR = re.compile(r"^(Create|Modify|Delete) `([^`]+)`(?: \(apply this diff\))?:\n\n", re.M)


def blocks(task_text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in HDR.finditer(task_text):
        rest = task_text[m.end():]
        fence = re.match(r"(`{3,})[^\n]*\n", rest)
        assert fence, m.group(0)
        ticks = fence.group(1)
        end = rest.index("\n" + ticks + "\n", fence.end() - 1)
        body = rest[fence.end():end + 1]
        out.append((m.group(1), m.group(2), body))
    return out


def run(wt: str, *args: str, inp: str | None = None) -> str:
    r = subprocess.run(["git", "-C", wt, *args], input=inp, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def main() -> None:
    plan, wt, *shas = sys.argv[1:]
    text = Path(plan).read_text()
    tasks = re.split(r"^### Task \d+: ", text, flags=re.M)[1:]
    assert len(tasks) == len(shas), (len(tasks), len(shas))
    for n, (task, sha) in enumerate(zip(tasks, shas), 1):
        for kind, path, body in blocks(task):
            if kind == "Create":
                f = Path(wt) / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(body)
            elif kind == "Modify":
                run(wt, "apply", "--whitespace=nowarn", "-", inp=body)
            else:
                run(wt, "rm", "-q", path)
        run(wt, "add", "-A")
        diff = run(wt, "diff", "--cached", "--stat", sha)
        status = "OK" if not diff.strip() else f"MISMATCH\n{diff}"
        print(f"Task {n} vs {sha[:7]}: {status}")
        if diff.strip():
            raise SystemExit(1)
        run(wt, "commit", "-q", "-m", f"replay task {n}")


if __name__ == "__main__":
    main()
