"""Build the pymarxan documentation PDF from the repo's own docs plus a generated API reference.

usage (weasyprint is in the shiny env):
    /opt/micromamba/envs/shiny/bin/python scripts/dev/docs_pdf/build.py \
        . docs/pymarxan-documentation.pdf

Sources: README.md, .github/copilot-instructions.md, docs/TUTORIAL.md, docs/USER_MANUAL.md,
docs/VALIDATION.md, the two latest CHANGELOG sections and the package docstrings. Needs
``pandoc`` on PATH.
"""
from __future__ import annotations

import datetime
import importlib
import inspect
import pkgutil
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
WORK = Path(tempfile.mkdtemp(prefix="pymarxan-docs-"))
sys.path.insert(0, str(REPO / "src"))

import pymarxan  # noqa: E402

_PYPROJECT = (REPO / "pyproject.toml").read_text()
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', _PYPROJECT, re.M).group(1)
TODAY = datetime.date.today().isoformat()


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def drop_section(md: str, title: str) -> str:
    """Remove a heading section titled ``title`` (up to the next heading of the same or a
    higher level). Lines inside fenced code blocks are never read as headings."""
    out, skip, level, fenced = [], False, 0, False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            if not skip:
                out.append(line)
            continue
        m = None if fenced else re.match(r"^(#+)\s+(.*)", line)
        if m:
            if skip and len(m.group(1)) <= level:
                skip = False
            if m.group(2).strip().lower() == title.lower():
                skip, level = True, len(m.group(1))
                continue
        if not skip:
            out.append(line)
    return "\n".join(out)


def chapter(md: str, title: str) -> str:
    """Make the file's own H1 the chapter title (replace it with ``title``)."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            lines[i] = f"# {title}"
            break
    else:
        lines.insert(0, f"# {title}")
    return "\n".join(lines)


def fix_images(md: str, base: str) -> str:
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith("http"):
            return "" if "shields.io" in src else m.group(0)
        p = (REPO / base / src).resolve()
        return f"![{alt}]({p.as_uri()})" if p.exists() else ""
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def api_reference() -> str:
    out = ["# API reference",
           "",
           f"Generated from the docstrings of pymarxan {VERSION}. Each entry gives the public "
           "classes and functions a module defines (names not starting with an underscore), "
           "with their signatures and the first paragraph of their documentation. The full "
           "docstrings are available with `help()` or the pdoc site (`make docs`).",
           ""]
    mods = sorted(m.name for m in pkgutil.walk_packages(pymarxan.__path__, "pymarxan."))
    for name in ["pymarxan", *mods]:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # optional dependency missing
            out += [f"## `{name}`", "", f"*Not importable here: {type(exc).__name__}.*", ""]
            continue
        if any(part.startswith("_") for part in name.split(".")[1:]):
            continue
        members = []
        for attr, obj in vars(mod).items():
            if attr.startswith("_"):
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if getattr(obj, "__module__", None) != name:
                continue
            members.append((attr, obj))
        doc = inspect.getdoc(mod) or ""
        if not members and not doc:
            continue
        out += [f"## `{name}`", ""]
        if doc:
            out += [first_para(doc), ""]
        for attr, obj in sorted(members, key=lambda kv: (not inspect.isclass(kv[1]), kv[0])):
            kind = "class" if inspect.isclass(obj) else "function"
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            if len(sig) > 160:
                sig = sig[:157] + "...)"
            out += [f"### {kind} `{attr}`", "", "```python", f"{attr}{sig}", "```", ""]
            d = inspect.getdoc(obj)
            if d:
                out += [first_para(d), ""]
            if inspect.isclass(obj):
                wrapped = (classmethod, staticmethod)
                meths = [(k, v) for k, v in vars(obj).items()
                         if not k.startswith("_")
                         and (inspect.isfunction(v) or isinstance(v, wrapped))]
                for k, v in meths:
                    f = v.__func__ if isinstance(v, wrapped) else v
                    md = inspect.getdoc(f)
                    try:
                        s = str(inspect.signature(f))
                    except (TypeError, ValueError):
                        s = "(...)"
                    if len(s) > 120:
                        s = s[:117] + "...)"
                    line = f"- `{k}{s}`"
                    if md:
                        line += " — " + first_para(md).replace("\n", " ")
                    out.append(line)
                if meths:
                    out.append("")
    return "\n".join(out)


def first_para(doc: str) -> str:
    para = doc.strip().split("\n\n")[0].strip()
    para = re.sub(r":[a-z]+:`~?([^`]+)`", r"`\1`", para)  # Sphinx roles
    para = re.sub(r"``([^`]+)``", r"`\1`", para)  # RST literals
    return re.sub(r"\s*\n\s*", " ", para)


def changelog_latest(n: int = 2) -> str:
    text = read("CHANGELOG.md")
    parts = re.split(r"(?m)^## ", text)
    secs = [p for p in parts[1:] if p.startswith("[0")][:n]
    body = "\n".join("## " + s.replace("### ", "#### ") for s in secs)
    return "# Release notes (latest)\n\n" + body


readme = read("README.md")
readme = re.sub(r"(?s)^# pymarxan\n", "", readme)
readme = drop_section(readme, "Documentation")
readme = fix_images(readme, ".")
arch = drop_section(read(".github/copilot-instructions.md"), "Build, Test, and Lint")
manual = drop_section(read("docs/USER_MANUAL.md"), "Table of Contents")
tutorial = drop_section(read("docs/TUTORIAL.md"), "Table of Contents")
tutorial = re.sub(r"(?m)^(##+ )\d+\.\s+", r"\1", tutorial)

parts = [
    chapter("# x\n" + readme, "Overview"),
    chapter(arch, "Architecture and conventions"),
    chapter(fix_images(tutorial, "docs"), "Tutorial: the Python API"),
    chapter(fix_images(manual, "docs"), "User manual: the web application"),
    chapter(fix_images(read("docs/VALIDATION.md"), "docs"), "Validation"),
    changelog_latest(),
    api_reference(),
]
md = "\n\n".join(parts)
(WORK / "book.md").write_text(md, encoding="utf-8")

subprocess.run([
    "pandoc", str(WORK / "book.md"), "-f", "gfm+tex_math_dollars", "-t", "html5",
    "-s", "--toc", "--toc-depth=2", "--number-sections", "--section-divs",
    "--metadata", "title=pymarxan",
    "-o", str(WORK / "body.html"),
], check=True)
full = (WORK / "body.html").read_text(encoding="utf-8")
toc = re.search(r'(?s)<nav id="TOC"[^>]*>(.*?)</nav>', full).group(1)
body = re.search(r'(?s)<body>(.*)</body>', full).group(1)
body = re.sub(r'(?s)<header id="title-block-header">.*?</header>', '', body)
body = re.sub(r'(?s)<nav id="TOC"[^>]*>.*?</nav>', '', body)
# Wide diagrams get their own landscape page (book.css: .landscape / @page wide).
body = re.sub(
    r'(?s)(<p><img[^>]*pymarxan_workflow\.svg[^>]*></p>|<figure>\s*<img[^>]*pymarxan_workflow\.svg[^>]*>.*?</figure>)',
    r'<div class="landscape">\1</div>',
    body,
)

css = (Path(__file__).parent / "book.css").read_text()
title = f"""
<section class="titlepage">
  <div class="tp-kicker">Systematic conservation planning in Python</div>
  <h1 class="tp-title">pymarxan</h1>
  <div class="tp-sub">Documentation &mdash; version {VERSION}</div>
  <div class="tp-desc">A Python toolkit and Shiny web application for Marxan-style reserve
  design: simulated annealing, exact MIP, heuristics, zones, connectivity, probability
  (chance) constraints, Zonation-style prioritisation and more, with Marxan file
  compatibility.</div>
  <div class="tp-meta">Generated {TODAY} from the repository's documentation and source
  docstrings &middot; MIT licence</div>
</section>
"""
html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>pymarxan {VERSION} documentation</title><style>{css}</style></head><body>{title}
<section class="toc-page"><h1 class="toc-title">Contents</h1><nav class="toc">{toc}</nav></section>
{body}</body></html>"""
(WORK / "book.html").write_text(html, encoding="utf-8")

from weasyprint import HTML  # noqa: E402

HTML(string=html, base_url=str(REPO)).write_pdf(str(OUT))
print("wrote", OUT)
