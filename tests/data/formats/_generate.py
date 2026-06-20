"""Deterministically generate the synthetic real-world-format fixtures.

These reproduce the file-format variations of real public Marxan datasets
(double-tab / quoted-CSV delimiters, a ``bound``-vs-``boundary`` column name)
on a small synthetic 5×5-grid project — without bundling any third-party data.
Run from the repo root: ``python tests/data/formats/_generate.py``.

The project: a 5×5 grid of 25 planning units, 4 features. Each feature is
present in ~2/3 of the PUs so a ``prop`` target of 0.2 is comfortably
feasible. Boundaries connect 4-neighbours in the grid. The numbers are fully
deterministic (no RNG) so the committed fixtures never drift.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
GRID = 5
N_PU = GRID * GRID
N_FEAT = 4
PROP = 0.2


def _pus():
    """(id, cost, status, xloc, yloc) per PU. One locked-in, one locked-out."""
    rows = []
    for pid in range(1, N_PU + 1):
        row, col = divmod(pid - 1, GRID)
        cost = 1.0 + (pid % 5)
        status = 2 if pid == 1 else (3 if pid == N_PU else 0)
        rows.append((pid, cost, status, float(col), float(row)))
    return rows


def _puvspr():
    """(species, pu, amount) — feature f present where (pu + f) % 3 != 0."""
    rows = []
    for f in range(1, N_FEAT + 1):
        for pid in range(1, N_PU + 1):
            if (pid + f) % 3 != 0:
                rows.append((f, pid, 1.0 + ((pid * f) % 5)))
    return rows


def _specs():
    """(id, prop, spf, name) per feature.

    SPF is high (like real datasets — MarxanConnect uses 1000) so the
    shortfall penalty dominates cost and the heuristic reliably reaches a
    feasible reserve, matching how practitioners calibrate Marxan.
    """
    return [(f, PROP, 100.0, f"feature_{f}") for f in range(1, N_FEAT + 1)]


def _bounds():
    """(id1, id2, value) — 4-neighbour grid adjacency, id1 < id2."""
    rows = []
    for pid in range(1, N_PU + 1):
        row, col = divmod(pid - 1, GRID)
        if col + 1 < GRID:
            rows.append((pid, pid + 1, 1.0))
        if row + 1 < GRID:
            rows.append((pid, pid + GRID, 1.0))
    return rows


_INPUT_DAT = """\
Input file for the synthetic real-world-format fixture.
BLM 1
PROP 0.5
RANDSEED -1
NUMREPS 10
NUMITNS 1000000
NUMTEMP 10000
INPUTDIR input
SPECNAME spec.dat
PUNAME pu.dat
PUVSPRNAME puvspr.dat
BOUNDNAME bound.dat
SCENNAME output
OUTPUTDIR output
RUNMODE 1
MISSLEVEL 1.0
"""


def _write_double_tab(base: Path) -> None:
    """MarOpt / AdrienBrunel style: columns padded with double tabs, boundary
    column named ``bound``."""
    inp = base / "input"
    inp.mkdir(parents=True, exist_ok=True)
    (base / "input.dat").write_text(_INPUT_DAT)

    def tt(rows, header):
        lines = ["\t\t".join(header)]
        lines += ["\t\t".join(str(v) for v in r) for r in rows]
        return "\n".join(lines) + "\n"

    (inp / "pu.dat").write_text(tt(_pus(), ["id", "cost", "status", "xloc", "yloc"]))
    (inp / "spec.dat").write_text(tt(_specs(), ["id", "prop", "spf", "name"]))
    (inp / "puvspr.dat").write_text(tt(_puvspr(), ["species", "pu", "amount"]))
    (inp / "bound.dat").write_text(tt(_bounds(), ["id1", "id2", "bound"]))


def _write_quoted_csv(base: Path) -> None:
    """MarxanConnect style: quoted puvspr header, comma rows, ``boundary`` col."""
    inp = base / "input"
    inp.mkdir(parents=True, exist_ok=True)
    (base / "input.dat").write_text(_INPUT_DAT)

    def csv(rows, header):
        lines = [",".join(header)]
        lines += [",".join(str(v) for v in r) for r in rows]
        return "\n".join(lines) + "\n"

    (inp / "pu.dat").write_text(csv(_pus(), ["id", "cost", "status", "xloc", "yloc"]))
    (inp / "spec.dat").write_text(csv(_specs(), ["id", "prop", "spf", "name"]))
    # Quoted header (as MarxanConnect emits), unquoted numeric rows.
    puvspr = ['"species","pu","amount"']
    puvspr += [",".join(str(v) for v in r) for r in _puvspr()]
    (inp / "puvspr.dat").write_text("\n".join(puvspr) + "\n")
    (inp / "bound.dat").write_text(csv(_bounds(), ["id1", "id2", "boundary"]))


def main() -> None:
    _write_double_tab(HERE / "double_tab")
    _write_quoted_csv(HERE / "quoted_csv")
    print(f"wrote fixtures under {HERE}/double_tab and {HERE}/quoted_csv")


if __name__ == "__main__":
    main()
