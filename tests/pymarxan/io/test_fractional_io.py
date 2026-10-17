"""Stage 4 (spec §6.1, §6.2): writers / readers with fractional solutions."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pymarxan.io.exporters import export_solution_csv, export_summary_csv
from pymarxan.io.output import OutputController
from pymarxan.io.readers import load_project, read_mvbest, read_ssoln
from pymarxan.io.spatial_export import export_frequency_spatial, export_solution_spatial
from pymarxan.io.writers import write_mvbest, write_ssoln
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import FractionalSolutionError
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import build_solution, check_targets
from pymarxan.zones.writers import write_zone_solution

DATA = Path(__file__).parent.parent.parent / "data"
X = np.array([0.2744680851063830, 1.0, 0.0, 0.11702127659574468, 0.10638297872340426, 1.0])


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X, 1.0)


@pytest.fixture
def binary(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X >= 0.5, 1.0)


def test_write_run_best_matrix(
    tmp_path: Path, simple: ConservationProblem, frac: Solution, binary: Solution,
) -> None:
    out = OutputController(params={})
    out.write_run(simple, frac, 1, tmp_path)
    out.write_best(simple, binary, tmp_path)
    out.write_solution_matrix(simple, [binary, frac], tmp_path)
    run = (tmp_path / "output_r001.csv").read_text().splitlines()
    assert run[1:3] == ["1,0.2744680851", "2,1"]
    assert (tmp_path / "output_best.csv").read_text().splitlines()[1:3] == ["1,0", "2,1"]
    matrix = (tmp_path / "output_solutionsmatrix.csv").read_text().splitlines()
    assert matrix[0] == "Planning_Unit,S1,S2"
    assert matrix[1] == "1,0,0.2744680851"
    assert matrix[4] == "4,0,0.1170212766"


def test_write_mvbest_fractional(
    tmp_path: Path, simple: ConservationProblem, frac: Solution,
) -> None:
    path = tmp_path / "mvbest.csv"
    write_mvbest(simple, frac, path)
    df = read_mvbest(path)
    pu_index = {int(p): i for i, p in enumerate(simple.planning_units["id"])}
    met = check_targets(simple, frac.selected, pu_index)
    assert [bool(v) for v in df["Target_Met"]] == [met[int(f)] for f in df["Feature_ID"]]
    assert all(df["Target_Met"])  # held ≈ R at the LP optimum: met within the tolerance
    held1 = 10 * X[0] + 15 + 12 * X[3] + 8 * X[4] + 10
    assert df.loc[df["Feature_ID"] == 1, "Amount_Held"].iloc[0] == pytest.approx(held1)
    assert (df["Shortfall"] == 0.0).all()


def test_write_mvbest_follows_misslevel_change(
    tmp_path: Path, simple: ConservationProblem, frac: Solution,
) -> None:
    """Target_Met is recomputed against the problem at write time, as for binary."""
    y = frac.selected * 0.5
    half = build_solution(simple, y, 1.0)
    write_mvbest(simple, half, tmp_path / "a.csv")
    assert not read_mvbest(tmp_path / "a.csv")["Target_Met"].any()
    simple.parameters["MISSLEVEL"] = 0.5
    write_mvbest(simple, half, tmp_path / "b.csv")
    assert read_mvbest(tmp_path / "b.csv")["Target_Met"].all()


def test_ssoln_roundtrip(
    tmp_path: Path, simple: ConservationProblem, frac: Solution, binary: Solution,
) -> None:
    path = tmp_path / "ssoln.csv"
    write_ssoln(simple, [frac, binary], path)
    df = read_ssoln(path)
    assert df["Number"].dtype == np.float64
    assert df["Number"].tolist() == pytest.approx((X + (X >= 0.5)).tolist())
    write_ssoln(simple, [binary, binary], path)
    df = read_ssoln(path)
    assert df["Number"].dtype.kind == "i"
    assert df["Number"].tolist() == [0, 2, 0, 0, 0, 2]
    assert path.read_text().splitlines()[2] == "2,2"


def test_read_ssoln_integral_floats_stay_int(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("Planning_Unit,Number\n1,1.0\n2,0.0\n")
    assert read_ssoln(path)["Number"].tolist() == [1, 0]
    assert read_ssoln(path)["Number"].dtype.kind == "i"
    path.write_text("Planning_Unit,Number\n1,1.5\n2,0.0\n")
    assert read_ssoln(path)["Number"].tolist() == [1.5, 0.0]


@pytest.mark.parametrize("body", ["1,1.5\n2,\n", "1,\n2,0\n", "1,abc\n2,1\n"])
def test_read_ssoln_malformed_still_raises(tmp_path: Path, body: str) -> None:
    path = tmp_path / "s.csv"
    path.write_text("Planning_Unit,Number\n" + body)
    with pytest.raises((ValueError, TypeError)):
        read_ssoln(path)


def test_export_solution_csv(tmp_path: Path, simple: ConservationProblem, frac: Solution,
                             binary: Solution) -> None:
    export_solution_csv(simple, frac, tmp_path / "f.csv")
    export_solution_csv(simple, binary, tmp_path / "b.csv")
    assert (tmp_path / "f.csv").read_text().splitlines()[1] == "1,10.0,0.2744680851"
    assert (tmp_path / "b.csv").read_text().splitlines()[1] == "1,10.0,0"


def test_export_summary_csv(tmp_path: Path, simple: ConservationProblem, frac: Solution) -> None:
    export_summary_csv(simple, frac, tmp_path / "s.csv")
    df = pd.read_csv(tmp_path / "s.csv")
    assert df["met"].all()
    assert df.loc[0, "achieved"] == pytest.approx(10 * X[0] + 15 + 12 * X[3] + 8 * X[4] + 10)


def _gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4, 5, 6]},
        geometry=[box(i, 0, i + 1, 1) for i in range(6)], crs="EPSG:3857",
    )


def test_spatial_exports(tmp_path: Path, frac: Solution, binary: Solution) -> None:
    export_solution_spatial(_gdf(), frac, tmp_path / "f.gpkg")
    got = gpd.read_file(tmp_path / "f.gpkg")
    assert got["selected"].tolist() == pytest.approx(X.tolist(), abs=1e-10)
    export_solution_spatial(_gdf(), binary, tmp_path / "b.gpkg")
    assert gpd.read_file(tmp_path / "b.gpkg")["selected"].tolist() == [0, 1, 0, 0, 0, 1]
    export_frequency_spatial(_gdf(), [frac, binary], tmp_path / "q.gpkg")
    q = gpd.read_file(tmp_path / "q.gpkg")
    assert q["count"].tolist() == pytest.approx((X + (X >= 0.5)).tolist())
    assert q["frequency"].tolist() == pytest.approx(((X + (X >= 0.5)) / 2).tolist())


def test_zone_writer_fallback_guarded(tmp_path: Path, frac: Solution) -> None:
    with pytest.raises(FractionalSolutionError, match="write_zone_solution"):
        write_zone_solution(frac, tmp_path / "z.csv")


def test_zone_summary_fallback_guarded(tmp_path: Path) -> None:
    import warnings

    from pymarxan.zones import load_zone_project
    from pymarxan.zones.writers import write_zone_summary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zp = load_zone_project(DATA / "zones")
    sel = np.full(zp.n_planning_units, 0.5)
    sol = Solution(selected=sel, cost=0.0, boundary=0.0, objective=0.0, targets_met={})
    with pytest.raises(FractionalSolutionError, match="write_zone_summary"):
        write_zone_summary(zp, [sol], tmp_path / "zs.csv")


def test_run_best_values_round_trip(
    tmp_path: Path, simple: ConservationProblem, binary: Solution,
) -> None:
    """Review focus: .10g loses at most 1e-9 relative; binary files keep ints."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 6)
    x[2] = 0.0
    x[4] = 1.0 / 3.0
    sol = build_solution(simple, x, 1.0)
    out = OutputController(params={})
    out.write_run(simple, sol, 1, tmp_path)
    out.write_best(simple, sol, tmp_path)
    for name in ("output_r001.csv", "output_best.csv"):
        got = pd.read_csv(tmp_path / name)["Solution"].to_numpy()
        assert got == pytest.approx(x, rel=1e-9, abs=0.0)
    out.write_run(simple, binary, 2, tmp_path)
    col = pd.read_csv(tmp_path / "output_r002.csv")["Solution"]
    assert col.dtype.kind == "i"
    assert col.tolist() == [int(v) for v in binary.selected]


def test_compute_held_public(simple: ConservationProblem, frac: Solution) -> None:
    """Review R7: the public held-amount helper the readers share."""
    from pymarxan.solvers import compute_held

    held = compute_held(simple, frac.selected)
    assert held[1] == pytest.approx(10 * X[0] + 15 + 12 * X[3] + 8 * X[4] + 10)
    pu_index = {int(q): i for i, q in enumerate(simple.planning_units["id"])}
    assert compute_held(simple, frac.selected, pu_index) == held
