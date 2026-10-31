"""Stage 4 (spec §6.4): every module reading ``Solution.selected`` / ``n_selected`` is
classified, so a new reader cannot slip in without deciding how it treats proportions.

Classification is per module. ``MIXED`` modules list their guarded functions (empty when
the module mixes fraction-aware and count-only readers); the per-function tests in
``tests/pymarxan/analysis/test_fractional_analysis.py`` and
``tests/pymarxan/io/test_fractional_io.py`` pin the split.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PATTERN = re.compile(r"\.selected\b|n_selected")

FRACTION_AWARE = {
    "pymarxan/analysis/equity.py",
    "pymarxan/analysis/representation.py",
    "pymarxan/analysis/robustness.py",
    "pymarxan/analysis/selection_freq.py",
    "pymarxan/io/exporters.py",
    "pymarxan/io/output.py",
    "pymarxan/io/spatial_export.py",
    "pymarxan/solvers/_decisions.py",  # extracts proportions
    "pymarxan/solvers/mip_solver.py",  # produces proportions
    "pymarxan/solvers/utils.py",
    "pymarxan_shiny/modules/mapping/solution_map.py",
    "pymarxan_shiny/modules/results/summary_table.py",
}
GUARDED = {
    "pymarxan/analysis/portfolio_cuts.py",
    "pymarxan/analysis/posthoc_clusters.py",
    "pymarxan/analysis/rank_importance.py",
    "pymarxan/analysis/replacement_cost.py",
    "pymarxan/phylo/diversity.py",
    "pymarxan/solvers/iterative_improvement.py",
    "pymarxan/zones/writers.py",
    "pymarxan_shiny/modules/mapping/comparison_map.py",
}
INTERNAL_OR_COUNT = {
    "pymarxan/calibration/sensitivity.py",
    "pymarxan/calibration/sweep.py",
    "pymarxan/solvers/base.py",
    "pymarxan/solvers/chance_state.py",
    "pymarxan/solvers/clumping.py",
    "pymarxan/solvers/separation.py",
    "pymarxan/solvers/space_state.py",
    "pymarxan_shiny/modules/results/export.py",
    "pymarxan_shiny/modules/run_control/run_panel.py",
    "pymarxan_shiny/modules/spatial_export/spatial_export.py",
}
MIXED: dict[str, tuple[str, ...]] = {
    "pymarxan/analysis/portfolio.py": ("solution_diversity",),
    "pymarxan/analysis/scenarios.py": ("ScenarioSet.overlap_matrix",),
    "pymarxan/io/writers.py": (),  # write_mvbest / write_ssoln aware, write_sum counts
}


def _readers() -> set[str]:
    return {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if PATTERN.search(path.read_text(encoding="utf-8"))
    }


def test_tables_are_disjoint() -> None:
    tables = [FRACTION_AWARE, GUARDED, INTERNAL_OR_COUNT, set(MIXED)]
    seen: set[str] = set()
    for table in tables:
        assert not (seen & table), seen & table
        seen |= table


def test_every_reader_is_classified() -> None:
    classified = FRACTION_AWARE | GUARDED | INTERNAL_OR_COUNT | set(MIXED)
    readers = _readers()
    assert sorted(readers - classified) == [], "classify new Solution.selected readers"
    assert sorted(classified - readers) == [], "stale entries: module no longer reads it"


def _module(rel: str) -> object:
    return importlib.import_module(rel.removesuffix(".py").replace("/", "."))


def test_guarded_modules_guard() -> None:
    for rel in GUARDED:
        text = (SRC / rel).read_text(encoding="utf-8")
        assert "require_binary" in text, rel


def test_mixed_guarded_functions_guard() -> None:
    for rel, names in MIXED.items():
        module = _module(rel)
        for name in names:
            obj: object = module
            for part in name.split("."):
                obj = getattr(obj, part)
            assert "require_binary" in inspect.getsource(obj), (rel, name)  # type: ignore[arg-type]
