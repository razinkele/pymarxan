"""Tests for the pure-Python PhylogeneticTree structure."""
from __future__ import annotations

import pytest

from pymarxan.phylo.tree import PhylogeneticTree

# ((A:1,B:1):2,C:3); with the internal node named "I".
REFERENCE_EDGES = [
    ("root", "I", 2.0),
    ("I", "A", 1.0),
    ("I", "B", 1.0),
    ("root", "C", 3.0),
]


def test_from_edges_tips_and_total_pd():
    tree = PhylogeneticTree.from_edges(REFERENCE_EDGES)
    assert tree.tips == ["A", "B", "C"]  # sorted by str
    assert tree.n_tips == 3
    assert tree.total_pd == pytest.approx(7.0)


def test_branches_carry_descendant_tips():
    tree = PhylogeneticTree.from_edges(REFERENCE_EDGES)
    desc = {child: (length, tips) for child, length, tips in tree.branches}
    assert desc["A"] == (1.0, frozenset({"A"}))
    assert desc["C"] == (3.0, frozenset({"C"}))
    assert desc["I"] == (2.0, frozenset({"A", "B"}))


def test_validate_accepts_well_formed_tree():
    assert PhylogeneticTree.from_edges(REFERENCE_EDGES).validate() == []


def test_validate_flags_two_roots():
    # X and Y are both never children → two roots.
    edges = [("X", "A", 1.0), ("Y", "B", 1.0)]
    errors = PhylogeneticTree.from_edges(edges).validate()
    assert any("root" in e for e in errors)


def test_validate_flags_negative_length():
    edges = [("root", "A", -1.0), ("root", "B", 1.0)]
    errors = PhylogeneticTree.from_edges(edges).validate()
    assert any("negative" in e for e in errors)


def test_validate_flags_empty_tree():
    errors = PhylogeneticTree.from_edges([]).validate()
    assert any("tip" in e for e in errors)


def test_from_edges_raises_on_two_parents():
    edges = [("root", "A", 1.0), ("other", "A", 1.0)]
    with pytest.raises(ValueError, match="more than one parent"):
        PhylogeneticTree.from_edges(edges)


def test_from_edges_raises_on_cycle():
    edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
    with pytest.raises(ValueError, match="cycle"):
        PhylogeneticTree.from_edges(edges)


def test_single_tip_and_zero_length():
    tree = PhylogeneticTree.from_edges([("root", "A", 0.0)])
    assert tree.tips == ["A"]
    assert tree.total_pd == pytest.approx(0.0)
    assert tree.validate() == []


def test_from_newick_matches_reference_structure():
    tree = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")
    assert tree.tips == ["A", "B", "C"]
    assert tree.total_pd == pytest.approx(7.0)
    desc = {child: (length, tips) for child, length, tips in tree.branches}
    # A and B share an internal parent; C hangs off the root.
    assert desc["A"] == (1.0, frozenset({"A"}))
    assert desc["B"] == (1.0, frozenset({"B"}))
    assert desc["C"] == (3.0, frozenset({"C"}))
    # exactly one internal branch of length 2 spanning {A, B}
    internal = [(ln, tips) for ch, ln, tips in tree.branches if tips == frozenset({"A", "B"})]
    assert internal == [(2.0, frozenset({"A", "B"}))]


def test_from_newick_requires_semicolon():
    with pytest.raises(ValueError, match="';'"):
        PhylogeneticTree.from_newick("(A:1,B:1)")


def test_from_newick_rejects_quoted_or_comment():
    with pytest.raises(ValueError, match="not supported"):
        PhylogeneticTree.from_newick("('A':1,B:1);")


def test_from_newick_named_internal_node():
    tree = PhylogeneticTree.from_newick("((A:1,B:1)clade:2,C:3);")
    # the named internal node "clade" spans {A, B}
    assert any(
        ch == "clade" and tips == frozenset({"A", "B"})
        for ch, ln, tips in tree.branches
    )


def test_from_newick_handles_multiline_whitespace():
    # real phylo tools emit multi-line / space-padded Newick.
    tree = PhylogeneticTree.from_newick("(\n  (A:1, B:1):2,\n  C:3\n);")
    assert tree.tips == ["A", "B", "C"]
    assert tree.total_pd == pytest.approx(7.0)
