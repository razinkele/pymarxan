"""Pure-Python rooted phylogenetic tree (no networkx / dendropy dependency)."""
from __future__ import annotations

from collections.abc import Iterable

NodeId = int | str


class PhylogeneticTree:
    """A rooted phylogeny: branch lengths keyed by child node, with cached
    descendant-tip sets. Structural impossibilities (a node with two parents,
    a cycle) raise at construction; softer issues (wrong root count, negative
    lengths, no tips) are reported by :meth:`validate`, mirroring
    ``ConservationProblem.validate``.
    """

    def __init__(
        self,
        length: dict[NodeId, float],
        children: dict[NodeId, list[NodeId]],
        parent: dict[NodeId, NodeId],
    ) -> None:
        self._length = length
        self._children = children
        self._parent = parent
        self._nodes: set[NodeId] = set(parent) | set(children)
        self._roots = [n for n in self._nodes if n not in parent]
        self._tips = [n for n in self._nodes if n not in children]
        self._descendant_tips = self._compute_descendant_tips()

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[NodeId, NodeId, float]]
    ) -> PhylogeneticTree:
        length: dict[NodeId, float] = {}
        children: dict[NodeId, list[NodeId]] = {}
        parent: dict[NodeId, NodeId] = {}
        for p, c, ln in edges:
            if c in parent:
                raise ValueError(
                    f"node {c!r} has more than one parent "
                    f"({parent[c]!r} and {p!r})"
                )
            parent[c] = p
            length[c] = float(ln)
            children.setdefault(p, []).append(c)
        return cls(length=length, children=children, parent=parent)

    @classmethod
    def from_newick(cls, text: str) -> PhylogeneticTree:
        s = text.strip()
        if "'" in s or "[" in s:
            raise ValueError(
                "quoted labels and NHX/comment brackets are not supported"
            )
        s = "".join(s.split())  # collapse internal whitespace (multi-line Newick)
        if not s.endswith(";"):
            raise ValueError("Newick string must end with ';'")
        s = s[:-1]

        pos = 0
        internal_counter = 0
        edges: list[tuple[NodeId, NodeId, float]] = []

        def parse_name() -> str:
            nonlocal pos
            start = pos
            while pos < len(s) and s[pos] not in "(),:;":
                pos += 1
            return s[start:pos].strip()

        def parse_length() -> float:
            nonlocal pos
            if pos < len(s) and s[pos] == ":":
                pos += 1
                start = pos
                while pos < len(s) and s[pos] not in "(),:;":
                    pos += 1
                return float(s[start:pos])
            return 0.0

        def parse_clade() -> tuple[str, float]:
            nonlocal pos, internal_counter
            if pos < len(s) and s[pos] == "(":
                pos += 1  # consume '('
                kids: list[tuple[str, float]] = []
                while True:
                    kids.append(parse_clade())
                    if pos < len(s) and s[pos] == ",":
                        pos += 1
                        continue
                    break
                if pos >= len(s) or s[pos] != ")":
                    raise ValueError(f"expected ')' at position {pos} in Newick")
                pos += 1  # consume ')'
                name = parse_name()
                if not name:
                    internal_counter += 1
                    name = f"__node{internal_counter}"
                length = parse_length()
                for child_name, child_len in kids:
                    edges.append((name, child_name, child_len))
                return name, length
            name = parse_name()
            if not name:
                raise ValueError(f"empty tip name at position {pos} in Newick")
            length = parse_length()
            return name, length

        parse_clade()  # root; its own :length (if any) is ignored
        if pos != len(s):
            raise ValueError(f"trailing characters in Newick at position {pos}")
        return cls.from_edges(edges)

    def _compute_descendant_tips(self) -> dict[NodeId, frozenset[NodeId]]:
        result: dict[NodeId, frozenset[NodeId]] = {}
        state: dict[NodeId, int] = {}  # 1 = in-progress, 2 = done

        def visit(n: NodeId) -> frozenset[NodeId]:
            if state.get(n) == 2:
                return result[n]
            if state.get(n) == 1:
                raise ValueError(f"cycle detected in phylogeny at node {n!r}")
            state[n] = 1
            kids = self._children.get(n)
            if not kids:
                tips = frozenset({n})
            else:
                acc: set[NodeId] = set()
                for k in kids:
                    acc |= visit(k)
                tips = frozenset(acc)
            result[n] = tips
            state[n] = 2
            return tips

        for node in self._nodes:
            visit(node)
        return result

    @property
    def tips(self) -> list[NodeId]:
        return sorted(self._tips, key=str)

    @property
    def n_tips(self) -> int:
        return len(self._tips)

    @property
    def branches(self) -> list[tuple[NodeId, float, frozenset[NodeId]]]:
        return [
            (child, self._length[child], self._descendant_tips[child])
            for child in sorted(self._length, key=str)
        ]

    @property
    def total_pd(self) -> float:
        return float(sum(self._length.values()))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self._tips:
            errors.append("tree has no tips")
        if len(self._roots) == 0:
            errors.append("tree has no root (every node has a parent)")
        elif len(self._roots) > 1:
            roots = sorted(map(str, self._roots))
            errors.append(f"tree has {len(self._roots)} roots, expected 1: {roots}")
        for child, ln in self._length.items():
            if ln < 0:
                errors.append(f"branch to {child!r} has negative length {ln}")
        return errors
