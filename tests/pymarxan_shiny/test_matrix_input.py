"""Tests for connectivity matrix input Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.connectivity.matrix_input import (
    matrix_input_server,
    matrix_input_ui,
    parse_format_label,
)


def test_matrix_input_ui_returns_tag():
    elem = matrix_input_ui("test_conn")
    assert elem is not None


def test_matrix_input_server_callable():
    assert callable(matrix_input_server)


def test_parse_format_label_edge_list():
    assert parse_format_label("edge_list") == "Edge List (id1, id2, value)"


def test_parse_format_label_full_matrix():
    assert parse_format_label("full_matrix") == "Full Matrix (NxN)"


def test_parse_format_label_unknown():
    assert parse_format_label("unknown") == "Unknown"
