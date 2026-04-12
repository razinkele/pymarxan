"""Tests for data_input/directory_browser Shiny module."""
import tempfile
from pathlib import Path

from pymarxan_shiny.modules.data_input.directory_browser import (
    _list_directory,
    _looks_like_marxan_project,
    directory_browser_server,
    directory_browser_ui,
)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestListDirectory:
    def test_returns_sorted_subdirs(self, tmp_path):
        (tmp_path / "beta").mkdir()
        (tmp_path / "alpha").mkdir()
        (tmp_path / "gamma").mkdir()
        result = _list_directory(tmp_path)
        names = [e["name"] for e in result]
        assert names == ["alpha", "beta", "gamma"]

    def test_excludes_hidden_dirs(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        result = _list_directory(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "visible"

    def test_excludes_files(self, tmp_path):
        (tmp_path / "somefile.txt").write_text("hi")
        (tmp_path / "subdir").mkdir()
        result = _list_directory(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "subdir"

    def test_returns_empty_for_nonexistent(self, tmp_path):
        result = _list_directory(tmp_path / "nope")
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = _list_directory(tmp_path)
        assert result == []

    def test_entry_has_correct_path(self, tmp_path):
        (tmp_path / "child").mkdir()
        result = _list_directory(tmp_path)
        assert result[0]["path"] == str(tmp_path / "child")


class TestLooksLikeMarxanProject:
    def test_true_when_input_dat_exists(self, tmp_path):
        (tmp_path / "input.dat").write_text("dummy")
        assert _looks_like_marxan_project(tmp_path) is True

    def test_false_when_no_input_dat(self, tmp_path):
        assert _looks_like_marxan_project(tmp_path) is False


# ---------------------------------------------------------------------------
# Module callable tests
# ---------------------------------------------------------------------------

def test_directory_browser_ui_callable():
    assert callable(directory_browser_ui)


def test_directory_browser_server_callable():
    assert callable(directory_browser_server)
