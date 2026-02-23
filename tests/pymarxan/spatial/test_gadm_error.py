"""Tests for gadm.py error handling."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGadmErrorHandling:
    @patch("pymarxan.spatial.gadm.requests.get")
    def test_missing_download_url_gives_clear_error(self, mock_get):
        """API response without gjDownloadURL should give descriptive error."""
        from pymarxan.spatial.gadm import fetch_gadm

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"otherKey": "value"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="gjDownloadURL"):
            fetch_gadm("USA", admin_level=0)
