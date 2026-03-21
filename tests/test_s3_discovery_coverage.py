# tests/test_s3_discovery_coverage.py
# Coverage gaps for s3_discovery.py
import pytest

pytestmark = pytest.mark.unit

from unittest.mock import MagicMock, patch

from src.exceptions import DiscoveryError
from src.s3_discovery import (
    _get_continuation_token,
    _is_truncated,
    discover_latest_bulk_files,
    list_s3_files,
)

LISTING_XML = (
    '<?xml version="1.0"?>'
    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    "<IsTruncated>{truncated}</IsTruncated>"
    "{token}"
    "{contents}"
    "</ListBucketResult>"
)


def _xml(truncated="false", token="", contents=""):
    tok = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return LISTING_XML.format(truncated=truncated, token=tok, contents=contents)


ITEM = "<Contents><Key>bulk-data/opinions-2025-12-31.csv.bz2</Key><Size>100</Size></Contents>"
ITEM2 = "<Contents><Key>bulk-data/courts-2025-12-31.csv.bz2</Key><Size>50</Size></Contents>"
ITEM3 = "<Contents><Key>bulk-data/dockets-2025-12-31.csv.bz2</Key><Size>60</Size></Contents>"
ITEM4 = "<Contents><Key>bulk-data/opinion-clusters-2025-12-31.csv.bz2</Key><Size>70</Size></Contents>"


class TestIsTruncated:
    def test_true(self):
        assert _is_truncated(_xml(truncated="true")) is True

    def test_false(self):
        assert _is_truncated(_xml(truncated="false")) is False


class TestGetContinuationToken:
    def test_present(self):
        assert _get_continuation_token(_xml(token="abc123")) == "abc123"

    def test_absent(self):
        assert _get_continuation_token(_xml()) is None


class TestListS3FilesPagination:
    @patch("src.s3_discovery.time.sleep")
    @patch("src.s3_discovery.requests.get")
    def test_single_page(self, mock_get, mock_sleep):
        mock_get.return_value = MagicMock(text=_xml(contents=ITEM), raise_for_status=lambda: None)
        files = list_s3_files()
        assert len(files) == 1

    @patch("src.s3_discovery.time.sleep")
    @patch("src.s3_discovery.requests.get")
    def test_pagination(self, mock_get, mock_sleep):
        page1 = MagicMock(text=_xml(truncated="true", token="tok2", contents=ITEM), raise_for_status=lambda: None)
        page2 = MagicMock(text=_xml(contents=ITEM2), raise_for_status=lambda: None)
        mock_get.side_effect = [page1, page2]
        files = list_s3_files()
        assert len(files) == 2
        assert mock_get.call_count == 2

    @patch("src.s3_discovery.time.sleep")
    @patch("src.s3_discovery.requests.get")
    def test_pagination_no_token_breaks(self, mock_get, mock_sleep):
        page1 = MagicMock(text=_xml(truncated="true", contents=ITEM), raise_for_status=lambda: None)
        mock_get.return_value = page1
        files = list_s3_files()
        assert len(files) == 1


class TestDiscoverLatestBulkFiles:
    @patch("src.s3_discovery.time.sleep")
    @patch("src.s3_discovery.requests.get")
    def test_finds_all_four(self, mock_get, mock_sleep):
        all_items = ITEM + ITEM2 + ITEM3 + ITEM4
        mock_get.return_value = MagicMock(text=_xml(contents=all_items), raise_for_status=lambda: None)
        result = discover_latest_bulk_files()
        assert set(result.keys()) == {"courts", "dockets", "clusters", "opinions"}

    @patch("src.s3_discovery.time.sleep")
    @patch("src.s3_discovery.requests.get")
    def test_missing_file_raises(self, mock_get, mock_sleep):
        mock_get.return_value = MagicMock(text=_xml(contents=ITEM), raise_for_status=lambda: None)
        with pytest.raises(DiscoveryError, match="Missing"):
            discover_latest_bulk_files()
