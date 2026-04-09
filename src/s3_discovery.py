# src/s3_discovery.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/s3_discovery.py
"""S3 bulk-data discovery for the CourtListener public bucket.

Lists the ``bulk-data/`` prefix of the public CourtListener S3 bucket,
parses the dated filename convention (e.g.
``opinions-2024-10-01.csv.bz2``), and returns the most recent object
for each corpus label the pipeline needs (``courts``, ``dockets``,
``clusters``, ``opinions``).

Design notes
------------
* **Pure XML parsing**: S3 returns a standard XML listing. Parsing is
  isolated into pure functions (:func:`parse_s3_listing`,
  :func:`_is_truncated`, :func:`_get_continuation_token`) so they can
  be unit-tested against fixtures without hitting the network.
* **Retry with backoff**: :func:`_request_with_retry` retries only on
  transient errors (timeouts, 429, 5xx) with exponential backoff; 4xx
  responses other than 429 surface immediately so misconfiguration
  fails loudly.
* **Pagination**: listings over 1000 keys are paginated via
  ``NextContinuationToken``; :func:`list_s3_files` walks every page.
* **No credentials**: the bucket is public; all requests are anonymous.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from src.config import PipelineConfig
from src.exceptions import DiscoveryError

#: Matches CourtListener bulk filenames of the form
#: ``bulk-data/<name>-YYYY-MM-DD.csv`` or ``.csv.bz2``, capturing the
#: corpus name and date components as named groups.
BULK_FILE_PATTERN = re.compile(
    r"^bulk-data/(?P<name>[a-z\-]+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.csv(?:\.bz2)?$"
)

#: XML namespace map for parsing the S3 ``ListBucketResult`` response.
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

#: Maximum retry attempts for transient HTTP errors.
MAX_RETRIES: int = 5

#: Base delay (seconds) for exponential backoff: delay = BASE * 2**attempt.
BACKOFF_BASE: float = 1.0


def _request_with_retry(
    url: str,
    timeout: int = 30,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
) -> requests.Response:
    """GET ``url`` with exponential backoff on transient failures.

    Retries on :class:`Timeout`, :class:`ConnectionError`, and HTTP
    status codes 429, 500, 502, 503, 504. All other HTTP errors raise
    immediately so misconfiguration (e.g. bad bucket name → 403)
    surfaces without pointless retries.

    Args:
        url: Fully-qualified URL to fetch.
        timeout: Per-attempt connect+read timeout in seconds.
        max_retries: Total attempts (not additional retries).
        backoff_base: Base delay in seconds; doubled each attempt.

    Returns:
        The successful :class:`requests.Response`.

    Raises:
        requests.RequestException: The last attempt's exception, after
            all retries are exhausted.
        requests.HTTPError: A non-retryable 4xx status.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2**attempt))
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                last_exc = exc
                time.sleep(backoff_base * (2**attempt))
            else:
                raise
    raise last_exc  # type: ignore[misc]


def parse_s3_listing(xml_text: str) -> List[Dict[str, Any]]:
    """Parse an S3 ``ListBucketResult`` XML body into file metadata dicts.

    Pure function — no I/O, no side effects. Safe to unit-test against
    fixture strings.

    Args:
        xml_text: Raw XML response body from an S3 list-objects call.

    Returns:
        List of ``{"key", "size", "size_mb"}`` dicts, one per
        ``<Contents>`` element. Malformed entries (missing ``Key`` or
        ``Size``) are silently skipped.
    """
    root = ET.fromstring(xml_text)
    files: List[Dict[str, Any]] = []
    for content in root.findall("s3:Contents", S3_NS):
        key_el = content.find("s3:Key", S3_NS)
        size_el = content.find("s3:Size", S3_NS)
        if key_el is None or key_el.text is None or size_el is None or size_el.text is None:
            continue
        key = key_el.text
        size = int(size_el.text)
        files.append({"key": key, "size": size, "size_mb": size / 1e6})
    return files


def _is_truncated(xml_text: str) -> bool:
    """Return ``True`` if the S3 listing indicates more pages follow."""
    root = ET.fromstring(xml_text)
    el = root.find("s3:IsTruncated", S3_NS)
    return el is not None and el.text == "true"


def _get_continuation_token(xml_text: str) -> Optional[str]:
    """Return the ``NextContinuationToken`` from a truncated listing, or ``None``."""
    root = ET.fromstring(xml_text)
    el = root.find("s3:NextContinuationToken", S3_NS)
    return el.text if el is not None else None


def list_s3_files(config: Optional[PipelineConfig] = None) -> List[Dict[str, Any]]:
    """Enumerate every object under ``config.s3_prefix``, walking all pages.

    Issues successive ``list-objects-v2`` requests with continuation
    tokens until ``IsTruncated`` is false or no next token is returned.

    Args:
        config: Pipeline configuration supplying ``s3_bucket_url`` and
            ``s3_prefix``. Defaults to :class:`PipelineConfig`.

    Returns:
        Flat list of every file dict from every page, in listing order.
    """
    if config is None:
        config = PipelineConfig()

    files: List[Dict[str, Any]] = []
    continuation_token: Optional[str] = None

    while True:
        url = f"{config.s3_bucket_url}?prefix={config.s3_prefix}&delimiter=/"
        if continuation_token:
            url += f"&continuation-token={continuation_token}"

        resp = _request_with_retry(url)
        files.extend(parse_s3_listing(resp.text))

        if _is_truncated(resp.text):
            continuation_token = _get_continuation_token(resp.text)
            if not continuation_token:
                break
        else:
            break

    return files


def _parse_bulk_file(key: str) -> Optional[Dict[str, Any]]:
    """Parse a ``bulk-data/<name>-YYYY-MM-DD.csv[.bz2]`` key.

    Returns:
        ``{"name", "date"}`` on success, or ``None`` if the key does
        not match the pattern or the parsed date is invalid
        (e.g. Feb 30).
    """
    match = BULK_FILE_PATTERN.match(key)
    if not match:
        return None
    try:
        parsed_date = date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        return None
    return {"name": match.group("name"), "date": parsed_date}


def find_latest_file(files: List[Dict[str, Any]], name_prefix: str) -> Optional[Dict[str, Any]]:
    """Return the newest file whose parsed name starts with ``name_prefix``.

    Pure function — accepts a pre-fetched file list so tests do not
    need network access. The trailing ``-`` on prefixes like
    ``"courts-"`` is stripped before comparison.

    Args:
        files: File list from :func:`list_s3_files` or a fixture.
        name_prefix: Corpus name prefix (from
            :attr:`PipelineConfig.needed_files`).

    Returns:
        A dict with ``key``, ``size``, ``size_mb``, ``date`` (ISO
        string), and ``name``, or ``None`` if no file matched.
    """
    candidates: List[Dict[str, Any]] = []
    for f in files:
        parsed = _parse_bulk_file(f["key"])
        if parsed is None:
            continue
        if parsed["name"].startswith(name_prefix.rstrip("-")):
            candidates.append({**f, "parsed_date": parsed["date"], "parsed_name": parsed["name"]})

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["parsed_date"], reverse=True)
    best = candidates[0]
    return {
        "key": best["key"],
        "size": best["size"],
        "size_mb": best["size_mb"],
        "date": best["parsed_date"].isoformat(),
        "name": best["parsed_name"],
    }


def discover_latest_bulk_files(config: Optional[PipelineConfig] = None) -> Dict[str, Dict[str, Any]]:
    """List the bucket and return the newest file for every required corpus.

    Consumed by :func:`src.pipeline.run_pipeline` when no pinned
    snapshot is configured. The return shape matches
    :attr:`PipelineConfig.pinned_files` so the two code paths
    converge.

    Args:
        config: Pipeline configuration supplying ``needed_files``.

    Returns:
        Mapping of corpus label → file metadata dict.

    Raises:
        DiscoveryError: One or more required corpus labels had no
            matching file on the bucket.
    """
    if config is None:
        config = PipelineConfig()

    bulk_files = list_s3_files(config=config)
    latest: Dict[str, Dict[str, Any]] = {}
    for label, prefix in config.needed_files.items():
        found = find_latest_file(bulk_files, prefix)
        if found:
            latest[label] = found
    missing = set(config.needed_files) - set(latest)
    if missing:
        raise DiscoveryError(f"Missing bulk files on S3: {missing}")
    return latest
