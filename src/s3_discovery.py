# src/s3_discovery.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/s3_discovery.py
# SRP: Discover CourtListener bulk data files on S3.

import re  # used: filename pattern matching
import time  # used: retry backoff sleep
import xml.etree.ElementTree as ET  # used: parse S3 XML listing
from datetime import date  # used: parse dates from filenames
from typing import Any, Dict, List, Optional

import requests  # used: HTTP GET to S3

from src.config import PipelineConfig
from src.exceptions import DiscoveryError

BULK_FILE_PATTERN = re.compile(
    r"^bulk-data/(?P<name>[a-z\-]+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.csv(?:\.bz2)?$"
)

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

MAX_RETRIES: int = 5
BACKOFF_BASE: float = 1.0


def _request_with_retry(
    url: str,
    timeout: int = 30,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
) -> requests.Response:
    """GET with exponential backoff on transient errors."""
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
    """Parse S3 XML listing into file dicts. Pure function — no I/O."""
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
    """Check if S3 listing is truncated. Pure function."""
    root = ET.fromstring(xml_text)
    el = root.find("s3:IsTruncated", S3_NS)
    return el is not None and el.text == "true"


def _get_continuation_token(xml_text: str) -> Optional[str]:
    """Extract continuation token from truncated S3 listing. Pure function."""
    root = ET.fromstring(xml_text)
    el = root.find("s3:NextContinuationToken", S3_NS)
    return el.text if el is not None else None


def list_s3_files(config: Optional[PipelineConfig] = None) -> List[Dict[str, Any]]:
    """List ALL files in S3 bucket, handling pagination with retry."""
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
    """Parse a bulk-data key into (name, date). Returns None if no match or invalid date."""
    match = BULK_FILE_PATTERN.match(key)
    if not match:
        return None
    try:
        parsed_date = date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        return None
    return {"name": match.group("name"), "date": parsed_date}


def find_latest_file(files: List[Dict[str, Any]], name_prefix: str) -> Optional[Dict[str, Any]]:
    """Find most recent file whose parsed name starts with prefix. Pure function."""
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
    """Discover latest version of all required bulk CSV files."""
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
