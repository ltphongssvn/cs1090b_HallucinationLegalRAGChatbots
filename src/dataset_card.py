# src/dataset_card.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/dataset_card.py
"""Hugging Face dataset card generator from pipeline manifest.

Produces a HF Hub-compatible ``README.md`` (YAML frontmatter + Markdown
body) describing a CourtListener shard directory. Consumed by the
notebook after :func:`src.manifest.write_manifest` returns.

Sections produced:

* YAML frontmatter — license, language, task_categories, size_categories,
  tags, source_datasets.
* Dataset Summary — row count, shard count, shard size.
* Provenance — git revision, timestamp, source files.
* Data Distribution — court breakdown, text length percentiles.

Design notes
------------
* **Pure**: every function is a string transformer. No I/O except
  :func:`write_dataset_card`, which writes to a single file.
* **Tolerant**: missing optional fields fall back to "(not recorded)"
  rather than raising — manifests from older pipeline versions still
  produce a usable card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


class DatasetCardError(ValueError):
    """Raised when manifest input is unusable for card generation."""


def _size_category(num_rows: int) -> str:
    """Map row count to HF Hub ``size_categories`` bucket."""
    if num_rows < 1_000:
        return "n<1K"
    if num_rows < 10_000:
        return "1K<n<10K"
    if num_rows < 100_000:
        return "10K<n<100K"
    if num_rows < 1_000_000:
        return "100K<n<1M"
    if num_rows < 10_000_000:
        return "1M<n<10M"
    if num_rows < 100_000_000:
        return "10M<n<100M"
    return "100M<n<1B"


def build_card_yaml_frontmatter(manifest: Dict[str, Any]) -> str:
    """Build the HF Hub YAML frontmatter block from a manifest dict.

    Args:
        manifest: Pipeline manifest as returned by
            :func:`src.manifest.write_manifest`.

    Returns:
        YAML block delimited by ``---\\n`` lines, ready to prepend to
        a Markdown body.
    """
    num_rows = int(manifest.get("num_cases", 0))
    size_cat = _size_category(num_rows)
    return (
        "---\n"
        "license: cc-by-nd-4.0\n"
        "language:\n"
        "- en\n"
        "task_categories:\n"
        "- text-retrieval\n"
        "- question-answering\n"
        "size_categories:\n"
        f"- {size_cat}\n"
        "source_datasets:\n"
        "- original\n"
        "tags:\n"
        "- legal\n"
        "- federal-appellate\n"
        "- courtlistener\n"
        "- legal-rag\n"
        "---\n"
    )


def build_card_markdown(manifest: Dict[str, Any]) -> str:
    """Build the full dataset card (frontmatter + Markdown body).

    Args:
        manifest: Pipeline manifest dict.

    Returns:
        Complete README.md content as a single string.
    """
    fm = build_card_yaml_frontmatter(manifest)
    parts = [fm, "\n# CourtListener Federal Appellate Subset\n\n"]

    # Dataset Summary
    num_cases = int(manifest.get("num_cases", 0))
    num_shards = int(manifest.get("num_shards", 0))
    shard_size = int(manifest.get("shard_size", 0))
    parts.append("## Dataset Summary\n\n")
    parts.append(
        f"This dataset contains **{num_cases:,}** opinions from U.S. federal "
        f"appellate courts, extracted from the public CourtListener bulk data.\n\n"
    )
    parts.append(f"- **Total opinions**: {num_cases:,}\n")
    parts.append(f"- **Number of shards**: {num_shards}\n")
    parts.append(f"- **Shard size**: {shard_size:,} rows per shard\n")
    fed_courts = manifest.get("federal_courts", [])
    if fed_courts:
        parts.append(f"- **Courts covered**: {len(fed_courts)} federal circuits\n")
    parts.append("\n")

    # Provenance
    parts.append("## Provenance\n\n")
    rm = manifest.get("run_metadata", {})
    git_rev = rm.get("git_revision", "(not recorded)")
    ts = rm.get("timestamp", "(not recorded)")
    py_ver = rm.get("python_version", "(not recorded)")
    parts.append(f"- **Build timestamp (UTC)**: {ts}\n")
    parts.append(f"- **Git revision**: `{git_rev}`\n")
    parts.append(f"- **Python version**: {py_ver}\n")
    sf = manifest.get("source_files", {})
    if sf:
        parts.append("- **Source files**:\n")
        for label, name in sf.items():
            parts.append(f"  - `{label}`: `{name}`\n")
    parts.append("\n")

    # Filter chain
    fc = manifest.get("filter_chain", {})
    if fc:
        parts.append("## Filter Chain\n\n")
        parts.append(
            f"- Courts: {fc.get('courts', 0)} → "
            f"Dockets: {fc.get('dockets', 0):,} → "
            f"Clusters: {fc.get('clusters', 0):,}\n\n"
        )

    # Court Distribution
    cd = manifest.get("court_distribution", {})
    if cd:
        parts.append("## Court Distribution\n\n")
        parts.append("| Court | Opinions |\n|---|---|\n")
        for court_id in sorted(cd, key=lambda k: -cd[k]):
            parts.append(f"| `{court_id}` | {cd[court_id]:,} |\n")
        parts.append("\n")

    # Text Length Stats
    tls = manifest.get("text_length_stats", {})
    if tls:
        parts.append("## Text Length Distribution (characters)\n\n")
        parts.append("| Statistic | Value |\n|---|---|\n")
        for key in ("mean", "median", "min", "p25", "p75", "p90", "p95", "p99", "max"):
            if key in tls:
                parts.append(f"| {key} | {int(tls[key]):,} |\n")
        parts.append("\n")

    # License
    parts.append("## License\n\n")
    parts.append(
        "Data is derived from CourtListener and is made available under the "
        "[Creative Commons Attribution-NoDerivs 4.0 International License]"
        "(https://creativecommons.org/licenses/by-nd/4.0/) (CC BY-ND 4.0).\n"
    )

    return "".join(parts)


def write_dataset_card(manifest: Dict[str, Any], target_dir: PathLike) -> Path:
    """Write the dataset card to ``<target_dir>/README.md``.

    Creates ``target_dir`` if missing. Overwrites any existing
    ``README.md`` in the directory.

    Args:
        manifest: Pipeline manifest dict.
        target_dir: Directory in which to write ``README.md``.

    Returns:
        Path to the written README.md.

    Raises:
        DatasetCardError: ``manifest`` is empty (``not manifest``).
    """
    if not manifest:
        raise DatasetCardError("empty manifest — cannot generate dataset card")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "README.md"
    out.write_text(build_card_markdown(manifest))
    return out
