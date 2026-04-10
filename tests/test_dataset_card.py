# tests/test_dataset_card.py
# TDD RED: Hugging Face dataset card generator from manifest.
import pytest

pytestmark = pytest.mark.unit

from src.dataset_card import (
    DatasetCardError,
    build_card_yaml_frontmatter,
    build_card_markdown,
    write_dataset_card,
)


@pytest.fixture
def sample_manifest():
    return {
        "version": 2,
        "num_cases": 1465484,
        "num_shards": 159,
        "shard_size": 10000,
        "federal_courts": ["ca1", "ca2", "ca9", "cadc"],
        "filter_chain": {"courts": 13, "dockets": 1557637, "clusters": 1465484},
        "court_distribution": {"ca9": 300000, "ca5": 200000, "ca1": 100000},
        "text_length_stats": {"mean": 5000, "median": 3500, "p95": 18000, "max": 250000},
        "source_files": {
            "courts": "courts-2026-03-31.csv.bz2",
            "dockets": "dockets-2026-03-31.csv.bz2",
            "clusters": "opinion-clusters-2026-03-31.csv.bz2",
            "opinions": "opinions-2026-03-31.csv.bz2",
        },
        "run_metadata": {
            "timestamp": "2026-04-09T12:00:00+00:00",
            "git_revision": "abc123def456",
            "python_version": "3.11.9",
        },
    }


class TestBuildCardYamlFrontmatter:
    def test_includes_required_fields(self, sample_manifest):
        """Frontmatter contains license, language, task_categories."""
        fm = build_card_yaml_frontmatter(sample_manifest)
        assert "license:" in fm
        assert "language:" in fm
        assert "task_categories:" in fm

    def test_starts_and_ends_with_yaml_delimiters(self, sample_manifest):
        """Wraps content in --- delimiters."""
        fm = build_card_yaml_frontmatter(sample_manifest)
        assert fm.startswith("---\n")
        assert fm.endswith("---\n")

    def test_includes_size_category(self, sample_manifest):
        """1.46M rows maps to size_categories: 1M<n<10M."""
        fm = build_card_yaml_frontmatter(sample_manifest)
        assert "1M<n<10M" in fm


class TestBuildCardMarkdown:
    def test_includes_yaml_frontmatter(self, sample_manifest):
        """Output begins with YAML frontmatter."""
        md = build_card_markdown(sample_manifest)
        assert md.startswith("---\n")

    def test_includes_dataset_summary_section(self, sample_manifest):
        """Has '## Dataset Summary' heading."""
        md = build_card_markdown(sample_manifest)
        assert "## Dataset Summary" in md

    def test_includes_row_count(self, sample_manifest):
        """Reports total row count."""
        md = build_card_markdown(sample_manifest)
        assert "1,465,484" in md

    def test_includes_shard_count(self, sample_manifest):
        """Reports number of shards."""
        md = build_card_markdown(sample_manifest)
        assert "159" in md

    def test_includes_provenance_section(self, sample_manifest):
        """Has '## Provenance' heading with git revision."""
        md = build_card_markdown(sample_manifest)
        assert "## Provenance" in md
        assert "abc123def456" in md

    def test_includes_court_distribution(self, sample_manifest):
        """Lists court distribution."""
        md = build_card_markdown(sample_manifest)
        assert "ca9" in md and "300,000" in md

    def test_includes_text_length_stats(self, sample_manifest):
        """Reports text length percentiles."""
        md = build_card_markdown(sample_manifest)
        assert "5,000" in md  # mean
        assert "p95" in md.lower() or "P95" in md

    def test_handles_missing_optional_fields(self):
        """Tolerates manifests without court_distribution or text_length_stats."""
        minimal = {
            "num_cases": 100,
            "num_shards": 1,
            "shard_size": 100,
            "federal_courts": ["ca1"],
            "source_files": {"opinions": "x.csv"},
            "run_metadata": {"git_revision": "abc", "timestamp": "2026-01-01"},
        }
        md = build_card_markdown(minimal)
        assert "## Dataset Summary" in md


class TestWriteDatasetCard:
    def test_writes_readme_md_to_target_dir(self, sample_manifest, tmp_path):
        """Writes README.md to the target directory."""
        out = write_dataset_card(sample_manifest, tmp_path)
        assert out == tmp_path / "README.md"
        assert out.is_file()
        assert "## Dataset Summary" in out.read_text()

    def test_creates_target_dir_if_missing(self, sample_manifest, tmp_path):
        """Creates parent directory if it doesn't exist."""
        target = tmp_path / "nested" / "deep"
        out = write_dataset_card(sample_manifest, target)
        assert out.is_file()

    def test_overwrites_existing_card(self, sample_manifest, tmp_path):
        """Overwrites README.md if it already exists."""
        existing = tmp_path / "README.md"
        existing.write_text("OLD CONTENT")
        out = write_dataset_card(sample_manifest, tmp_path)
        assert "OLD CONTENT" not in out.read_text()
        assert "## Dataset Summary" in out.read_text()

    def test_raises_on_empty_manifest(self, tmp_path):
        """Raises DatasetCardError on empty manifest."""
        with pytest.raises(DatasetCardError, match="empty manifest"):
            write_dataset_card({}, tmp_path)
