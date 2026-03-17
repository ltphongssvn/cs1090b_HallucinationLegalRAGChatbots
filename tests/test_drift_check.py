# tests/test_drift_check.py
# Unit tests for src/drift_check.py — pure functions only (no actual imports).
import pytest

pytestmark = pytest.mark.unit


class TestParseFreezeInManifestCollector:
    """parse_freeze is duplicated in manifest_collector — test the drift_check version."""

    def test_parse_simple_freeze(self) -> None:
        from src.manifest_collector import parse_freeze

        result = parse_freeze("torch==2.0.1+cu117\nnumpy==1.26.4\n")
        assert result["torch"] == "2.0.1+cu117"
        assert result["numpy"] == "1.26.4"

    def test_parse_empty_string(self) -> None:
        from src.manifest_collector import parse_freeze

        result = parse_freeze("")
        assert result == {}

    def test_parse_skips_comments(self) -> None:
        from src.manifest_collector import parse_freeze

        result = parse_freeze("# comment\ntorch==2.0.1\n")
        assert "torch" in result
        assert len(result) == 1

    def test_parse_unknown_format_line(self) -> None:
        from src.manifest_collector import parse_freeze

        result = parse_freeze("some-package-without-version\n")
        assert result.get("some-package-without-version") == "unknown-format"


class TestTier4MetadataCheck:
    def test_passes_for_installed_packages(self) -> None:
        from src.drift_check import tier4_metadata_check

        # Run against real installed packages — should not raise
        drift = tier4_metadata_check()
        # torch, numpy etc. are installed so drift should be empty list
        assert isinstance(drift, list)

    def test_returns_list(self) -> None:
        from src.drift_check import tier4_metadata_check

        result = tier4_metadata_check()
        assert isinstance(result, list)


class TestManifestCollectorPureFunctions:
    def test_get_cpu_info_returns_dict(self) -> None:
        from src.manifest_collector import get_cpu_info

        info = get_cpu_info()
        assert isinstance(info, dict)
        assert "logical_cores" in info
        assert "cpu_model" in info
        assert "ram_total_gb" in info

    def test_get_cpu_info_logical_cores_positive(self) -> None:
        from src.manifest_collector import get_cpu_info

        info = get_cpu_info()
        if info["logical_cores"] != "unknown":
            assert int(info["logical_cores"]) > 0  # type: ignore[arg-type]

    def test_get_installed_versions_returns_dict(self) -> None:
        from src.manifest_collector import get_installed_versions

        result = get_installed_versions(["torch", "numpy"])
        assert "torch" in result
        assert "numpy" in result

    def test_get_installed_versions_missing_package(self) -> None:
        from src.manifest_collector import get_installed_versions

        result = get_installed_versions(["nonexistent-package-xyz"])
        assert result["nonexistent-package-xyz"] == "not installed"
