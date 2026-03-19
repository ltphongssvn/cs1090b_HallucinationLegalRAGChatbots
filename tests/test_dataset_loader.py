# tests/test_dataset_loader.py
# Unit tests for DatasetConfig, RowValidator, RowNormalizer, DatasetLoader.
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_config import DatasetConfig
from src.dataset_loader import HEX_REVISION_RE, DatasetLoader
from src.row_normalizer import RowNormalizer
from src.row_validator import RowValidator

pytestmark = pytest.mark.unit

PINNED_REVISION = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret


@pytest.fixture
def config() -> DatasetConfig:
    return DatasetConfig()


@pytest.fixture
def validator(config: DatasetConfig) -> RowValidator:
    return RowValidator(config)


@pytest.fixture
def normalizer(config: DatasetConfig, validator: RowValidator) -> RowNormalizer:
    return RowNormalizer(config, validator)


@pytest.fixture
def loader(config: DatasetConfig) -> DatasetLoader:
    return DatasetLoader(config)


@pytest.fixture
def valid_row() -> dict:
    return {
        "text": "The court held that the defendant failed to establish a genuine issue. " * 2,
        "created_timestamp": "2022-01-15",
        "downloaded_timestamp": "2022-06-01",
        "url": "https://courtlistener.com/opinion/123/",
    }


def _mock_ds(rows: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock


class TestDatasetConfig:
    def test_default_revision_is_40char_sha(self, config: DatasetConfig) -> None:
        assert HEX_REVISION_RE.fullmatch(config.revision) is not None

    def test_default_reproducible_is_true(self, config: DatasetConfig) -> None:
        assert config.reproducible is True

    def test_config_is_injectable(self) -> None:
        custom = DatasetConfig(subset="atticus_contracts", min_text_length=100)
        assert custom.subset == "atticus_contracts"
        assert custom.min_text_length == 100


class TestRowValidator:
    def test_valid_row_returns_no_errors(self, validator: RowValidator, valid_row: dict) -> None:
        assert validator.validate(valid_row) == []

    def test_missing_url_caught(self, validator: RowValidator) -> None:
        row = {"text": "A" * 60, "created_timestamp": "", "downloaded_timestamp": ""}
        assert any("Missing required fields" in e for e in validator.validate(row))

    def test_resolve_text_field_prefers_text(self, validator: RowValidator) -> None:
        assert validator.resolve_text_field({"text": "x", "contents": "y"}) == "text"

    def test_resolve_text_field_falls_back_to_contents(self, validator: RowValidator) -> None:
        assert validator.resolve_text_field({"contents": "y"}) == "contents"


class TestRowNormalizer:
    def test_normalize_valid_row(self, normalizer: RowNormalizer, valid_row: dict) -> None:
        result = normalizer.normalize(valid_row)
        assert "text" in result and "source_url" in result

    def test_normalize_raises_on_invalid(self, normalizer: RowNormalizer) -> None:
        with pytest.raises(ValueError, match="normalize\\(\\) called on invalid row"):
            normalizer.normalize({"url": "x"})

    def test_source_text_field_recorded(self, normalizer: RowNormalizer, valid_row: dict) -> None:
        assert normalizer.normalize(valid_row)["_source_text_field"] == "text"

    def test_contents_renamed_to_text(self, normalizer: RowNormalizer) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = normalizer.normalize(row)
        assert "text" in result and "contents" not in result


class TestDatasetLoader:
    @patch("datasets.load_dataset")
    def test_load_passes_revision(self, mock_load, loader: DatasetLoader, valid_row: dict) -> None:
        mock_load.return_value = _mock_ds([valid_row])
        list(loader.load())
        assert mock_load.call_args.kwargs["revision"] == loader._config.revision

    @patch("datasets.load_dataset")
    def test_load_no_trust_remote_code(self, mock_load, loader: DatasetLoader, valid_row: dict) -> None:
        mock_load.return_value = _mock_ds([valid_row])
        list(loader.load())
        assert "trust_remote_code" not in mock_load.call_args.kwargs

    def test_load_raises_on_mutable_revision(self, config: DatasetConfig) -> None:
        config.revision = "main"  # type: ignore[misc]
        loader = DatasetLoader(config)
        with pytest.raises(RuntimeError, match="Reproducibility violation"):
            list(loader.load())

    @patch("datasets.load_dataset")
    def test_iter_valid_rows_filters_invalid(self, mock_load, loader: DatasetLoader, valid_row: dict) -> None:
        rows = list(loader.iter_valid_rows([valid_row, {"url": "x"}, valid_row]))
        assert len(rows) == 2

    def test_get_provenance_has_required_keys(self, loader: DatasetLoader) -> None:
        prov = loader.get_provenance()
        assert {"dataset", "subset", "split", "revision", "reproducible"}.issubset(prov.keys())


class TestLightningDataModuleImport:
    def test_import_without_lightning_does_not_crash(self) -> None:
        from src.lightning_datamodule import CourtListenerDataModule, CourtListenerIterableDataset

        assert CourtListenerDataModule is not None
        assert CourtListenerIterableDataset is not None


class TestLogStats:
    def test_log_stats_returns_required_keys(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 42)
        mock_tokenizer.name_or_path = "bert-base-uncased"

        stats = loader.log_stats([valid_row, valid_row], mock_tokenizer, max_samples=10)
        required = {
            "n_valid",
            "n_skipped",
            "avg_token_length",
            "min_token_length",
            "max_token_length",
            "avg_text_length_chars",
            "court_distribution",
            "token_length_histogram",
            "tokenizer_name",
        }
        assert required <= set(stats.keys())

    def test_log_stats_counts_valid_and_skipped(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        stats = loader.log_stats([valid_row, {"url": "x"}, valid_row], mock_tokenizer)
        assert stats["n_valid"] == 2
        assert stats["n_skipped"] == 1

    def test_log_stats_respects_max_samples(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        stats = loader.log_stats([valid_row] * 100, mock_tokenizer, max_samples=5)
        assert stats["n_valid"] <= 5

    def test_log_stats_includes_provenance(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        stats = loader.log_stats([valid_row], mock_tokenizer)
        assert "revision" in stats
        assert "dataset" in stats

    def test_log_stats_court_distribution(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        row_with_court = {**valid_row, "court_id": "ca9"}
        stats = loader.log_stats([row_with_court, row_with_court], mock_tokenizer)
        assert stats["court_distribution"].get("ca9") == 2

    def test_log_stats_empty_source(self, loader: DatasetLoader) -> None:
        mock_tokenizer = MagicMock()
        stats = loader.log_stats([], mock_tokenizer)
        assert stats["n_valid"] == 0
        assert stats["avg_token_length"] == 0
        assert stats["token_length_histogram"] == []


class TestDatasetConfigHydra:
    def test_from_hydra_with_plain_dict(self) -> None:
        d = {
            "dataset_id": "pile-of-law/pile-of-law",
            "subset": "atticus_contracts",
            "split": "train",
            "revision": "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b",  # pragma: allowlist secret
            "reproducible": True,
            "min_text_length": 100,
            "streaming": True,
            "text_fields": ["text", "contents"],
            "required_fields": ["created_timestamp", "downloaded_timestamp", "url"],
        }
        cfg = DatasetConfig.from_hydra(d)
        assert cfg.subset == "atticus_contracts"
        assert cfg.min_text_length == 100
        assert isinstance(cfg.text_fields, tuple)
        assert isinstance(cfg.required_fields, frozenset)

    def test_from_hydra_overrides_min_text_length(self) -> None:
        d = {
            "dataset_id": "pile-of-law/pile-of-law",
            "subset": "r_courtlistener_opinions",
            "split": "train",
            "revision": "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b",  # pragma: allowlist secret
            "reproducible": False,
            "min_text_length": 20,
            "streaming": True,
            "text_fields": ["text"],
            "required_fields": ["url"],
        }
        cfg = DatasetConfig.from_hydra(d)
        assert cfg.min_text_length == 20
        assert cfg.reproducible is False

    def test_hydra_yaml_files_exist(self) -> None:
        from pathlib import Path

        assert Path("configs/data/legal_rag.yaml").exists()
        assert Path("configs/data/legal_rag_explore.yaml").exists()

    def test_hydra_yaml_parseable(self) -> None:
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("configs/data/legal_rag.yaml").read_text())
        assert cfg["subset"] == "r_courtlistener_opinions"
        assert cfg["reproducible"] is True
        assert cfg["min_text_length"] == 50

    def test_hydra_explore_yaml_sets_reproducible_false(self) -> None:
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("configs/data/legal_rag_explore.yaml").read_text())
        assert cfg["reproducible"] is False


class TestFilteringAPI:
    def test_filter_by_date_range_includes_matching(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "created_timestamp": "2022-03-15"}
        results = list(loader.filter_by_date_range([row], "2022-01-01", "2022-12-31"))
        assert len(results) == 1

    def test_filter_by_date_range_excludes_out_of_range(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "created_timestamp": "2020-01-01"}
        results = list(loader.filter_by_date_range([row], "2022-01-01", "2022-12-31"))
        assert results == []

    def test_filter_by_date_range_excludes_missing_timestamp(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "created_timestamp": ""}
        results = list(loader.filter_by_date_range([row], "2022-01-01", "2022-12-31"))
        assert results == []

    def test_filter_by_court_includes_matching(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "court_id": "ca9"}
        results = list(loader.filter_by_court([row], ["ca9", "ca1"]))
        assert len(results) == 1

    def test_filter_by_court_excludes_non_matching(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "court_id": "ca5"}
        results = list(loader.filter_by_court([row], ["ca9", "ca1"]))
        assert results == []

    def test_filter_by_court_excludes_missing_court_field(self, loader: DatasetLoader, valid_row: dict) -> None:
        results = list(loader.filter_by_court([valid_row], ["ca9"]))
        assert results == []

    def test_filter_min_text_tokens_includes_long_enough(self, loader: DatasetLoader, valid_row: dict) -> None:
        results = list(loader.filter_min_text_tokens([valid_row], min_tokens=5, tokenizer=None))
        assert len(results) == 1

    def test_filter_min_text_tokens_excludes_short(self, loader: DatasetLoader, valid_row: dict) -> None:
        row = {**valid_row, "text": "short"}
        results = list(loader.filter_min_text_tokens([row], min_tokens=100, tokenizer=None))
        assert results == []

    def test_filter_min_text_tokens_uses_tokenizer_when_provided(self, loader: DatasetLoader, valid_row: dict) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 200)
        results = list(loader.filter_min_text_tokens([valid_row], min_tokens=150, tokenizer=mock_tokenizer))
        assert len(results) == 1
        mock_tokenizer.encode.assert_called_once()

    def test_filters_composable(self, loader: DatasetLoader, valid_row: dict) -> None:
        """Filters are composable — pipe output of one into another."""
        rows = [
            {**valid_row, "court_id": "ca9", "created_timestamp": "2022-06-01"},
            {**valid_row, "court_id": "ca1", "created_timestamp": "2022-06-01"},
            {**valid_row, "court_id": "ca9", "created_timestamp": "2020-01-01"},
        ]
        by_court = loader.filter_by_court(rows, ["ca9"])
        by_date = loader.filter_by_date_range(by_court, "2022-01-01", "2022-12-31")
        results = list(by_date)
        assert len(results) == 1
        assert results[0]["court_id"] == "ca9"
