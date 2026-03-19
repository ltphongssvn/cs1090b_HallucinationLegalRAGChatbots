# src/dataset_config.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    """
    Hydra-injectable configuration for a pile-of-law subset.
    All fields overridable via configs/data/legal_rag.yaml.

    data_source:
      "artifact" — load from preprocessed local artifact (training default)
      "hf"       — load from HF Hub with trust_remote_code (ingestion only)
    """

    dataset_id: str = "pile-of-law/pile-of-law"
    subset: str = "r_courtlistener_opinions"
    split: str = "train"
    revision: str = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret
    reproducible: bool = True
    min_text_length: int = 50
    streaming: bool = True
    text_fields: tuple[str, ...] = field(default_factory=lambda: ("text", "contents"))
    required_fields: frozenset[str] = field(
        default_factory=lambda: frozenset({"created_timestamp", "downloaded_timestamp", "url"})
    )
    data_source: Literal["hf", "artifact"] = "artifact"
    artifact_path: str | None = None
    # Streaming adds non-determinism in artifact mode; disable for exact reruns.
    allow_streaming_in_artifact_mode: bool = False

    @classmethod
    def from_hydra(cls, cfg: object) -> "DatasetConfig":
        """Instantiate from a Hydra DictConfig or plain dict."""
        try:
            from omegaconf import OmegaConf  # type: ignore[import]

            d = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            d = dict(cfg)  # type: ignore[call-overload]
        if "text_fields" in d:
            d["text_fields"] = tuple(d["text_fields"])
        if "required_fields" in d:
            d["required_fields"] = frozenset(d["required_fields"])
        return cls(**d)


@dataclass
class ArtifactManifest:
    """
    Provenance metadata written alongside every materialized artifact.
    Stored as artifact_manifest.json next to data.
    """

    source_dataset_id: str
    subset: str
    revision: str  # pragma: allowlist secret
    ingestion_timestamp: str
    loader_version: str
    schema_version: str
    row_count: int
    artifact_checksum: str  # pragma: allowlist secret
    hf_datasets_version: str

    def is_compatible_with(self, config: "DatasetConfig") -> list[str]:
        """Return list of incompatibility reasons. Empty = compatible."""
        issues: list[str] = []
        if self.source_dataset_id != config.dataset_id:
            issues.append(f"dataset_id mismatch: {self.source_dataset_id!r} vs {config.dataset_id!r}")
        if self.subset != config.subset:
            issues.append(f"subset mismatch: {self.subset!r} vs {config.subset!r}")
        if self.revision != config.revision:
            issues.append(f"revision mismatch: {self.revision!r} vs {config.revision!r}")
        return issues
