# src/dataset_config.py
# Hydra-compatible dataset configuration — injectable, subset-agnostic.
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """
    Immutable configuration for a pile-of-law subset.
    Designed for Hydra injection — override any field via config YAML.
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
