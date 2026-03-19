# src/dataset_config.py
# Hydra-compatible dataset configuration — injectable, subset-agnostic.
# Use as a Hydra structured config or instantiate directly.
#
# Hydra usage:
#   @hydra.main(config_path="../configs", config_name="train")
#   def main(cfg: DictConfig) -> None:
#       data_cfg = DatasetConfig(**cfg.data)
#
# Direct usage:
#   config = DatasetConfig(subset="atticus_contracts", min_text_length=100)
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """
    Immutable configuration for a pile-of-law subset.
    All fields overridable via Hydra YAML — see configs/data/legal_rag.yaml.
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

    @classmethod
    def from_hydra(cls, cfg: object) -> "DatasetConfig":
        """Instantiate from a Hydra DictConfig or plain dict.

        Usage:
            data_cfg = DatasetConfig.from_hydra(cfg.data)
        """
        try:
            from omegaconf import OmegaConf  # type: ignore[import]

            d = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            d = dict(cfg)  # type: ignore[call-overload]

        # Convert list fields to correct types
        if "text_fields" in d:
            d["text_fields"] = tuple(d["text_fields"])
        if "required_fields" in d:
            d["required_fields"] = frozenset(d["required_fields"])
        return cls(**d)
