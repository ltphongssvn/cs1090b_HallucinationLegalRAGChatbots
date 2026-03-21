# src/validation.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/validation.py
# SRP: TDD contract tests for bulk data acquisition pipeline.
# NOTE: Functions prefixed check_ (not test_) to avoid pytest auto-collection.

import json  # used: parse JSONL
import random  # used: random sampling
from pathlib import Path  # used: shard paths
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

from src.config import PipelineConfig
from src.exceptions import ValidationError
from src.manifest import file_checksum


def _count_lines(filepath: Union[str, Path]) -> int:
    with open(filepath) as fh:
        return sum(1 for _ in fh)


def _iter_cases(
    filepath: Union[str, Path],
    count: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    read = 0
    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
                read += 1
                if count and read >= count:
                    return


def _random_sample_cases(
    filepath: Union[str, Path],
    sample_size: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    total = _count_lines(filepath)
    if total <= sample_size:
        return list(_iter_cases(filepath))
    rng = random.Random(seed)
    sampled_indices: Set[int] = set(rng.sample(range(total), sample_size))
    cases: List[Dict[str, Any]] = []
    with open(filepath) as fh:
        for idx, line in enumerate(fh):
            if idx in sampled_indices:
                stripped = line.strip()
                if stripped:
                    cases.append(json.loads(stripped))
    return cases


def _select_shards(config: PipelineConfig, strategy: str = "all") -> List[Path]:
    all_shards = sorted(config.shard_dir.glob("shard_*.jsonl"))
    if strategy == "all":
        return all_shards
    elif strategy == "head":
        return all_shards[:3]
    elif strategy == "sample":
        rng = random.Random(42)
        sample_count = min(max(3, len(all_shards) // 5), len(all_shards))
        return rng.sample(all_shards, sample_count)
    return all_shards


def check_shard_dir_exists(config: PipelineConfig) -> None:
    if not config.shard_dir.exists():
        raise ValidationError(f"Shard dir missing: {config.shard_dir}")


def check_manifest_exists(config: PipelineConfig) -> None:
    if not config.manifest_path.exists():
        raise ValidationError(f"Manifest missing: {config.manifest_path}")


def check_shards_exist(config: PipelineConfig) -> None:
    if len(sorted(config.shard_dir.glob("shard_*.jsonl"))) < 1:
        raise ValidationError("No shards found")


def check_total_count(config: PipelineConfig) -> None:
    shards = sorted(config.shard_dir.glob("shard_*.jsonl"))
    total = sum(_count_lines(shard_path) for shard_path in shards)
    if total < config.min_expected_total:
        raise ValidationError(f"Only {total}, need >={config.min_expected_total:,}")


def check_valid_json(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    for shard_path in _select_shards(config, shard_strategy):
        _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)


def check_text_present(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)
        missing = [i for i, case in enumerate(cases) if not case.get("text")]
        if missing:
            raise ValidationError(f"{shard_path.name}: {len(missing)}/{len(cases)} lack text")


def check_text_substantive(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)
        stubs = [i for i, case in enumerate(cases) if len(case.get("text", "")) < config.min_text_length]
        if cases and (1.0 - len(stubs) / len(cases)) < 0.9:
            raise ValidationError(f"{shard_path.name}: {len(stubs)}/{len(cases)} are stubs")


def check_provenance_fields(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    required: Set[str] = {"court_id", "docket_id", "case_name", "date_filed", "precedential_status", "text_source"}
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)
        for idx, case in enumerate(cases):
            missing_fields = required - set(case.keys())
            if missing_fields:
                raise ValidationError(f"{shard_path.name} record {idx} missing: {missing_fields}")


def check_raw_and_normalized_text(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    """Records must have both raw_text and text (normalized) plus cleaning_flags."""
    required_text_fields: Set[str] = {"raw_text", "text", "cleaning_flags"}
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)
        for idx, case in enumerate(cases):
            missing_fields = required_text_fields - set(case.keys())
            if missing_fields:
                raise ValidationError(f"{shard_path.name} record {idx} missing: {missing_fields}")


def check_text_source_tracked(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    valid: Set[str] = set(config.text_source_fields)
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=config.schema_audit_per_shard)
        for idx, case in enumerate(cases):
            if case.get("text_source") not in valid:
                raise ValidationError(f"Record {idx}: bad text_source={case.get('text_source')}")


def check_multiple_circuits(config: PipelineConfig) -> None:
    courts: Set[str] = set()
    for shard_path in sorted(config.shard_dir.glob("shard_*.jsonl")):
        cases = _random_sample_cases(shard_path, sample_size=50)
        for case in cases:
            courts.add(case.get("court_id", ""))
    if len(courts) < 5:
        raise ValidationError(f"Only {len(courts)} circuits")


def check_schema_consistent(config: PipelineConfig, shard_strategy: str = "sample") -> None:
    reference_schema: Optional[Set[str]] = None
    for shard_path in _select_shards(config, shard_strategy):
        cases = _random_sample_cases(shard_path, sample_size=10)
        for idx, case in enumerate(cases):
            if reference_schema is None:
                reference_schema = set(case.keys())
            elif set(case.keys()) != reference_schema:
                raise ValidationError(f"{shard_path.name} record {idx} schema mismatch")


def check_checksums(config: PipelineConfig, manifest_data: Dict[str, Any]) -> None:
    if "checksum" not in manifest_data:
        return
    for name, expected in list(manifest_data["checksum"].items())[:3]:
        filepath = config.shard_dir / name
        if filepath.exists() and file_checksum(filepath) != expected:
            raise ValidationError(f"{name} checksum mismatch")


def run_contract_tests(
    config: Optional[PipelineConfig] = None,
    manifest_data: Optional[Dict[str, Any]] = None,
    logger: Any = None,
    shard_strategy: str = "sample",
) -> bool:
    if config is None:
        config = PipelineConfig()
    if manifest_data is None:
        manifest_data = {}

    tests: List[Tuple[str, Any]] = [
        ("Shard directory must exist", lambda: check_shard_dir_exists(config)),
        ("Manifest must exist", lambda: check_manifest_exists(config)),
        ("At least one shard present", lambda: check_shards_exist(config)),
        ("Sufficient opinions", lambda: check_total_count(config)),
        ("Valid JSON", lambda: check_valid_json(config, shard_strategy)),
        ("Text present", lambda: check_text_present(config, shard_strategy)),
        ("Text substantive", lambda: check_text_substantive(config, shard_strategy)),
        ("Provenance metadata", lambda: check_provenance_fields(config, shard_strategy)),
        ("Raw + normalized text + flags", lambda: check_raw_and_normalized_text(config, shard_strategy)),
        ("Text source tracked", lambda: check_text_source_tracked(config, shard_strategy)),
        ("Multiple circuits", lambda: check_multiple_circuits(config)),
        ("Schema consistent", lambda: check_schema_consistent(config, shard_strategy)),
        ("Checksums match", lambda: check_checksums(config, manifest_data)),
    ]

    all_passed = True
    for description, test_function in tests:
        try:
            test_function()
            if logger:
                logger.info(f"✓ PASS: {description}")
        except ValidationError as error:
            if logger:
                logger.error(f"✗ FAIL: {description} — {error}")
            all_passed = False

    return all_passed
