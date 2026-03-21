# tests/unit/test_repro_placeholder.py
# Placeholder unit tests — verify src/repro.py constants are sane.
# These run in CI without requiring a GPU or a built venv.
import pytest


@pytest.mark.unit
def test_random_seed_is_int() -> None:
    """RANDOM_SEED must be an integer — non-integer seeds break numpy/torch seeding."""
    from src.repro import _RANDOM_SEED

    assert isinstance(_RANDOM_SEED, int)


@pytest.mark.unit
def test_expected_pythonhashseed_is_string_zero() -> None:
    from src.repro import _EXPECTED_PYTHONHASHSEED

    assert _EXPECTED_PYTHONHASHSEED == "0"


@pytest.mark.unit
def test_expected_cublas_cfg_is_set() -> None:
    from src.repro import _EXPECTED_CUBLAS_CFG

    assert _EXPECTED_CUBLAS_CFG.startswith(":")


@pytest.mark.unit
def test_expected_tokenizers_parallelism_is_false() -> None:
    from src.repro import _EXPECTED_TOKENIZERS_PAR

    assert _EXPECTED_TOKENIZERS_PAR == "false"
