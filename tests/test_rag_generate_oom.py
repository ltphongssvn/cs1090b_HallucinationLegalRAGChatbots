# tests/test_rag_generate_oom.py
"""Test rag_generate sbatch defaults are conservative enough for retrieval-augmented prompts."""

from pathlib import Path

import pytest


@pytest.mark.contract
class TestSbatchDefaults:
    def test_batch_size_default_4_or_lower(self):
        sbatch = Path("scripts/rag_generate_multigpu.sbatch").read_text()
        # Look for BATCH_SIZE:-N pattern
        idx = sbatch.find("BATCH_SIZE:-")
        assert idx >= 0, "BATCH_SIZE default not found"
        # Extract digits after :-
        rest = sbatch[idx + len("BATCH_SIZE:-") :]
        digits = ""
        for c in rest:
            if c.isdigit():
                digits += c
            else:
                break
        bs = int(digits)
        assert bs <= 4, (
            f"BATCH_SIZE default {bs} too high for retrieval-augmented prompts on L4 "
            f"(measured OOM at 8); use 4 or lower"
        )

    def test_max_length_default_appropriate(self):
        sbatch = Path("scripts/rag_generate_multigpu.sbatch").read_text()
        idx = sbatch.find("MAX_LENGTH:-")
        assert idx >= 0, "MAX_LENGTH default not found"
        rest = sbatch[idx + len("MAX_LENGTH:-") :]
        digits = ""
        for c in rest:
            if c.isdigit():
                digits += c
            else:
                break
        ml = int(digits)
        assert ml <= 2048, f"MAX_LENGTH default {ml} too high for L4 22GB at batch=4"
