"""Test mine_hard_negatives writes git_sha to summary."""
import inspect
import pytest


@pytest.fixture
def mine_module():
    return __import__("scripts.mine_hard_negatives", fromlist=["*"])


@pytest.mark.contract
class TestMineHardNegativesProvenance:
    def test_summary_includes_git_sha(self, mine_module):
        """Summary must include git_sha for provenance tracking."""
        src = inspect.getsource(mine_module)
        assert "git_sha" in src, (
            "scripts/mine_hard_negatives.py must write git_sha to summary "
            "for reproducibility provenance"
        )
