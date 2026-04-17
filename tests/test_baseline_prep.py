"""Contract + unit + property tests for scripts/baseline_prep.py.

Follows tests/test_eda_ms3_lepard.py + test_eda_ms3_corpus.py conventions:
class-based, module-scoped import fixture, Hypothesis for property tests.
"""

from __future__ import annotations

import ast
import gzip
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline_module() -> Any:
    return importlib.import_module("scripts.baseline_prep")


class DummyTokenizer:
    """Whitespace tokenizer — air-gaps unit tests from HF Hub."""

    model_max_length = 8192

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(t) % 30000 for t in text.split()]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{t}" for t in tokens)

    def __call__(self, text: str, **kw: Any) -> dict:
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


# ---------- contract tier ----------


@pytest.mark.contract
class TestFileExists:
    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts" / "baseline_prep.py").is_file()


@pytest.mark.contract
class TestModuleLevelConstants:
    def test_constants(self, baseline_module: Any) -> None:
        m = baseline_module
        assert isinstance(m.CHUNK_SIZE_SUBWORDS, int) and m.CHUNK_SIZE_SUBWORDS == 1024
        assert isinstance(m.CHUNK_OVERLAP_SUBWORDS, int) and m.CHUNK_OVERLAP_SUBWORDS == 128
        assert isinstance(m.ENCODER_MODEL, str) and m.ENCODER_MODEL == "BAAI/bge-m3"
        assert isinstance(m.SCHEMA_VERSION, str) and len(m.SCHEMA_VERSION.split(".")) == 3
        assert m.VAL_SIZE == 2000
        assert m.TEST_SIZE == 45000


@pytest.mark.contract
class TestMainSignature:
    def test_main_has_expected_parameters(self, baseline_module: Any) -> None:
        sig = inspect.signature(baseline_module.main)
        expected = {
            "shard_dir",
            "lepard_path",
            "cl_ids_path",
            "court_map_path",
            "out_dir",
            "log_to_wandb",
            "resume",
            "seed",
            "val_size",
            "test_size",
        }
        assert expected.issubset(set(sig.parameters))


@pytest.mark.contract
class TestSummaryModelIsPydantic:
    def test_is_basemodel(self) -> None:
        from pydantic import BaseModel

        from src.eda_schemas import BaselinePrepSummary

        assert issubclass(BaselinePrepSummary, BaseModel)


@pytest.mark.contract
class TestNoBasicConfigAtImport:
    def test_no_logging_basicconfig(self) -> None:
        src = (REPO_ROOT / "scripts" / "baseline_prep.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "logging"
                    and fn.attr == "basicConfig"
                ):
                    raise AssertionError("logging.basicConfig at module level")


@pytest.mark.contract
class TestArgparseCLI:
    def test_build_arg_parser(self, baseline_module: Any) -> None:
        parser = baseline_module._build_arg_parser()
        # Matches tests/test_eda_ms3_lepard.py:367 repo convention
        actions = {a.dest: a for a in parser._actions}
        for dest in (
            "shard_dir",
            "lepard_path",
            "cl_ids_path",
            "court_map_path",
            "out_dir",
            "log_to_wandb",
            "resume",
            "seed",
        ):
            assert dest in actions


@pytest.mark.contract
class TestWandbIsolation:
    def test_log_to_wandb_callable(self, baseline_module: Any) -> None:
        assert callable(getattr(baseline_module, "_log_to_wandb", None))


@pytest.mark.contract
class TestAtomicCheckpointWrite:
    def test_script_uses_atomic_rename(self) -> None:
        """Checkpoint writer must use os.replace() or tempfile + rename.

        Repo convention per src/bulk_download.py:141 (os.replace) and
        scripts/ingest_lepard.py:578 (tempfile.mkstemp + os.replace).
        """
        src = (REPO_ROOT / "scripts" / "baseline_prep.py").read_text(encoding="utf-8")
        assert "os.replace" in src or "tempfile.mkstemp" in src, (
            "checkpoint must be atomic (os.replace or tempfile.mkstemp pattern)"
        )


# ---------- unit tier ----------


@pytest.fixture
def mini_shard(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    records = [
        {"id": 1001, "court_id": "ca5", "text": "alpha " * 300, "text_source": "plain_text", "text_length": 1800},
        {"id": 1002, "court_id": "ca9", "text": "beta " * 600, "text_source": "plain_text", "text_length": 3000},
        {"id": 1003, "court_id": "ca1", "text": "gamma " * 100, "text_source": "plain_text", "text_length": 600},
        {"id": 1004, "court_id": "cadc", "text": "delta " * 400, "text_source": "plain_text", "text_length": 2400},
    ]
    p = shard_dir / "shard_0000.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return shard_dir


@pytest.fixture
def mini_lepard(tmp_path: Path) -> Path:
    pairs = [
        {"source_id": 1001, "dest_id": 1002, "passage": "p1"},
        {"source_id": 1002, "dest_id": 1003, "passage": "p2"},
        {"source_id": 1003, "dest_id": 1004, "passage": "p3"},
        {"source_id": 1001, "dest_id": 1004, "passage": "p4"},
        {"source_id": 9999, "dest_id": 1001, "passage": "p5"},
        {"source_id": 1002, "dest_id": 8888, "passage": "p6"},
    ]
    p = tmp_path / "lepard.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def mini_cl_ids(tmp_path: Path) -> Path:
    p = tmp_path / "cl_ids.txt.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for oid in (1001, 1002, 1003, 1004):
            f.write(f"{oid}\n")
    return p


@pytest.fixture
def mini_court_map(tmp_path: Path) -> Path:
    p = tmp_path / "court_map.json"
    p.write_text(json.dumps({"1001": "ca5", "1002": "ca9", "1003": "ca1", "1004": "cadc"}))
    return p


@pytest.mark.unit
class TestGoldPairExtraction:
    def test_filters_to_both_in_cl(self, baseline_module: Any, mini_lepard: Path, mini_cl_ids: Path) -> None:
        cl_ids = baseline_module._load_cl_ids(mini_cl_ids)
        pairs = list(baseline_module._iter_usable_gold(mini_lepard, cl_ids))
        assert len(pairs) == 4
        ids = {(p["source_id"], p["dest_id"]) for p in pairs}
        assert (1001, 1002) in ids
        assert (9999, 1001) not in ids
        assert (1002, 8888) not in ids


@pytest.mark.unit
class TestSourceCourtLookup:
    def test_annotates_source_court(
        self, baseline_module: Any, mini_lepard: Path, mini_cl_ids: Path, mini_court_map: Path
    ) -> None:
        cl_ids = baseline_module._load_cl_ids(mini_cl_ids)
        court_map = baseline_module._load_court_map(mini_court_map)
        pairs = list(baseline_module._iter_usable_gold(mini_lepard, cl_ids))
        annotated = baseline_module._annotate_source_court(pairs, court_map)
        assert all("source_court" in p for p in annotated)
        assert annotated[0]["source_court"] in {"ca5", "ca9", "ca1", "cadc"}


@pytest.mark.unit
class TestStratifiedSplit:
    def test_split_sizes(self, baseline_module: Any) -> None:
        pairs = [
            {"source_id": i, "dest_id": i + 10000, "source_court": "ca5" if i % 2 == 0 else "ca9"} for i in range(1000)
        ]
        val, test = baseline_module._stratified_split(
            pairs,
            val_size=100,
            test_size=500,
            seed=0,
        )
        assert len(val) == 100
        assert len(test) == 500
        val_courts = {p["source_court"] for p in val}
        assert val_courts == {"ca5", "ca9"}

    def test_deterministic_under_same_seed(self, baseline_module: Any) -> None:
        pairs = [{"source_id": i, "dest_id": i + 10000, "source_court": "ca5" if i < 50 else "ca9"} for i in range(100)]
        v1, t1 = baseline_module._stratified_split(
            pairs,
            val_size=10,
            test_size=30,
            seed=42,
        )
        v2, t2 = baseline_module._stratified_split(
            pairs,
            val_size=10,
            test_size=30,
            seed=42,
        )
        assert [p["source_id"] for p in v1] == [p["source_id"] for p in v2]
        assert [p["source_id"] for p in t1] == [p["source_id"] for p in t2]

    def test_proportional_stratification(self, baseline_module: Any) -> None:
        """Val/test court proportions must match input within ±5pp tolerance."""
        # 60% ca5, 30% ca9, 10% ca1
        pairs = (
            [{"source_id": i, "dest_id": i + 10000, "source_court": "ca5"} for i in range(600)]
            + [{"source_id": i + 600, "dest_id": i + 10600, "source_court": "ca9"} for i in range(300)]
            + [{"source_id": i + 900, "dest_id": i + 10900, "source_court": "ca1"} for i in range(100)]
        )
        val, test = baseline_module._stratified_split(
            pairs,
            val_size=200,
            test_size=500,
            seed=0,
        )

        def _share(lst: list[dict], court: str) -> float:
            return sum(1 for p in lst if p["source_court"] == court) / len(lst)

        for court, expected in [("ca5", 0.60), ("ca9", 0.30), ("ca1", 0.10)]:
            assert abs(_share(val, court) - expected) <= 0.05, (
                f"val share for {court} off: {_share(val, court)} vs {expected}"
            )
            assert abs(_share(test, court) - expected) <= 0.05, (
                f"test share for {court} off: {_share(test, court)} vs {expected}"
            )


@pytest.mark.unit
class TestChunkerBoundaries:
    """Chunking arithmetic tested with DummyTokenizer — air-gapped from HF Hub."""

    def test_single_chunk_for_short_text(self, baseline_module: Any) -> None:
        tok = DummyTokenizer()
        chunks = baseline_module._chunk_text(
            "hello world " * 20,
            opinion_id=1,
            tok=tok,
        )
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["opinion_id"] == 1

    def test_multi_chunk_with_overlap(self, baseline_module: Any) -> None:
        tok = DummyTokenizer()
        # 2500 tokens at 1024 window + 128 overlap → stride=896 → ~3 chunks
        text = "lorem " * 2500
        chunks = baseline_module._chunk_text(text, opinion_id=42, tok=tok)
        assert len(chunks) >= 2
        assert all(c["opinion_id"] == 42 for c in chunks)
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


@pytest.mark.unit
class TestCheckpointResume:
    def test_skip_completed_shards(self, baseline_module: Any, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        ck = out_dir / "chunking_checkpoint.json"
        ck.write_text(json.dumps({"completed": ["shard_0000.jsonl"]}))
        completed = baseline_module._load_checkpoint(ck)
        assert "shard_0000.jsonl" in completed


@pytest.mark.unit
class TestSummarySchema:
    def test_summary_fields(self) -> None:
        from src.eda_schemas import BaselinePrepSummary

        data = {
            "schema_version": "1.0.0",
            "corpus_chunks": 150,
            "n_opinions_chunked": 4,
            "gold_pairs_total": 4,
            "gold_pairs_train": 0,
            "gold_pairs_val": 2,
            "gold_pairs_test": 2,
            "val_court_distribution": {"ca5": 1, "ca9": 1},
            "test_court_distribution": {"ca1": 1, "cadc": 1},
            "seed": 0,
            "git_sha": "abc123def456",
            "corpus_manifest_sha": "a" * 64,
        }
        mod = BaselinePrepSummary(**data)
        assert mod.gold_pairs_test == 2
        j1 = json.dumps(mod.model_dump(), sort_keys=True)
        j2 = json.dumps(mod.model_dump(), sort_keys=True)
        assert j1 == j2


@pytest.mark.unit
class TestAtomicArtifactWrite:
    def test_summary_written_at_end(
        self,
        baseline_module: Any,
        tmp_path: Path,
        mini_shard: Path,
        mini_lepard: Path,
        mini_cl_ids: Path,
        mini_court_map: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mock tokenizer to air-gap this integration test from HF Hub
        monkeypatch.setattr(
            baseline_module,
            "_get_tokenizer",
            lambda: DummyTokenizer(),
        )
        out_dir = tmp_path / "out"
        baseline_module.main(
            shard_dir=mini_shard,
            lepard_path=mini_lepard,
            cl_ids_path=mini_cl_ids,
            court_map_path=mini_court_map,
            out_dir=out_dir,
            log_to_wandb=False,
            resume=False,
            seed=0,
            val_size=2,
            test_size=2,
        )
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "gold_pairs_test.jsonl").exists()
        assert (out_dir / "gold_pairs_val.jsonl").exists()


# ---------- property tier ----------


@pytest.mark.property
class TestStratificationProperty:
    """True property test: Hypothesis-generated court distributions."""

    @given(
        court_counts=st.dictionaries(
            keys=st.sampled_from(["ca1", "ca2", "ca3", "ca5", "ca9", "cadc"]),
            values=st.integers(min_value=2, max_value=50),
            min_size=2,
            max_size=6,
        ),
        val_frac=st.floats(min_value=0.05, max_value=0.3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_minority_groups_preserved(
        self,
        baseline_module: Any,
        court_counts: dict[str, int],
        val_frac: float,
        seed: int,
    ) -> None:
        """Every court with ≥2 members must appear in val OR test."""
        pairs = []
        idx = 0
        for court, n in court_counts.items():
            for _ in range(n):
                pairs.append(
                    {
                        "source_id": idx,
                        "dest_id": idx + 10**7,
                        "source_court": court,
                    }
                )
                idx += 1
        total = len(pairs)
        val_size = max(len(court_counts), int(total * val_frac))
        test_size = max(len(court_counts), int(total * 0.3))
        if val_size + test_size > total:
            return  # skip invalid splits
        val, test = baseline_module._stratified_split(
            pairs,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        combined_courts = {p["source_court"] for p in val} | {p["source_court"] for p in test}
        # Every input court must appear somewhere in val ∪ test
        assert set(court_counts.keys()).issubset(combined_courts)
