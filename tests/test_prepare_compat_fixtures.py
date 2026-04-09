"""Tests for scripts/prepare_compat_fixtures.py (TDD Red-first)."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_compat_fixtures import prepare_cl, prepare_lepard  # noqa: E402


class TestPrepareLepard:
    def test_copies_jsonl_to_fixtures_dir(self, tmp_path):
        src = tmp_path / "src.jsonl"
        src.write_text('{"source_id": 1, "dest_id": 2}\n')
        out = tmp_path / "fixtures"
        prepare_lepard(src, out)
        dest = out / "lepard_sample_1k.jsonl"
        assert dest.exists()
        assert dest.read_text() == '{"source_id": 1, "dest_id": 2}\n'

    def test_creates_out_dir_if_missing(self, tmp_path):
        src = tmp_path / "src.jsonl"
        src.write_text("{}\n")
        out = tmp_path / "new" / "nested"
        prepare_lepard(src, out)
        assert (out / "lepard_sample_1k.jsonl").exists()


class TestPrepareCl:
    def _make_shards(self, cl_dir: Path, shard_data: list[list[dict]]) -> None:
        cl_dir.mkdir(parents=True, exist_ok=True)
        for i, rows in enumerate(shard_data):
            path = cl_dir / f"shard_{i:04d}.jsonl"
            with path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    def test_writes_gzipped_id_set(self, tmp_path):
        cl_dir = tmp_path / "cl"
        self._make_shards(
            cl_dir,
            [
                [{"id": 100, "court_id": "ca9"}, {"id": 200, "court_id": "ca5"}],
                [{"id": 300, "court_id": "ca1"}],
            ],
        )
        out = tmp_path / "fixtures"
        prepare_cl(cl_dir, out, lepard_sample=None)
        ids_path = out / "cl_ids.txt.gz"
        assert ids_path.exists()
        with gzip.open(ids_path, "rt") as f:
            ids = {int(line) for line in f}
        assert ids == {100, 200, 300}

    def test_writes_matched_court_map_when_lepard_provided(self, tmp_path):
        cl_dir = tmp_path / "cl"
        self._make_shards(
            cl_dir,
            [[{"id": 100, "court_id": "ca9"}, {"id": 200, "court_id": "ca5"}, {"id": 999, "court_id": "ca1"}]],
        )
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps({"source_id": 100, "dest_id": 200})
            + "\n"
            + json.dumps({"source_id": 100, "dest_id": 500})
            + "\n"
        )
        out = tmp_path / "fixtures"
        prepare_cl(cl_dir, out, lepard_sample=lepard)
        courts_path = out / "cl_matched_courts.json"
        assert courts_path.exists()
        data = json.loads(courts_path.read_text())
        assert data == {"100": "ca9", "200": "ca5"}

    def test_skips_court_map_when_no_lepard(self, tmp_path):
        cl_dir = tmp_path / "cl"
        self._make_shards(cl_dir, [[{"id": 1, "court_id": "ca9"}]])
        out = tmp_path / "fixtures"
        prepare_cl(cl_dir, out, lepard_sample=None)
        assert not (out / "cl_matched_courts.json").exists()

    def test_raises_if_no_shards(self, tmp_path):
        cl_dir = tmp_path / "empty"
        cl_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No shards found"):
            prepare_cl(cl_dir, tmp_path / "fixtures", lepard_sample=None)
