"""TDD for scripts/clean_gold_pairs.py — orchestrator that cleans
destination_context in gold_pairs_{val,test}.jsonl files.

Tests call main() directly (matches repo convention test_baseline_prep.py).
One smoke test invokes the CLI via subprocess to verify __main__ wiring.

Usage:
    uv run python scripts/clean_gold_pairs.py \\
        --in-dir data/processed/baseline \\
        --out-dir data/processed/baseline \\
        --files gold_pairs_test.jsonl gold_pairs_val.jsonl
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "clean_gold_pairs.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def make_gold_dir(tmp_path: Path):
    """Factory: build an `in/` dir with named JSONL files containing rows."""

    def _build(files: dict[str, list[dict]]) -> tuple[Path, Path]:
        in_dir = tmp_path / "in"
        in_dir.mkdir(exist_ok=True)
        out_dir = tmp_path / "out"
        for fname, rows in files.items():
            _write_jsonl(in_dir / fname, rows)
        return in_dir, out_dir

    return _build


def _row(src_id: int, ctx: str, cluster: int = 100, dest: int = 999) -> dict:
    return {
        "source_id": src_id,
        "source_cluster_id": cluster,
        "source_court": "ca9",
        "dest_id": dest,
        "quote": "q",
        "destination_context": ctx,
    }


# ----- in-process CLI tests (fast, repo convention) -----------------------


@pytest.mark.unit
class TestCleanGoldPairsInProcess:
    def test_script_exists(self) -> None:
        assert SCRIPT.exists(), f"missing {SCRIPT}"

    @pytest.mark.parametrize(
        "files_to_process",
        [
            (["gold_pairs_test.jsonl"]),
            (["gold_pairs_test.jsonl", "gold_pairs_val.jsonl"]),
        ],
    )
    def test_cleans_files_to_out_dir(self, make_gold_dir, files_to_process) -> None:
        from clean_gold_pairs import main

        files_dict = {
            fname: [
                _row(1, "Brown v. Board, 347 U.S. 483 (1954)"),
                _row(2, "Plain prose without citation."),
            ]
            for fname in files_to_process
        }
        in_dir, out_dir = make_gold_dir(files_dict)
        main(in_dir=in_dir, out_dir=out_dir, files=files_to_process)
        for fname in files_to_process:
            out_file = out_dir / fname
            assert out_file.exists()
            rows = [json.loads(line) for line in out_file.open()]
            assert len(rows) == 2
            assert "347 U.S." not in rows[0]["destination_context"]
            assert "Brown" not in rows[0]["destination_context"]
            assert "Plain prose" in rows[1]["destination_context"]
            assert rows[0]["source_cluster_id"] == 100
            assert rows[0]["dest_id"] == 999

    def test_writes_summary_json(self, make_gold_dir) -> None:
        from clean_gold_pairs import main

        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [
                    _row(1, "Brown v. Board, 347 U.S. 483 (1954)"),
                ],
            }
        )
        main(
            in_dir=in_dir,
            out_dir=out_dir,
            files=["gold_pairs_test.jsonl"],
        )
        summary = out_dir / "clean_gold_pairs_summary.json"
        assert summary.exists()
        meta = json.loads(summary.read_text())
        assert meta["files_processed"] == ["gold_pairs_test.jsonl"]
        assert meta["total_rows_cleaned"] == 1
        assert "git_sha" in meta
        assert "input_sha256" in meta
        assert "gold_pairs_test.jsonl" in meta["input_sha256"]
        assert "output_sha256" in meta

    def test_raises_on_missing_input_file(self, tmp_path: Path) -> None:
        from clean_gold_pairs import main

        with pytest.raises(FileNotFoundError):
            main(
                in_dir=tmp_path,
                out_dir=tmp_path / "out",
                files=["missing.jsonl"],
            )


# ----- subprocess smoke test (verifies __main__ wiring) -------------------


@pytest.mark.integration
class TestCleanGoldPairsCLI:
    def test_dry_run_via_subprocess(self, make_gold_dir) -> None:
        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [
                    _row(1, "Brown v. Board, 347 U.S. 483 (1954)"),
                ],
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-dir",
                str(in_dir),
                "--out-dir",
                str(out_dir),
                "--files",
                "gold_pairs_test.jsonl",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "DRY RUN" in result.stdout
        assert "gold_pairs_test.jsonl" in result.stdout


# ----- additional contracts -------------------------------------------


@pytest.mark.unit
class TestCleanGoldPairsContracts:
    def test_preserves_non_citation_prose(self, make_gold_dir) -> None:
        """Cleaner must not destroy semantic content like verbs, nouns."""
        from clean_gold_pairs import main

        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [
                    _row(1, "The Court held that Brown v. Board, 347 U.S. 483, controls."),
                ],
            }
        )
        main(in_dir=in_dir, out_dir=out_dir, files=["gold_pairs_test.jsonl"])
        row = json.loads((out_dir / "gold_pairs_test.jsonl").read_text().strip())
        # Citation removed
        assert "347 U.S." not in row["destination_context"]
        # Prose survived
        ctx = row["destination_context"].lower()
        assert "court" in ctx
        assert "held" in ctx
        assert "controls" in ctx

    def test_summary_hashes_match_actual_files(self, make_gold_dir) -> None:
        """input_sha256[file] == actual sha256 of input file."""
        import hashlib

        from clean_gold_pairs import main

        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [_row(1, "Brown v. Board, 347 U.S. 483")],
            }
        )
        main(in_dir=in_dir, out_dir=out_dir, files=["gold_pairs_test.jsonl"])
        meta = json.loads((out_dir / "clean_gold_pairs_summary.json").read_text())

        actual_in = hashlib.sha256((in_dir / "gold_pairs_test.jsonl").read_bytes()).hexdigest()
        actual_out = hashlib.sha256((out_dir / "gold_pairs_test.jsonl").read_bytes()).hexdigest()
        assert meta["input_sha256"]["gold_pairs_test.jsonl"] == actual_in
        assert meta["output_sha256"]["gold_pairs_test.jsonl"] == actual_out
        assert len(actual_in) == 64
        assert len(actual_out) == 64

    def test_git_sha_is_valid_string(self, make_gold_dir) -> None:
        from clean_gold_pairs import main

        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [_row(1, "Plain text.")],
            }
        )
        main(in_dir=in_dir, out_dir=out_dir, files=["gold_pairs_test.jsonl"])
        meta = json.loads((out_dir / "clean_gold_pairs_summary.json").read_text())
        assert isinstance(meta["git_sha"], str)
        assert len(meta["git_sha"]) >= 7

    def test_schema_preservation_all_fields_except_cleaned(self, make_gold_dir) -> None:
        """All fields other than destination_context must be byte-identical."""
        from clean_gold_pairs import main

        original = _row(42, "Brown v. Board, 347 U.S. 483 (1954)", cluster=777, dest=88)
        in_dir, out_dir = make_gold_dir({"gold_pairs_test.jsonl": [original]})
        main(in_dir=in_dir, out_dir=out_dir, files=["gold_pairs_test.jsonl"])
        cleaned_row = json.loads((out_dir / "gold_pairs_test.jsonl").read_text().strip())
        # destination_context changed; everything else preserved
        for key in original:
            if key == "destination_context":
                continue
            assert cleaned_row[key] == original[key], (
                f"field {key!r} corrupted: {original[key]!r} -> {cleaned_row[key]!r}"
            )

    @pytest.mark.parametrize(
        "rows",
        [
            # Empty file
            [],
            # Row with None destination_context
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 100,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "q",
                    "destination_context": None,
                }
            ],
            # Row missing destination_context entirely
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 100,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "q",
                }
            ],
        ],
        ids=["empty_file", "null_field", "missing_field"],
    )
    def test_handles_edge_case_inputs(self, make_gold_dir, rows) -> None:
        """Cleaner must not crash on empty file or None/missing field."""
        from clean_gold_pairs import main

        in_dir, out_dir = make_gold_dir({"gold_pairs_test.jsonl": rows})
        main(in_dir=in_dir, out_dir=out_dir, files=["gold_pairs_test.jsonl"])
        out_file = out_dir / "gold_pairs_test.jsonl"
        assert out_file.exists()
        # Output row count matches input
        out_rows = [json.loads(line) for line in out_file.open() if line.strip()]
        assert len(out_rows) == len(rows)

    def test_idempotent_at_cli_level(self, make_gold_dir) -> None:
        """Run twice via CLI: second run produces identical bytes."""
        import hashlib

        from clean_gold_pairs import main

        in_dir, out_dir1 = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [
                    _row(1, "Brown v. Board, 347 U.S. 483 (1954)"),
                    _row(2, "Plain prose."),
                ],
            }
        )
        main(in_dir=in_dir, out_dir=out_dir1, files=["gold_pairs_test.jsonl"])
        first = (out_dir1 / "gold_pairs_test.jsonl").read_bytes()

        # Second run: feed first cleaned output back in
        out_dir2 = out_dir1.parent / "out2"
        main(
            in_dir=out_dir1,
            out_dir=out_dir2,
            files=["gold_pairs_test.jsonl"],
        )
        second = (out_dir2 / "gold_pairs_test.jsonl").read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


@pytest.mark.integration
class TestCleanGoldPairsDryRunSafety:
    def test_dry_run_creates_no_output_files(self, make_gold_dir) -> None:
        """Dry-run must not write any output files."""
        in_dir, out_dir = make_gold_dir(
            {
                "gold_pairs_test.jsonl": [_row(1, "Brown v. Board, 347 U.S. 483")],
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-dir",
                str(in_dir),
                "--out-dir",
                str(out_dir),
                "--files",
                "gold_pairs_test.jsonl",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert not (out_dir / "gold_pairs_test.jsonl").exists()
        assert not (out_dir / "clean_gold_pairs_summary.json").exists()
