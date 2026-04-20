# ADR 0001: BM25 SLURM job requires gpu partition, not general

Date: 2026-04-18
Status: Accepted

## Context
BM25 baseline retrieval (`scripts/baseline_bm25.py`) over 5.79M-chunk /
1.47M-opinion CourtListener corpus needs to build an in-memory bm25s index
and retrieve top-100 for 45K LePaRD queries.

## Observation
SLURM job 95226 submitted to `general` partition (48 CPU / 93GB mem cap)
failed with OOM after 39 minutes mid-index-build:

    JobID        State   ExitCode  MaxRSS      Elapsed   ReqMem
    95226        FAILED  9:0                   00:39:16  90G
    95226.batch  FAILED  9:0       89616288K   00:39:16

MaxRSS 89.6GB ≈ requested 90G → killed by cgroup. Memory breakdown:
corpus texts in RAM (~26GB) + bm25s inverted index (~50GB) + tokenization
workspace (~10GB) + Python runtime (~4GB) ≈ 90GB peak.

## Decision
Move sbatch to `gpu` partition (48 CPU / 186GB cap) with `--mem=160G`.
BM25 is CPU-only; GPUs remain idle (acceptable resource trade-off given
general partition headroom exhausted at 5.79M chunks).

## Consequences
- BM25 baseline runs reliably end-to-end (~35-45 min walltime).
- Future dense baselines (BGE-M3) will use same gpu partition for actual
  GPU compute — no migration cost.
- If corpus grows beyond ~10M chunks, memory may again exceed 160G;
  chunking into shards will be required.

## Related
- Commit e521818
- SLURM job 95226 (failed) → 95254 (recovery)
- scripts/baseline_bm25.sbatch
