"""
EDA: diagnose text_length_distribution contract FAIL (p5=0).

Cell 6 Gate A8 PASSED using stored text_length metadata; Cell 4 data_contracts
FAILED using len(text) live. This script reconciles the contradiction by
comparing stored text_length vs actual character length of the text field.

Outputs:
    - Console: counts, percentiles, court/shard breakdown of anomalies.
    - logs/eda_text_length_anomalies.csv: full anomaly list for notebook reuse.
"""

from pathlib import Path

import polars as pl

SHARD_GLOB = "data/raw/cl_federal_appellate_bulk/shard_*.jsonl"
OUT_CSV = Path("logs/eda_text_length_anomalies.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

print(f"Scanning {SHARD_GLOB} via Polars lazy scan...")
df = pl.scan_ndjson(SHARD_GLOB, low_memory=True)

stats = df.select(
    [
        pl.col("text_length").min().alias("min"),
        pl.col("text_length").quantile(0.01).alias("p1"),
        pl.col("text_length").quantile(0.05).alias("p5"),
        pl.col("text_length").quantile(0.50).alias("p50"),
        pl.col("text_length").quantile(0.95).alias("p95"),
        pl.col("text_length").max().alias("max"),
        pl.len().alias("n_total"),
    ]
).collect()
print("\n=== Stored text_length distribution (full corpus) ===")
print(stats)

anomalies = (
    df.filter(pl.col("text_length") < 100)
    .with_columns(pl.col("text").str.len_chars().alias("actual_len"))
    .select(["id", "court_id", "text_length", "actual_len"])
    .collect()
)
n_anom = anomalies.height
print("\n=== Anomalies: text_length < 100 ===")
print(f"Total anomalies: {n_anom:,}")

if n_anom > 0:
    print("\nFirst 10 anomalies (stored vs actual length):")
    print(anomalies.head(10))
    stale = anomalies.filter(pl.col("actual_len") >= 100).height
    truly_empty = anomalies.filter(pl.col("actual_len") < 100).height
    print(f"\nStale metadata (actual_len >= 100): {stale:,}")
    print(f"Truly short/empty (actual_len < 100): {truly_empty:,}")
    by_court = anomalies.group_by("court_id").agg(pl.len().alias("n")).sort("n", descending=True).head(15)
    print("\nTop 15 courts by anomaly count:")
    print(by_court)
    anomalies.write_csv(OUT_CSV)
    print(f"\nFull anomaly list written to {OUT_CSV}")
else:
    print("No anomalies found — contract threshold may need review.")
