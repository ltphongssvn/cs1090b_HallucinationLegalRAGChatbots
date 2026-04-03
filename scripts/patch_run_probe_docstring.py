"""Patch run_probe() docstring to document full_scan semantics accurately."""
from pathlib import Path

path = Path("src/dataset_probe.py")
content = path.read_text(encoding="utf-8")

OLD = '    """\n    Run all gates on a record subset. Returns a typed ProbeReport.\n    No side effects on corpus shards \u2014 all output written to --output only.\n\n    log_to_wandb is intentionally absent from this signature (obs 4/17).\n    W&B telemetry is exclusively a main() concern \u2014 call _log_report_to_wandb\n    after run_probe() returns from main() only.\n    """'

NEW = '    """\n    Run all gates on a record subset. Returns a typed ProbeReport.\n    No side effects on corpus shards \u2014 all output written to --output only.\n\n    full_scan parameter semantics (important):\n      Polars _full_scan_with_polars() is ALWAYS called regardless of full_scan value.\n      full_scan=True  (default) \u2014 all loaded records passed to gates.\n      full_scan=False           \u2014 loaded records reservoir-subsampled to `subset`\n                                  before gates run.\n      The flag controls post-load retention, not scan scope.\n      provenance["full_scan"] records the flag value passed by the caller.\n\n    log_to_wandb is intentionally absent from this signature.\n    W&B telemetry is exclusively a main() concern \u2014 call _log_report_to_wandb\n    after run_probe() returns from main() only.\n    """'

assert OLD in content, f"Target not found. Check exact bytes."
path.write_text(content.replace(OLD, NEW, 1), encoding="utf-8")
print("Docstring patched.")
