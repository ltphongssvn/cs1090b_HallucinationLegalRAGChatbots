"""Report S3 remote artifact count + total size for DVC-tracked files.

Usage:
    .venv/bin/python scripts/inspect_dvc_remote.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import yaml


def main() -> int:
    # Get remote URL from .dvc/config
    cfg_path = Path(".dvc/config")
    if not cfg_path.is_file():
        print("ERROR: .dvc/config not found")
        return 1
    cfg = cfg_path.read_text()
    remote_url = None
    for line in cfg.splitlines():
        line = line.strip()
        if line.startswith("url ="):
            remote_url = line.split("=", 1)[1].strip()
            break
    if not remote_url:
        print("ERROR: no remote URL in .dvc/config")
        return 1
    print(f"Remote: {remote_url}\n")

    # Walk all .dvc files, parse outs (md5 + size)
    rows: list[tuple[int, str, str]] = []
    total_size = 0
    for p in Path(".").rglob("*.dvc"):
        s = str(p)
        if ".venv" in s or ".git" in s:
            continue
        try:
            d = yaml.safe_load(p.read_text())
            for out in d.get("outs", []):
                size = out.get("size", 0)
                md5 = out.get("md5", "")
                path = out.get("path", str(p)[:-4])
                total_size += size
                rows.append((size, path, md5))
        except Exception as e:
            print(f"WARN: {p}: {e}")

    rows.sort(reverse=True)
    print(f"{'SIZE':>12}  {'PATH':<60}  MD5")
    print("=" * 100)
    for size, path, md5 in rows:
        gb = size / 1024**3
        mb = size / 1024**2
        sz = f"{gb:.2f} GB" if gb >= 1 else f"{mb:.2f} MB"
        print(f"{sz:>12}  {path:<60}  {md5[:16]}")

    print("=" * 100)
    print(f"  Total artifacts: {len(rows)}")
    print(f"  Total size:      {total_size / 1024**3:.2f} GB ({total_size:,} bytes)")

    # Cross-check vs actual S3 usage
    print(f"\nQuerying actual S3 usage at {remote_url} ...")
    proc = subprocess.run(
        ["aws", "s3", "ls", "--recursive", "--summarize", remote_url],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode == 0:
        for line in proc.stdout.splitlines()[-3:]:
            print(f"  {line.strip()}")
    else:
        print(f"  (aws cli not available or no creds; skipping S3 check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
