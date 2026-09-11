#!/usr/bin/env python3
"""Reduce a directory of `cabt` episode replays to the decision-point dataset.

    python scripts/build_dataset.py --replays ~/Downloads/archive --out data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cabt.dataset import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays", required=True, help="directory of *.json / *.json.gz replays")
    ap.add_argument("--out", default="data/processed", help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="only parse the first N episodes")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count)")
    ap.add_argument("--parquet", action="store_true", help="write Parquet instead of gzipped CSV")
    args = ap.parse_args()

    summary = build(args.replays, args.out, args.limit, args.workers, args.parquet)
    failures = summary.pop("failures", [])
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"\n{len(failures)} replay(s) could not be parsed:", file=sys.stderr)
        for f in failures[:20]:
            print(f"  {f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
