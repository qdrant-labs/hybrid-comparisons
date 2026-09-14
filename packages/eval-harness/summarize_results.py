#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Summarize eval-harness JSON reports into a single CSV/table.

Usage:
    uv run python summarize_results.py [results_dir]
"""

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../../results")
    reports = sorted(results_dir.glob("*.json"))
    if not reports:
        print(f"No reports found in {results_dir}")
        return

    rows = []
    for path in reports:
        with open(path) as f:
            r = json.load(f)
        rows.append(
            {
                "collection": r["collection_name"],
                "config": r["config"],
                "k": r["k"],
                "prefetch_limit": r["prefetch_limit"],
                "prefetch_mode": "hybrid"
                if r.get("use_dense_prefetch", True)
                else "sparseonly",
                "rescorer": r.get("rescorer", "colbert"),
                "recall@k": round(r["quality"]["recall@k"], 4),
                "mrr@k": round(r["quality"]["mrr@k"], 4),
                "ndcg@k": round(r["quality"]["ndcg@k"], 4),
                "p50_ms": round(r["latency"]["p50_ms"], 2),
                "p95_ms": round(r["latency"]["p95_ms"], 2),
                "p99_ms": round(r["latency"]["p99_ms"], 2),
                "qps": round(r["throughput_qps"], 2),
            }
        )

    rows.sort(
        key=lambda row: (
            row["collection"],
            row["prefetch_mode"],
            row["rescorer"],
            row["k"],
            row["prefetch_limit"],
        )
    )

    out_path = results_dir / "summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widths = {
        key: max(len(key), *(len(str(row[key])) for row in rows)) for key in rows[0]
    }
    header = "  ".join(key.ljust(widths[key]) for key in rows[0])
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in row))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
