"""Epic 1 Definition of Done demo: parse a DEGIRO CSV end-to-end and print state.

Usage: python scripts/run_epic1_demo.py <path-to-degiro-transactions.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.base_parser import parse_degiro_csv


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_epic1_demo.py <path-to-degiro-transactions.csv>")
        raise SystemExit(1)

    portfolio = parse_degiro_csv(sys.argv[1])
    transactions = portfolio["transactions"]
    skipped = portfolio["skipped_rows"]

    print(f"Broker: {portfolio['broker']} | Base currency: {portfolio['base_currency']}")
    print(f"Parsed {len(transactions)} transactions, skipped {len(skipped)} row(s)\n")

    print("First 10 transactions:")
    for txn in transactions[:10]:
        print(txn.model_dump_json(indent=2))

    unmapped = sorted({t.isin for t in transactions if t.ticker is None})
    if unmapped:
        print(f"\n{len(unmapped)} ISIN(s) with no ticker mapping (static table + OpenFIGI resolver both missed):")
        for isin in unmapped:
            print(f"  {isin}")

    if skipped:
        print(f"\n{len(skipped)} skipped row(s):")
        for row in skipped:
            print(f"  {row}")

    print("\nLogged 'portfolio_ingested' to telemetry.jsonl")


if __name__ == "__main__":
    main()
