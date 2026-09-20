"""Local PII / internal-broker-ID scrubbing (CLAUDE.md Section 3A/3D).

Runs immediately on each parsed CSV row, before a Transaction is built, so
scrubbed values never exist in PortfolioState and are never eligible to reach
an external LLM. Only tickers and aggregate share counts should ever leave
the local process.
"""

from __future__ import annotations

# Column names (normalized: lowercased, spaces -> underscores) that must never
# survive past ingestion. "order_id" is DEGIRO's own internal transaction
# reference. Future broker adapters (IBKR/Schwab CSVs carry an account
# name/number column) register their PII column names here too.
PII_FIELDS = {
    "order_id",
    "account_name",
    "account_number",
    "account_id",
}


def _normalize(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def scrub_row(raw_row: dict[str, object]) -> dict[str, object]:
    """Strip known PII/internal-broker-ID fields from a single parsed CSV row."""
    return {k: v for k, v in raw_row.items() if _normalize(k) not in PII_FIELDS}
