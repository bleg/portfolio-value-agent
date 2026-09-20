"""Global state definitions for the portfolio-value-agent LangGraph pipeline.

Epic 1 only declares the fields it owns (raw parsed transactions). Later
epics (holdings/cost-basis aggregation, benchmark comparison, risk report,
HITL flags) extend `PortfolioState` as they're built — see CLAUDE.md Section 7
on working one epic at a time.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import TypedDict

from pydantic import BaseModel


class Transaction(BaseModel):
    """A single, PII-scrubbed, broker-agnostic transaction row."""

    isin: str
    ticker: str | None
    product_name: str
    broker: str
    trade_date: date
    trade_time: time | None
    quantity: int
    price_local: Decimal
    price_currency: str
    local_value: Decimal
    value_eur: Decimal
    exchange_rate: Decimal | None
    fees_eur: Decimal
    total_eur: Decimal
    is_corporate_action: bool = False


class PortfolioState(TypedDict, total=False):
    transactions: list[Transaction]
    broker: str
    base_currency: str
    skipped_rows: list[dict]
