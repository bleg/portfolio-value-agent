"""Historical FX rate lookup.

Not called on the DEGIRO happy path in base_parser.py — DEGIRO's own export
already reports the EUR-converted value and the exchange rate it executed at
(the historical, broker-executed rate, more accurate than re-deriving one
after the fact). This module exists for two things instead:

1. A pytest sanity check that DEGIRO's embedded rate is within tolerance of
   the market historical close for that date (see tests/test_flows.py).
2. The fallback conversion path Epic 3 will need for brokers (IBKR/Schwab)
   whose exports don't hand back a pre-converted total.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import yfinance as yf

from src.resilience import resilient_tool


class FxRateUnavailableError(Exception):
    """Raised when no historical FX data can be found for a pair/date."""


def get_historical_fx_rate(pair: str, on_date: date) -> Decimal:
    """Return the most recent close for `pair` (e.g. "EURUSD=X") on/before `on_date`."""
    start = on_date - timedelta(days=7)
    end = on_date + timedelta(days=1)
    history = yf.Ticker(pair).history(start=start.isoformat(), end=end.isoformat())
    if history.empty:
        raise FxRateUnavailableError(f"No FX data for {pair} around {on_date}")
    last_close = history["Close"].iloc[-1]
    return Decimal(str(round(float(last_close), 6)))


@resilient_tool(tool_name="fx_history")
def get_fx_history(ticker: str, period: str = "max") -> list[dict]:
    """Historical close-price series for an FX pair (e.g. "EURUSD=X").

    Used by Epic 4's quant_agent to convert multi-currency holdings to EUR at
    arbitrary historical dates (TWR sub-period breakpoints), where a single
    point-in-time lookup via `get_historical_fx_rate` isn't enough. Same
    `{"date": iso, "close": float}` shape as `mcp_server._price_history`, so
    both can share a single on-or-before lookup helper.

    The parameter is named `ticker` (not `pair`) so `resilience.resilient_tool`'s
    ticker-extraction (which inspects the signature for a `ticker` argument)
    attaches it to `mcp_tool_failure` telemetry on exhaustion.
    """
    history = yf.Ticker(ticker).history(period=period, timeout=5.0)
    if history.empty:
        raise ValueError(f"No FX history available for {ticker!r}")
    return [
        {"date": index.date().isoformat(), "close": round(float(row["Close"]), 6)}
        for index, row in history.iterrows()
    ]
