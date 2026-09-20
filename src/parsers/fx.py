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
