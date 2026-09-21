from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src import db_controller
from src.parsers import fx, isin_resolver
from src.parsers.base_parser import _parse_decimal, parse_degiro_csv
from src.parsers.pii_scrubber import scrub_row

FIXTURE = Path(__file__).parent / "fixtures" / "degiro_sample.csv"


@pytest.fixture(autouse=True)
def _stub_isin_resolver(monkeypatch):
    """Without this, any fixture ISIN missing from the static ISIN_TO_TICKER
    table would trigger a real OpenFIGI/yfinance network call during tests."""
    monkeypatch.setattr(isin_resolver, "resolve_isins", lambda isin_currency_map: {})

ORDER_IDS = [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
]


@pytest.fixture
def parsed(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", telemetry_path)
    portfolio = parse_degiro_csv(FIXTURE)
    return portfolio, telemetry_path


def test_parses_european_decimal_strings():
    assert _parse_decimal("1743,30") == Decimal("1743.30")
    assert _parse_decimal("-58,1100") == Decimal("-58.1100")
    assert _parse_decimal("") is None
    assert _parse_decimal(None) is None


def test_valid_transactions_are_extracted(parsed):
    portfolio, _ = parsed
    assert len(portfolio["transactions"]) == 5
    assert len(portfolio["skipped_rows"]) == 1


def test_corporate_action_is_tagged_not_dropped(parsed):
    portfolio, _ = parsed
    rights_issue = next(
        t for t in portfolio["transactions"]
        if t.isin == "NL0000000001" and t.quantity == 20
    )
    assert rights_issue.is_corporate_action is True


def test_ordinary_trade_is_not_a_corporate_action(parsed):
    portfolio, _ = parsed
    apple = next(t for t in portfolio["transactions"] if t.isin == "US0378331005")
    assert apple.is_corporate_action is False


def test_malformed_continuation_row_is_skipped_not_raised(parsed):
    portfolio, _ = parsed
    assert len(portfolio["skipped_rows"]) == 1
    assert "CONTINUED ETF NAME" in str(portfolio["skipped_rows"][0])


def test_known_isin_resolves_ticker(parsed):
    portfolio, _ = parsed
    apple = next(t for t in portfolio["transactions"] if t.isin == "US0378331005")
    assert apple.ticker == "AAPL"


def test_unknown_isin_leaves_ticker_none(parsed):
    portfolio, _ = parsed
    unmapped = next(t for t in portfolio["transactions"] if t.isin == "US0000000002")
    assert unmapped.ticker is None


def test_scrub_row_strips_order_id_and_similar_fields():
    raw_row = {
        "Date": "01-01-2024",
        "Order ID": "secret-uuid-1234",
        "ISIN": "US0000000000",
        "Account Number": "NL01ABCD1234567890",
    }
    scrubbed = scrub_row(raw_row)
    assert "Order ID" not in scrubbed
    assert "Account Number" not in scrubbed
    assert scrubbed == {"Date": "01-01-2024", "ISIN": "US0000000000"}


def test_order_id_never_reaches_portfolio_state(parsed):
    portfolio, _ = parsed
    serialized = json.dumps(
        [t.model_dump(mode="json") for t in portfolio["transactions"]]
        + portfolio["skipped_rows"],
        default=str,
    )
    for order_id in ORDER_IDS:
        assert order_id not in serialized


def test_portfolio_ingested_event_fires_once(parsed):
    _, telemetry_path = parsed
    events = db_controller.read_telemetry_events(db_path=telemetry_path)
    ingested = [e for e in events if e["event"] == "portfolio_ingested"]
    assert len(ingested) == 1
    assert ingested[0]["transaction_count"] == 5
    assert ingested[0]["skipped_row_count"] == 1


def test_fx_cross_check_matches_embedded_rate(monkeypatch):
    fake_history = pd.DataFrame({"Close": [1.1105]})

    class FakeTicker:
        def __init__(self, pair: str) -> None:
            self.pair = pair

        def history(self, start: str, end: str) -> pd.DataFrame:
            return fake_history

    class FakeYf:
        Ticker = FakeTicker

    monkeypatch.setattr(fx, "yf", FakeYf())

    rate = fx.get_historical_fx_rate("EURUSD=X", date(2024, 1, 20))
    degiro_embedded_rate = Decimal("1.1100")
    assert abs(rate - degiro_embedded_rate) / degiro_embedded_rate < Decimal("0.01")
