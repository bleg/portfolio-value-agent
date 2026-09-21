from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src import db_controller, mcp_server
from src.agents import quant_agent
from src.agents.quant_agent import InsufficientHoldingsError, run_quant_agent
from src.parsers import fx
from src.resilience import McpToolError
from src.state import Transaction

# --- shared fixtures ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    return db_controller.read_telemetry_events(db_path=telemetry_path)


def _history_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))


class _FakeYfModule:
    def __init__(self, ticker_cls) -> None:
        self.Ticker = ticker_cls


def _make_fake_yf(ticker_data: dict[str, dict]) -> _FakeYfModule:
    """ticker_data[symbol] = {"info": {...}, "history": <DataFrame>}"""

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            canned = ticker_data[symbol]
            self.info = canned.get("info", {})
            self.cashflow = pd.DataFrame()
            self._history = canned["history"]

        def history(self, period: str, timeout: float) -> pd.DataFrame:
            return self._history

    return _FakeYfModule(_Ticker)


def _raising_fake_yf() -> _FakeYfModule:
    class _Ticker:
        def __init__(self, symbol: str) -> None:
            raise AssertionError(f"FX should not be fetched for an EUR-only portfolio ({symbol})")

    return _FakeYfModule(_Ticker)


def _txn(
    *,
    isin: str,
    ticker: str | None,
    trade_date: date,
    quantity: int,
    price_currency: str,
    total_eur: str,
    exchange_rate: str | None = None,
    fees_eur: str = "0",
    is_corporate_action: bool = False,
) -> Transaction:
    """Only quantity/total_eur/price_currency/trade_date/ticker/isin are read
    by quant_agent; other Decimal fields are harmless realistic placeholders."""
    return Transaction(
        isin=isin,
        ticker=ticker,
        product_name=f"{ticker or isin} product",
        broker="DEGIRO",
        trade_date=trade_date,
        trade_time=time(10, 0),
        quantity=quantity,
        price_local=Decimal(total_eur).copy_abs(),
        price_currency=price_currency,
        local_value=Decimal(total_eur),
        value_eur=Decimal(total_eur),
        exchange_rate=Decimal(exchange_rate) if exchange_rate else None,
        fees_eur=Decimal(fees_eur),
        total_eur=Decimal(total_eur),
        is_corporate_action=is_corporate_action,
    )


# --- Portfolio 1: simple buy-and-hold, EUR-only, no FX ----------------------


def test_buy_and_hold_eur_only(monkeypatch, telemetry_path):
    transactions = [
        _txn(
            isin="TEST0000AAA1",
            ticker="AAA",
            trade_date=date(2024, 1, 1),
            quantity=10,
            price_currency="EUR",
            total_eur="-1010.00",
            fees_eur="10.00",
        )
    ]
    fake_yf = _make_fake_yf(
        {
            "AAA": {
                "info": {
                    "regularMarketPrice": 120.0,
                    "trailingPE": 15.0,
                    "priceToBook": 3.0,
                    "debtToEquity": 50.0,
                    "freeCashflow": 5_000_000,
                    "marketCap": 100_000_000,
                },
                "history": _history_df(["2024-01-01", "2024-07-01"], [100.0, 120.0]),
            },
            "^GSPC": {
                "history": _history_df(["2024-01-01", "2024-07-01"], [4000.0, 4200.0]),
            },
        }
    )
    monkeypatch.setattr(mcp_server, "yf", fake_yf)
    monkeypatch.setattr(fx, "yf", _raising_fake_yf())

    state = run_quant_agent({"transactions": transactions}, as_of=date(2024, 7, 1))
    metrics = state["quant_metrics"]

    assert metrics.total_cost_basis_eur == Decimal("1010.00")
    holding = metrics.holdings[0]
    assert holding.ticker == "AAA"
    assert holding.avg_cost_basis_eur == Decimal("101.00")
    assert holding.current_price_eur == Decimal("120.0")
    assert holding.market_value_eur == Decimal("1200.0")
    assert holding.unrealized_return_pct == pytest.approx(190 / 1010)
    assert metrics.net_return_pct == pytest.approx(190 / 1010)
    assert metrics.weighted_pe == pytest.approx(15.0)
    assert metrics.twr_pct == pytest.approx(0.20)
    assert metrics.benchmark_return_pct == pytest.approx(0.05)
    assert metrics.unresolved_isins == []

    assert metrics.value_history[0].portfolio_index == pytest.approx(100.0)
    assert metrics.value_history[0].benchmark_index == pytest.approx(100.0)
    assert metrics.value_history[-1].portfolio_index / 100 - 1 == pytest.approx(metrics.twr_pct)
    assert metrics.value_history[-1].benchmark_index == pytest.approx(105.0)

    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert failures == []
    compared = [e for e in _events(telemetry_path) if e["event"] == "benchmark_compared"]
    assert len(compared) == 1
    assert compared[0]["twr"] == pytest.approx(0.20)
    assert compared[0]["benchmark_return"] == pytest.approx(0.05)


# --- Portfolio 2: mid-period buy/sell across a currency (TWR edge case) ----


def test_mid_period_buy_sell_across_currency(monkeypatch, telemetry_path):
    transactions = [
        _txn(
            isin="TEST0000BBB1",
            ticker="BBB",
            trade_date=date(2024, 1, 1),
            quantity=20,
            price_currency="USD",
            total_eur="-800.00",
            exchange_rate="1.25",
        ),
        _txn(
            isin="TEST0000BBB1",
            ticker="BBB",
            trade_date=date(2024, 4, 1),
            quantity=-8,
            price_currency="USD",
            total_eur="384.00",
            exchange_rate="1.25",
        ),
    ]
    fake_yf = _make_fake_yf(
        {
            "BBB": {
                "info": {
                    "regularMarketPrice": 70.0,
                    "trailingPE": 22.5,
                },
                "history": _history_df(
                    ["2024-01-01", "2024-04-01", "2024-07-01"], [50.0, 60.0, 70.0]
                ),
            },
            "^GSPC": {
                "history": _history_df(["2024-01-01", "2024-07-01"], [4000.0, 4400.0]),
            },
        }
    )
    fake_fx = _make_fake_yf(
        {
            "EURUSD=X": {
                "history": _history_df(
                    ["2024-01-01", "2024-04-01", "2024-07-01"], [1.25, 1.25, 1.25]
                ),
            },
        }
    )
    monkeypatch.setattr(mcp_server, "yf", fake_yf)
    monkeypatch.setattr(fx, "yf", fake_fx)

    state = run_quant_agent({"transactions": transactions}, as_of=date(2024, 7, 1))
    metrics = state["quant_metrics"]

    holding = metrics.holdings[0]
    assert holding.quantity == 12
    assert holding.avg_cost_basis_eur == Decimal("40.00")
    assert holding.total_cost_basis_eur == Decimal("480.00")
    assert holding.current_price_eur == Decimal("56.00")
    assert holding.market_value_eur == Decimal("672.00")
    assert holding.unrealized_return_pct == pytest.approx(0.40)

    assert metrics.net_return_pct == pytest.approx(0.32)
    assert metrics.twr_pct == pytest.approx(0.40)
    assert metrics.benchmark_return_pct == pytest.approx(0.10)
    # TWR (chain-linked, accounts for mid-period sell timing) genuinely
    # differs from net return (a flat point-to-point measure) — that
    # divergence is exactly what this test case is meant to catch.
    assert metrics.twr_pct != pytest.approx(metrics.net_return_pct)


# --- guardrails and edge cases ----------------------------------------------


def test_oversell_raises_insufficient_holdings_error():
    ticker_txns = {
        "CCC": [
            _txn(
                isin="TEST0000CCC1",
                ticker="CCC",
                trade_date=date(2024, 1, 1),
                quantity=5,
                price_currency="EUR",
                total_eur="-500.00",
            ),
            _txn(
                isin="TEST0000CCC1",
                ticker="CCC",
                trade_date=date(2024, 2, 1),
                quantity=-10,
                price_currency="EUR",
                total_eur="900.00",
            ),
        ]
    }
    with pytest.raises(InsufficientHoldingsError):
        quant_agent._replay_positions(ticker_txns)


def test_mcp_tool_failure_propagates_as_mcp_tool_error(monkeypatch, telemetry_path):
    transactions = [
        _txn(
            isin="TEST0000FAIL1",
            ticker="FAIL",
            trade_date=date(2024, 1, 1),
            quantity=1,
            price_currency="EUR",
            total_eur="-100.00",
        )
    ]

    class _EmptyTicker:
        def __init__(self, symbol: str) -> None:
            self.info: dict = {}
            self.cashflow = pd.DataFrame()

        def history(self, period: str, timeout: float) -> pd.DataFrame:
            return pd.DataFrame({"Close": []})

    monkeypatch.setattr(mcp_server, "yf", _FakeYfModule(_EmptyTicker))

    with pytest.raises(McpToolError):
        run_quant_agent({"transactions": transactions}, as_of=date(2024, 7, 1))

    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["ticker"] == "FAIL"
    # No benchmark_compared event on failure.
    assert [e for e in _events(telemetry_path) if e["event"] == "benchmark_compared"] == []


def test_unresolved_ticker_excluded_but_reported(monkeypatch, telemetry_path):
    transactions = [
        _txn(
            isin="TEST0000UNK1",
            ticker=None,
            trade_date=date(2024, 1, 1),
            quantity=5,
            price_currency="EUR",
            total_eur="-500.00",
        )
    ]
    fake_yf = _make_fake_yf(
        {"^GSPC": {"history": _history_df(["2024-01-01", "2024-07-01"], [4000.0, 4200.0])}}
    )
    monkeypatch.setattr(mcp_server, "yf", fake_yf)
    monkeypatch.setattr(fx, "yf", _raising_fake_yf())

    state = run_quant_agent({"transactions": transactions}, as_of=date(2024, 7, 1))
    metrics = state["quant_metrics"]

    assert metrics.unresolved_isins == ["TEST0000UNK1"]
    assert metrics.holdings == []
    assert metrics.weighted_pe is None
    assert metrics.net_return_pct == 0.0
    assert metrics.twr_pct == 0.0


def test_value_on_or_before_forward_fills_and_rejects_missing_data():
    series = [{"date": "2024-01-01", "close": 100.0}, {"date": "2024-01-05", "close": 110.0}]

    assert quant_agent._value_on_or_before(series, date(2024, 1, 3)) == Decimal("100.0")
    assert quant_agent._value_on_or_before(series, date(2024, 1, 5)) == Decimal("110.0")
    with pytest.raises(ValueError):
        quant_agent._value_on_or_before(series, date(2023, 12, 31))
