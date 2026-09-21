from __future__ import annotations

import pandas as pd
import pytest

from src.parsers import isin_resolver
from src.parsers.base_parser import INTERNAL_COLUMNS, _build_transactions


@pytest.fixture(autouse=True)
def _telemetry_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", tmp_path / "test.db")


def _row(isin: str, currency: str = "EUR", **overrides) -> dict:
    base = {
        "date": "01-01-2024",
        "time": "10:00",
        "product_name": "TEST PRODUCT",
        "isin": isin,
        "reference_exchange": "",
        "venue": "",
        "quantity": "1",
        "price_local": "10,00",
        "price_currency": currency,
        "local_value": "10,00",
        "local_currency": currency,
        "value_eur": "10,00",
        "exchange_rate": "1,00",
        "autofx_fee": "",
        "fees_eur": "0,00",
        "total_eur": "10,00",
        "order_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    }
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=INTERNAL_COLUMNS)


def test_build_transactions_batches_missing_isins_in_one_call(monkeypatch):
    calls: list[dict] = []

    def spy(isin_currency_map):
        calls.append(dict(isin_currency_map))
        return {}

    monkeypatch.setattr(isin_resolver, "resolve_isins", spy)

    df = _df(
        [
            _row("US0378331005"),  # in the static table (AAPL) — must not reach the resolver
            _row("NL0010273215", "EUR"),  # missing — must reach the resolver
            _row("NL0010273215", "EUR"),  # duplicate of the same missing ISIN — deduped
        ]
    )

    _build_transactions(df, broker="DEGIRO")

    assert len(calls) == 1
    assert calls[0] == {"NL0010273215": "EUR"}


def test_resolver_result_is_merged_into_transactions(monkeypatch):
    monkeypatch.setattr(
        isin_resolver, "resolve_isins", lambda isin_currency_map: {"NL0010273215": "ASML.AS"}
    )

    df = _df([_row("NL0010273215", "EUR")])
    transactions, _ = _build_transactions(df, broker="DEGIRO")

    assert transactions[0].ticker == "ASML.AS"


def test_resolver_not_called_when_every_isin_is_in_static_table(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(isin_resolver, "resolve_isins", lambda m: calls.append(m) or {})

    df = _df([_row("US0378331005")])
    _build_transactions(df, broker="DEGIRO")

    assert calls == []
