from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import db_controller
from src.parsers import isin_resolver

# --- shared fixtures ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", path)
    return path


@pytest.fixture
def cache_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "isin_cache.json"
    monkeypatch.setattr(isin_resolver, "DEFAULT_CACHE_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    return db_controller.read_telemetry_events(db_path=telemetry_path)


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeRequests:
    """Fakes the `requests` module's `.post` for OpenFIGI calls."""

    def __init__(self, responder) -> None:
        self.calls: list[dict] = []
        self._responder = responder

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self._responder(json, headers)


class _FakeTickerInstance:
    def __init__(self, ticker: str, verifies_by_ticker: dict[str, bool]) -> None:
        self.ticker = ticker
        self._verifies_by_ticker = verifies_by_ticker

    def history(self, period: str, timeout: float) -> pd.DataFrame:
        if self._verifies_by_ticker.get(self.ticker):
            return pd.DataFrame({"Close": [100.0, 101.0]})
        return pd.DataFrame({"Close": []})


class FakeYfModule:
    """Fakes `yf.Ticker(...).history(...)` for verify-before-cache checks."""

    def __init__(self, verifies_by_ticker: dict[str, bool]) -> None:
        self._verifies_by_ticker = verifies_by_ticker

    def Ticker(self, ticker: str) -> _FakeTickerInstance:
        return _FakeTickerInstance(ticker, self._verifies_by_ticker)


# --- cache hit ----------------------------------------------------------------


def test_cache_hit_skips_network_call(monkeypatch, telemetry_path, cache_path):
    cache_path.write_text(
        json.dumps(
            {"US0378331005": {"ticker": "AAPL", "source": "openfigi", "resolved_at": "2026-01-01T00:00:00+00:00"}}
        )
    )
    fake_requests = FakeRequests(lambda jobs, headers: FakeResponse([]))
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)

    result = isin_resolver.resolve_isins({"US0378331005": "USD"})

    assert result == {"US0378331005": "AAPL"}
    assert fake_requests.calls == []
    assert _events(telemetry_path) == []


# --- cache miss + success ------------------------------------------------------


def test_cache_miss_success_resolves_and_caches(monkeypatch, telemetry_path, cache_path):
    def responder(jobs, headers):
        assert jobs == [{"idType": "ID_ISIN", "idValue": "NL0010273215"}]
        return FakeResponse(
            [{"data": [{"ticker": "ASML", "exchCode": "NA"}, {"ticker": "ASME", "exchCode": "GY"}]}]
        )

    fake_requests = FakeRequests(responder)
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)
    monkeypatch.setattr(isin_resolver, "yf", FakeYfModule({"ASML.AS": True, "ASME.DE": True}))

    result = isin_resolver.resolve_isins({"NL0010273215": "EUR"})

    assert result == {"NL0010273215": "ASML.AS"}
    assert len(fake_requests.calls) == 1

    cache = json.loads(cache_path.read_text())
    assert cache["NL0010273215"]["ticker"] == "ASML.AS"
    assert cache["NL0010273215"]["source"] == "openfigi"

    events = [e for e in _events(telemetry_path) if e["event"] == "isin_resolved"]
    assert len(events) == 1
    assert events[0]["isins_resolved"] == 1
    assert events[0]["isins_queried"] == 1
    assert events[0]["isins_still_unresolved"] == 0


# --- cache miss + OpenFIGI failure ---------------------------------------------


def test_cache_miss_openfigi_failure_leaves_unresolved_and_uncached(monkeypatch, telemetry_path, cache_path):
    fake_requests = FakeRequests(lambda jobs, headers: FakeResponse({"error": "boom"}, status_code=500))
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)

    result = isin_resolver.resolve_isins({"BADISIN0001": "EUR"})

    assert result == {}
    assert not cache_path.exists()

    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "openfigi_mapping"

    resolved_events = [e for e in _events(telemetry_path) if e["event"] == "isin_resolved"]
    assert len(resolved_events) == 1
    assert resolved_events[0]["isins_resolved"] == 0


# --- OpenFIGI match but yfinance verification fails ----------------------------


def test_openfigi_match_but_yfinance_verification_fails(monkeypatch, telemetry_path, cache_path):
    fake_requests = FakeRequests(
        lambda jobs, headers: FakeResponse([{"data": [{"ticker": "FAKE", "exchCode": "NA"}]}])
    )
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)
    monkeypatch.setattr(isin_resolver, "yf", FakeYfModule({}))  # nothing verifies

    result = isin_resolver.resolve_isins({"XX0000000001": "EUR"})

    assert result == {}
    assert not cache_path.exists()


# --- multi-listing: priority order, not array order ----------------------------


def test_multi_listing_isin_picks_priority_order_not_array_order(monkeypatch, telemetry_path, cache_path):
    # German cluster listed FIRST in the array, Amsterdam SECOND — both verify.
    # The resolver must still prefer Amsterdam (higher cluster priority).
    fake_requests = FakeRequests(
        lambda jobs, headers: FakeResponse(
            [{"data": [{"ticker": "ASME", "exchCode": "GY"}, {"ticker": "ASML", "exchCode": "NA"}]}]
        )
    )
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)
    monkeypatch.setattr(isin_resolver, "yf", FakeYfModule({"ASME.DE": True, "ASML.AS": True}))

    result = isin_resolver.resolve_isins({"NL0010273215": "EUR"})

    assert result == {"NL0010273215": "ASML.AS"}


# --- same-cluster, multi-currency candidates -----------------------------------


def test_same_cluster_currency_hint_picks_matching_currency(monkeypatch, telemetry_path, cache_path):
    fake_requests = FakeRequests(
        lambda jobs, headers: FakeResponse(
            [
                {
                    "data": [
                        {"ticker": "BITC", "exchCode": "SW"},
                        {"ticker": "BITCGBP", "exchCode": "SW"},
                        {"ticker": "BITCEUR", "exchCode": "SW"},
                    ]
                }
            ]
        )
    )
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)
    monkeypatch.setattr(
        isin_resolver, "yf", FakeYfModule({"BITC.SW": True, "BITCGBP.SW": True, "BITCEUR.SW": True})
    )

    result = isin_resolver.resolve_isins({"GB00BLD4ZL17": "EUR"})

    assert result == {"GB00BLD4ZL17": "BITCEUR.SW"}


def test_same_cluster_no_currency_match_picks_shortest_bare_ticker(monkeypatch, telemetry_path, cache_path):
    fake_requests = FakeRequests(
        lambda jobs, headers: FakeResponse(
            [{"data": [{"ticker": "BITCGBP", "exchCode": "SW"}, {"ticker": "BITC", "exchCode": "SW"}]}]
        )
    )
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)
    monkeypatch.setattr(isin_resolver, "yf", FakeYfModule({"BITC.SW": True, "BITCGBP.SW": True}))

    result = isin_resolver.resolve_isins({"GB00BLD4ZL17": "USD"})  # no matching currency suffix

    assert result == {"GB00BLD4ZL17": "BITC.SW"}


# --- batching -------------------------------------------------------------------


def test_batch_chunking_respects_batch_size(monkeypatch, telemetry_path, cache_path):
    monkeypatch.delenv("OPENFIGI_API_KEY", raising=False)
    isins = [f"XX00000{i:04d}" for i in range(12)]  # more than BATCH_SIZE_NO_KEY

    fake_requests = FakeRequests(
        lambda jobs, headers: FakeResponse([{"warning": "No identifier found."} for _ in jobs])
    )
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)

    isin_resolver.resolve_isins({isin: "EUR" for isin in isins})

    assert len(fake_requests.calls) == 2
    assert len(fake_requests.calls[0]["json"]) == isin_resolver.BATCH_SIZE_NO_KEY
    assert len(fake_requests.calls[1]["json"]) == 2


def test_openfigi_api_key_widens_batch_size_and_sets_header(monkeypatch, telemetry_path, cache_path):
    monkeypatch.setenv("OPENFIGI_API_KEY", "test-key-123")
    isins = [f"XX00000{i:04d}" for i in range(12)]

    def responder(jobs, headers):
        assert headers.get("X-OPENFIGI-APIKEY") == "test-key-123"
        return FakeResponse([{"warning": "No identifier found."} for _ in jobs])

    fake_requests = FakeRequests(responder)
    monkeypatch.setattr(isin_resolver, "requests", fake_requests)

    isin_resolver.resolve_isins({isin: "EUR" for isin in isins})

    assert len(fake_requests.calls) == 1  # all 12 fit in one batch of BATCH_SIZE_WITH_KEY (100)
