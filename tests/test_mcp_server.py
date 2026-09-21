from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import db_controller, mcp_server
from src.resilience import MAX_ATTEMPTS, McpToolError, resilient_tool

# --- shared fixtures -------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def _reset_cik_cache(monkeypatch):
    monkeypatch.setattr(mcp_server, "_cik_cache", None)


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    return db_controller.read_telemetry_events(db_path=telemetry_path)


class FakeInfoTicker:
    """Fakes `yf.Ticker` for a well-formed, fully-populated ticker."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.info = {
            "regularMarketPrice": 150.0,
            "shortName": "Apple Inc.",
            "trailingPE": 38.59,
            "priceToBook": 45.67,
            "debtToEquity": 78.44,
            "freeCashflow": 100_000_000,
            "marketCap": 4_000_000_000,
        }
        self.cashflow = pd.DataFrame()

    def history(self, period: str, timeout: float) -> pd.DataFrame:
        idx = pd.date_range("2026-09-01", periods=3, freq="D")
        return pd.DataFrame({"Close": [100.0, 101.5, 102.25]}, index=idx)


class FakeYfModule:
    def __init__(self, ticker_cls) -> None:
        self.Ticker = ticker_cls


# --- yfinance_fundamentals ---------------------------------------------------


def test_yfinance_fundamentals_returns_expected_shape(monkeypatch, telemetry_path):
    monkeypatch.setattr(mcp_server, "yf", FakeYfModule(FakeInfoTicker))

    result = mcp_server.yfinance_fundamentals("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["pe_ratio"] == 38.59
    assert result["pb_ratio"] == 45.67
    assert result["debt_to_equity"] == 78.44
    assert result["fcf_yield"] == pytest.approx(100_000_000 / 4_000_000_000)
    assert result["price_history"] == [
        {"date": "2026-09-01", "close": 100.0},
        {"date": "2026-09-02", "close": 101.5},
        {"date": "2026-09-03", "close": 102.25},
    ]
    assert "as_of" in result
    assert _events(telemetry_path) == []


def test_yfinance_fundamentals_bad_ticker_retries_and_logs_failure(monkeypatch, telemetry_path):
    call_count = 0

    class EmptyInfoTicker:
        def __init__(self, ticker: str) -> None:
            nonlocal call_count
            call_count += 1
            self.info = {}
            self.cashflow = pd.DataFrame()

        def history(self, period: str, timeout: float) -> pd.DataFrame:
            return pd.DataFrame({"Close": []})

    monkeypatch.setattr(mcp_server, "yf", FakeYfModule(EmptyInfoTicker))

    with pytest.raises(McpToolError):
        mcp_server.yfinance_fundamentals("BADTICKER")

    assert call_count == MAX_ATTEMPTS
    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "yfinance_fundamentals"
    assert failures[0]["ticker"] == "BADTICKER"


def test_yfinance_fundamentals_degrades_when_info_is_crumb_blocked(monkeypatch, telemetry_path):
    """A real ticker whose `.info` call is rate-limited (Yahoo's crumb auth,
    routinely exhausted on shared cloud hosts) returns empty `.info` even
    though the ticker is valid and `.history()` still resolves - distinct
    from a genuinely bad ticker, where both calls come back empty."""

    class CrumbBlockedTicker:
        def __init__(self, ticker: str) -> None:
            self.info = {}
            self.cashflow = pd.DataFrame()

        def history(self, period: str, timeout: float) -> pd.DataFrame:
            idx = pd.date_range("2026-09-01", periods=2, freq="D")
            return pd.DataFrame({"Close": [100.0, 101.5]}, index=idx)

    monkeypatch.setattr(mcp_server, "yf", FakeYfModule(CrumbBlockedTicker))

    result = mcp_server.yfinance_fundamentals("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["pe_ratio"] is None
    assert result["pb_ratio"] is None
    assert result["debt_to_equity"] is None
    assert result["fcf_yield"] is None
    assert result["price_history"] == [
        {"date": "2026-09-01", "close": 100.0},
        {"date": "2026-09-02", "close": 101.5},
    ]
    assert _events(telemetry_path) == []


# --- benchmark_data_fetcher --------------------------------------------------


def test_benchmark_data_fetcher_returns_price_series(monkeypatch, telemetry_path):
    monkeypatch.setattr(mcp_server, "yf", FakeYfModule(FakeInfoTicker))

    result = mcp_server.benchmark_data_fetcher("^GSPC")

    assert result["ticker"] == "^GSPC"
    assert len(result["price_history"]) == 3
    assert result["price_history"][0] == {"date": "2026-09-01", "close": 100.0}
    assert _events(telemetry_path) == []


def test_benchmark_data_fetcher_network_error_retries_and_logs_failure(
    monkeypatch, telemetry_path
):
    call_count = 0

    class FailingTicker:
        def __init__(self, ticker: str) -> None:
            pass

        def history(self, period: str, timeout: float) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("simulated network failure")

    monkeypatch.setattr(mcp_server, "yf", FakeYfModule(FailingTicker))

    with pytest.raises(McpToolError):
        mcp_server.benchmark_data_fetcher("^GSPC")

    assert call_count == MAX_ATTEMPTS
    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "benchmark_data_fetcher"
    assert failures[0]["ticker"] == "^GSPC"


# --- sec_edgar_lookup ---------------------------------------------------------

TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "10-K", "4"],
            "filingDate": ["2026-08-01", "2026-07-15", "2025-11-01", "2026-07-01"],
            "accessionNumber": [
                "0000320193-26-000079",
                "0000320193-26-000070",
                "0000320193-25-000100",
                "0000320193-26-000060",
            ],
            "primaryDocDescription": ["10-Q", "8-K", "10-K", "FORM 4"],
        }
    }
}


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "portfolio-value-agent test@example.com")


def test_sec_edgar_lookup_returns_filing_metadata(monkeypatch, telemetry_path):
    def fake_get(url, headers, timeout):
        assert headers["User-Agent"] == "portfolio-value-agent test@example.com"
        if url == mcp_server.SEC_TICKERS_URL:
            return FakeResponse(TICKERS_PAYLOAD)
        return FakeResponse(SUBMISSIONS_PAYLOAD)

    monkeypatch.setattr(mcp_server.requests, "get", fake_get)

    result = mcp_server.sec_edgar_lookup("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["cik"] == "0000320193"
    assert result["filings"] == [
        {
            "form": "10-Q",
            "filing_date": "2026-08-01",
            "accession_number": "0000320193-26-000079",
            "primary_doc_description": "10-Q",
        },
        {
            "form": "10-K",
            "filing_date": "2025-11-01",
            "accession_number": "0000320193-25-000100",
            "primary_doc_description": "10-K",
        },
    ]
    assert _events(telemetry_path) == []


def test_sec_edgar_lookup_forced_network_error_retries_and_logs_failure(
    monkeypatch, telemetry_path
):
    call_count = 0

    def fake_get(url, headers, timeout):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(mcp_server.requests, "get", fake_get)

    with pytest.raises(McpToolError):
        mcp_server.sec_edgar_lookup("AAPL")

    assert call_count == MAX_ATTEMPTS
    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "sec_edgar_lookup"
    assert failures[0]["ticker"] == "AAPL"


def test_sec_edgar_lookup_missing_user_agent_fails_immediately_without_retry(
    monkeypatch, telemetry_path
):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    call_count = 0

    def fake_get(url, headers, timeout):
        nonlocal call_count
        call_count += 1
        return FakeResponse(TICKERS_PAYLOAD)

    monkeypatch.setattr(mcp_server.requests, "get", fake_get)

    with pytest.raises(RuntimeError):
        mcp_server.sec_edgar_lookup("AAPL")

    assert call_count == 0
    assert _events(telemetry_path) == []


# --- duckduckgo_search ---------------------------------------------------------

NEWS_RESULTS = [
    {
        "date": "2026-09-18T00:00:00+00:00",
        "title": "Apple supply chain update",
        "body": "Some snippet text.",
        "url": "https://example.com/article",
        "source": "Example News",
    }
]


class FakeDDGS:
    news_call_count = 0
    raise_error = False

    def __init__(self, timeout: float) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def news(self, query: str, max_results: int):
        type(self).news_call_count += 1
        if type(self).raise_error:
            raise RuntimeError("simulated ddgs failure")
        return NEWS_RESULTS


@pytest.fixture(autouse=True)
def _reset_fake_ddgs():
    FakeDDGS.news_call_count = 0
    FakeDDGS.raise_error = False


def test_duckduckgo_search_returns_results(monkeypatch, telemetry_path):
    monkeypatch.setattr(mcp_server, "DDGS", FakeDDGS)

    result = mcp_server.duckduckgo_search("AAPL supply chain", ticker="AAPL")

    assert result["query"] == "AAPL supply chain"
    assert result["results"] == [
        {
            "title": "Apple supply chain update",
            "url": "https://example.com/article",
            "source": "Example News",
            "date": "2026-09-18T00:00:00+00:00",
            "snippet": "Some snippet text.",
        }
    ]
    assert _events(telemetry_path) == []


def test_duckduckgo_search_forced_network_error_retries_and_logs_failure(
    monkeypatch, telemetry_path
):
    FakeDDGS.raise_error = True
    monkeypatch.setattr(mcp_server, "DDGS", FakeDDGS)

    with pytest.raises(McpToolError):
        mcp_server.duckduckgo_search("AAPL supply chain", ticker="AAPL")

    assert FakeDDGS.news_call_count == MAX_ATTEMPTS
    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "duckduckgo_search"
    assert failures[0]["ticker"] == "AAPL"


# --- resilience decorator, in isolation ---------------------------------------


def test_resilient_tool_fires_telemetry_exactly_once_on_exhaustion(monkeypatch, telemetry_path):
    call_count = 0

    @resilient_tool(tool_name="dummy_tool")
    def always_fails(ticker: str) -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("boom")

    with pytest.raises(McpToolError):
        always_fails("XYZ")

    assert call_count == MAX_ATTEMPTS
    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "dummy_tool"
    assert failures[0]["ticker"] == "XYZ"
    assert failures[0]["attempts"] == MAX_ATTEMPTS
