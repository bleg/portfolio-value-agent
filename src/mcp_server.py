"""MCP Tool Server (CLAUDE.md Section 3B).

Exposes the data-fetching layer as MCP tools, decoupled from the reasoning
engine (the agents built in later epics). Every tool is wrapped in
`resilience.resilient_tool` for retry-with-backoff + a hard timeout; on
exhaustion the wrapper fires `mcp_tool_failure` telemetry and raises
`McpToolError`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf
from ddgs import DDGS
from mcp.server.mcpserver import MCPServer

if __name__ == "__main__":
    # Allow `python src/mcp_server.py` to resolve the `src` package, matching
    # the sys.path bootstrap already used by scripts/run_epic1_demo.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db_controller
from src.resilience import resilient_tool

mcp = MCPServer("portfolio-value-agent")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_cik_cache: dict[str, int] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _price_history(t, period: str, timeout: float) -> list[dict]:
    history = t.history(period=period, timeout=timeout)
    return [
        {"date": index.date().isoformat(), "close": round(float(row["Close"]), 4)}
        for index, row in history.iterrows()
    ]


@mcp.tool()
@resilient_tool(tool_name="yfinance_fundamentals")
def yfinance_fundamentals(ticker: str, history_period: str = "6mo") -> dict:
    """P/E, P/B, Debt-to-Equity, FCF Yield, and historical close prices for `ticker`.

    `.info` and `.history()` hit separate Yahoo endpoints with different auth
    requirements: `.info` needs a "crumb" token that Yahoo rate-limits per-IP
    (routinely exhausted on shared cloud hosts like Streamlit Community
    Cloud), while the `/v8/finance/chart` endpoint behind `.history()` does
    not. So `.info` failing doesn't mean the ticker is invalid - only that
    fundamentals are temporarily unavailable. Degrade to `None` fundamentals
    rather than failing the whole call whenever price history still resolves;
    `quant_agent.py` prices holdings off `price_history` already (not
    `.info`), and `_weighted_pe`/`risk_agent.detect_anomalies` already treat
    `None` fields as "skip this holding for that metric".
    """
    t = yf.Ticker(ticker)
    price_history = _price_history(t, history_period, timeout=5.0)

    try:
        info = t.info
    except Exception:
        info = {}
    has_fundamentals = bool(info.get("regularMarketPrice") or info.get("shortName"))

    if not has_fundamentals and not price_history:
        raise ValueError(f"No data available for ticker {ticker!r}")

    fcf_yield = None
    if has_fundamentals:
        market_cap = info.get("marketCap")
        free_cash_flow = info.get("freeCashflow")
        if free_cash_flow is None:
            try:
                free_cash_flow = float(t.cashflow.loc["Free Cash Flow"].iloc[0])
            except Exception:
                free_cash_flow = None
        if free_cash_flow is not None and market_cap:
            fcf_yield = round(float(free_cash_flow) / float(market_cap), 4)

    return {
        "ticker": ticker,
        "pe_ratio": info.get("trailingPE") if has_fundamentals else None,
        "pb_ratio": info.get("priceToBook") if has_fundamentals else None,
        "debt_to_equity": info.get("debtToEquity") if has_fundamentals else None,
        "fcf_yield": fcf_yield,
        "price_history": price_history,
        "as_of": _now_iso(),
    }


@mcp.tool()
@resilient_tool(tool_name="benchmark_data_fetcher")
def benchmark_data_fetcher(ticker: str = "^GSPC", history_period: str = "1y") -> dict:
    """Historical price series for a benchmark index (S&P 500 `^GSPC`, MSCI World `URTH`)."""
    price_history = _price_history(yf.Ticker(ticker), history_period, timeout=5.0)
    if not price_history:
        raise ValueError(f"No benchmark data available for {ticker!r}")
    return {
        "ticker": ticker,
        "price_history": price_history,
        "as_of": _now_iso(),
    }


def _sec_headers() -> dict[str, str]:
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT is not set. SEC EDGAR's fair-access policy requires a "
            "compliant User-Agent (name + contact email) — set it in .env before calling "
            "sec_edgar_lookup."
        )
    return {"User-Agent": user_agent}


def _resolve_cik(ticker: str, headers: dict[str, str]) -> str:
    global _cik_cache
    if _cik_cache is None:
        response = requests.get(SEC_TICKERS_URL, headers=headers, timeout=5.0)
        response.raise_for_status()
        _cik_cache = {
            entry["ticker"].upper(): entry["cik_str"] for entry in response.json().values()
        }
    cik = _cik_cache.get(ticker.upper())
    if cik is None:
        raise ValueError(f"No SEC CIK found for ticker {ticker!r}")
    return f"{cik:010d}"


@resilient_tool(tool_name="sec_edgar_lookup")
def _fetch_sec_filings(
    ticker: str,
    filing_types: tuple[str, ...],
    limit: int,
    headers: dict[str, str],
) -> dict:
    cik = _resolve_cik(ticker, headers)
    response = requests.get(
        SEC_SUBMISSIONS_URL.format(cik=int(cik)), headers=headers, timeout=5.0
    )
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]

    filings = []
    for form, filing_date, accession_number, primary_doc_description in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent["primaryDocDescription"],
    ):
        if form in filing_types:
            filings.append(
                {
                    "form": form,
                    "filing_date": filing_date,
                    "accession_number": accession_number,
                    "primary_doc_description": primary_doc_description,
                }
            )
        if len(filings) >= limit:
            break

    return {"ticker": ticker, "cik": cik, "filings": filings}


@mcp.tool()
def sec_edgar_lookup(
    ticker: str,
    filing_types: tuple[str, ...] = ("10-K", "10-Q"),
    limit: int = 5,
) -> dict:
    """Recent 10-K/10-Q filing metadata for `ticker` (no LLM summarization — Epic 5's job).

    The User-Agent config check runs outside the retry wrapper: a missing
    `SEC_EDGAR_USER_AGENT` is a config error, not a transient failure, so it
    fails immediately instead of burning retry attempts.
    """
    headers = _sec_headers()
    return _fetch_sec_filings(ticker, filing_types, limit, headers)


@mcp.tool()
@resilient_tool(tool_name="duckduckgo_search")
def duckduckgo_search(query: str, ticker: str | None = None, max_results: int = 5) -> dict:
    """Recent qualitative news for `query` (e.g. a flagged ticker's risk anomaly)."""
    with DDGS(timeout=5.0) as ddgs:
        raw_results = ddgs.news(query, max_results=max_results)

    results = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "source": r.get("source"),
            "date": r.get("date"),
            "snippet": r.get("body"),
        }
        for r in raw_results
    ]
    return {"query": query, "results": results}


@mcp.tool()
@resilient_tool(tool_name="historical_db_read")
def historical_db_read(user_id: str, limit: int = 5) -> dict:
    """Scoped, parameterized read of past audit runs for `user_id` (no LLM SQL generation)."""
    return {"user_id": user_id, "audits": db_controller.read_audits_for_user(user_id, limit)}


if __name__ == "__main__":
    mcp.run()
