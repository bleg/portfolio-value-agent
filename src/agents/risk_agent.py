"""The Risk & Performance Analyst Agent (Epic 5): a gpt-4o ReAct agent.

Reviews Epic 4's `quant_agent.py` output. Two responsibilities, run as two
independent `create_react_agent` loops sharing the same tool set
(`sec_edgar_lookup_tool`, `duckduckgo_search_tool`), so each path's
tool-call routing stays independently testable:

1. Risk-check: `detect_anomalies` deterministically (plain Python, no LLM -
   CLAUDE.md Section 3C's "never let the LLM calculate math" applies here
   too) flags holdings whose already-computed fundamentals
   (`Holding.fcf_yield`/`debt_to_equity`/`pe_ratio` - already on
   `state["quant_metrics"].holdings` from Epic 4, never re-fetched) breach a
   threshold. One ReAct investigation runs per flagged anomaly.
2. Performance attribution: one ReAct investigation explains the portfolio's
   TWR vs. the benchmark return (also already computed by Epic 4).

Failure-propagation contract - deliberately NOT the same as quant_agent.py's:
- MCP tool failures (`resilience.McpToolError`, and `sec_edgar_lookup`'s
  config-error `RuntimeError` for a missing `SEC_EDGAR_USER_AGENT`) are
  caught *inside* the `@tool` wrappers below and turned into a text
  observation fed back into the ReAct loop. This is the "agent-node
  boundary" catch the roadmap's Epic 5 task 1 asks for, scoped to tool
  calls - it lets the agent keep reasoning instead of crashing.
- Everything else - a missing `quant_metrics` in state (`ValueError`), or a
  `ChatOpenAI`-level failure (auth/network/rate-limit) - propagates
  uncaught out of `run_risk_agent`, exactly like `quant_agent.py`. Epic 6's
  Supervisor remains the single catch boundary for agent-level failures.
"""

from __future__ import annotations

import json
from typing import Literal, NamedTuple

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src import mcp_server, telemetry
from src.resilience import McpToolError
from src.state import (
    AnomalyFinding,
    Holding,
    PerformanceAttribution,
    PortfolioState,
    QuantMetrics,
    RiskReport,
    ToolCallRecord,
)

RISK_CHECK_MODEL = "gpt-4o"
PERFORMANCE_MODEL = "gpt-4o"

# Thresholds are illustrative for this portfolio demo, not investment advice.
LOW_FCF_YIELD_THRESHOLD = 0.03
# yfinance's raw `debtToEquity` field (mcp_server.yfinance_fundamentals ->
# Holding.debt_to_equity) is already a percentage (e.g. 50.0 means a D/E
# ratio of 0.5), not a unitless ratio - 200.0 here means "D/E ratio > 2.0".
HIGH_DEBT_TO_EQUITY_THRESHOLD = 200.0
NEGATIVE_PE_THRESHOLD = 0.0

AnomalyType = Literal["low_fcf_yield", "high_debt_to_equity", "negative_pe"]


class _DetectedAnomaly(NamedTuple):
    ticker: str
    anomaly_type: AnomalyType
    metric_value: float
    threshold: float


def detect_anomalies(holdings: list[Holding]) -> list[_DetectedAnomaly]:
    """Pure-Python threshold checks over per-holding fundamentals already on
    `QuantMetrics.holdings` (Epic 4) - no re-fetching, no LLM math. A single
    holding can trigger more than one anomaly type."""
    findings: list[_DetectedAnomaly] = []
    for h in holdings:
        if h.fcf_yield is not None and h.fcf_yield < LOW_FCF_YIELD_THRESHOLD:
            findings.append(
                _DetectedAnomaly(h.ticker, "low_fcf_yield", h.fcf_yield, LOW_FCF_YIELD_THRESHOLD)
            )
        if h.debt_to_equity is not None and h.debt_to_equity > HIGH_DEBT_TO_EQUITY_THRESHOLD:
            findings.append(
                _DetectedAnomaly(
                    h.ticker, "high_debt_to_equity", h.debt_to_equity, HIGH_DEBT_TO_EQUITY_THRESHOLD
                )
            )
        if h.pe_ratio is not None and h.pe_ratio < NEGATIVE_PE_THRESHOLD:
            findings.append(
                _DetectedAnomaly(h.ticker, "negative_pe", h.pe_ratio, NEGATIVE_PE_THRESHOLD)
            )
    return findings


@tool
def sec_edgar_lookup_tool(ticker: str, limit: int = 5) -> str:
    """Look up recent SEC 10-K/10-Q filing metadata for a ticker, to
    investigate a flagged risk anomaly or a performance-attribution
    question."""
    try:
        result = mcp_server.sec_edgar_lookup(ticker, filing_types=("10-K", "10-Q"), limit=limit)
    except RuntimeError as exc:  # SEC_EDGAR_USER_AGENT unset - config error, not retried
        return f"sec_edgar_lookup unavailable (configuration error): {exc}"
    except McpToolError as exc:  # exhausted retries
        return f"sec_edgar_lookup failed after retries: {exc}"
    return json.dumps(result, default=str)


@tool
def duckduckgo_search_tool(query: str, ticker: str | None = None, max_results: int = 5) -> str:
    """Search recent web/news coverage relevant to a ticker or a portfolio
    performance question (e.g. sector headwinds)."""
    try:
        result = mcp_server.duckduckgo_search(query, ticker=ticker, max_results=max_results)
    except McpToolError as exc:
        return f"duckduckgo_search failed after retries: {exc}"
    return json.dumps(result, default=str)


_TOOLS = [sec_edgar_lookup_tool, duckduckgo_search_tool]

_RISK_CHECK_SYSTEM_PROMPT = (
    "You are a value-investing risk analyst reviewing ONE already-detected "
    "quantitative anomaly on a single holding: its ticker, anomaly_type (one "
    "of low_fcf_yield / high_debt_to_equity / negative_pe), the metric value, "
    "and the threshold it breached. All of these numbers are already computed "
    "deterministically - never restate them as if verifying them, and never "
    "compute or estimate any new number yourself. Your only job is to "
    "investigate WHY this anomaly might be occurring, using the "
    "sec_edgar_lookup_tool (recent SEC filings) and/or duckduckgo_search_tool "
    "(recent news) tools. Call at least one tool before concluding. Finish "
    "with a concise (2-4 sentence) plain-English explanation of the likely "
    "cause."
)

_PERFORMANCE_SYSTEM_PROMPT = (
    "You are a portfolio performance analyst. You are given the portfolio's "
    "time-weighted return, the S&P 500 benchmark return over the same "
    "period, the gap between them, and the list of current holding tickers. "
    "There is no sector/industry field available in the data, so if you "
    "suspect sector allocation mismatch (or any other holding-specific "
    "driver) you must research it yourself via sec_edgar_lookup_tool and/or "
    "duckduckgo_search_tool. Call at least one tool before concluding. Do "
    "not invent or recompute the return numbers - only explain them. Finish "
    "with a concise explanation of the likely driver(s) of over- or "
    "under-performance."
)


def _build_react_agent(model_name: str, system_prompt: str):
    model = ChatOpenAI(model=model_name, temperature=0)
    return create_agent(model, _TOOLS, system_prompt=system_prompt)


def _extract_tool_calls(messages: list) -> list[ToolCallRecord]:
    pending: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls or []:
                pending[call["id"]] = {"tool_name": call["name"], "tool_input": call["args"]}
        elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
            pending[m.tool_call_id]["output_summary"] = str(m.content)[:1000]
    return [ToolCallRecord(**entry) for entry in pending.values() if "output_summary" in entry]


def _run_react_agent(agent, user_message: str) -> tuple[str, list[ToolCallRecord]]:
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    messages = result["messages"]
    final_text = str(messages[-1].content)
    return final_text, _extract_tool_calls(messages)


def _format_anomaly_prompt(anomaly: _DetectedAnomaly) -> str:
    return (
        f"Ticker: {anomaly.ticker}\n"
        f"Anomaly type: {anomaly.anomaly_type}\n"
        f"Metric value: {anomaly.metric_value}\n"
        f"Threshold breached: {anomaly.threshold}\n"
        "Investigate why this anomaly might be occurring."
    )


def _format_performance_prompt(quant_metrics: QuantMetrics) -> str:
    tickers = ", ".join(h.ticker for h in quant_metrics.holdings)
    relative_return_pct = quant_metrics.twr_pct - quant_metrics.benchmark_return_pct
    return (
        f"Portfolio time-weighted return: {quant_metrics.twr_pct:.4f}\n"
        f"Benchmark ({quant_metrics.benchmark_ticker}) return: "
        f"{quant_metrics.benchmark_return_pct:.4f}\n"
        f"Relative return (portfolio - benchmark): {relative_return_pct:.4f}\n"
        f"Current holdings: {tickers}\n"
        "Explain the likely driver(s) of this over- or under-performance."
    )


def run_risk_agent(state: PortfolioState) -> PortfolioState:
    """Detect anomalies deterministically, then investigate anomalies and
    performance attribution via two independent ReAct tool-calling loops.

    Raises `ValueError` if `quant_metrics` is missing from `state`. MCP tool
    failures (`McpToolError`/`RuntimeError`) never propagate out of this
    function - they're caught in the `@tool` wrappers above and surfaced to
    the LLM as text observations. `ChatOpenAI`-level failures (auth,
    network, OpenAI rate limits) DO propagate uncaught, matching
    `quant_agent.py`'s contract - Epic 6's Supervisor is the catch boundary
    for agent-level failures, not this module.
    """
    quant_metrics = state.get("quant_metrics")
    if quant_metrics is None:
        raise ValueError("PortfolioState has no quant_metrics to run risk analysis on")

    detected = detect_anomalies(quant_metrics.holdings)

    anomalies: list[AnomalyFinding] = []
    if detected:
        risk_check_agent = _build_react_agent(RISK_CHECK_MODEL, _RISK_CHECK_SYSTEM_PROMPT)
        for d in detected:
            explanation, tool_calls = _run_react_agent(risk_check_agent, _format_anomaly_prompt(d))
            anomalies.append(
                AnomalyFinding(
                    ticker=d.ticker,
                    anomaly_type=d.anomaly_type,
                    metric_value=d.metric_value,
                    threshold=d.threshold,
                    explanation=explanation,
                    tool_calls=tool_calls,
                )
            )

    performance_agent = _build_react_agent(PERFORMANCE_MODEL, _PERFORMANCE_SYSTEM_PROMPT)
    perf_explanation, perf_tool_calls = _run_react_agent(
        performance_agent, _format_performance_prompt(quant_metrics)
    )
    relative_return_pct = quant_metrics.twr_pct - quant_metrics.benchmark_return_pct
    performance_attribution = PerformanceAttribution(
        portfolio_return_pct=quant_metrics.twr_pct,
        benchmark_return_pct=quant_metrics.benchmark_return_pct,
        relative_return_pct=relative_return_pct,
        explanation=perf_explanation,
        tool_calls=perf_tool_calls,
    )

    risk_report = RiskReport(
        as_of=quant_metrics.as_of,
        anomalies=anomalies,
        performance_attribution=performance_attribution,
    )

    telemetry.log_event(
        "risk_audit_completed",
        anomalies_flagged=len(anomalies),
        tools_called=sum(len(a.tool_calls) for a in anomalies) + len(perf_tool_calls),
        relative_return_pct=relative_return_pct,
    )

    return {**state, "risk_report": risk_report}
