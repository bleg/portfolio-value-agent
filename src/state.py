"""Global state definitions for the portfolio-value-agent LangGraph pipeline.

Epic 1 only declares the fields it owns (raw parsed transactions). Later
epics (holdings/cost-basis aggregation, benchmark comparison, risk report,
HITL flags) extend `PortfolioState` as they're built — see CLAUDE.md Section 7
on working one epic at a time.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Literal, TypedDict

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


class Holding(BaseModel):
    """A currently-held, aggregated position (Epic 4's quant_agent output).

    Cost basis uses average-cost lot accounting. Raw per-holding fundamentals
    (pe_ratio/pb_ratio/debt_to_equity/fcf_yield) are surfaced unweighted, not
    just via the portfolio-level `QuantMetrics.weighted_pe` aggregate, because
    Epic 5's Risk Agent needs per-holding fcf_yield for its anomaly check.
    """

    ticker: str
    isin: str
    quantity: int
    avg_cost_basis_eur: Decimal
    total_cost_basis_eur: Decimal
    current_price_eur: Decimal
    market_value_eur: Decimal
    unrealized_return_pct: float
    pe_ratio: float | None
    pb_ratio: float | None
    debt_to_equity: float | None
    fcf_yield: float | None


class ValueHistoryPoint(BaseModel):
    """One TWR breakpoint's rebased index value (Epic 7's Portfolio vs
    benchmark chart). Both indices start at 100 at the first breakpoint so
    they're directly comparable regardless of absolute portfolio size vs.
    index level."""

    date: date
    portfolio_index: float
    benchmark_index: float


class QuantMetrics(BaseModel):
    """Deterministic valuation output of Epic 4's quant_agent. No LLM ever
    computes a number here (CLAUDE.md Section 3C).

    Known limitation: dividends aren't captured anywhere upstream yet (no
    parser support), so net_return_pct/twr_pct understate true return for
    dividend-paying holdings until a future epic adds dividend-row parsing.
    """

    as_of: date
    holdings: list[Holding]
    total_cost_basis_eur: Decimal
    total_market_value_eur: Decimal
    weighted_pe: float | None
    net_return_pct: float
    twr_pct: float
    benchmark_ticker: str
    benchmark_return_pct: float
    unresolved_isins: list[str]
    value_history: list[ValueHistoryPoint] = []


class ToolCallRecord(BaseModel):
    """One tool invocation captured from a ReAct loop (Epic 5's risk_agent),
    kept for standalone inspectability and tool-routing test/eval
    assertions - not surfaced to end users directly."""

    tool_name: str
    tool_input: dict
    output_summary: str


class AnomalyFinding(BaseModel):
    """One deterministically-detected anomaly (risk_agent.detect_anomalies),
    enriched with the ReAct agent's qualitative investigation. The three
    anomaly_type values and their thresholds are illustrative for this demo,
    not investment advice - see risk_agent.py's threshold constants."""

    ticker: str
    anomaly_type: Literal["low_fcf_yield", "high_debt_to_equity", "negative_pe"]
    metric_value: float
    threshold: float
    explanation: str
    tool_calls: list[ToolCallRecord]


class PerformanceAttribution(BaseModel):
    """Qualitative explanation of portfolio vs. benchmark performance.
    portfolio_return_pct/benchmark_return_pct/relative_return_pct are copied
    from QuantMetrics (deterministic) - the LLM only supplies `explanation`,
    never recomputes the numbers (CLAUDE.md Section 3C)."""

    portfolio_return_pct: float
    benchmark_return_pct: float
    relative_return_pct: float
    explanation: str
    tool_calls: list[ToolCallRecord]


class RiskReport(BaseModel):
    """Epic 5's output: src.agents.risk_agent.run_risk_agent's return value,
    attached to PortfolioState.risk_report."""

    as_of: date
    anomalies: list[AnomalyFinding]
    performance_attribution: PerformanceAttribution


class SupervisorError(BaseModel):
    """An unrecoverable failure caught at Epic 6's Supervisor boundary
    (supervisor.quant_node/risk_node), rather than left to propagate and
    crash the graph. `error_type` is what the routing functions switch on;
    `message` is the human-readable text rendered into the final report."""

    stage: Literal["quant", "risk"]
    error_type: Literal[
        "insufficient_holdings",
        "mcp_tool_error",
        "value_error",
        "missing_quant_metrics",
        "risk_agent_failure",
    ]
    message: str


class HitlRecord(BaseModel):
    """A *resolved* Human-In-The-Loop breakpoint (Epic 6). Only ever
    constructed after the human's decision is known - PortfolioState never
    holds a HitlRecord while a graph run is paused mid-interrupt, only after
    resume - so `decision` is required, not Optional."""

    reason: Literal["unresolved_isins", "insufficient_holdings"]
    details: str
    decision: Literal["approve", "abort"]


class PortfolioState(TypedDict, total=False):
    transactions: list[Transaction]
    broker: str
    base_currency: str
    skipped_rows: list[dict]
    quant_metrics: QuantMetrics
    risk_report: RiskReport
    quant_error: SupervisorError
    risk_error: SupervisorError
    hitl_record: HitlRecord
    report_markdown: str
