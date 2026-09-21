"""The Supervisor Agent (Epic 6): wires the Quant Agent (Epic 4) and Risk
Agent (Epic 5) into one LangGraph `StateGraph`, adds the Human-In-The-Loop
(HITL) approval breakpoint, and compiles the final Markdown report.

Scoped to routing and orchestration only, per epics_roadmap.md's Epic 6
goal - no new agent logic lives here.

Graph shape:

    quant --[unresolved_isins or oversell?]--> hitl --[approved?]--> risk -> report -> END
         \\--[clean]-----------------------------------------------> risk /
         \\--[unrecoverable non-HITL error, e.g. McpToolError]-----------> report

HITL sits *before* the Risk Agent, gated purely on the Quant Agent's own
output - cheaper (skips a paid gpt-4o ReAct call on data that might be
garbage) and matches the two triggering signals the roadmap actually names:
`QuantMetrics.unresolved_isins` and `quant_agent.InsufficientHoldingsError`.
Nothing the Risk Agent finds is a HITL trigger.

Failure-catching contract, mirroring quant_agent.py/risk_agent.py's own
module docstrings: both agent modules deliberately let certain exceptions
propagate uncaught, on the understanding that this module is the single
catch boundary. `quant_node` and `risk_node` below are that boundary.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from src import telemetry
from src.agents.quant_agent import InsufficientHoldingsError, run_quant_agent
from src.agents.risk_agent import run_risk_agent
from src.resilience import McpToolError
from src.state import HitlRecord, PortfolioState, QuantMetrics, RiskReport, SupervisorError

# --- nodes -------------------------------------------------------------------


def quant_node(state: PortfolioState) -> PortfolioState:
    """Runs the Quant Agent. Catches exactly the exceptions quant_agent.py's
    module docstring documents as propagating uncaught - nothing else.

    `InsufficientHoldingsError` is caught before the bare `ValueError`
    clause deliberately: it's a `ValueError` subclass, and catching order
    matters here or an oversell would fall into the generic "value_error"
    branch and lose its HITL eligibility.
    """
    try:
        return run_quant_agent(state)
    except InsufficientHoldingsError as exc:
        return {
            **state,
            "quant_error": SupervisorError(
                stage="quant", error_type="insufficient_holdings", message=str(exc)
            ),
        }
    except McpToolError as exc:
        return {
            **state,
            "quant_error": SupervisorError(
                stage="quant", error_type="mcp_tool_error", message=str(exc)
            ),
        }
    except ValueError as exc:
        return {
            **state,
            "quant_error": SupervisorError(
                stage="quant", error_type="value_error", message=str(exc)
            ),
        }


def route_after_quant(state: PortfolioState) -> Literal["hitl", "risk", "report"]:
    quant_error = state.get("quant_error")
    if quant_error is not None:
        # Only an oversell is HITL-eligible - McpToolError/bare ValueError
        # skip straight to an error report, no human pause.
        return "hitl" if quant_error.error_type == "insufficient_holdings" else "report"

    quant_metrics = state.get("quant_metrics")
    if quant_metrics is not None and quant_metrics.unresolved_isins:
        return "hitl"
    return "risk"


def hitl_node(state: PortfolioState) -> PortfolioState:
    """Pauses for human approval via `interrupt()`.

    LangGraph re-runs this node from the top on resume. `reason`/`details`
    are recomputed identically both times since they're pure functions of
    already-settled state, so the payload shown to the human and the
    HitlRecord built after resume never diverge. Everything below the
    `interrupt()` call is skipped while paused, which is what makes
    `hitl_override_triggered` fire exactly once - only once the decision is
    known, never on the initial pause.
    """
    quant_error = state.get("quant_error")
    if quant_error is not None and quant_error.error_type == "insufficient_holdings":
        reason: Literal["unresolved_isins", "insufficient_holdings"] = "insufficient_holdings"
        details = quant_error.message
    else:
        quant_metrics = state["quant_metrics"]  # guaranteed by route_after_quant
        reason = "unresolved_isins"
        details = "Unresolved ISINs (no ticker mapping): " + ", ".join(
            quant_metrics.unresolved_isins
        )

    decision: Literal["approve", "abort"] = interrupt({"reason": reason, "details": details})

    telemetry.log_event("hitl_override_triggered", reason=reason, user_decision=decision)

    return {**state, "hitl_record": HitlRecord(reason=reason, details=details, decision=decision)}


def route_after_hitl(state: PortfolioState) -> Literal["risk", "report"]:
    quant_error = state.get("quant_error")
    if quant_error is not None and quant_error.error_type == "insufficient_holdings":
        # No valid quant_metrics exists no matter what the human decided.
        return "report"
    return "risk" if state["hitl_record"].decision == "approve" else "report"


def risk_node(state: PortfolioState) -> PortfolioState:
    """Runs the Risk Agent. `ValueError` (missing quant_metrics) shouldn't
    happen in normal graph flow - quant always runs first and only routes
    here when quant_metrics exists - but the catch is required per the
    roadmap's explicit instruction. `McpToolError` is deliberately not
    special-cased: risk_agent.py already catches it inside its own @tool
    wrappers and it should never reach here; the broad `except Exception`
    below still covers it defensively, alongside genuine ChatOpenAI-level
    failures (auth/network/rate-limit), which have no custom exception type
    in this codebase.
    """
    try:
        return run_risk_agent(state)
    except ValueError as exc:
        return {
            **state,
            "risk_error": SupervisorError(
                stage="risk", error_type="missing_quant_metrics", message=str(exc)
            ),
        }
    except Exception as exc:
        return {
            **state,
            "risk_error": SupervisorError(
                stage="risk", error_type="risk_agent_failure", message=str(exc)
            ),
        }


def report_node(state: PortfolioState) -> PortfolioState:
    """Deterministic Markdown compilation - plain string templating, no LLM
    call (epics_roadmap.md's Epic 6 goal: routing/orchestration only, and
    all qualitative text here was already generated by Epic 5's Risk
    Agent). Covers all four end states: clean, HITL-aborted, unrecoverable
    quant error, unrecoverable risk error.
    """
    sections: list[str] = ["# Portfolio Audit Report", ""]

    hitl_record = state.get("hitl_record")
    if hitl_record is not None:
        sections += _render_hitl_section(hitl_record)

    quant_error = state.get("quant_error")
    if quant_error is not None:
        sections += _render_error_section("Quant Agent", quant_error)
        return {**state, "report_markdown": "\n".join(sections)}

    quant_metrics = state.get("quant_metrics")
    if quant_metrics is not None:
        sections += _render_quant_section(quant_metrics)

    if hitl_record is not None and hitl_record.decision == "abort":
        sections += ["", "_Run aborted by human reviewer before Risk Agent analysis._"]
        return {**state, "report_markdown": "\n".join(sections)}

    risk_error = state.get("risk_error")
    if risk_error is not None:
        sections += _render_error_section("Risk Agent", risk_error)
        return {**state, "report_markdown": "\n".join(sections)}

    risk_report = state.get("risk_report")
    if risk_report is not None:
        sections += _render_risk_section(risk_report)

    return {**state, "report_markdown": "\n".join(sections)}


# --- markdown helpers ---------------------------------------------------------


def _render_hitl_section(hitl_record: HitlRecord) -> list[str]:
    return [
        "## Human Review",
        f"- Reason: `{hitl_record.reason}`",
        f"- Details: {hitl_record.details}",
        f"- Decision: **{hitl_record.decision}**",
        "",
    ]


def _render_error_section(agent: str, error: SupervisorError) -> list[str]:
    return ["## Error", f"{agent} failed (`{error.error_type}`): {error.message}", ""]


def _render_quant_section(m: QuantMetrics) -> list[str]:
    lines = [
        "## Quant Metrics",
        f"As of {m.as_of}",
        f"- Total cost basis: €{m.total_cost_basis_eur}",
        f"- Total market value: €{m.total_market_value_eur}",
        f"- Net return: {m.net_return_pct:.2%}",
        f"- Time-weighted return: {m.twr_pct:.2%} vs {m.benchmark_ticker} {m.benchmark_return_pct:.2%}",
        f"- Weighted P/E: {m.weighted_pe if m.weighted_pe is not None else 'n/a'}",
        "",
        "| Ticker | Qty | Avg Cost (EUR) | Market Value (EUR) | Return |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {h.ticker} | {h.quantity} | {h.avg_cost_basis_eur} | {h.market_value_eur} | "
        f"{h.unrealized_return_pct:.2%} |"
        for h in m.holdings
    ]
    if m.unresolved_isins:
        lines += ["", f"_Unresolved ISINs excluded from valuation: {', '.join(m.unresolved_isins)}_"]
    lines.append("")
    return lines


def _render_risk_section(r: RiskReport) -> list[str]:
    lines = ["## Risk & Performance Analysis", ""]
    if r.anomalies:
        lines.append("### Anomalies")
        for a in r.anomalies:
            lines.append(
                f"- **{a.ticker}** ({a.anomaly_type}, {a.metric_value} vs threshold "
                f"{a.threshold}): {a.explanation}"
            )
        lines.append("")
    else:
        lines += ["No anomalies detected.", ""]

    pa = r.performance_attribution
    lines += [
        "### Performance Attribution",
        f"Portfolio {pa.portfolio_return_pct:.2%} vs benchmark {pa.benchmark_return_pct:.2%} "
        f"(relative {pa.relative_return_pct:.2%})",
        pa.explanation,
        "",
    ]
    return lines


# --- graph ---------------------------------------------------------------


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Builds and compiles the Epic 6 Supervisor graph. A checkpointer is
    required for the HITL `interrupt()`/`Command(resume=...)` breakpoint to
    work; defaults to a fresh `MemorySaver()` per call so callers (tests,
    `main.py`) never accidentally share state across separate runs.
    """
    graph = StateGraph(PortfolioState)
    graph.add_node("quant", quant_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("risk", risk_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("quant")
    graph.add_conditional_edges(
        "quant", route_after_quant, {"hitl": "hitl", "risk": "risk", "report": "report"}
    )
    graph.add_conditional_edges("hitl", route_after_hitl, {"risk": "risk", "report": "report"})
    graph.add_edge("risk", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
