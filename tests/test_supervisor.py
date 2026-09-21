from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from langgraph.types import Command

from src import db_controller
from src.agents import supervisor
from src.agents.quant_agent import InsufficientHoldingsError
from src.resilience import McpToolError
from src.state import Holding, PerformanceAttribution, QuantMetrics, RiskReport, SupervisorError

# --- shared fixtures ---------------------------------------------------------


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    return db_controller.read_telemetry_events(db_path=telemetry_path)


def _holding(*, ticker: str = "AAA") -> Holding:
    return Holding(
        ticker=ticker,
        isin=f"TEST{ticker}0001",
        quantity=10,
        avg_cost_basis_eur=Decimal("100"),
        total_cost_basis_eur=Decimal("1000"),
        current_price_eur=Decimal("120"),
        market_value_eur=Decimal("1200"),
        unrealized_return_pct=0.2,
        pe_ratio=15.0,
        pb_ratio=3.0,
        debt_to_equity=50.0,
        fcf_yield=0.10,
    )


def _quant_metrics(*, unresolved_isins: list[str] | None = None) -> QuantMetrics:
    return QuantMetrics(
        as_of=date(2024, 12, 31),
        holdings=[_holding()],
        total_cost_basis_eur=Decimal("1000"),
        total_market_value_eur=Decimal("1200"),
        weighted_pe=15.0,
        net_return_pct=0.2,
        twr_pct=0.10,
        benchmark_ticker="^GSPC",
        benchmark_return_pct=0.08,
        unresolved_isins=unresolved_isins or [],
    )


def _risk_report() -> RiskReport:
    return RiskReport(
        as_of=date(2024, 12, 31),
        anomalies=[],
        performance_attribution=PerformanceAttribution(
            portfolio_return_pct=0.10,
            benchmark_return_pct=0.08,
            relative_return_pct=0.02,
            explanation="Outperformed on strong holding selection.",
            tool_calls=[],
        ),
    )


def _initial_state() -> dict:
    return {"transactions": [], "broker": "DEGIRO", "base_currency": "EUR"}


def _fake_quant_ok(quant_metrics: QuantMetrics):
    def _fake(state):
        return {**state, "quant_metrics": quant_metrics}

    return _fake


def _fake_quant_raises(exc: Exception):
    def _fake(state):
        raise exc

    return _fake


def _fake_risk_ok(risk_report: RiskReport, calls: list):
    def _fake(state):
        calls.append(state)
        return {**state, "risk_report": risk_report}

    return _fake


def _fake_risk_raises(exc: Exception, calls: list):
    def _fake(state):
        calls.append(state)
        raise exc

    return _fake


def _config() -> dict:
    return {"configurable": {"thread_id": "test-thread"}}


# --- graph-level tests --------------------------------------------------------


def test_clean_run_routes_quant_risk_report_no_interrupt(monkeypatch, telemetry_path):
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_ok(_quant_metrics()))
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    result = compiled.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert len(risk_calls) == 1
    assert result["risk_report"] is not None
    assert "quant_error" not in result
    assert "risk_error" not in result
    assert "hitl_record" not in result
    assert result["report_markdown"]
    assert "## Quant Metrics" in result["report_markdown"]
    assert "## Risk & Performance Analysis" in result["report_markdown"]
    assert _events(telemetry_path) == []


def test_unresolved_isins_triggers_hitl_and_approve_continues_to_risk(monkeypatch, telemetry_path):
    monkeypatch.setattr(
        supervisor, "run_quant_agent", _fake_quant_ok(_quant_metrics(unresolved_isins=["XX1234567890"]))
    )
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    config = _config()
    result = compiled.invoke(_initial_state(), config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["reason"] == "unresolved_isins"
    assert "XX1234567890" in payload["details"]
    assert len(risk_calls) == 0
    assert [e for e in _events(telemetry_path) if e["event"] == "hitl_override_triggered"] == []

    result = compiled.invoke(Command(resume="approve"), config)

    assert "__interrupt__" not in result
    assert len(risk_calls) == 1
    assert result["hitl_record"].decision == "approve"
    hitl_events = [e for e in _events(telemetry_path) if e["event"] == "hitl_override_triggered"]
    assert len(hitl_events) == 1
    assert hitl_events[0]["reason"] == "unresolved_isins"
    assert hitl_events[0]["user_decision"] == "approve"


def test_unresolved_isins_abort_skips_risk_goes_to_error_report(monkeypatch, telemetry_path):
    monkeypatch.setattr(
        supervisor, "run_quant_agent", _fake_quant_ok(_quant_metrics(unresolved_isins=["XX1234567890"]))
    )
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    config = _config()
    compiled.invoke(_initial_state(), config)
    result = compiled.invoke(Command(resume="abort"), config)

    assert "__interrupt__" not in result
    assert len(risk_calls) == 0
    assert "risk_report" not in result
    assert result["report_markdown"]
    assert "aborted" in result["report_markdown"].lower()
    hitl_events = [e for e in _events(telemetry_path) if e["event"] == "hitl_override_triggered"]
    assert len(hitl_events) == 1
    assert hitl_events[0]["user_decision"] == "abort"


@pytest.mark.parametrize("decision", ["approve", "abort"])
def test_insufficient_holdings_triggers_hitl_routes_to_report_regardless_of_decision(
    decision, monkeypatch, telemetry_path
):
    exc = InsufficientHoldingsError("AAA: sell of 10 on 2024-01-01 exceeds held quantity 5")
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_raises(exc))
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    config = _config()
    result = compiled.invoke(_initial_state(), config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["reason"] == "insufficient_holdings"
    assert "exceeds held quantity" in payload["details"]

    result = compiled.invoke(Command(resume=decision), config)

    assert "__interrupt__" not in result
    assert len(risk_calls) == 0
    assert "risk_report" not in result
    assert result["quant_error"].error_type == "insufficient_holdings"
    assert result["report_markdown"]
    hitl_events = [e for e in _events(telemetry_path) if e["event"] == "hitl_override_triggered"]
    assert len(hitl_events) == 1
    assert hitl_events[0]["reason"] == "insufficient_holdings"
    assert hitl_events[0]["user_decision"] == decision


def test_mcp_tool_error_routes_straight_to_report_no_hitl(monkeypatch, telemetry_path):
    exc = McpToolError("yfinance_fundamentals failed after 2 attempts")
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_raises(exc))
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    result = compiled.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert len(risk_calls) == 0
    assert result["quant_error"].error_type == "mcp_tool_error"
    assert result["report_markdown"]
    assert _events(telemetry_path) == []


def test_bare_value_error_no_transactions_routes_straight_to_report_no_hitl(monkeypatch, telemetry_path):
    exc = ValueError("PortfolioState has no transactions to compute quant metrics for")
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_raises(exc))
    risk_calls: list = []
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_ok(_risk_report(), risk_calls))

    compiled = supervisor.build_graph()
    result = compiled.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert len(risk_calls) == 0
    assert result["quant_error"].error_type == "value_error"
    assert result["report_markdown"]
    assert _events(telemetry_path) == []


def test_risk_missing_quant_metrics_value_error_routes_to_report(monkeypatch, telemetry_path):
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_ok(_quant_metrics()))
    risk_calls: list = []
    exc = ValueError("PortfolioState has no quant_metrics to run risk analysis on")
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_raises(exc, risk_calls))

    compiled = supervisor.build_graph()
    result = compiled.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert result["risk_error"].error_type == "missing_quant_metrics"
    assert "## Quant Metrics" in result["report_markdown"]
    assert "## Error" in result["report_markdown"]


def test_risk_generic_exception_routes_to_report(monkeypatch, telemetry_path):
    monkeypatch.setattr(supervisor, "run_quant_agent", _fake_quant_ok(_quant_metrics()))
    risk_calls: list = []
    exc = RuntimeError("OpenAI rate limit exceeded")
    monkeypatch.setattr(supervisor, "run_risk_agent", _fake_risk_raises(exc, risk_calls))

    compiled = supervisor.build_graph()
    result = compiled.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert result["risk_error"].error_type == "risk_agent_failure"
    assert result["report_markdown"]


# --- routing unit tests --------------------------------------------------------


def test_route_after_quant_unit():
    assert supervisor.route_after_quant({"quant_metrics": _quant_metrics()}) == "risk"
    assert (
        supervisor.route_after_quant({"quant_metrics": _quant_metrics(unresolved_isins=["X"])})
        == "hitl"
    )
    assert (
        supervisor.route_after_quant(
            {"quant_error": SupervisorError(stage="quant", error_type="insufficient_holdings", message="x")}
        )
        == "hitl"
    )
    assert (
        supervisor.route_after_quant(
            {"quant_error": SupervisorError(stage="quant", error_type="mcp_tool_error", message="x")}
        )
        == "report"
    )
    assert (
        supervisor.route_after_quant(
            {"quant_error": SupervisorError(stage="quant", error_type="value_error", message="x")}
        )
        == "report"
    )


def test_route_after_hitl_unit():
    insufficient_error = SupervisorError(
        stage="quant", error_type="insufficient_holdings", message="x"
    )
    assert (
        supervisor.route_after_hitl(
            {
                "quant_error": insufficient_error,
                "hitl_record": {"decision": "approve"},
            }
        )
        == "report"
    )
    assert (
        supervisor.route_after_hitl(
            {
                "quant_error": insufficient_error,
                "hitl_record": {"decision": "abort"},
            }
        )
        == "report"
    )

    from src.state import HitlRecord

    approved = HitlRecord(reason="unresolved_isins", details="x", decision="approve")
    aborted = HitlRecord(reason="unresolved_isins", details="x", decision="abort")
    assert supervisor.route_after_hitl({"hitl_record": approved}) == "risk"
    assert supervisor.route_after_hitl({"hitl_record": aborted}) == "report"
