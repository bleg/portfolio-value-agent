from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src import mcp_server
from src.agents import risk_agent
from src.agents.risk_agent import (
    detect_anomalies,
    duckduckgo_search_tool,
    run_risk_agent,
    sec_edgar_lookup_tool,
)
from src.resilience import McpToolError
from src.state import Holding, QuantMetrics

# --- shared fixtures ---------------------------------------------------------


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr("src.telemetry.DEFAULT_LOG_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    if not telemetry_path.exists():
        return []
    return [json.loads(line) for line in telemetry_path.read_text().splitlines()]


def _holding(
    *,
    ticker: str = "AAA",
    fcf_yield: float | None = 0.10,
    debt_to_equity: float | None = 50.0,
    pe_ratio: float | None = 15.0,
) -> Holding:
    return Holding(
        ticker=ticker,
        isin=f"TEST{ticker}0001",
        quantity=10,
        avg_cost_basis_eur=Decimal("100"),
        total_cost_basis_eur=Decimal("1000"),
        current_price_eur=Decimal("120"),
        market_value_eur=Decimal("1200"),
        unrealized_return_pct=0.2,
        pe_ratio=pe_ratio,
        pb_ratio=3.0,
        debt_to_equity=debt_to_equity,
        fcf_yield=fcf_yield,
    )


def _quant_metrics(
    holdings: list[Holding],
    *,
    twr_pct: float = 0.10,
    benchmark_return_pct: float = 0.08,
) -> QuantMetrics:
    return QuantMetrics(
        as_of=date(2024, 12, 31),
        holdings=holdings,
        total_cost_basis_eur=Decimal("1000"),
        total_market_value_eur=Decimal("1200"),
        weighted_pe=15.0,
        net_return_pct=0.2,
        twr_pct=twr_pct,
        benchmark_ticker="^GSPC",
        benchmark_return_pct=benchmark_return_pct,
        unresolved_isins=[],
    )


class _FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel doesn't implement bind_tools (it's abstract on
    BaseChatModel) - create_agent calls model.bind_tools(tools) internally,
    so this override just returns self and lets the scripted messages drive
    tool-call routing directly."""

    def bind_tools(self, tools, **kwargs):
        return self


def _scripted_model(turns: list[AIMessage]) -> _FakeToolCallingModel:
    return _FakeToolCallingModel(messages=iter(turns))


def _tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


_FINAL_ANSWER = AIMessage(content="A concise explanation of the likely cause.")


# --- detect_anomalies (pure Python, no mocking) ------------------------------


def test_detect_anomalies_flags_low_fcf_yield():
    findings = detect_anomalies([_holding(ticker="LOW", fcf_yield=0.01)])
    assert findings == [risk_agent._DetectedAnomaly("LOW", "low_fcf_yield", 0.01, 0.03)]


def test_detect_anomalies_flags_high_debt_to_equity():
    findings = detect_anomalies([_holding(ticker="LEV", debt_to_equity=350.0)])
    assert findings == [risk_agent._DetectedAnomaly("LEV", "high_debt_to_equity", 350.0, 200.0)]


def test_detect_anomalies_flags_negative_pe():
    findings = detect_anomalies([_holding(ticker="NEG", pe_ratio=-4.2)])
    assert findings == [risk_agent._DetectedAnomaly("NEG", "negative_pe", -4.2, 0.0)]


def test_detect_anomalies_flags_multiple_types_on_one_holding():
    findings = detect_anomalies(
        [_holding(ticker="BAD", fcf_yield=0.01, debt_to_equity=500.0, pe_ratio=-1.0)]
    )
    assert {f.anomaly_type for f in findings} == {
        "low_fcf_yield",
        "high_debt_to_equity",
        "negative_pe",
    }
    assert all(f.ticker == "BAD" for f in findings)


def test_detect_anomalies_ignores_none_fields():
    findings = detect_anomalies(
        [_holding(ticker="UNK", fcf_yield=None, debt_to_equity=None, pe_ratio=None)]
    )
    assert findings == []


def test_detect_anomalies_returns_empty_for_healthy_portfolio():
    findings = detect_anomalies([_holding(ticker="OK")])
    assert findings == []


# --- tool wrapper failure handling (no agent needed) -------------------------


def test_sec_edgar_lookup_tool_catches_runtime_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("SEC_EDGAR_USER_AGENT is not set")

    monkeypatch.setattr(mcp_server, "sec_edgar_lookup", _raise)
    result = sec_edgar_lookup_tool.invoke({"ticker": "AAPL"})
    assert isinstance(result, str)
    assert "unavailable" in result


def test_sec_edgar_lookup_tool_catches_mcp_tool_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise McpToolError("sec_edgar_lookup failed after 2 attempts")

    monkeypatch.setattr(mcp_server, "sec_edgar_lookup", _raise)
    result = sec_edgar_lookup_tool.invoke({"ticker": "AAPL"})
    assert isinstance(result, str)
    assert "failed after retries" in result


def test_duckduckgo_search_tool_catches_mcp_tool_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise McpToolError("duckduckgo_search failed after 2 attempts")

    monkeypatch.setattr(mcp_server, "duckduckgo_search", _raise)
    result = duckduckgo_search_tool.invoke({"query": "AAPL news"})
    assert isinstance(result, str)
    assert "failed after retries" in result


def test_sec_edgar_lookup_tool_returns_json_on_success(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "sec_edgar_lookup",
        lambda ticker, filing_types, limit: {"ticker": ticker, "cik": "1", "filings": []},
    )
    result = sec_edgar_lookup_tool.invoke({"ticker": "AAPL"})
    assert json.loads(result) == {"ticker": "AAPL", "cik": "1", "filings": []}


# --- LLM-layer tool-call routing (mocked model, real graph) ------------------


def test_risk_check_agent_calls_sec_edgar_for_low_fcf_yield(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [_tool_call_message("sec_edgar_lookup_tool", {"ticker": "LOW"}), _FINAL_ANSWER]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "sec_edgar_lookup",
        lambda ticker, filing_types, limit: {"ticker": ticker, "cik": "1", "filings": []},
    )

    quant_metrics = _quant_metrics([_holding(ticker="LOW", fcf_yield=0.01)])
    result = run_risk_agent({"quant_metrics": quant_metrics})

    anomalies = result["risk_report"].anomalies
    assert len(anomalies) == 1
    assert anomalies[0].tool_calls[0].tool_name == "sec_edgar_lookup_tool"


def test_risk_check_agent_calls_duckduckgo_for_negative_pe(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [_tool_call_message("duckduckgo_search_tool", {"query": "NEG"}), _FINAL_ANSWER]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "duckduckgo_search",
        lambda query, ticker, max_results: {"query": query, "results": []},
    )

    quant_metrics = _quant_metrics([_holding(ticker="NEG", pe_ratio=-2.0)])
    result = run_risk_agent({"quant_metrics": quant_metrics})

    anomalies = result["risk_report"].anomalies
    assert len(anomalies) == 1
    assert anomalies[0].tool_calls[0].tool_name == "duckduckgo_search_tool"


def test_risk_check_agent_can_call_both_tools_in_sequence(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [
                _tool_call_message("sec_edgar_lookup_tool", {"ticker": "LOW"}, call_id="call_1"),
                _tool_call_message(
                    "duckduckgo_search_tool", {"query": "LOW"}, call_id="call_2"
                ),
                _FINAL_ANSWER,
            ]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "sec_edgar_lookup",
        lambda ticker, filing_types, limit: {"ticker": ticker, "cik": "1", "filings": []},
    )
    monkeypatch.setattr(
        mcp_server,
        "duckduckgo_search",
        lambda query, ticker, max_results: {"query": query, "results": []},
    )

    quant_metrics = _quant_metrics([_holding(ticker="LOW", fcf_yield=0.01)])
    result = run_risk_agent({"quant_metrics": quant_metrics})

    tool_names = [tc.tool_name for tc in result["risk_report"].anomalies[0].tool_calls]
    assert tool_names == ["sec_edgar_lookup_tool", "duckduckgo_search_tool"]


def test_performance_attribution_agent_calls_a_tool(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [_tool_call_message("duckduckgo_search_tool", {"query": "sector news"}), _FINAL_ANSWER]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "duckduckgo_search",
        lambda query, ticker, max_results: {"query": query, "results": []},
    )

    quant_metrics = _quant_metrics([_holding()], twr_pct=0.05, benchmark_return_pct=0.15)
    result = run_risk_agent({"quant_metrics": quant_metrics})

    attribution = result["risk_report"].performance_attribution
    assert len(attribution.tool_calls) == 1
    assert attribution.relative_return_pct == pytest.approx(0.05 - 0.15)


# --- full-loop resilience: tool failure fed back as observation, no crash ---


def test_risk_check_agent_continues_after_tool_failure(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [_tool_call_message("sec_edgar_lookup_tool", {"ticker": "LOW"}), _FINAL_ANSWER]
        ),
    )

    def _raise(*args, **kwargs):
        raise McpToolError("sec_edgar_lookup failed after 2 attempts")

    monkeypatch.setattr(mcp_server, "sec_edgar_lookup", _raise)

    quant_metrics = _quant_metrics([_holding(ticker="LOW", fcf_yield=0.01)])
    result = run_risk_agent({"quant_metrics": quant_metrics})

    anomalies = result["risk_report"].anomalies
    assert len(anomalies) == 1
    assert anomalies[0].explanation == "A concise explanation of the likely cause."
    assert "failed after retries" in anomalies[0].tool_calls[0].output_summary


# --- telemetry ---------------------------------------------------------------


def test_run_risk_agent_fires_risk_audit_completed_on_success(monkeypatch, telemetry_path):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model(
            [_tool_call_message("sec_edgar_lookup_tool", {"ticker": "LOW"}), _FINAL_ANSWER]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "sec_edgar_lookup",
        lambda ticker, filing_types, limit: {"ticker": ticker, "cik": "1", "filings": []},
    )

    quant_metrics = _quant_metrics([_holding(ticker="LOW", fcf_yield=0.01)])
    run_risk_agent({"quant_metrics": quant_metrics})

    events = [e for e in _events(telemetry_path) if e["event"] == "risk_audit_completed"]
    assert len(events) == 1
    assert events[0]["anomalies_flagged"] == 1
    # The fake ChatOpenAI factory hands out a fresh 1-tool-call script to
    # *every* agent it builds, so both the risk-check loop and the
    # performance-attribution loop each call one tool.
    assert events[0]["tools_called"] == 2


def test_run_risk_agent_missing_quant_metrics_raises_and_fires_no_telemetry(telemetry_path):
    with pytest.raises(ValueError):
        run_risk_agent({})
    assert _events(telemetry_path) == []


# --- entry-point contract -----------------------------------------------------


def test_run_risk_agent_returns_shallow_copy_with_risk_report(monkeypatch):
    monkeypatch.setattr(
        risk_agent,
        "ChatOpenAI",
        lambda **kwargs: _scripted_model([_FINAL_ANSWER]),
    )

    state = {"quant_metrics": _quant_metrics([_holding()]), "broker": "DEGIRO"}
    result = run_risk_agent(state)

    assert result is not state
    assert result["broker"] == "DEGIRO"
    assert result["risk_report"].anomalies == []
