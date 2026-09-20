"""Epic 5 LangSmith eval: seeded-portfolio anomaly/tool-routing grading.

Runs `risk_agent.run_risk_agent` against 3-5 seeded portfolios with known,
hand-picked anomalies, grading whether the right anomaly type was flagged
and the right tool got called - not prose quality (per CLAUDE.md's Testing
& Evals section and epics_roadmap.md Epic 5 task 4).

Usage: python scripts/risk_agent_eval.py

Requires LANGSMITH_API_KEY in the environment (see .env.example - sign up
free at https://smith.langchain.com). Prints a message and exits 0 (not a
failure) if it's unset, so CI/pytest never hard-depend on a LangSmith
account. This makes real `gpt-4o` calls (also requires OPENAI_API_KEY) -
it's the one place in the repo actual OpenAI spend happens outside normal
usage.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print(
            "LANGSMITH_API_KEY is not set - skipping the risk-agent LangSmith eval. "
            "See .env.example for setup instructions (free sign-up at "
            "https://smith.langchain.com)."
        )
        raise SystemExit(0)

    from langsmith import Client, evaluate

    from src.agents.risk_agent import run_risk_agent
    from src.state import Holding, QuantMetrics

    DATASET_NAME = "risk-agent-seeded-anomalies"
    PROJECT_NAME = os.environ.get("LANGSMITH_PROJECT", "portfolio-value-agent-risk-eval")

    @dataclass
    class SeededPortfolio:
        name: str
        quant_metrics: QuantMetrics
        expected_anomaly_types: set[str]
        expected_tools: set[str]

    def _holding(
        ticker: str,
        *,
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
        holdings: list[Holding], *, twr_pct: float = 0.10, benchmark_return_pct: float = 0.08
    ) -> QuantMetrics:
        from datetime import date

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

    _BOTH_TOOLS = {"sec_edgar_lookup_tool", "duckduckgo_search_tool"}

    SEEDED_PORTFOLIOS = [
        SeededPortfolio(
            name="low_fcf_yield_only",
            quant_metrics=_quant_metrics([_holding("LOWFCF", fcf_yield=0.01)]),
            expected_anomaly_types={"low_fcf_yield"},
            expected_tools=_BOTH_TOOLS,
        ),
        SeededPortfolio(
            name="high_debt_to_equity_only",
            quant_metrics=_quant_metrics([_holding("HILEV", debt_to_equity=400.0)]),
            expected_anomaly_types={"high_debt_to_equity"},
            expected_tools=_BOTH_TOOLS,
        ),
        SeededPortfolio(
            name="negative_pe_only",
            quant_metrics=_quant_metrics([_holding("NEGPE", pe_ratio=-2.5)]),
            expected_anomaly_types={"negative_pe"},
            expected_tools=_BOTH_TOOLS,
        ),
        SeededPortfolio(
            name="multiple_anomalies_one_holding",
            quant_metrics=_quant_metrics(
                [_holding("TRIPLE", fcf_yield=0.005, debt_to_equity=500.0, pe_ratio=-8.0)]
            ),
            expected_anomaly_types={"low_fcf_yield", "high_debt_to_equity", "negative_pe"},
            expected_tools=_BOTH_TOOLS,
        ),
        SeededPortfolio(
            name="healthy_portfolio_large_underperformance",
            quant_metrics=_quant_metrics(
                [_holding("HEALTHY")], twr_pct=-0.15, benchmark_return_pct=0.10
            ),
            expected_anomaly_types=set(),
            expected_tools=_BOTH_TOOLS,  # performance-attribution path must still research
        ),
    ]
    PORTFOLIOS_BY_NAME = {p.name: p for p in SEEDED_PORTFOLIOS}

    client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Epic 5 risk_agent seeded-anomaly / tool-routing eval portfolios.",
        )

    existing_names = {
        ex.inputs.get("portfolio_name")
        for ex in client.list_examples(dataset_name=DATASET_NAME)
    }
    for p in SEEDED_PORTFOLIOS:
        if p.name in existing_names:
            continue
        client.create_example(
            dataset_name=DATASET_NAME,
            inputs={"portfolio_name": p.name},
            outputs={
                "expected_anomaly_types": sorted(p.expected_anomaly_types),
                "expected_tools": sorted(p.expected_tools),
            },
        )

    def target(inputs: dict) -> dict:
        portfolio = PORTFOLIOS_BY_NAME[inputs["portfolio_name"]]
        risk_report = run_risk_agent({"quant_metrics": portfolio.quant_metrics})["risk_report"]
        actual_anomaly_types = sorted({a.anomaly_type for a in risk_report.anomalies})
        actual_tools = sorted(
            {tc.tool_name for a in risk_report.anomalies for tc in a.tool_calls}
            | {tc.tool_name for tc in risk_report.performance_attribution.tool_calls}
        )
        return {"anomaly_types": actual_anomaly_types, "tools_called": actual_tools}

    def grade_anomaly_and_tool_routing(run, example) -> dict:
        actual = run.outputs or {}
        expected = example.outputs or {}
        anomaly_types_correct = set(actual.get("anomaly_types", [])) == set(
            expected.get("expected_anomaly_types", [])
        )
        expected_tools = set(expected.get("expected_tools", []))
        tool_correct = (
            bool(set(actual.get("tools_called", [])) & expected_tools)
            if expected_tools
            else True
        )
        score = 1.0 if (anomaly_types_correct and tool_correct) else 0.0
        return {
            "key": "anomaly_and_tool_routing",
            "score": score,
            "comment": (
                f"anomaly_types_correct={anomaly_types_correct}, "
                f"expected_tool_called={tool_correct}"
            ),
        }

    print(f"Running risk_agent eval against {len(SEEDED_PORTFOLIOS)} seeded portfolios...")
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[grade_anomaly_and_tool_routing],
        experiment_prefix="risk-agent-eval",
        client=client,
        metadata={"project": PROJECT_NAME},
    )

    scores = [
        r["evaluation_results"]["results"][0].score
        for r in results
        if r.get("evaluation_results", {}).get("results")
    ]
    passed = sum(1 for s in scores if s == 1.0)
    print(f"\n{passed}/{len(SEEDED_PORTFOLIOS)} seeded portfolios correct.")
    print("Inspect full results in the LangSmith UI under your configured project.")


if __name__ == "__main__":
    main()
