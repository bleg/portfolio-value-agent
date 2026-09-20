"""Epic 5 Definition of Done demo: parse -> quant -> risk, printed end-to-end.

Runs entirely outside Epic 6's Supervisor (which doesn't exist yet), proving
risk_agent.py is runnable/inspectable standalone per the roadmap's
Definition of Done. Makes real gpt-4o calls (requires OPENAI_API_KEY) and
real MCP tool calls (yfinance/SEC EDGAR/DuckDuckGo).

Usage: python scripts/run_epic5_demo.py <path-to-degiro-transactions.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.quant_agent import run_quant_agent
from src.agents.risk_agent import run_risk_agent
from src.parsers.base_parser import parse_degiro_csv


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_epic5_demo.py <path-to-degiro-transactions.csv>")
        raise SystemExit(1)

    state = parse_degiro_csv(sys.argv[1])
    print(f"Parsed {len(state['transactions'])} transaction(s).\n")

    state = run_quant_agent(state)
    quant_metrics = state["quant_metrics"]
    print(f"Quant metrics: TWR={quant_metrics.twr_pct:.4f} vs benchmark "
          f"{quant_metrics.benchmark_return_pct:.4f}, {len(quant_metrics.holdings)} holding(s).\n")

    state = run_risk_agent(state)
    print("Risk report:")
    print(state["risk_report"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
