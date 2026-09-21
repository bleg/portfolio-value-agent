"""CLI entry point for the portfolio-value-agent (Epic 6).

Usage: python src/main.py <csv_path> [output_path]

Runs the full agentic workflow - parse -> quant -> risk (as needed) -> HITL
(as needed) -> report - as one LangGraph, pausing for a terminal y/n prompt
if a Human-In-The-Loop breakpoint fires, and writes the final Markdown
report to `output_path` (default `portfolio_audit.md`).

Supersedes scripts/run_epic5_demo.py, which existed only because Epic 6's
Supervisor didn't exist yet.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agents.supervisor import build_graph
from src.parsers.base_parser import parse_degiro_csv

DEFAULT_OUTPUT_PATH = "portfolio_audit.md"


def _prompt_decision(payload: dict) -> str:
    print("\n--- Human approval required ---")
    print(f"Reason: {payload['reason']}")
    print(f"Details: {payload['details']}")
    while True:
        raw = input("Approve and continue? [y/N]: ").strip().lower()
        if raw in ("y", "yes"):
            return "approve"
        if raw in ("n", "no", ""):
            return "abort"
        print("Please answer y or n.")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(f"Usage: python {sys.argv[0]} <csv_path> [output_path]")
        raise SystemExit(1)

    csv_path = sys.argv[1]
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(DEFAULT_OUTPUT_PATH)

    state = parse_degiro_csv(csv_path)
    print(f"Parsed {len(state['transactions'])} transaction(s).")

    compiled = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = compiled.invoke(state, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = _prompt_decision(payload)
        result = compiled.invoke(Command(resume=decision), config)

    output_path.write_text(result["report_markdown"], encoding="utf-8")
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
