"""Phase 1 Web UI (CLAUDE.md Section 4/Epic 7): Streamlit front end for the
LangGraph pipeline built in Epics 1-6.

Streamlit reruns this whole script on every widget interaction (unlike
`main.py`'s blocking `while "__interrupt__" in result` CLI loop), so
anything that must survive a rerun - the compiled graph, the LangGraph
thread id, the last invoke() result, and one-shot "have we already done
this" guards - lives in `st.session_state`.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src import db_controller, telemetry
from src.agents.supervisor import build_graph
from src.parsers.base_parser import parse_degiro_csv

st.set_page_config(page_title="Portfolio Value Agent", layout="wide")


def export_report(report_markdown: str, user_id: str) -> None:
    """Fires `report_exported` exactly once per call - kept as a standalone
    function so it's testable without simulating a Streamlit button click."""
    telemetry.log_event("report_exported", user_id=user_id, length=len(report_markdown))


def _render_chart(value_history: list) -> None:
    dates = [p.date for p in value_history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[p.portfolio_index for p in value_history], name="Portfolio"))
    fig.add_trace(go.Scatter(x=dates, y=[p.benchmark_index for p in value_history], name="Benchmark"))
    fig.update_layout(
        title="Portfolio vs. Benchmark (rebased to 100 at inception)",
        xaxis_title="Date",
        yaxis_title="Index",
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("Portfolio Value Agent")

    user_id = db_controller.get_or_create_local_user_id()

    if "compiled_graph" not in st.session_state:
        st.session_state["compiled_graph"] = build_graph(checkpointer=MemorySaver())

    uploaded_file = st.file_uploader("Upload a broker CSV", type=["csv"])
    if uploaded_file is None:
        return

    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("processed_file_id") != file_id:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            state = parse_degiro_csv(tmp_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Failed to parse CSV: {exc}")
            return

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = st.session_state["compiled_graph"].invoke(state, config)

        st.session_state["processed_file_id"] = file_id
        st.session_state["thread_id"] = thread_id
        st.session_state["graph_result"] = result
        st.session_state["saved_run_id"] = None

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    result = st.session_state["graph_result"]

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.warning(f"Human approval required — {payload['reason']}: {payload['details']}")
        col1, col2 = st.columns(2)
        if col1.button("Approve"):
            result = st.session_state["compiled_graph"].invoke(Command(resume="approve"), config)
            st.session_state["graph_result"] = result
            st.rerun()
        if col2.button("Abort"):
            result = st.session_state["compiled_graph"].invoke(Command(resume="abort"), config)
            st.session_state["graph_result"] = result
            st.rerun()
        return

    report_markdown = result.get("report_markdown")
    if not report_markdown:
        st.info("Processing...")
        return

    quant_metrics = result.get("quant_metrics")
    if quant_metrics is not None and quant_metrics.value_history:
        _render_chart(quant_metrics.value_history)

    st.markdown(report_markdown)

    if quant_metrics is not None and st.session_state.get("saved_run_id") is None:
        st.session_state["saved_run_id"] = db_controller.save_audit_run(
            user_id, quant_metrics, report_markdown, result.get("broker", "unknown")
        )

    if st.download_button(
        "Download report", data=report_markdown, file_name="portfolio_audit.md", mime="text/markdown"
    ):
        export_report(report_markdown, user_id)

    with st.expander("Past audits"):
        past_audits = db_controller.read_audits_for_user(user_id)
        if past_audits:
            st.table(past_audits)
        else:
            st.write("No past audits yet.")


if __name__ == "__main__":
    main()
