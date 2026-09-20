# Delivery Roadmap: Multi-Broker Value-Investing Portfolio Agent

This document breaks down the end-to-end architecture into 8 sequential Epics, dependency-ordered. Work through them one at a time in a fresh Claude Code session per epic (see `CLAUDE.md` Section 7). Epics that used to be bundled together (agent build, self-healing parser) have been split out below because they were too large and too interdependent to build reliably in one sitting.

Every epic includes a `pytest` task, not just Epic 1 — "deterministic execution guardrails" is a core selling point of this project, so test coverage needs to be continuous, not front-loaded.

---

## Epic 1: Data Foundations & State Management
**Goal:** Build the deterministic core. Extract data from the CSV into a structured Python format, define the global LangGraph state, and strip PII before anything reaches an LLM (No LLMs yet).

*   **Tasks:**
    1. Define `src/state.py` using Pydantic models and the LangGraph `TypedDict`.
    2. Build the deterministic DEGIRO CSV parser (`src/parsers/base_parser.py`) using `pandas`.
    3. Implement historical currency conversion: fetch the FX rate at each transaction's trade date (not spot) so cost basis is computed correctly for EUR/USD-denominated trades. Decide and document the FX data source here (e.g. `yfinance` currency pairs) — this was previously assumed but never scoped.
    4. Implement local PII scrubbing (`src/parsers/pii_scrubber.py`): strip account names and internal broker IDs from parsed rows immediately after ingestion, before the data enters `PortfolioState`. Only tickers and aggregate share counts should ever be eligible to leave the local process.
    5. Scaffold `src/telemetry.py`: a minimal `log_event(name, **props)` function, local file/console output for now (upgraded to DB-backed storage in Epic 7). Fire `portfolio_ingested` once a CSV successfully parses. Building this now, rather than in Epic 7, avoids every later epic needing a module that doesn't exist yet.
    6. Write `pytest` unit tests for currency conversion, ticker extraction, and PII scrubbing (assert scrubbed fields never appear in the resulting state object).
*   **Definition of Done:** Running a local test script ingests a sample DEGIRO CSV, prints a clean, validated, PII-free `PortfolioState` dictionary with correctly converted cost bases, and logs a `portfolio_ingested` event.

---

## Epic 2: The MCP Server & Financial Tools
**Goal:** Build the data-fetching layer in isolation using the Model Context Protocol (MCP).

*   **Tasks:**
    1. Scaffold `src/mcp_server.py`.
    2. Implement `yfinance_fundamentals` (P/E, P/B, Debt-to-Equity, FCF Yield, historical price series).
    3. Implement `benchmark_data_fetcher` (S&P 500 `^GSPC`, MSCI World `URTH`).
    4. Implement `sec_edgar_lookup` (recent 10-K/10-Q summaries) — this is required by the Risk Agent in Epic 5 and was missing from the original plan.
    5. Implement `duckduckgo_search` for qualitative news.
    6. Wrap every tool call in retry-with-backoff and a timeout (per CLAUDE.md Section 3B — these APIs are known to rate-limit/fail intermittently). On exhaustion, call `telemetry.log_event("mcp_tool_failure", tool=..., ticker=...)` from the `telemetry.py` scaffolded in Epic 1.
    7. Write `pytest` tests (or use the MCP inspector) confirming each tool returns a clean, correctly-shaped JSON response, including a failure case per tool (bad ticker, forced network error) to confirm retries happen and `mcp_tool_failure` fires exactly once on exhaustion.
    *   Note: `historical_db_read` is **not** built here — it depends on `db_controller.py`, which doesn't exist until Epic 7. It's scoped there instead of forcing a premature dependency.
*   **Definition of Done:** The MCP server runs locally, a test script can successfully request a stock's P/E ratio, an SEC filing summary, and recent news, and a forced-failure test confirms retry behavior and the `mcp_tool_failure` event firing.

---

## Epic 3: Self-Healing Parsing & Multi-Broker Support
**Goal:** Deliver the two things that make this a "multi-broker" tool with "self-healing" data ingestion — both are headline architecture claims and belong early, not as a Phase-5 afterthought.

*   **Tasks:**
    1. Build `src/parsers/broker_llm.py`: the `gpt-4o-mini` self-healing fallback. On a parser exception, pass the new CSV headers/sample rows to the LLM, have it map columns to the internal schema, and cache the mapping (to a local file or DB stub) for future runs. Reuse `src/resilience.py`'s `resilient_tool` decorator (built in Epic 2) for this call's retry-with-backoff rather than writing bespoke retry logic.
    2. Add a second broker adapter (Interactive Brokers or Schwab CSV export) to `base_parser.py`, using the deterministic path first and falling back to `broker_llm.py` on unrecognized formats. Without this, "multi-broker" is asserted in the README but never actually exercised.
    3. Fire `telemetry.log_event("schema_healed", broker=..., fields_mapped=...)` whenever the LLM fallback path is used (not the deterministic path) — this is the event named in the GTM/KPI doc and the fallback is the only place it's meaningful to log.
    4. Write `pytest` tests: a deliberately malformed/renamed-column CSV that must be caught by the fallback and correctly mapped, plus a test that the second broker's real export format parses deterministically.
*   **Definition of Done:** Feeding the parser a CSV from the second broker, and a DEGIRO CSV with intentionally renamed columns, both produce a correctly mapped `PortfolioState` — one via the deterministic path, one via the LLM fallback.

---

## Epic 4: The Quant Agent (Deterministic Math)
**Goal:** Build the deterministic math node. No LLM ever computes a number here — CLAUDE.md is explicit that this is a hard rule, not a preference.

*   **Tasks:**
    1. Build `quant_agent.py`: Weighted P/E, Cost Basis, Net Return.
    2. Implement Time-Weighted Return vs. S&P 500 using the benchmark data from Epic 2. Fire `telemetry.log_event("benchmark_compared", twr=..., benchmark_return=...)` once the comparison is computed.
    3. Write `pytest` tests with hand-calculated expected values for at least two portfolios (including one with mid-period buys/sells, to catch TWR edge cases).
*   **Definition of Done:** Given a `PortfolioState` from Epic 1/3 and live MCP data from Epic 2, `quant_agent.py` deterministically outputs weighted valuation metrics and benchmark-relative return, with test-verified math.

---

## Epic 5: The Risk & Performance Analyst Agent (ReAct)
**Goal:** Build the `gpt-4o`-powered ReAct agent in isolation, against the Quant Agent's output from Epic 4. Kept separate from the Supervisor because ReAct tool-calling loops are the most iterative, hardest-to-debug part of the build and deserve their own focused session.

*   **Tasks:**
    1. Build `risk_agent.py`'s risk-check path: detect anomalies (e.g. FCF Yield < 3%) and autonomously call `sec_edgar_lookup` / `duckduckgo_search` to investigate why. Note the failure contract from Epic 2: an exhausted MCP tool call raises `resilience.McpToolError`, it does not return an `{"error": ...}` dict — catch this exception at the agent-node boundary and feed the error text back to the LLM as an observation.
    2. Build the performance-attribution path: explain over/underperformance vs. S&P 500 (e.g. sector allocation mismatch).
    3. Write `pytest` tests that mock MCP tool responses and assert the agent calls the *correct* tool for a given anomaly type (not full end-to-end correctness of LLM prose, which isn't unit-testable — assert on tool-call routing).
    4. Build a small LangSmith eval dataset (per CLAUDE.md Testing & Evals): 3-5 seeded portfolios with known, hand-picked anomalies. Run the agent against each and grade on whether the right anomaly was flagged and the right tool called — not prose quality. This is the eval layer pytest can't cover.
*   **Definition of Done:** Given a Quant Agent output containing a seeded anomaly, `risk_agent.py` autonomously calls the right tool(s) and produces a qualitative explanation, runnable and inspectable outside the full graph, with the LangSmith eval dataset passing.

---

## Epic 6: The Supervisor Agent & HITL Orchestration
**Goal:** Wire Epics 1–5 together into a working LangGraph. This is now scoped to routing and orchestration only, not new agent logic.

*   **Tasks:**
    1. Build `supervisor.py`: route between Quant and Risk agents, and compile the final Markdown report.
    2. Implement the Human-In-The-Loop (HITL) breakpoint: trigger manual approval on extreme accounting variance or unknown tickers. Fire `telemetry.log_event("hitl_override_triggered", reason=..., user_decision=...)` when a breakpoint fires and is resolved — this is the event the "HITL Acceptance Rate" product metric is built from.
    3. Wire `src/main.py` as the CLI entry point.
    4. Write `pytest` / integration tests for graph routing logic (does an anomaly correctly route to the Risk Agent; does a HITL condition correctly pause execution).
*   **Definition of Done:** A fully working CLI application. Running `python src/main.py sample.csv` triggers the full agentic workflow — parse → quant → risk (as needed) → HITL (as needed) → report — and outputs `portfolio_audit.md`.

---

## Epic 7: Persistence & Web UI
**Goal:** Wrap the working backend in a user-friendly interface and save results safely. Five of the six KPI events (`portfolio_ingested`, `mcp_tool_failure`, `schema_healed`, `benchmark_compared`, `hitl_override_triggered`) were already wired into their originating epics as each feature was built — this epic only adds the last one and upgrades storage.

*   **Tasks:**
    1. Build `src/db_controller.py` using SQLite/SQLAlchemy for strict, deterministic database writes (no LLM SQL generation). The LLM only ever gets read access via the scoped `historical_db_read(user_id)` MCP tool — implement that tool here now that the DB exists.
    2. Build `src/app.py` using Streamlit: drag-and-drop CSV upload, interactive Plotly chart for the S&P 500 benchmark comparison. Fire `telemetry.log_event("report_exported", ...)` when a user downloads/saves a report from the UI.
    3. Upgrade `telemetry.py`'s backend from local file/console logging (Epic 1 stub) to writing events into the new DB via `db_controller.py`, so historical KPI data persists across runs.
    4. Write `pytest` tests confirming DB writes are correctly scoped/parameterized (no raw string interpolation) and that `report_exported` fires exactly once per export.
*   **Definition of Done:** Opening `localhost:8501` lets the user upload a CSV, view the performance chart, read the AI risk report, verify the run was saved to the local `.db` file, export a report, and confirm all six KPI events are now persisted to the DB rather than the local log.

---

## Epic 8: Enterprise Polish & GTM
**Goal:** Add tracing/observability and prepare the project for hiring-manager consumption.

*   **Tasks:**
    1. Enable LangSmith tracing via `.env` and confirm a full run produces an inspectable trace.
    2. Add a `LICENSE` file (MIT or Apache 2.0, per CLAUDE.md Section 1) — required for the open-core/BYOK PLG tier to make sense to anyone forking the repo.
    3. Finalize `README.md` with the Product-Led Growth (PLG) strategy, architecture diagram, and usage instructions.
    4. Deploy the Streamlit app to Streamlit Community Cloud.
*   **Definition of Done:** The project is live on the internet with LangSmith traces available for a sample run, the GitHub repository is polished, and a live demo link is ready to be shared with hiring managers.

---

See `CLAUDE.md` Section 7 ("Development Workflow") for how to work through these epics session-by-session.
