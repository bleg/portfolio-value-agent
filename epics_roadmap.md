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

## Epic 3: Self-Healing CSV Parsing (Multi-Broker Adapter Deferred)
**Goal:** Deliver the "self-healing data ingestion" headline claim: when a broker's CSV export format changes, the system heals itself via an LLM column-mapping fallback instead of hard-failing.

*   **Tasks:**
    1. Build `src/parsers/broker_llm.py`: the `gpt-4o-mini` self-healing fallback. On a parser exception, pass the new CSV's header (plus locally-computed, value-free type tags per column — never real cell values, since PII columns can't be identified by name until the mapping is known) to the LLM, have it map columns to the internal schema, and cache the mapping to a local JSON file for future runs. Reuse `src/resilience.py`'s `resilient_tool` decorator (built in Epic 2) for this call's retry-with-backoff rather than writing bespoke retry logic. Reject an oversized/adversarial header (too many columns, or an implausibly long column name) before ever calling the LLM, and cap the sample read used for type-profiling to a small fixed row count regardless of total file size, so cost and prompt-injection surface never scale with an uploaded file's size.
    2. Fire `telemetry.log_event("schema_healed", broker=..., fields_mapped=...)` only when the LLM is actually called (a cache miss) — not on cache hits, and not on the deterministic path — this is the event named in the GTM/KPI doc and only meaningful when healing actually occurred.
    3. Write `pytest` tests: a deliberately renamed-column DEGIRO CSV caught by the fallback and correctly mapped (LLM call mocked), a cache-hit test confirming a second parse of the same renamed header doesn't call the LLM again, a malformed-LLM-response test confirming retry-then-`McpToolError` behavior, and header-shape-guardrail tests confirming an oversized/adversarial header is rejected before any LLM call.
*   **Definition of Done:** Feeding the parser a DEGIRO CSV with intentionally renamed columns produces a correctly mapped `PortfolioState` via the LLM fallback, with `schema_healed` firing on the first (cache-miss) parse only, and header-shape guardrails test-verified.

> **Deferred:** Adding a second real broker adapter (e.g. Interactive Brokers or Schwab) to `base_parser.py` — so "multi-broker" is actually exercised, not just asserted — is deferred to a later session. It will also be the first real test of the fallback against genuinely different row-value conventions (different date/decimal formats), not just a renamed DEGIRO header.

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
    1. Build `risk_agent.py`'s risk-check path: detect anomalies (e.g. FCF Yield < 3%) and autonomously call `sec_edgar_lookup` / `duckduckgo_search` to investigate why. Note the failure contract from Epic 2: an exhausted MCP tool call raises `resilience.McpToolError`, it does not return an `{"error": ...}` dict — catch this exception at the agent-node boundary and feed the error text back to the LLM as an observation. Per-holding `fcf_yield`/`pe_ratio`/`pb_ratio`/`debt_to_equity` are already on `state["quant_metrics"].holdings` (Epic 4) — read them directly for the anomaly check rather than re-fetching via `yfinance_fundamentals`.
    2. Build the performance-attribution path: explain over/underperformance vs. S&P 500 (e.g. sector allocation mismatch).
    3. Write `pytest` tests that mock MCP tool responses and assert the agent calls the *correct* tool for a given anomaly type (not full end-to-end correctness of LLM prose, which isn't unit-testable — assert on tool-call routing).
    4. Build a small LangSmith eval dataset (per CLAUDE.md Testing & Evals): 3-5 seeded portfolios with known, hand-picked anomalies. Run the agent against each and grade on whether the right anomaly was flagged and the right tool called — not prose quality. This is the eval layer pytest can't cover.
*   **Definition of Done:** Given a Quant Agent output containing a seeded anomaly, `risk_agent.py` autonomously calls the right tool(s) and produces a qualitative explanation, runnable and inspectable outside the full graph, with the LangSmith eval dataset passing.

---

## Epic 6: The Supervisor Agent & HITL Orchestration
**Goal:** Wire Epics 1–5 together into a working LangGraph. This is now scoped to routing and orchestration only, not new agent logic.

*   **Tasks:**
    1. Build `supervisor.py`: route between Quant and Risk agents, and compile the final Markdown report.
    2. Implement the Human-In-The-Loop (HITL) breakpoint: trigger manual approval on extreme accounting variance or unknown tickers. Fire `telemetry.log_event("hitl_override_triggered", reason=..., user_decision=...)` when a breakpoint fires and is resolved — this is the event the "HITL Acceptance Rate" product metric is built from. Epic 4 already surfaces both triggering signals: `QuantMetrics.unresolved_isins` (unknown tickers) and a dedicated `quant_agent.InsufficientHoldingsError(ValueError)` raised on data-integrity issues like an oversell (extreme accounting variance) — wire HITL to these rather than re-deriving detection logic. Also note `quant_agent.py`, like the Epic 2 MCP tools, lets `McpToolError`/`ValueError` propagate uncaught rather than catching internally, so the graph wiring in task 1 must wrap the Quant Agent node in the same try/except boundary as the Risk Agent, not just the Risk Agent. `risk_agent.py`'s contract is narrower, though: it already catches `McpToolError`/`RuntimeError` from its own tool calls internally (feeding the error text back to the LLM as an observation, per Epic 5), so `McpToolError` should never actually reach the Risk Agent node's try/except in practice — the boundary still needs to exist, but for `ValueError` (missing `quant_metrics`) and `ChatOpenAI`-level failures (auth/network), not for MCP tool failures.
    3. Wire `src/main.py` as the CLI entry point.
    4. Write `pytest` / integration tests for graph routing logic (does an anomaly correctly route to the Risk Agent; does a HITL condition correctly pause execution).
*   **Definition of Done:** A fully working CLI application. Running `python src/main.py sample.csv` triggers the full agentic workflow — parse → quant → HITL (as needed, gated purely on the Quant Agent's own output per task 2 above — never on anything the Risk Agent finds, so an unapproved run never reaches a paid `gpt-4o` ReAct call) → risk (as needed) → report — and outputs `portfolio_audit.md`.

---

## Epic 7: Persistence & Web UI
**Goal:** Wrap the working backend in a user-friendly interface and save results safely. Six of the seven KPI events (`portfolio_ingested`, `mcp_tool_failure`, `schema_healed`, `benchmark_compared`, `risk_audit_completed`, `hitl_override_triggered`) were already wired into their originating epics as each feature was built — this epic only adds the last one (`report_exported`) and upgrades storage.

*   **Tasks:**
    1. Build `src/db_controller.py` using SQLite/SQLAlchemy for strict, deterministic database writes (no LLM SQL generation). The LLM only ever gets read access via the scoped `historical_db_read(user_id)` MCP tool — implement that tool here now that the DB exists. `user_id` didn't exist anywhere before this epic (Phase 1 has no auth) — `db_controller.get_or_create_local_user_id()` generates a UUID on first run and persists it locally (`~/.portfolio_agent/user_id`), so the "scoped" guarantee is real per-install now rather than a placeholder, and won't need reworking once Phase 2 adds real multi-tenant auth.
    2. Build `src/app.py` using Streamlit: drag-and-drop CSV upload, interactive Plotly chart for the S&P 500 benchmark comparison. Fire `telemetry.log_event("report_exported", ...)` when a user downloads/saves a report from the UI. Note: a genuine over-time comparison chart needs more than `QuantMetrics`'s final scalar `twr_pct`/`benchmark_return_pct` (Epic 4) — this epic additively extends `QuantMetrics` with `value_history` (rebased portfolio/benchmark index points at each TWR breakpoint, surfaced from `quant_agent.py`'s already-computed-but-previously-discarded intermediate values), rather than reopening Epic 4's math.
    3. Upgrade `telemetry.py`'s backend from local file/console logging (Epic 1 stub) to writing events into the new DB via `db_controller.py`, so historical KPI data persists across runs. Every existing test file's `telemetry_path`/`_events()` fixture (one per agent/parser test file) needs the same mechanical seam swap — from monkeypatching `telemetry.DEFAULT_LOG_PATH` + reading a JSONL file, to monkeypatching `db_controller.DEFAULT_DB_PATH` + `db_controller.read_telemetry_events()` — since the JSONL file stops being written entirely.
    4. Write `pytest` tests confirming DB writes are correctly scoped/parameterized (no raw string interpolation) and that `report_exported` fires exactly once per export.
*   **Definition of Done:** Opening `localhost:8501` lets the user upload a CSV, view the performance chart, read the AI risk report, verify the run was saved to the local `.db` file, export a report, and confirm all seven KPI events (including the newly-added `report_exported`) are now persisted to the DB rather than the local log.

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
