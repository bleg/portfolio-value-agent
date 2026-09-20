# Project Brief: Multi-Broker Value-Investing Portfolio Agent (`portfolio-value-agent`)

## 1. Overview & Purpose
**Target Persona & Problem Statement:** Retail value investors across European and US brokers (DEGIRO, Interactive Brokers, Schwab) spend hours manually extracting trade histories, converting currencies, and reading SEC filings to calculate valuation metrics and benchmark returns. This agent turns a 4-hour manual audit into a 30-second automated report across any broker CSV.

**Motivation & Career Goal:** 
This project serves as a portfolio demonstration piece for Senior Product Management roles specializing in Agentic AI and SaaS platforms. It demonstrates enterprise-grade product architecture: multi-agent orchestration, self-healing data pipelines, Model Context Protocol (MCP) tool integration, deterministic execution guardrails, multi-tenant security, and Open-Core Product-Led Growth (PLG) strategy.

**License:** Open-sourced under MIT (or Apache 2.0) to support the Bring-Your-Own-Key PLG tier described in Section 5 — pick one before the first public push and add a `LICENSE` file at the repo root.

---

## 2. Tech Stack & Frameworks
- **Language:** Python 3.10+
- **Agent Framework:** LangGraph (StateGraph) for multi-agent orchestration.
- **Primary Models (OpenAI-native):** 
  - `gpt-4o-mini`: Fast JSON parsing, tool routing, schema validation, and error correction.
  - `gpt-4o`: Deep qualitative value audit synthesis, SEC filing analysis, and performance attribution.
  - *Model Abstraction Layer:* Designed model-agnostically to reduce vendor lock-in — LangChain's chat model interface is swapped via `.env`. Note this covers basic prompting/completion; tool-calling schemas and structured-output behavior differ enough between OpenAI and Anthropic that a provider swap should still be smoke-tested against the agent nodes in `/src/agents/`, not assumed to be a zero-code change.
- **Protocol:** Model Context Protocol (MCP) Server/Client architecture.
- **Web & Visualization:** Streamlit & Plotly (Phase 1 Web UI), FastAPI (Phase 2 REST API).
- **Data & Persistence:** `pandas`, `pydantic`, SQLAlchemy with SQLite (Phase 1) / PostgreSQL (Phase 2).
- **Testing & Evals:** `pytest` for deterministic logic and tool-routing assertions; LangSmith (tracing enabled via `.env`) for run inspection. Qualitative LLM output (the Risk Agent's writeups) isn't pytest-testable — track it separately via a small LangSmith eval dataset of seeded portfolios with expected risk flags, graded on whether the right anomaly was caught and the right tool was called, not on prose quality.

---

## 3. Core System Architecture & Features

### A. Self-Healing CSV Parsing Engine (`/src/parsers/`)
- Implements an **LLM-Assisted Fallback Pattern** to ingest transactions and holdings from multiple brokers (starting with DEGIRO, extending to Interactive Brokers/Schwab) without brittle hardcoding.
- **Deterministic Primary Path:** Uses Pandas/Pydantic to parse CSVs against known column mappings — no LLM call, sub-second, for any recognized broker format.
- **Agentic Self-Healing:** If a broker changes their CSV export format, the system catches the exception and passes the new headers/sample rows to `gpt-4o-mini`. The LLM dynamically maps the new columns to the required internal schema (Ticker, Cost Basis, Date) and caches the updated mapping to the database for future runs.
- **Strategic Value:** Delivers "intelligent error correction" to eliminate ongoing engineering maintenance for data integration pipelines.
- **PII Scrubbing (`/src/parsers/pii_scrubber.py`):** Runs immediately after parsing, before data enters `PortfolioState`. Account names and internal broker IDs are stripped locally — only stock tickers and aggregate share counts are ever sent to external LLMs.

### B. MCP Tool Server (`/src/mcp_server.py`)
Data connections are abstracted into an MCP Server, decoupling the reasoning engine from the data layer:
- **`yfinance_fundamentals`:** Fetches quantitative metrics (P/E, P/B, Debt-to-Equity, FCF Yield) and historical price series.
- **`benchmark_data_fetcher`:** Pulls historical index performance data for S&P 500 (`^GSPC`) and MSCI World (`URTH`).
- **`sec_edgar_lookup`:** Retrieves recent 10-K/10-Q summaries.
- **`duckduckgo_search`:** Free web search for real-time qualitative news on flagged stocks.
- **`historical_db_read`:** Scoped, parameterized database read for past user audits.
- **Resilience:** All external calls (yfinance, SEC EDGAR, DuckDuckGo) are known to rate-limit or fail intermittently. Each tool wraps its call in a retry-with-backoff and a timeout, and emits the `mcp_tool_failure` telemetry event (Section 5B) on exhaustion so failures are visible in product analytics rather than surfacing only as a stack trace.

### C. Multi-Agent Orchestrator (`/src/agents/`)
- **The Quant Agent (Deterministic Execution):** Gathers ticker and index data via MCP. Runs deterministic Benjamin Graham math (Weighted P/E, Cost Basis, Net Return) AND calculates relative benchmark performance (Time-Weighted Return vs. S&P 500). *CRITICAL RULE: Never let the LLM calculate math or performance percentages.* Output is pushed directly to State.
- **The Risk & Performance Analyst Agent (ReAct Pattern):** Powered by `gpt-4o`. Reviews the Quant Agent's output. 
  - *Risk Check:* If an anomaly is detected (e.g., FCF Yield drops below 3%), it autonomously searches web/SEC tools to determine *why*.
  - *Performance Attribution:* Evaluates why the portfolio over/underperformed the S&P 500 (e.g., sector allocation mismatch).
- **The Supervisor Agent:** Evaluates all data, routes to final formatting, or triggers the Human-In-The-Loop (HITL) breakpoint.

### D. Data Governance & Multi-Tenant Security (`/src/db_controller.py`)
- **HITL Guardrail:** Triggers a manual user approval breakpoint if extreme accounting variance or unknown tickers are detected.
- **Deterministic Writes / Scoped Reads:** To prevent prompt injection and database corruption, the LLM is strictly firewalled from database write operations. All DB writes (saving audit runs) are executed via deterministic Python controllers (`db_controller.py`). The LLM is only granted read access via explicitly parameterized, scoped MCP tools (`fetch_past_audit(user_id)`).
- **PII Scrubbing:** See Section 3A — enforced at parse time, not at the DB boundary, so scrubbed data never exists in memory in the first place.

---

## 4. Product Delivery & UI/DB Evolution Roadmap

### Phase 1: Open-Source Portfolio & Interactive Web Demo (Current Build)
*   **Web UI & Interactive Charts (`/src/app.py`):** Built using **Streamlit** and **Plotly**. Provides a clean drag-and-drop CSV upload, an interactive benchmark performance comparison graph (Portfolio vs. S&P 500), and the qualitative AI risk report.
*   **Database:** Local **SQLite** managed via `db_controller.py`. Zero-configuration setup for developers who fork the open-source repo.
*   **Recruiter Access:** Deployed directly to Streamlit Community Cloud (free) with a "Live Demo" link in the GitHub `README.md` so hiring managers can test the agent in real-time without installing Python.

### Phase 2: Hosted SaaS & Monetization Architecture (Roadmap)
*   **API Gateway (`/src/api.py`):** Wraps the LangGraph workflow in **FastAPI** endpoints to decouple backend reasoning from any frontend consumer (e.g., Next.js / React web app).
*   **Production Database:** Migrates from local SQLite to **Supabase / PostgreSQL** to manage multi-tenant user authentication, encrypted historical snapshot storage, and strict row-level security (RLS).
*   **Monetization Integration:** Connects Stripe webhooks to trigger access control for the €5/month SaaS tier.

---

## 5. Product Strategy & Go-To-Market (GTM)

### A. Commercialization & Unit Economics
- **Tier 1 Routing (`gpt-4o-mini`):** Used for >80% of operations (parsing, routing, simple MCP calls) keeping Cost of Goods Sold (COGS) to fractions of a cent per audit.
- **Tier 2 Routing (`gpt-4o`):** Used strictly for qualitative performance attribution and final markdown report generation.
- **Open-Core PLG Strategy:** The local CLI/Streamlit version is open-sourced via GitHub (Bring Your Own Key). This zero-COGS tier drives top-of-funnel acquisition via organic sharing on developer and value-investing communities (`r/ValueInvesting`, Hacker News).
- **SaaS Monetization:** A hosted, frictionless web-app tier at €5/month targets non-technical retail investors, abstracting away API keys and Python setup, yielding high operating margins due to optimized model routing.

### B. Telemetry & KPIs
- **Product Analytics (`/src/telemetry.py`):** A single lightweight logging module, called from the Supervisor graph and `db_controller.py` at each triggering point, so events aren't scattered ad hoc across agent files. Logs key product usage metrics alongside engineering traces.
- **Key Events Tracked:** `portfolio_ingested`, `schema_healed`, `benchmark_compared`, `mcp_tool_failure`, `hitl_override_triggered`, `report_exported`.
- **AI Success Metrics:** "HITL Acceptance Rate" (tracking if users trust the AI's risk detection) and "Time-to-Value" (time from CSV upload to final benchmarked report).

---

## 6. Repository Structure
```text
portfolio-value-agent/
├── README.md               # Architecture diagram, GTM strategy, and live Streamlit demo link
├── LICENSE                 # MIT/Apache 2.0 — required for the open-core PLG tier (Section 5A)
├── .env.example            # Template for API keys (OPENAI_API_KEY, etc.)
├── requirements.txt        # langgraph, openai, pandas, yfinance, mcp, streamlit, plotly, fastapi, pytest
├── src/
│   ├── __init__.py
│   ├── app.py              # Phase 1: Streamlit Web UI (CSV upload, S&P 500 charts, AI audit viewer)
│   ├── api.py              # Phase 2: FastAPI REST endpoints (for SaaS web frontend integration)
│   ├── state.py            # LangGraph TypedDict state
│   ├── db_controller.py    # Safe, deterministic database manager (SQLite/PostgreSQL)
│   ├── mcp_server.py       # MCP Server hosting yfinance, S&P 500, DuckDuckGo, SEC tools
│   ├── telemetry.py        # Product analytics event logging (Section 5B KPI events)
│   ├── parsers/
│   │   ├── base_parser.py    # Abstract base class / Adapter (DEGIRO, IBKR/Schwab)
│   │   ├── broker_llm.py     # Self-healing LLM schema mapper (gpt-4o-mini)
│   │   └── pii_scrubber.py   # Strips account names/broker IDs before data reaches state (Section 3A/D)
│   └── agents/
│       ├── quant_agent.py      # Deterministic math node (Valuation metrics & S&P 500 TWR calculations)
│       ├── risk_agent.py       # ReAct tool-calling agent (gpt-4o attribution & risk audit)
│       └── supervisor.py       # Graph routing & HITL breakpoints
├── tests/
│   └── test_flows.py       # Unit tests for parsing, math, PII scrubbing, and DB interactions
└── docs/
    └── Product_Strategy_and_PRD.md  # Product Vision, Unit Economics, & Data Governance
```

---

## 7. Development Workflow
- The dependency-ordered build sequence lives in `epics_roadmap.md` (8 epics). This file (`CLAUDE.md`) loads automatically at the start of every Claude Code session in this repo — it never needs to be attached manually.
- Work one epic at a time, in a fresh session per epic (e.g. "Do Epic 1 from epics_roadmap.md"). Start in Plan Mode so the approach can be reviewed before code is written, then implement.
- Keep epics in separate sessions even though this file's full context is available throughout — it avoids scope creep into future epics and keeps each session's diff reviewable on its own.
- Epics 4–6 (Quant Agent, Risk Agent, Supervisor) are especially worth keeping separate despite all living in `/src/agents/`: each has its own debugging loop and should be validated in isolation before being wired together.