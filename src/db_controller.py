"""Safe, deterministic database manager (CLAUDE.md Section 3D/Epic 7).

SQLite via SQLAlchemy. The LLM never writes here and never generates SQL -
all queries are built with SQLAlchemy's parameterized ORM constructs, never
string-interpolated. Reads are only ever exposed to the LLM through the
scoped `historical_db_read(user_id)` MCP tool (src/mcp_server.py), which
delegates straight to `read_audits_for_user` below.

Epic 7 also upgrades `telemetry.py`'s backend to write into this database
(`record_telemetry_event`/`read_telemetry_events`) instead of a local JSONL
file, so historical KPI data persists across runs.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, Numeric, String, Text, create_engine, desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

if TYPE_CHECKING:
    from src.state import QuantMetrics

DEFAULT_DB_PATH = Path("portfolio_agent.db")
DEFAULT_CONFIG_DIR = Path.home() / ".portfolio_agent"

_engines: dict[str, Engine] = {}


class Base(DeclarativeBase):
    pass


class AuditRun(Base):
    __tablename__ = "audit_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    broker: Mapped[str] = mapped_column(String)
    total_cost_basis_eur: Mapped[Decimal] = mapped_column(Numeric)
    total_market_value_eur: Mapped[Decimal] = mapped_column(Numeric)
    net_return_pct: Mapped[float] = mapped_column(Float)
    twr_pct: Mapped[float] = mapped_column(Float)
    benchmark_ticker: Mapped[str] = mapped_column(String)
    benchmark_return_pct: Mapped[float] = mapped_column(Float)
    report_markdown: Mapped[str] = mapped_column(Text)


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_name: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    props_json: Mapped[str] = mapped_column(Text)


def get_engine(db_path: Path | str | None = None) -> Engine:
    """Returns a cached engine for `db_path` (or `DEFAULT_DB_PATH`), creating
    tables on first use for that path. Cached per resolved path so repeated
    calls (e.g. once per request in `app.py`) don't re-run `create_all` or
    open a fresh connection pool every time."""
    resolved = str(Path(db_path) if db_path is not None else DEFAULT_DB_PATH)
    engine = _engines.get(resolved)
    if engine is None:
        engine = create_engine(f"sqlite:///{resolved}")
        Base.metadata.create_all(engine)
        _engines[resolved] = engine
    return engine


def save_audit_run(
    user_id: str,
    quant_metrics: "QuantMetrics",
    report_markdown: str,
    broker: str,
    db_path: Path | str | None = None,
) -> int:
    """Deterministic, parameterized insert of one completed audit run.
    Returns the new row's id."""
    run = AuditRun(
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
        broker=broker,
        total_cost_basis_eur=quant_metrics.total_cost_basis_eur,
        total_market_value_eur=quant_metrics.total_market_value_eur,
        net_return_pct=quant_metrics.net_return_pct,
        twr_pct=quant_metrics.twr_pct,
        benchmark_ticker=quant_metrics.benchmark_ticker,
        benchmark_return_pct=quant_metrics.benchmark_return_pct,
        report_markdown=report_markdown,
    )
    with Session(get_engine(db_path)) as session:
        session.add(run)
        session.commit()
        return run.id


def read_audits_for_user(
    user_id: str, limit: int = 10, db_path: Path | str | None = None
) -> list[dict]:
    """Scoped, parameterized read of past audit runs for `user_id`, most
    recent first. Backs both `historical_db_read` (LLM-facing MCP tool) and
    `app.py`'s "Past audits" panel."""
    stmt = (
        select(AuditRun)
        .where(AuditRun.user_id == user_id)
        .order_by(desc(AuditRun.created_at))
        .limit(limit)
    )
    with Session(get_engine(db_path)) as session:
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(),
                "broker": row.broker,
                "total_cost_basis_eur": row.total_cost_basis_eur,
                "total_market_value_eur": row.total_market_value_eur,
                "net_return_pct": row.net_return_pct,
                "twr_pct": row.twr_pct,
                "benchmark_ticker": row.benchmark_ticker,
                "benchmark_return_pct": row.benchmark_return_pct,
            }
            for row in rows
        ]


def record_telemetry_event(
    name: str,
    props: dict,
    user_id: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Deterministic, parameterized insert of one telemetry event. The new
    backend for `telemetry.log_event` - props are stored as JSON text, same
    `default=str` convention `log_event` already used for its JSONL output."""
    event = TelemetryEvent(
        event_name=name,
        user_id=user_id,
        timestamp=datetime.now(timezone.utc),
        props_json=json.dumps(props, default=str),
    )
    with Session(get_engine(db_path)) as session:
        session.add(event)
        session.commit()


def read_telemetry_events(
    event_name: str | None = None, db_path: Path | str | None = None
) -> list[dict]:
    """Parameterized read of telemetry events, optionally filtered by
    `event_name`. Test-facing equivalent of the old JSONL-file `_events()`
    helpers."""
    stmt = select(TelemetryEvent).order_by(TelemetryEvent.id)
    if event_name is not None:
        stmt = stmt.where(TelemetryEvent.event_name == event_name)
    with Session(get_engine(db_path)) as session:
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "event": row.event_name,
                "user_id": row.user_id,
                "timestamp": row.timestamp.isoformat(),
                **json.loads(row.props_json),
            }
            for row in rows
        ]


def get_or_create_local_user_id(config_dir: Path | None = None) -> str:
    """Returns this local install's user id, generating and persisting a new
    UUID on first call. Phase 1 has no real auth (CLAUDE.md Section 4) - this
    is what makes `historical_db_read(user_id)`'s scoping real today, so it
    doesn't need reworking once Phase 2 adds actual multi-tenant auth."""
    resolved_dir = config_dir if config_dir is not None else DEFAULT_CONFIG_DIR
    id_path = resolved_dir / "user_id"
    if id_path.exists():
        return id_path.read_text(encoding="utf-8").strip()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    new_id = str(uuid.uuid4())
    id_path.write_text(new_id, encoding="utf-8")
    return new_id
