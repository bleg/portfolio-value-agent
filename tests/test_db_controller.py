from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src import db_controller, telemetry
from src.app import export_report
from src.state import Holding, QuantMetrics


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setattr("src.db_controller.DEFAULT_DB_PATH", path)
    return path


def _quant_metrics() -> QuantMetrics:
    return QuantMetrics(
        as_of=date(2024, 12, 31),
        holdings=[
            Holding(
                ticker="AAPL",
                isin="US0378331005",
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
        ],
        total_cost_basis_eur=Decimal("1000"),
        total_market_value_eur=Decimal("1200"),
        weighted_pe=15.0,
        net_return_pct=0.2,
        twr_pct=0.10,
        benchmark_ticker="^GSPC",
        benchmark_return_pct=0.08,
        unresolved_isins=[],
    )


def test_save_and_read_audit_run_round_trips(db_path):
    run_id = db_controller.save_audit_run(
        "user-a", _quant_metrics(), "# Report", "DEGIRO", db_path=db_path
    )
    assert isinstance(run_id, int)

    audits = db_controller.read_audits_for_user("user-a", db_path=db_path)
    assert len(audits) == 1
    assert audits[0]["broker"] == "DEGIRO"
    assert audits[0]["twr_pct"] == 0.10
    assert audits[0]["benchmark_return_pct"] == 0.08


def test_read_audits_for_user_scoped_no_cross_user_leakage(db_path):
    db_controller.save_audit_run("user-a", _quant_metrics(), "# A", "DEGIRO", db_path=db_path)
    db_controller.save_audit_run("user-b", _quant_metrics(), "# B", "DEGIRO", db_path=db_path)

    a_audits = db_controller.read_audits_for_user("user-a", db_path=db_path)
    b_audits = db_controller.read_audits_for_user("user-b", db_path=db_path)
    assert len(a_audits) == 1
    assert len(b_audits) == 1


def test_read_audits_for_user_treats_user_id_as_literal_value(db_path):
    db_controller.save_audit_run("user-a", _quant_metrics(), "# A", "DEGIRO", db_path=db_path)

    injection_attempt = "' OR '1'='1"
    audits = db_controller.read_audits_for_user(injection_attempt, db_path=db_path)
    assert audits == []

    still_scoped = db_controller.read_audits_for_user("user-a", db_path=db_path)
    assert len(still_scoped) == 1


def test_read_audits_for_user_respects_limit(db_path):
    for _ in range(3):
        db_controller.save_audit_run("user-a", _quant_metrics(), "# A", "DEGIRO", db_path=db_path)

    audits = db_controller.read_audits_for_user("user-a", limit=2, db_path=db_path)
    assert len(audits) == 2


def test_record_and_read_telemetry_event_round_trips(db_path):
    db_controller.record_telemetry_event(
        "portfolio_ingested", {"transaction_count": 5}, db_path=db_path
    )

    events = db_controller.read_telemetry_events(db_path=db_path)
    assert len(events) == 1
    assert events[0]["event"] == "portfolio_ingested"
    assert events[0]["transaction_count"] == 5


def test_read_telemetry_events_filters_by_event_name(db_path):
    db_controller.record_telemetry_event("portfolio_ingested", {}, db_path=db_path)
    db_controller.record_telemetry_event("report_exported", {}, db_path=db_path)

    exported = db_controller.read_telemetry_events(event_name="report_exported", db_path=db_path)
    assert len(exported) == 1
    assert exported[0]["event"] == "report_exported"


def test_get_or_create_local_user_id_persists_across_calls(tmp_path):
    config_dir = tmp_path / "config"
    first = db_controller.get_or_create_local_user_id(config_dir=config_dir)
    second = db_controller.get_or_create_local_user_id(config_dir=config_dir)
    assert first == second


def test_export_report_fires_report_exported_exactly_once(db_path):
    export_report("# Some report", "user-a")

    events = db_controller.read_telemetry_events(event_name="report_exported", db_path=db_path)
    assert len(events) == 1
    assert events[0]["user_id"] == "user-a"
