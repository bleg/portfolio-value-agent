from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.parsers import broker_llm
from src.parsers.base_parser import INTERNAL_COLUMNS
from src.parsers.broker_llm import (
    MAX_HEADER_CELL_LENGTH,
    MAX_HEADER_COLUMNS,
    UnsupportedCsvShapeError,
    parse_broker_csv,
)
from src.resilience import MAX_ATTEMPTS, McpToolError

FIXTURE = Path(__file__).parent / "fixtures" / "degiro_renamed_columns.csv"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture
def telemetry_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr("src.telemetry.DEFAULT_LOG_PATH", path)
    return path


@pytest.fixture
def cache_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "schema_cache.json"
    monkeypatch.setattr(broker_llm, "DEFAULT_CACHE_PATH", path)
    return path


def _events(telemetry_path: Path) -> list[dict]:
    if not telemetry_path.exists():
        return []
    return [json.loads(line) for line in telemetry_path.read_text().splitlines()]


class FakeCompletions:
    def __init__(self, mapping_provider) -> None:
        self.calls: list[dict] = []
        self._mapping_provider = mapping_provider

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        mapping = self._mapping_provider()
        content = json.dumps({"column_mapping": mapping})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _install_fake_openai(monkeypatch, mapping_provider) -> FakeCompletions:
    completions = FakeCompletions(mapping_provider)
    monkeypatch.setattr(broker_llm, "OpenAI", lambda: FakeOpenAIClient(completions))
    return completions


# --- fallback + correct mapping ---------------------------------------------


def test_fallback_maps_and_parses_correctly(monkeypatch, telemetry_path, cache_path):
    completions = _install_fake_openai(monkeypatch, lambda: list(INTERNAL_COLUMNS))

    portfolio = parse_broker_csv(FIXTURE)

    assert len(portfolio["transactions"]) == 5
    assert len(portfolio["skipped_rows"]) == 1
    apple = next(t for t in portfolio["transactions"] if t.isin == "US0378331005")
    assert apple.ticker == "AAPL"

    assert len(completions.calls) == 1

    events = _events(telemetry_path)
    healed = [e for e in events if e["event"] == "schema_healed"]
    assert len(healed) == 1
    assert healed[0]["broker"] == "DEGIRO"
    assert healed[0]["fields_mapped"] == len(INTERNAL_COLUMNS)

    ingested = [e for e in events if e["event"] == "portfolio_ingested"]
    assert len(ingested) == 1

    assert cache_path.exists()


# --- cache hit ---------------------------------------------------------------


def test_cache_hit_skips_second_llm_call(monkeypatch, telemetry_path, cache_path):
    completions = _install_fake_openai(monkeypatch, lambda: list(INTERNAL_COLUMNS))

    parse_broker_csv(FIXTURE)
    parse_broker_csv(FIXTURE)

    assert len(completions.calls) == 1

    events = _events(telemetry_path)
    healed = [e for e in events if e["event"] == "schema_healed"]
    assert len(healed) == 1

    ingested = [e for e in events if e["event"] == "portfolio_ingested"]
    assert len(ingested) == 2


# --- malformed mapping ---------------------------------------------------------


def test_malformed_mapping_retries_then_raises_mcp_tool_error(
    monkeypatch, telemetry_path, cache_path
):
    bad_mapping = list(INTERNAL_COLUMNS)
    bad_mapping[-1] = bad_mapping[0]  # duplicate field, omits another -> invalid coverage
    completions = _install_fake_openai(monkeypatch, lambda: bad_mapping)

    with pytest.raises(McpToolError):
        parse_broker_csv(FIXTURE)

    assert len(completions.calls) == MAX_ATTEMPTS

    failures = [e for e in _events(telemetry_path) if e["event"] == "mcp_tool_failure"]
    assert len(failures) == 1
    assert failures[0]["tool"] == "broker_llm_schema_mapping"

    assert not cache_path.exists()


# --- header-shape guardrails ---------------------------------------------------


def test_oversized_column_count_is_rejected_without_llm_call(
    monkeypatch, telemetry_path, cache_path, tmp_path
):
    completions = _install_fake_openai(monkeypatch, lambda: list(INTERNAL_COLUMNS))

    header = [f"col_{i}" for i in range(MAX_HEADER_COLUMNS + 1)]
    csv_path = tmp_path / "too_many_columns.csv"
    csv_path.write_text(",".join(header) + "\n" + ",".join(["x"] * len(header)) + "\n")

    with pytest.raises(UnsupportedCsvShapeError):
        parse_broker_csv(csv_path)

    assert len(completions.calls) == 0
    assert _events(telemetry_path) == []


def test_oversized_header_cell_is_rejected_without_llm_call(
    monkeypatch, telemetry_path, cache_path, tmp_path
):
    completions = _install_fake_openai(monkeypatch, lambda: list(INTERNAL_COLUMNS))

    injected_cell = "x" * (MAX_HEADER_CELL_LENGTH + 1)
    header = ["Date", "Time", injected_cell]
    csv_path = tmp_path / "oversized_cell.csv"
    csv_path.write_text(",".join(header) + "\n" + "a,b,c\n")

    with pytest.raises(UnsupportedCsvShapeError):
        parse_broker_csv(csv_path)

    assert len(completions.calls) == 0
    assert _events(telemetry_path) == []
