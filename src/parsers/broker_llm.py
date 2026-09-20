"""Self-healing LLM CSV-schema fallback (CLAUDE.md Section 3A).

When `base_parser.py`'s deterministic DEGIRO parser hits a header it doesn't
recognize (e.g. a broker renamed/reordered columns), `parse_broker_csv` here
catches `UnrecognizedBrokerFormatError` and asks `gpt-4o-mini` to map the new
header onto the internal schema instead of hard-failing. The mapping is
cached locally (`schema_cache.json`) keyed by a hash of the raw header, so
the same renamed export never costs a second LLM call.

Data-minimization: only the header text and locally-computed, value-free
type tags are ever sent to the LLM — never real cell values. Before a
mapping exists, `pii_scrubber.scrub_row`'s exact-name matching can't yet
tell which raw column is an account number or order ID, so no unscrubbed
sample value is safe to send at this stage.

Guardrails: `_check_header_shape` rejects an oversized/adversarial header
(too many columns, or an implausibly long column name — e.g. a
prompt-injection payload stuffed into a header cell) before any LLM call is
attempted, and the type-profiling read is capped to `SAMPLE_ROW_COUNT` rows
regardless of total file size, so cost never scales with how large an
uploaded CSV is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.parsers.base_parser import (
    INTERNAL_COLUMNS,
    UnrecognizedBrokerFormatError,
    _build_transactions,
    _read_header,
    parse_degiro_csv,
)
from src.resilience import resilient_tool
from src.state import PortfolioState
from src.telemetry import log_event

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("schema_cache.json")
DEFAULT_MODEL = "gpt-4o-mini"

MAX_HEADER_COLUMNS = 40  # DEGIRO's real header has 17; generous headroom, still
                          # rejects a garbage/adversarial file with hundreds of "columns"
MAX_HEADER_CELL_LENGTH = 100  # DEGIRO's longest real header cell is ~40 chars
SAMPLE_ROW_COUNT = 5  # rows read for type-profiling, regardless of total file size

_SYSTEM_PROMPT = f"""You map a broker CSV export's column headers onto a fixed internal schema.

The internal schema (target field names) is exactly:
{json.dumps(INTERNAL_COLUMNS)}

You will be given the CSV's raw header row and a locally-computed type
profile for each column (never real cell values). Respond with a JSON
object of the form {{"column_mapping": [...]}} where the list is positionally
aligned to the input header: each entry is either one of the internal
schema field names above, or null if that column doesn't correspond to any
of them. Every internal schema field name must appear exactly once across
the whole list (no duplicates, no omissions)."""


class UnsupportedCsvShapeError(Exception):
    """Raised when a CSV's header shape is rejected before any LLM call is attempted.

    Distinct from UnrecognizedBrokerFormatError (an expected trigger for the
    fallback): this means the fallback itself refuses to even try, because
    the header looks adversarial (e.g. crafted to smuggle prompt-injection
    text) or malformed rather than like a real, renamed broker export.
    """


def _check_header_shape(header: list[str]) -> None:
    if len(header) > MAX_HEADER_COLUMNS:
        raise UnsupportedCsvShapeError(
            f"{len(header)} columns exceeds the {MAX_HEADER_COLUMNS}-column limit"
        )
    for cell in header:
        if len(cell) > MAX_HEADER_CELL_LENGTH:
            raise UnsupportedCsvShapeError(
                f"header cell exceeds {MAX_HEADER_CELL_LENGTH} chars: {cell[:50]!r}..."
            )


_DATE_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
_DECIMAL_COMMA_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*,\d+$")
_DECIMAL_DOT_RE = re.compile(r"^-?\d+(,\d{3})*\.\d+$")
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INT_RE = re.compile(r"^-?\d+$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def _profile_column(sample_values: list[str]) -> str:
    """Locally categorize a column's sample values into a value-free type tag.

    Never returns an actual cell value — only a coarse category string, so
    this is safe to send to an external LLM even before PII columns can be
    identified by name.
    """
    non_blank = [v for v in sample_values if v]
    if not non_blank:
        return "blank"
    if all(_DATE_RE.match(v) for v in non_blank):
        return "date (DD-MM-YYYY or similar)"
    if all(_UUID_RE.match(v) for v in non_blank):
        return "UUID-like identifier"
    if all(_ISIN_RE.match(v) for v in non_blank):
        return "ISIN-like identifier"
    if all(_CURRENCY_RE.match(v) for v in non_blank):
        return "3-letter currency code"
    if all(_DECIMAL_COMMA_RE.match(v) for v in non_blank):
        return "decimal number (comma decimal separator)"
    if all(_DECIMAL_DOT_RE.match(v) for v in non_blank):
        return "decimal number (dot decimal separator)"
    if all(_INT_RE.match(v) for v in non_blank):
        return "integer"
    return "free text"


def _validate_mapping(header: list[str], mapping: object) -> None:
    if not isinstance(mapping, list) or len(mapping) != len(header):
        raise ValueError(
            f"column_mapping must be a list of length {len(header)}, got {mapping!r}"
        )
    mapped_fields = [m for m in mapping if m is not None]
    if sorted(mapped_fields) != sorted(INTERNAL_COLUMNS):
        raise ValueError(
            f"column_mapping must cover each of {INTERNAL_COLUMNS} exactly once, "
            f"got {mapping!r}"
        )


@resilient_tool(tool_name="broker_llm_schema_mapping")
def _request_column_mapping(header: list[str], profiles: list[str], model: str) -> list[str | None]:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"header": header, "column_profiles": profiles})},
        ],
    )
    mapping = json.loads(response.choices[0].message.content)["column_mapping"]
    _validate_mapping(header, mapping)
    return mapping


def _hash_header(header: list[str]) -> str:
    return hashlib.sha256(json.dumps(header).encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if not DEFAULT_CACHE_PATH.exists():
        return {}
    return json.loads(DEFAULT_CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict) -> None:
    DEFAULT_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _apply_mapping(df: pd.DataFrame, mapping: list[str | None]) -> pd.DataFrame:
    keep_mask = [m is not None for m in mapping]
    mapped_df = df.loc[:, keep_mask].copy()
    mapped_df.columns = [m for m in mapping if m is not None]
    return mapped_df[INTERNAL_COLUMNS]


def _parse_with_llm_fallback(path: Path) -> PortfolioState:
    header = _read_header(path)
    _check_header_shape(header)

    cache_key = _hash_header(header)
    cache = _load_cache()
    entry = cache.get(cache_key)

    if entry is not None:
        mapping = entry["column_mapping"]
    else:
        sample_df = pd.read_csv(
            path, header=None, skiprows=1, nrows=SAMPLE_ROW_COUNT,
            dtype=str, keep_default_na=False,
        )
        profiles = [_profile_column(sample_df.iloc[:, i].tolist()) for i in range(len(header))]
        model = os.environ.get("BROKER_LLM_MODEL", DEFAULT_MODEL)
        mapping = _request_column_mapping(header, profiles, model)
        cache[cache_key] = {
            "broker": "DEGIRO",
            "raw_header": header,
            "column_mapping": mapping,
            "model": model,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_cache(cache)
        log_event(
            "schema_healed",
            broker="DEGIRO",
            fields_mapped=sum(1 for m in mapping if m is not None),
        )

    df = pd.read_csv(path, header=None, skiprows=1, dtype=str, keep_default_na=False)
    mapped_df = _apply_mapping(df, mapping)
    transactions, skipped_rows = _build_transactions(mapped_df, broker="DEGIRO")

    state: PortfolioState = {
        "transactions": transactions,
        "broker": "DEGIRO",
        "base_currency": "EUR",
        "skipped_rows": skipped_rows,
    }

    log_event(
        "portfolio_ingested",
        broker="DEGIRO",
        transaction_count=len(transactions),
        skipped_row_count=len(skipped_rows),
    )

    return state


def parse_broker_csv(path: str | Path) -> PortfolioState:
    """Parse a broker CSV, falling back to the LLM schema-mapping healer
    if the deterministic DEGIRO parser doesn't recognize the header."""
    path = Path(path)
    try:
        return parse_degiro_csv(path)
    except UnrecognizedBrokerFormatError:
        logger.info("%s: unrecognized header, attempting LLM schema-mapping fallback", path.name)
        return _parse_with_llm_fallback(path)
