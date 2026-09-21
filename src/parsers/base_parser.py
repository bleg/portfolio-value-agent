"""Deterministic DEGIRO transactions-CSV parser (CLAUDE.md Section 3A).

No LLM, no network calls — sub-second, for any recognized broker format.
If a broker changes its export format, `UnrecognizedBrokerFormatError` is
raised here and caught by Epic 3's `broker_llm.py` self-healing fallback.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.parsers import isin_resolver
from src.parsers.pii_scrubber import scrub_row
from src.state import PortfolioState, Transaction
from src.telemetry import log_event

logger = logging.getLogger(__name__)

# DEGIRO's English "Transactions" export header. Two columns are unnamed in
# the raw file — they hold the currency code for the preceding Price and
# Local value columns respectively.
EXPECTED_DEGIRO_HEADER = [
    "Date", "Time", "Product", "ISIN", "Reference exchange", "Venue",
    "Quantity", "Price", "", "Local value", "", "Value EUR",
    "Exchange rate", "AutoFX Fee", "Transaction and/or third party fees EUR",
    "Total EUR", "Order ID",
]

INTERNAL_COLUMNS = [
    "date", "time", "product_name", "isin", "reference_exchange", "venue",
    "quantity", "price_local", "price_currency", "local_value", "local_currency",
    "value_eur", "exchange_rate", "autofx_fee", "fees_eur", "total_eur", "order_id",
]

# Known single-stock ISIN -> ticker mappings — a fast, zero-network first
# tier checked before falling back to `isin_resolver.py`'s OpenFIGI lookup
# for anything not covered here. Also serves as an authoritative override
# for any ISIN where the live resolver's exchange-priority heuristics might
# otherwise pick a less-preferred (but still valid) cross-listing.
ISIN_TO_TICKER: dict[str, str] = {
    "US70450Y1038": "PYPL",
    "US00724F1012": "ADBE",
    "US5949181045": "MSFT",
    "US8887871080": "TOST",
    "US02079K3059": "GOOGL",
    "US57636Q1040": "MA",
    "US79466L3024": "CRM",
    "US90353T1007": "UBER",
    "US58933Y1055": "MRK",
    "KYG6683N1034": "NU",
    "US4781601046": "JNJ",
    "GB00BN7SWP63": "GSK",
    "US26603R1068": "DUOL",
    "US58733R1023": "MELI",
    "US78409V1044": "SPGI",
    "MHY235921357": "ESEA",
    "US0231351067": "AMZN",
    "US15135B1017": "CNC",
    "US01609W1027": "BABA",
    "US19260Q1076": "COIN",
    "US4581401001": "INTC",
    "US25243Q2057": "DEO",
    "US3453708600": "F",
    "US92826C8394": "V",
    "US30303M1027": "META",
    "US00287Y1091": "ABBV",
    "US91332U1016": "U",
    "US1689051076": "PLCE",
    "US9311421039": "WMT",
    "US5801351017": "MCD",
    "US0846707026": "BRK-B",
    "US5705351048": "MKL",
    "US81141R1005": "SE",
    "US1011371077": "BSX",
    "US0605051046": "BAC",
    "US2546871060": "DIS",
    "US0378331005": "AAPL",
}


class UnrecognizedBrokerFormatError(Exception):
    """Raised when a CSV's header doesn't match a known broker format.

    Epic 3's broker_llm.py self-healing fallback catches this and asks an
    LLM to map the new columns instead. Epic 1 only needs to raise it.
    """


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _parse_decimal(raw: str | float | int | None) -> Decimal | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    cleaned = str(raw).strip().replace(".", "").replace(",", ".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _date_range(transactions: list[Transaction]) -> list[str] | None:
    if not transactions:
        return None
    dates = sorted(t.trade_date for t in transactions)
    return [dates[0].isoformat(), dates[-1].isoformat()]


def _build_transactions(df: pd.DataFrame, broker: str) -> tuple[list[Transaction], list[dict]]:
    """Parse a dataframe already renamed to INTERNAL_COLUMNS into Transactions.

    Shared by the deterministic DEGIRO path and Epic 3's broker_llm.py LLM
    fallback path — this function only knows the internal schema, never any
    broker-specific header text.

    Ticker resolution happens in two phases so `isin_resolver.py`'s OpenFIGI
    lookup can run as one batched call per parse rather than once per row:
    phase A parses every row and notes which ISINs the static ISIN_TO_TICKER
    table missed; phase B resolves all of those misses in a single call;
    phase C builds the Transactions using the merged static+resolved mapping.
    """
    parsed_rows: list[dict] = []
    skipped_rows: list[dict] = []
    missing_isin_currency: dict[str, str | None] = {}

    for raw_row in df.to_dict(orient="records"):
        row = scrub_row(raw_row)

        trade_date_raw = row["date"]
        isin = row["isin"]
        quantity_raw = row["quantity"]

        if not trade_date_raw or not isin or not quantity_raw:
            skipped_rows.append({"raw_row": row, "reason": "missing date/isin/quantity"})
            continue

        try:
            trade_date = datetime.strptime(trade_date_raw, "%d-%m-%Y").date()
            quantity = int(quantity_raw)
        except ValueError as exc:
            skipped_rows.append({"raw_row": row, "reason": f"unparseable value: {exc}"})
            continue

        trade_time: time | None = None
        if row["time"]:
            trade_time = datetime.strptime(row["time"], "%H:%M").time()

        price_local = _parse_decimal(row["price_local"]) or Decimal("0")
        total_eur = _parse_decimal(row["total_eur"]) or Decimal("0")
        price_currency = row["price_currency"] or row["local_currency"] or "EUR"

        if isin not in ISIN_TO_TICKER:
            missing_isin_currency[isin] = price_currency

        parsed_rows.append(
            {
                "isin": isin,
                "product_name": row["product_name"],
                "trade_date": trade_date,
                "trade_time": trade_time,
                "quantity": quantity,
                "price_local": price_local,
                "price_currency": price_currency,
                "local_value": _parse_decimal(row["local_value"]) or Decimal("0"),
                "value_eur": _parse_decimal(row["value_eur"]) or Decimal("0"),
                "exchange_rate": _parse_decimal(row["exchange_rate"]),
                "fees_eur": _parse_decimal(row["fees_eur"]) or Decimal("0"),
                "total_eur": total_eur,
                "is_corporate_action": (price_local == 0 and total_eur == 0),
            }
        )

    resolved = isin_resolver.resolve_isins(missing_isin_currency) if missing_isin_currency else {}

    transactions: list[Transaction] = []
    for parsed in parsed_rows:
        isin = parsed["isin"]
        ticker = ISIN_TO_TICKER.get(isin) or resolved.get(isin)

        if ticker is None:
            logger.warning(
                "No ticker mapping for ISIN %s (%s); leaving ticker=None",
                isin, parsed["product_name"],
            )

        transactions.append(Transaction(ticker=ticker, broker=broker, **parsed))

    return transactions, skipped_rows


def parse_degiro_csv(path: str | Path) -> PortfolioState:
    path = Path(path)
    header = _read_header(path)
    if header != EXPECTED_DEGIRO_HEADER:
        raise UnrecognizedBrokerFormatError(
            f"{path.name} header does not match the known DEGIRO transactions "
            f"export format (got {header!r})"
        )

    df = pd.read_csv(
        path, skiprows=1, names=INTERNAL_COLUMNS, dtype=str, keep_default_na=False
    )

    transactions, skipped_rows = _build_transactions(df, broker="DEGIRO")

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
        date_range=_date_range(transactions),
    )

    return state
