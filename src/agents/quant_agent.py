"""The Quant Agent (Epic 4): deterministic valuation math, no LLM involved.

CLAUDE.md Section 3C is explicit that no LLM may ever compute a number here.
This module aggregates `PortfolioState.transactions` into current holdings
(average-cost lot accounting), fetches live fundamentals/price history via
the Epic 2 MCP tools, and computes Weighted P/E, Cost Basis, Net Return, and
Time-Weighted Return (TWR) vs a benchmark index.

Known limitations:
- Dividends aren't captured anywhere upstream yet (no parser support), so
  net_return_pct/twr_pct understate true return for dividend-paying holdings
  until a future epic adds dividend-row parsing. A `quantity == 0` row (which
  a dividend would be) is not a case `_replay_positions` handles today.
- Transactions whose ISIN has no ticker mapping (see `unresolved_isins`) are
  excluded from valuation and TWR cash-flow breakpoints entirely — their cash
  flows aren't represented in the computed metrics.
- `_fx_pair` assumes `Transaction.price_currency` is a standard ISO 4217 code
  usable directly in a yfinance pair (e.g. "USD" -> "EURUSD=X"). DEGIRO can
  report UK line items in GBX (pence), which this does not special-case —
  deferred alongside the roadmap's other multi-broker currency-convention
  work (see epics_roadmap.md's Epic 3 deferred note).

`McpToolError` (exhausted MCP-tool/FX-history retries) and `ValueError` /
`InsufficientHoldingsError` (data-integrity guardrails) are allowed to
propagate uncaught out of `run_quant_agent` — per the Epic 5 roadmap note,
catching happens at the agent-node/Supervisor boundary built in Epic 6, not
inside individual agent modules.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import NamedTuple

from src import mcp_server, telemetry
from src.parsers.fx import get_fx_history
from src.state import Holding, PortfolioState, QuantMetrics, Transaction

DEFAULT_BENCHMARK_TICKER = "^GSPC"

# Must cover from the earliest transaction to `as_of`, not
# yfinance_fundamentals'/benchmark_data_fetcher's own shorter defaults
# (6mo/1y) — TWR needs the full history.
_HISTORY_PERIOD = "max"


class InsufficientHoldingsError(ValueError):
    """Raised when a sell's quantity exceeds recorded holdings for that
    ticker — a data-integrity guardrail, not a transient failure."""


class _Position(NamedTuple):
    quantity: int
    total_cost_eur: Decimal


def run_quant_agent(
    state: PortfolioState,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    as_of: date | None = None,
) -> PortfolioState:
    """Compute Weighted P/E, Cost Basis, Net Return, and TWR vs `benchmark_ticker`.

    Returns a shallow copy of `state` with `quant_metrics` populated, so it
    composes cleanly as a future LangGraph node (Epic 6). Fires
    `benchmark_compared` telemetry exactly once, at the end, only on success.
    """
    resolved_as_of = as_of if as_of is not None else date.today()
    transactions = state.get("transactions", [])
    if not transactions:
        raise ValueError("PortfolioState has no transactions to compute quant metrics for")

    ticker_txns, unresolved_isins = _group_by_ticker(transactions)

    # Fetch fundamentals for every ticker ever transacted, not just current
    # holdings — a ticker fully sold mid-period still needs its price history
    # to value the sub-periods while it was held, or TWR breaks.
    fundamentals_by_ticker: dict[str, dict] = {}
    price_histories: dict[str, list[dict]] = {}
    currencies: dict[str, str] = {}
    for ticker, txns in ticker_txns.items():
        fundamentals = mcp_server.yfinance_fundamentals(ticker, history_period=_HISTORY_PERIOD)
        fundamentals_by_ticker[ticker] = fundamentals
        price_histories[ticker] = fundamentals["price_history"]
        currencies[ticker] = txns[0].price_currency

    fx_histories: dict[str, list[dict]] = {}
    for currency in set(currencies.values()):
        if currency == "EUR":
            continue
        fx_histories[currency] = get_fx_history(_fx_pair(currency), period=_HISTORY_PERIOD)

    positions = _replay_positions(ticker_txns)

    holdings: list[Holding] = []
    for ticker, position in positions.items():
        if position.quantity <= 0:
            continue
        fundamentals = fundamentals_by_ticker[ticker]
        current_price_eur = _price_eur_on(
            ticker, resolved_as_of, price_histories, currencies, fx_histories
        )
        market_value_eur = current_price_eur * position.quantity
        avg_cost_basis_eur = position.total_cost_eur / position.quantity
        unrealized_return_pct = (
            float((market_value_eur - position.total_cost_eur) / position.total_cost_eur)
            if position.total_cost_eur != 0
            else 0.0
        )
        holdings.append(
            Holding(
                ticker=ticker,
                isin=ticker_txns[ticker][0].isin,
                quantity=position.quantity,
                avg_cost_basis_eur=avg_cost_basis_eur,
                total_cost_basis_eur=position.total_cost_eur,
                current_price_eur=current_price_eur,
                market_value_eur=market_value_eur,
                unrealized_return_pct=unrealized_return_pct,
                pe_ratio=fundamentals["pe_ratio"],
                pb_ratio=fundamentals["pb_ratio"],
                debt_to_equity=fundamentals["debt_to_equity"],
                fcf_yield=fundamentals["fcf_yield"],
            )
        )

    total_cost_basis_eur = sum((h.total_cost_basis_eur for h in holdings), Decimal("0"))
    total_market_value_eur = sum((h.market_value_eur for h in holdings), Decimal("0"))
    weighted_pe = _weighted_pe(holdings)

    total_invested_eur = sum(
        (abs(t.total_eur) for txns in ticker_txns.values() for t in txns if t.quantity > 0),
        Decimal("0"),
    )
    total_realized_eur = sum(
        (abs(t.total_eur) for txns in ticker_txns.values() for t in txns if t.quantity < 0),
        Decimal("0"),
    )
    net_return_pct = (
        float(
            (total_market_value_eur + total_realized_eur - total_invested_eur)
            / total_invested_eur
        )
        if total_invested_eur != 0
        else 0.0
    )

    first_txn_date = min(t.trade_date for t in transactions)
    benchmark_data = mcp_server.benchmark_data_fetcher(
        benchmark_ticker, history_period=_HISTORY_PERIOD
    )
    benchmark_return_pct = _compute_benchmark_return(
        benchmark_data["price_history"], first_txn_date, resolved_as_of
    )

    twr_pct = _compute_twr(ticker_txns, price_histories, currencies, fx_histories, resolved_as_of)

    quant_metrics = QuantMetrics(
        as_of=resolved_as_of,
        holdings=holdings,
        total_cost_basis_eur=total_cost_basis_eur,
        total_market_value_eur=total_market_value_eur,
        weighted_pe=weighted_pe,
        net_return_pct=net_return_pct,
        twr_pct=twr_pct,
        benchmark_ticker=benchmark_ticker,
        benchmark_return_pct=benchmark_return_pct,
        unresolved_isins=unresolved_isins,
    )

    telemetry.log_event("benchmark_compared", twr=twr_pct, benchmark_return=benchmark_return_pct)

    return {**state, "quant_metrics": quant_metrics}


def _group_by_ticker(
    transactions: list[Transaction],
) -> tuple[dict[str, list[Transaction]], list[str]]:
    """Group transactions by resolved ticker, sorted chronologically per
    ticker. ISINs with no ticker mapping are collected separately."""
    ticker_txns: dict[str, list[Transaction]] = {}
    unresolved_isins: list[str] = []
    for txn in transactions:
        if txn.ticker is None:
            unresolved_isins.append(txn.isin)
            continue
        ticker_txns.setdefault(txn.ticker, []).append(txn)
    for txns in ticker_txns.values():
        txns.sort(key=lambda t: (t.trade_date, t.trade_time or time.min))
    return ticker_txns, unresolved_isins


def _replay_positions(
    ticker_txns: dict[str, list[Transaction]],
    cutoff: date | None = None,
    inclusive: bool = True,
) -> dict[str, _Position]:
    """Average-cost lot accounting, replayed up to `cutoff` (or in full if
    `cutoff` is None). `inclusive` controls whether `cutoff`'s own
    transactions are included — used to build the "before"/"after" snapshots
    TWR needs at each breakpoint date.

    Direction is taken from `quantity`'s sign only (the only reliable buy/
    sell signal — `total_eur`'s sign is not consistent in real broker data).
    A corporate-action row (`total_eur == 0`) falls into the buy branch for
    free: it adds quantity at zero cost, diluting the average cost per share,
    with no special-casing needed.
    """
    positions: dict[str, _Position] = {}
    for ticker, txns in ticker_txns.items():
        qty = 0
        cost = Decimal("0")
        for txn in txns:
            if cutoff is not None:
                if inclusive and txn.trade_date > cutoff:
                    break
                if not inclusive and txn.trade_date >= cutoff:
                    break
            if txn.quantity > 0:
                qty += txn.quantity
                cost += abs(txn.total_eur)
            elif txn.quantity < 0:
                sell_qty = -txn.quantity
                if sell_qty > qty:
                    raise InsufficientHoldingsError(
                        f"{ticker}: sell of {sell_qty} on {txn.trade_date} exceeds "
                        f"held quantity {qty}"
                    )
                avg_cost = cost / qty
                cost -= avg_cost * sell_qty
                qty -= sell_qty
            # quantity == 0 (e.g. a future dividend row) is not a position event.
        positions[ticker] = _Position(quantity=qty, total_cost_eur=cost)
    return positions


def _value_on_or_before(series: list[dict], on_date: date) -> Decimal:
    """Forward-fill lookup shared by ticker price series and FX series (both
    are `{"date": iso, "close": float}` lists)."""
    candidates = [entry for entry in series if date.fromisoformat(entry["date"]) <= on_date]
    if not candidates:
        raise ValueError(f"No data on or before {on_date} in series")
    latest = max(candidates, key=lambda entry: entry["date"])
    return Decimal(str(latest["close"]))


def _fx_pair(currency: str) -> str:
    return f"EUR{currency}=X"


def _price_eur_on(
    ticker: str,
    on_date: date,
    price_histories: dict[str, list[dict]],
    currencies: dict[str, str],
    fx_histories: dict[str, list[dict]],
) -> Decimal:
    native_price = _value_on_or_before(price_histories[ticker], on_date)
    currency = currencies[ticker]
    if currency == "EUR":
        return native_price
    fx_rate = _value_on_or_before(fx_histories[currency], on_date)
    return native_price / fx_rate


def _portfolio_value_eur(
    positions: dict[str, int],
    on_date: date,
    price_histories: dict[str, list[dict]],
    currencies: dict[str, str],
    fx_histories: dict[str, list[dict]],
) -> Decimal:
    total = Decimal("0")
    for ticker, quantity in positions.items():
        if quantity <= 0:
            continue
        total += quantity * _price_eur_on(
            ticker, on_date, price_histories, currencies, fx_histories
        )
    return total


def _compute_twr(
    ticker_txns: dict[str, list[Transaction]],
    price_histories: dict[str, list[dict]],
    currencies: dict[str, str],
    fx_histories: dict[str, list[dict]],
    as_of: date,
) -> float:
    """Sub-period (chain-linked) TWR, using external-cash-flow dates as
    breakpoints. Corporate-action rows (`total_eur == 0`) don't get their own
    breakpoint — they just change quantity, which is picked up automatically
    by the position snapshot at the next real breakpoint.

    The first breakpoint is only ever used as a sub-period's `start`, never
    an `end` — there's no "value before the first cash flow" to compute a
    return over, so the inception boundary needs no special-casing.
    """
    cash_flow_dates = {
        t.trade_date for txns in ticker_txns.values() for t in txns if t.total_eur != 0
    }
    breakpoints = sorted(cash_flow_dates | {as_of})
    if len(breakpoints) < 2:
        return 0.0

    twr = Decimal("1")
    for start, end in zip(breakpoints, breakpoints[1:]):
        start_positions = {
            ticker: pos.quantity
            for ticker, pos in _replay_positions(ticker_txns, cutoff=start, inclusive=True).items()
        }
        end_positions = {
            ticker: pos.quantity
            for ticker, pos in _replay_positions(
                ticker_txns, cutoff=end, inclusive=False
            ).items()
        }
        v_start = _portfolio_value_eur(start_positions, start, price_histories, currencies, fx_histories)
        v_end = _portfolio_value_eur(end_positions, end, price_histories, currencies, fx_histories)
        if v_start == 0:
            continue
        twr *= v_end / v_start
    return float(twr - 1)


def _compute_benchmark_return(price_history: list[dict], first_txn_date: date, as_of: date) -> float:
    """Simple buy-and-hold return over the same window as the portfolio's
    TWR — not TWR itself, since a benchmark index has no cash flows."""
    start = _value_on_or_before(price_history, first_txn_date)
    end = _value_on_or_before(price_history, as_of)
    return float((end - start) / start)


def _weighted_pe(holdings: list[Holding]) -> float | None:
    """Market-value-weighted average P/E, excluding holdings with no P/E and
    renormalizing over the rest. None if no holding has a P/E."""
    priced = [h for h in holdings if h.pe_ratio is not None]
    total_value = sum((h.market_value_eur for h in priced), Decimal("0"))
    if not priced or total_value == 0:
        return None
    weighted_sum = sum(
        (Decimal(str(h.pe_ratio)) * h.market_value_eur for h in priced), Decimal("0")
    )
    return float(weighted_sum / total_value)
