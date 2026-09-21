"""OpenFIGI-based ISIN -> ticker resolver, for ISINs `base_parser.py`'s
static `ISIN_TO_TICKER` table doesn't cover (mostly European ETFs and
foreign-listed stocks).

Follows the same self-healing shape as `broker_llm.py`: a deterministic
attempt first (the static table, checked by the caller before this module is
even reached), then an external lookup on a miss, cached locally
(`isin_cache.json`, keyed by ISIN this time rather than a hashed header) so
the same ISIN never costs a second network round trip.

OpenFIGI's mapping API returns every listing for an ISIN - often 100+ across
global exchanges, OTC feeds, and dark pools, most of them irrelevant. Two
real hazards found by querying live data against this project's own test
portfolio during planning:

1. Wrong-symbol risk: a company's foreign OTC cross-listings can trade under
   a *different ticker* than its primary listing (ASML's German OTC feeds
   are ticker "ASME", not "ASML"). Exchange-code priority must be ranked
   deliberately, not just suffix-mapped.
2. Same-exchange, different-currency risk: some ETFs list several tickers
   under the *same* exchange code, one per currency share class (a Bitcoin
   ETC lists BITC/BITCGBP/BITCCHF/BITCEUR all on the same Swiss code) -
   exchange code alone can't disambiguate; the transaction's own currency
   can.

To guard against both, a candidate ticker is only ever cached after it's
independently verified to resolve via yfinance's `.history()` call - the
`/v8/finance/chart` endpoint, confirmed not crumb-gated (see
`src/mcp_server.py`'s `yfinance_fundamentals`), so this verification isn't
exposed to the same Yahoo rate-limiting that endpoint's `.info` call is.

An ISIN that never resolves (OpenFIGI has no match, or no candidate
verifies) is left out of the cache entirely - never negatively cached - so
falling through to today's unresolved/HITL behavior costs nothing, and a
later improvement to EXCHANGE_CLUSTERS_PRIORITY can pick it up for free on
the next run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf

from src import telemetry
from src.resilience import McpToolError, resilient_tool

DEFAULT_CACHE_PATH = Path("isin_cache.json")

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
BATCH_SIZE_NO_KEY = 10  # OpenFIGI's unauthenticated per-request job cap
BATCH_SIZE_WITH_KEY = 100  # cap with a free OPENFIGI_API_KEY

# (exchCode cluster, yfinance ticker suffix), in fallback priority order.
# Deliberately scoped to codes verified live against real portfolio ISINs
# while planning this module - not a global OpenFIGI exchCode catalog. An
# unmatched exchCode just leaves that candidate unused (safe default: falls
# through to unresolved, same as today's behavior) - extend this table as
# new exchanges are encountered rather than guessing ahead of time.
#
# Codes within a cluster are treated as equivalent, same-ticker regional
# venues/MTFs for one primary listing (confirmed for the US and German
# clusters against live data - e.g. Volkswagen's ticker "VOW" is identical
# across all of GR/GF/GD/GY/GS/GM/GI/GH), not distinct alternative listings.
#
# This static order is only a FALLBACK. A candidate's exchange cluster alone
# isn't a reliable "this is the real listing" signal in either direction:
# OpenFIGI tags foreign stocks' US OTC/pink-sheet cross-listings under the
# same generic "US"-family exchCodes as real domestic listings, but a
# different ticker (ASML's OTC equivalent is "ASMLF", exchCode "US" -
# confirmed live) - and conversely a US stock can have a spurious-but-real
# cross-listing on a European venue that also passes yfinance verification.
# `_pick_verified_ticker` below tries the ISIN's own home-country cluster
# FIRST (derived from the ISIN's 2-letter country prefix, e.g. "US"/"NL"/
# "DE" - a much stronger signal than any exchCode heuristic), then falls
# back to this list for anything without a home-cluster match or where the
# home listing doesn't verify.
EXCHANGE_CLUSTERS_PRIORITY: list[tuple[frozenset[str], str]] = [
    (frozenset({"NA"}), ".AS"),  # Euronext Amsterdam
    (frozenset({"LN"}), ".L"),  # London Stock Exchange
    (frozenset({"SM", "SQ"}), ".MC"),  # Bolsa de Madrid
    (frozenset({"IM"}), ".MI"),  # Borsa Italiana (Milan)
    (frozenset({"SW"}), ".SW"),  # SIX Swiss Exchange
    (frozenset({"FP", "PA"}), ".PA"),  # Euronext Paris
    (frozenset({"GR", "GF", "GD", "GY", "GS", "GM", "GH", "GI", "GT"}), ".DE"),  # German venue cluster
    (frozenset({"US", "UN", "UW", "UA", "UC", "UP", "UB", "UM", "UX"}), ""),  # US composite
]

# ISIN country-code prefix (ISO 3166-1 alpha-2, always the ISIN's first 2
# characters) -> the matching cluster above, tried first for that ISIN.
_COUNTRY_TO_CLUSTER: dict[str, tuple[frozenset[str], str]] = {
    "NL": EXCHANGE_CLUSTERS_PRIORITY[0],
    "GB": EXCHANGE_CLUSTERS_PRIORITY[1],
    "ES": EXCHANGE_CLUSTERS_PRIORITY[2],
    "IT": EXCHANGE_CLUSTERS_PRIORITY[3],
    "CH": EXCHANGE_CLUSTERS_PRIORITY[4],
    "FR": EXCHANGE_CLUSTERS_PRIORITY[5],
    "DE": EXCHANGE_CLUSTERS_PRIORITY[6],
    "US": EXCHANGE_CLUSTERS_PRIORITY[7],
}


def _load_cache() -> dict:
    if not DEFAULT_CACHE_PATH.exists():
        return {}
    return json.loads(DEFAULT_CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict) -> None:
    DEFAULT_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@resilient_tool(tool_name="openfigi_mapping")
def _request_mapping_batch(isins: list[str], api_key: str | None) -> list[dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    jobs = [{"idType": "ID_ISIN", "idValue": isin} for isin in isins]
    response = requests.post(OPENFIGI_MAPPING_URL, json=jobs, headers=headers, timeout=5.0)
    response.raise_for_status()
    results = response.json()
    if not isinstance(results, list) or len(results) != len(isins):
        raise ValueError(f"OpenFIGI response length mismatch for batch of {len(isins)}")
    return results


@resilient_tool(tool_name="isin_ticker_verification")
def _verify_via_yfinance(ticker: str) -> bool:
    history = yf.Ticker(ticker).history(period="5d", timeout=5.0)
    return not history.empty


def _cluster_order(isin: str) -> list[tuple[frozenset[str], str]]:
    """The ISIN's home-country cluster first (if we have one), then the
    static fallback order, home cluster not repeated."""
    home = _COUNTRY_TO_CLUSTER.get(isin[:2])
    if home is None:
        return EXCHANGE_CLUSTERS_PRIORITY
    return [home] + [c for c in EXCHANGE_CLUSTERS_PRIORITY if c != home]


def _pick_verified_ticker(isin: str, data_entries: list[dict], currency_hint: str | None) -> str | None:
    """Walk exchange clusters, home-country cluster first, then fallback
    priority order; within a matched cluster, prefer a ticker ending in the
    transaction's currency, else the shortest/bare ticker; return the first
    candidate that verifies via yfinance."""
    for codes, suffix in _cluster_order(isin):
        candidates = [d for d in data_entries if d.get("exchCode") in codes and d.get("ticker")]
        if not candidates:
            continue
        candidates.sort(
            key=lambda d: (
                0 if currency_hint and d["ticker"].endswith(currency_hint) else 1,
                len(d["ticker"]),
            )
        )
        for entry in candidates:
            raw_ticker = entry["ticker"]
            candidate = raw_ticker if raw_ticker.endswith(suffix) or not suffix else raw_ticker + suffix
            try:
                if _verify_via_yfinance(candidate):
                    return candidate
            except McpToolError:
                continue
    return None


def resolve_isins(isin_currency_map: dict[str, str | None]) -> dict[str, str | None]:
    """Resolve ISINs to yfinance-compatible tickers via OpenFIGI, caching
    successful, yfinance-verified mappings in isin_cache.json.

    Never raises - an OpenFIGI or yfinance failure just leaves the affected
    ISIN out of the returned dict, falling through to the caller's existing
    unresolved-ISIN handling rather than crashing the whole parse.
    """
    unique_isins = sorted(isin_currency_map)
    cache = _load_cache()
    resolved: dict[str, str | None] = {}
    to_query: list[str] = []

    for isin in unique_isins:
        entry = cache.get(isin)
        if entry is not None:
            resolved[isin] = entry["ticker"]
        else:
            to_query.append(isin)

    if not to_query:
        return resolved

    api_key = os.environ.get("OPENFIGI_API_KEY")
    batch_size = BATCH_SIZE_WITH_KEY if api_key else BATCH_SIZE_NO_KEY
    newly_resolved = 0

    for chunk in _chunks(to_query, batch_size):
        try:
            results = _request_mapping_batch(chunk, api_key)
        except McpToolError:
            continue

        for isin, result in zip(chunk, results):
            data_entries = result.get("data")
            if not data_entries:
                continue
            ticker = _pick_verified_ticker(isin, data_entries, isin_currency_map.get(isin))
            if ticker is None:
                continue
            cache[isin] = {
                "ticker": ticker,
                "source": "openfigi",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            resolved[isin] = ticker
            newly_resolved += 1

    if newly_resolved:
        _save_cache(cache)

    telemetry.log_event(
        "isin_resolved",
        source="openfigi",
        isins_queried=len(to_query),
        isins_resolved=newly_resolved,
        isins_still_unresolved=len(to_query) - newly_resolved,
    )

    return resolved
