"""Retry-with-backoff and timeout wrapper for MCP tool calls (CLAUDE.md Section 3B).

External calls (yfinance, SEC EDGAR, DuckDuckGo) are known to rate-limit or
fail intermittently. `resilient_tool` wraps a tool function so every call
gets a hard timeout and a short retry-with-backoff, and fires
`telemetry.log_event("mcp_tool_failure", ...)` exactly once on exhaustion.

This is the first retry/backoff primitive in the codebase; it lives here
(not under an MCP-only subpackage) because it's a generic cross-cutting
concern later epics (e.g. Epic 3's `broker_llm.py` OpenAI calls) will
plausibly reuse.
"""

from __future__ import annotations

import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

from src import telemetry

MAX_ATTEMPTS = 2
WAIT_MULTIPLIER_SECONDS = 1
WAIT_MAX_SECONDS = 3
DEFAULT_TIMEOUT_SECONDS = 6.0

F = TypeVar("F", bound=Callable[..., Any])


class McpToolError(Exception):
    """Raised when an MCP tool call fails after exhausting retries."""


def resilient_tool(tool_name: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Callable[[F], F]:
    """Wrap an MCP tool function with retry-with-backoff and a hard timeout.

    On final exhaustion, fires `mcp_tool_failure` exactly once (via
    tenacity's `retry_error_callback`, not `after=`, which would fire once
    per attempt) and raises `McpToolError`.
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        def _extract_ticker(args: tuple, kwargs: dict) -> str | None:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return bound.arguments.get("ticker")
            except TypeError:
                return None

        def _on_exhausted(retry_state: RetryCallState) -> Any:
            exc = retry_state.outcome.exception()
            ticker = _extract_ticker(retry_state.args, retry_state.kwargs)
            telemetry.log_event(
                "mcp_tool_failure",
                tool=tool_name,
                ticker=ticker,
                attempts=retry_state.attempt_number,
                error=repr(exc),
            )
            raise McpToolError(
                f"{tool_name} failed after {retry_state.attempt_number} attempts"
            ) from exc

        @retry(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=WAIT_MULTIPLIER_SECONDS, max=WAIT_MAX_SECONDS),
            retry_error_callback=_on_exhausted,
        )
        def _call_with_timeout(*args: Any, **kwargs: Any) -> Any:
            # A single worker thread enforces a hard wall-clock timeout even
            # for clients (e.g. yfinance's `.info`) with no native timeout
            # kwarg. Python threads can't be forcibly killed, so a timed-out
            # call keeps running in the background after we give up on it —
            # an accepted trade-off for a local demo, not a production queue.
            # `shutdown(wait=False)` (not the `with` form, which blocks in
            # `__exit__` until the thread finishes) is what makes "give up"
            # actually mean the caller returns after `timeout`, not after
            # the hung call eventually completes.
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            finally:
                pool.shutdown(wait=False)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _call_with_timeout(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
