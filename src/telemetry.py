"""Product analytics event logging (CLAUDE.md Section 5B KPI events).

Epic 1 stub: console + local JSONL file. Epic 7 upgrades this to write into
the SQLite/Postgres DB via db_controller.py once that exists, without
changing this function's call sites.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("telemetry")

DEFAULT_LOG_PATH = Path("telemetry.jsonl")


def log_event(name: str, log_path: Path | str | None = None, **props: object) -> None:
    target = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    event = {
        "event": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **props,
    }
    logger.info("telemetry_event %s", event)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
