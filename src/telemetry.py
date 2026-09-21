"""Product analytics event logging (CLAUDE.md Section 5B KPI events).

Epic 1 stub was console + local JSONL file. Epic 7 upgrades the backend to
write into the SQLite DB via db_controller.py, without changing this
function's call-site contract (`log_event(name, **props)`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src import db_controller

logger = logging.getLogger("telemetry")


def log_event(
    name: str,
    db_path: Path | str | None = None,
    user_id: str | None = None,
    **props: object,
) -> None:
    logger.info("telemetry_event %s", {"event": name, "user_id": user_id, **props})
    db_controller.record_telemetry_event(name, props, user_id=user_id, db_path=db_path)
