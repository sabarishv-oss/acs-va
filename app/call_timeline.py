from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger


def log_call_timeline(event: str, **fields) -> None:
    """
    Lightweight event logger used by concurrent dialers.

    This repo previously referenced a richer timeline module; for now we keep
    a stable API so `new_dialer.py` can run without changing its behavior.
    """
    try:
        payload = {
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        logger.info(f"[timeline] {payload}")
    except Exception:
        # Timeline logging must never break dialing.
        logger.debug(f"[timeline] {event}")

