"""
opening_guard.py
----------------
Stores monotonic timestamps when ACS signals CallConnected, so the inbound
opening-guard window can be anchored to call connection instead of the first
media frame. Cleared when the call ends.
"""

import time
from typing import Dict

# session_id -> time.monotonic() at Microsoft.Communication.CallConnected
_call_connected_monotonic: Dict[str, float] = {}


def record_call_connected(session_id: str) -> None:
    """Call when ACS CallConnected fires (HTTP callback). Idempotent: first wins."""
    if not session_id or session_id in _call_connected_monotonic:
        return
    _call_connected_monotonic[session_id] = time.monotonic()


def get_call_connected_anchor(session_id: str) -> float | None:
    """Return monotonic time at CallConnected, or None if not recorded yet."""
    return _call_connected_monotonic.get(session_id)


def clear_call_connected_anchor(session_id: str) -> None:
    """Remove anchor when the call ends (disconnect / WS close)."""
    _call_connected_monotonic.pop(session_id, None)
