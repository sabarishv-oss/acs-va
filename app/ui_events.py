"""
ui_events.py
------------
Minimal event bus for streaming call lifecycle events to the local UI via SSE.
All functions are synchronous and safe to call from any coroutine in the
same asyncio event loop (asyncio.Queue.put_nowait is loop-thread-safe).
"""

import asyncio
import json
from datetime import datetime, timezone

# Each SSE connection gets its own Queue entry here.
_subscribers: list[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Returns a queue to read events from."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a subscriber queue when the SSE connection closes."""
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


def emit(event_type: str, session_id: str = "", **kwargs) -> None:
    """
    Push a structured event to all active SSE subscribers.
    Non-blocking: drops the event for any subscriber whose queue is full.
    """
    payload = json.dumps({
        "type": event_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        **{k: v for k, v in kwargs.items() if v is not None},
    })
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass
