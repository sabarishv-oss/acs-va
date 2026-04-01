"""
new_dialer.py
-------------
Production-grade ACS outbound dialer — fault-tolerant, concurrent,
resumable. Designed to behave like a Twilio/Retell dialer.

Key features vs the previous version
--------------------------------------
  Concurrency          asyncio.Semaphore pool — N calls live at once
  Resume               Tracks every placed/completed/failed unique_id in
                       dialer_state.json; skipping already-done rows on restart
  Retry with backoff   Failed ACS placements are retried up to MAX_RETRIES
                       times with exponential backoff + jitter
  Circuit breaker      After CIRCUIT_BREAKER_THRESHOLD consecutive ACS errors,
                       the dialer pauses for CIRCUIT_BREAKER_RESET_SECONDS
                       before trying again (prevents hammering a downed service)
  DNC list             Reads an optional DNC_LIST_FILE (one E.164 per line);
                       matching numbers are skipped before any API call
  Skip already done    Reads call_results/ directory at startup; unique_ids
                       that already have a final result JSON are skipped
  Per-call watchdog    Each call slot has its own asyncio timeout independent
                       of the slot hold timer; prevents zombie slots
  Graceful shutdown    SIGINT/SIGTERM cancel in-flight tasks cleanly and flush
                       state to disk before exit
  Rate limiter         Token-bucket limits ACS create_call requests per second
                       (configurable via MAX_CALLS_PER_SECOND)
  Live progress        Prints a summary line every PROGRESS_LOG_INTERVAL seconds
  Persistent stats     dialer_state.json survives crashes; counters accumulate
                       across runs

Environment variables
---------------------
  Required:
    ACS_CONNECTION_STRING
    ACS_SOURCE_PHONE_NUMBER
    CALLBACK_URI_HOST

  Optional:
    CAMPAIGN_INPUT_CSV          default ./campaign_input.csv
    CALL_RESULTS_DIR            default ./call_results
    MAX_CONCURRENT_CALLS        default 2
    MAX_CALL_DURATION_SECONDS   default 180   (per-call watchdog ceiling)
    SLOT_RELEASE_MODE           duration | poll   default poll
    STATUS_POLL_INTERVAL        default 4.0  (seconds, poll mode)
    BETWEEN_CALL_DELAY          default 0.5  (seconds between task launches)
    MAX_CALLS_PER_SECOND        default 2.0  (rate limiter)
    MAX_RETRIES                 default 3    (per call, ACS placement only)
    RETRY_BASE_DELAY            default 2.0  (seconds, exponential backoff base)
    CIRCUIT_BREAKER_THRESHOLD   default 5    (consecutive errors before open)
    CIRCUIT_BREAKER_RESET_SECS  default 60   (seconds before half-open retry)
    DNC_LIST_FILE               default ""   (path to DNC file, one E.164/line)
    PROGRESS_LOG_INTERVAL       default 30   (seconds between live stats logs)
    DIALER_STATE_FILE           default ./dialer_state.json
    DIALER_LOG_FILE             default ./dialer_logs.txt
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import csv
import json
import math
import os
import random
import signal
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse, urlunparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from app.call_timeline import log_call_timeline
from azure.communication.callautomation import (
    CallAutomationClient,
    PhoneNumberIdentifier,
    MediaStreamingOptions,
    AudioFormat,
    MediaStreamingTransportType,
    MediaStreamingContentType,
    MediaStreamingAudioChannelType,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACS_CONNECTION_STRING   = os.getenv("ACS_CONNECTION_STRING", "")
ACS_SOURCE_PHONE_NUMBER = os.getenv("ACS_SOURCE_PHONE_NUMBER", "")
CALLBACK_URI_HOST       = os.getenv("CALLBACK_URI_HOST", "")
CALLBACK_EVENTS_URI     = CALLBACK_URI_HOST + "/api/callbacks"
CALL_RESULTS_DIR        = Path(os.getenv("CALL_RESULTS_DIR", "./call_results"))

CAMPAIGN_INPUT_CSV        = os.getenv("CAMPAIGN_INPUT_CSV", "./campaign_input.csv")
MAX_CONCURRENT_CALLS      = int(os.getenv("MAX_CONCURRENT_CALLS", "2"))
MAX_CALL_DURATION_SECONDS = int(os.getenv("MAX_CALL_DURATION_SECONDS", "180"))
SLOT_RELEASE_MODE         = os.getenv("SLOT_RELEASE_MODE", "poll").strip().lower()
STATUS_POLL_INTERVAL      = float(os.getenv("STATUS_POLL_INTERVAL", "4.0"))
BETWEEN_CALL_DELAY        = float(os.getenv("BETWEEN_CALL_DELAY", "0.5"))
MAX_CALLS_PER_SECOND      = float(os.getenv("MAX_CALLS_PER_SECOND", "2.0"))
MAX_RETRIES               = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY          = float(os.getenv("RETRY_BASE_DELAY", "2.0"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_RESET_SECS = int(os.getenv("CIRCUIT_BREAKER_RESET_SECS", "60"))
DNC_LIST_FILE             = os.getenv("DNC_LIST_FILE", "").strip()
PROGRESS_LOG_INTERVAL     = int(os.getenv("PROGRESS_LOG_INTERVAL", "30"))
DIALER_STATE_FILE         = Path(os.getenv("DIALER_STATE_FILE", "./dialer_state.json"))
DIALER_LOG_FILE           = os.getenv("DIALER_LOG_FILE", "./dialer_logs.txt")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_env() -> None:
    missing = [
        name for name, val in [
            ("ACS_CONNECTION_STRING",   ACS_CONNECTION_STRING),
            ("ACS_SOURCE_PHONE_NUMBER", ACS_SOURCE_PHONE_NUMBER),
            ("CALLBACK_URI_HOST",       CALLBACK_URI_HOST),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    if SLOT_RELEASE_MODE not in {"duration", "poll"}:
        raise RuntimeError("SLOT_RELEASE_MODE must be 'duration' or 'poll'")

    if SLOT_RELEASE_MODE == "duration" and MAX_CALL_DURATION_SECONDS <= 0:
        raise RuntimeError("MAX_CALL_DURATION_SECONDS must be > 0 for duration mode")


# ---------------------------------------------------------------------------
# Persistent state  (survives crashes / restarts)
# ---------------------------------------------------------------------------

class DialerState:
    """
    JSON-backed state file that tracks every call across restarts.

    Schema:
      {
        "placed":    { unique_id: session_id },
        "completed": { unique_id: session_id },
        "failed":    { unique_id: error_str  },
        "skipped":   { unique_id: reason     },
        "stats":     { placed, completed, failed, skipped, retried }
      }
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = asyncio.Lock()
        self._data: dict = {
            "placed":    {},
            "completed": {},
            "failed":    {},
            "skipped":   {},
            "stats":     {
                "placed": 0, "completed": 0,
                "failed": 0, "skipped": 0, "retried": 0,
            },
        }
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
                # Merge — keep existing keys, fill missing ones
                for key in self._data:
                    if key in loaded:
                        self._data[key] = loaded[key]
                logger.info(f"[state] Loaded from {self._path}")
            except Exception as e:
                logger.warning(f"[state] Could not load state file: {e} — starting fresh")

    def _save_sync(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[state] Save failed: {e}")

    async def save(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_sync)

    def is_done(self, unique_id: str) -> bool:
        """True if this unique_id was already completed or is in results dir."""
        return (
            unique_id in self._data["completed"]
            or unique_id in self._data["placed"]  # placed but not yet completed = in-flight from last run
        )

    def already_has_result(self, unique_id: str) -> bool:
        """True if a final .json result already exists on disk."""
        result_path = CALL_RESULTS_DIR / f"{unique_id}.json"
        return result_path.exists()

    async def mark_placed(self, unique_id: str, session_id: str) -> None:
        async with self._lock:
            self._data["placed"][unique_id] = session_id
            self._data["stats"]["placed"] += 1
        await asyncio.to_thread(self._save_sync)

    async def mark_completed(self, unique_id: str, session_id: str) -> None:
        async with self._lock:
            self._data["completed"][unique_id] = session_id
            self._data["placed"].pop(unique_id, None)
            self._data["stats"]["completed"] += 1
        await asyncio.to_thread(self._save_sync)

    async def mark_failed(self, unique_id: str, error: str) -> None:
        async with self._lock:
            self._data["failed"][unique_id] = error
            self._data["placed"].pop(unique_id, None)
            self._data["stats"]["failed"] += 1
        await asyncio.to_thread(self._save_sync)

    async def mark_skipped(self, unique_id: str, reason: str) -> None:
        async with self._lock:
            self._data["skipped"][unique_id] = reason
            self._data["stats"]["skipped"] += 1
        await asyncio.to_thread(self._save_sync)

    async def mark_retried(self) -> None:
        async with self._lock:
            self._data["stats"]["retried"] += 1

    @property
    def stats(self) -> dict:
        return dict(self._data["stats"])


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Three-state circuit breaker: CLOSED → OPEN → HALF-OPEN → CLOSED.

    CLOSED:    Normal operation. Consecutive failures tracked.
    OPEN:      Too many consecutive failures. All calls blocked.
               Transitions to HALF-OPEN after reset_seconds.
    HALF-OPEN: One trial call allowed. Success → CLOSED. Failure → OPEN.
    """

    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int, reset_seconds: int):
        self._threshold      = threshold
        self._reset_seconds  = reset_seconds
        self._state          = self.CLOSED
        self._failure_count  = 0
        self._opened_at: float | None = None
        self._lock           = asyncio.Lock()

    async def call_succeeded(self) -> None:
        async with self._lock:
            self._failure_count = 0
            if self._state != self.CLOSED:
                logger.info("[circuit] CLOSED — service recovered")
            self._state = self.CLOSED

    async def call_failed(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._state == self.HALF_OPEN or self._failure_count >= self._threshold:
                self._state     = self.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    f"[circuit] OPEN after {self._failure_count} consecutive failures — "
                    f"pausing {self._reset_seconds}s before retry"
                )

    async def wait_if_open(self) -> None:
        """
        Block the caller until the circuit is not OPEN.
        Transitions OPEN → HALF-OPEN after reset_seconds.
        """
        while True:
            async with self._lock:
                if self._state == self.CLOSED:
                    return
                if self._state == self.HALF_OPEN:
                    return
                # OPEN — check if reset window has passed
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed >= self._reset_seconds:
                    self._state = self.HALF_OPEN
                    logger.info("[circuit] HALF-OPEN — allowing one trial call")
                    return
                wait = self._reset_seconds - elapsed

            logger.info(f"[circuit] OPEN — waiting {wait:.0f}s before retry")
            await asyncio.sleep(min(wait, 5.0))


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter.
    Allows up to `rate` calls per second with bursting up to `burst`.
    """

    def __init__(self, rate: float, burst: int | None = None):
        self._rate     = rate
        self._burst    = burst or max(1, int(rate))
        self._tokens   = float(self._burst)
        self._last     = time.monotonic()
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
        await asyncio.sleep(wait)
        async with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0 + wait * self._rate)


# ---------------------------------------------------------------------------
# DNC list
# ---------------------------------------------------------------------------

def _load_dnc(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        logger.warning(f"[DNC] File not found: {path} — DNC list empty")
        return set()
    dnc = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            num = line.strip()
            if num:
                dnc.add(num)
    logger.info(f"[DNC] Loaded {len(dnc)} numbers from {path}")
    return dnc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _e164(number: str) -> str:
    number = (
        number.strip()
        .replace("-", "").replace(" ", "")
        .replace("(", "").replace(")", "")
    )
    if not number.startswith("+"):
        number = "+1" + number
    return number


def _build_websocket_url(params: str) -> str:
    parsed = urlparse(CALLBACK_EVENTS_URI)
    return urlunparse(("wss", parsed.netloc, "/ws", "", params, ""))


def _build_media_streaming_options(websocket_url: str) -> MediaStreamingOptions:
    return MediaStreamingOptions(
        transport_url=websocket_url,
        transport_type=MediaStreamingTransportType.WEBSOCKET,
        content_type=MediaStreamingContentType.AUDIO,
        audio_channel_type=MediaStreamingAudioChannelType.UNMIXED,
        start_media_streaming=True,
        enable_bidirectional=True,
        audio_format=AudioFormat.PCM16_K_MONO,
    )


def _load_targets(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# ACS call placement  (sync SDK wrapped in thread)
# ---------------------------------------------------------------------------

def _sdk_create_call(
    phone_number: str,
    callback_uri: str,
    websocket_url: str,
    ws_params: str,
) -> str:
    client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)
    result = client.create_call(
        target_participant=PhoneNumberIdentifier(phone_number),
        source_caller_id_number=PhoneNumberIdentifier(ACS_SOURCE_PHONE_NUMBER),
        callback_url=callback_uri,
        media_streaming=_build_media_streaming_options(websocket_url),
        operation_context=ws_params,
    )
    return result.call_connection_id


async def _place_call_with_retry(
    phone_number: str,
    org_name: str,
    services: str,
    unique_id: str,
    session_id: str,
    circuit: CircuitBreaker,
    rate_limiter: RateLimiter,
    state: DialerState,
) -> str:
    """
    Place an ACS call with retry + exponential backoff + circuit breaker.
    Returns call_connection_id on success.
    Raises on permanent failure (all retries exhausted or circuit open).
    """
    ws_params     = urlencode({
        "org_name":     org_name,
        "phone_number": phone_number,
        "services":     services,
        "unique_id":    unique_id,
        "session_id":   session_id,
    })
    callback_uri  = f"{CALLBACK_EVENTS_URI}/{session_id}?{ws_params}"
    websocket_url = _build_websocket_url(ws_params)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        # Wait if circuit is open
        await circuit.wait_if_open()

        # Rate limit
        await rate_limiter.acquire()

        log_call_timeline(
            "acs_outbound_create_call_start",
            session_id=session_id,
            to_=phone_number,
            unique_id=unique_id,
            org_name=org_name,
            attempt=attempt,
        )

        try:
            conn_id = await asyncio.wait_for(
                asyncio.to_thread(
                    _sdk_create_call,
                    phone_number,
                    callback_uri,
                    websocket_url,
                    ws_params,
                ),
                timeout=30.0,   # ACS SDK call should never take >30s
            )
            await circuit.call_succeeded()
            log_call_timeline(
                "acs_outbound_create_call_ok",
                session_id=session_id,
                call_connection_id=conn_id,
                unique_id=unique_id,
                attempt=attempt,
            )
            return conn_id

        except asyncio.TimeoutError:
            last_error = TimeoutError("ACS SDK call timed out after 30s")
            await circuit.call_failed()
            logger.warning(
                f"[retry] Timeout on attempt {attempt}/{MAX_RETRIES} | "
                f"unique_id={unique_id}"
            )
        except Exception as e:
            last_error = e
            await circuit.call_failed()
            logger.warning(
                f"[retry] Error on attempt {attempt}/{MAX_RETRIES} | "
                f"unique_id={unique_id} | {e}"
            )

        if attempt < MAX_RETRIES:
            await state.mark_retried()
            # Exponential backoff with full jitter
            cap   = 30.0
            delay = min(cap, RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            jitter = random.uniform(0, delay)
            logger.info(
                f"[retry] Waiting {jitter:.1f}s before attempt {attempt + 1} | "
                f"unique_id={unique_id}"
            )
            await asyncio.sleep(jitter)

    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed for {phone_number} ({unique_id}): {last_error}"
    )


# ---------------------------------------------------------------------------
# Slot hold strategies
# ---------------------------------------------------------------------------

async def _hold_duration() -> None:
    await asyncio.sleep(MAX_CALL_DURATION_SECONDS)


async def _hold_poll(http: aiohttp.ClientSession, session_id: str) -> None:
    # Poll localhost directly — avoids routing through ngrok (SSL errors on Windows).
    # The public CALLBACK_URI_HOST is only needed by ACS; the dialer talks to
    # the server on the same machine.
    poll_host  = os.getenv("STATUS_POLL_HOST", "http://localhost:8000")
    status_url = f"{poll_host}/api/call-status/{session_id}"
    deadline   = (
        None if MAX_CALL_DURATION_SECONDS <= 0
        else (time.monotonic() + MAX_CALL_DURATION_SECONDS)
    )

    consecutive_errors = 0
    max_poll_errors    = 10   # stop polling if server is persistently unreachable

    while deadline is None or time.monotonic() < deadline:
        await asyncio.sleep(STATUS_POLL_INTERVAL)
        try:
            async with http.get(
                status_url,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                consecutive_errors = 0
                if resp.status == 200:
                    data = await resp.json()
                    if not data.get("active", True):
                        logger.debug(f"[slot] Ended confirmed | session={session_id[:8]}")
                        return
                elif resp.status == 404:
                    # Server doesn't know this session — call must have ended
                    logger.debug(f"[slot] 404 from server — treating as ended | session={session_id[:8]}")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            consecutive_errors += 1
            logger.debug(f"[slot] Poll error #{consecutive_errors}: {e}")
            if consecutive_errors >= max_poll_errors:
                logger.warning(
                    f"[slot] {max_poll_errors} consecutive poll errors — "
                    f"falling back to duration ceiling | session={session_id[:8]}"
                )
                # Fall through to duration ceiling logic below
                break

    if deadline is not None:
        logger.warning(
            f"[slot] Duration ceiling hit | "
            f"session={session_id[:8]} | force-releasing"
        )


# ---------------------------------------------------------------------------
# Per-call task
# ---------------------------------------------------------------------------

async def _run_call_slot(
    semaphore: asyncio.Semaphore,
    http: aiohttp.ClientSession,
    row: dict,
    index: int,
    total: int,
    state: DialerState,
    circuit: CircuitBreaker,
    rate_limiter: RateLimiter,
    dnc: set[str],
    shutdown_event: asyncio.Event,
) -> None:
    """
    Full lifecycle for one call row:
      1. Parse + validate row
      2. DNC check
      3. Resume check (already done / result on disk)
      4. Block on semaphore until slot free
      5. Place call with retry + circuit breaker
      6. Hold slot until call ends (poll or duration)
      7. Release slot
    """
    # ── Parse row ────────────────────────────────────────────────────────────
    raw_number = (
        row.get("phone_number") or row.get("phone") or row.get("to") or ""
    ).strip()

    if not raw_number:
        logger.warning(f"[{index}/{total}] Skipping — no phone number: {row}")
        await state.mark_skipped(str(index), "no_phone_number")
        return

    phone_number = _e164(raw_number)
    org_name     = (row.get("org_name") or "").strip()
    services     = (
        row.get("services") or
        row.get("services_listed") or
        row.get("services_list") or ""
    ).strip()
    unique_id    = (row.get("unique_id") or row.get("id") or str(uuid.uuid4())).strip()

    # ── DNC check ────────────────────────────────────────────────────────────
    if phone_number in dnc:
        logger.info(f"[{index}/{total}] DNC skip | {phone_number} | unique_id={unique_id}")
        await state.mark_skipped(unique_id, "dnc")
        return

    # ── Resume check — skip if already completed or result exists ─────────
    if state.already_has_result(unique_id):
        logger.info(
            f"[{index}/{total}] Skip — result already on disk | unique_id={unique_id}"
        )
        await state.mark_skipped(unique_id, "result_exists")
        return

    if state.is_done(unique_id):
        logger.info(
            f"[{index}/{total}] Skip — already in state | unique_id={unique_id}"
        )
        await state.mark_skipped(unique_id, "already_in_state")
        return

    # ── Honour shutdown signal before acquiring slot ──────────────────────
    if shutdown_event.is_set():
        logger.info(f"[{index}/{total}] Shutdown — skipping {unique_id}")
        return

    session_id = str(uuid.uuid4())

    # ── Acquire slot ─────────────────────────────────────────────────────────
    async with semaphore:
        if shutdown_event.is_set():
            logger.info(f"[{index}/{total}] Shutdown inside slot — aborting {unique_id}")
            return

        placed_at  = time.monotonic()
        active_now = MAX_CONCURRENT_CALLS - semaphore._value

        logger.info(
            f"[{index}/{total}] Placing → {phone_number} | "
            f"org={org_name} | unique_id={unique_id} | "
            f"session={session_id[:8]} | "
            f"slots={active_now}/{MAX_CONCURRENT_CALLS}"
        )

        # ── Place the call (with retry + circuit breaker) ─────────────────
        try:
            conn_id = await _place_call_with_retry(
                phone_number, org_name, services, unique_id, session_id,
                circuit, rate_limiter, state,
            )
            logger.info(
                f"[{index}/{total}] Placed | conn_id={conn_id} | "
                f"session={session_id[:8]} | unique_id={unique_id}"
            )
            await state.mark_placed(unique_id, session_id)

        except asyncio.CancelledError:
            logger.info(f"[{index}/{total}] Cancelled before placement | unique_id={unique_id}")
            raise

        except Exception as e:
            logger.error(f"[{index}/{total}] Permanent failure | unique_id={unique_id} | {e}")
            await state.mark_failed(unique_id, str(e))
            return   # slot released by async with exit

        # ── Hold slot until call ends ────────────────────────────────────
        try:
            hold_coro = (
                _hold_poll(http, session_id)
                if SLOT_RELEASE_MODE == "poll"
                else _hold_duration()
            )
            # Per-call watchdog: never hold a slot longer than MAX_CALL_DURATION_SECONDS
            # regardless of poll mode — prevents zombie slots from leaked connections.
            if MAX_CALL_DURATION_SECONDS > 0:
                try:
                    await asyncio.wait_for(
                        hold_coro,
                        timeout=MAX_CALL_DURATION_SECONDS + 10,  # small grace buffer
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{index}/{total}] Watchdog timeout — force-releasing slot | "
                        f"unique_id={unique_id} | session={session_id[:8]}"
                    )
            else:
                await hold_coro

        except asyncio.CancelledError:
            logger.info(
                f"[{index}/{total}] Cancelled during hold | "
                f"unique_id={unique_id} | session={session_id[:8]}"
            )
            raise

        held = time.monotonic() - placed_at
        logger.info(
            f"[{index}/{total}] Slot released | "
            f"session={session_id[:8]} | unique_id={unique_id} | held={held:.1f}s"
        )
        await state.mark_completed(unique_id, session_id)


# ---------------------------------------------------------------------------
# Progress logger
# ---------------------------------------------------------------------------

async def _log_progress(state: DialerState, total: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(PROGRESS_LOG_INTERVAL)
        if stop.is_set():
            break
        s = state.stats
        logger.info(
            f"[progress] total={total} | "
            f"placed={s['placed']} | completed={s['completed']} | "
            f"failed={s['failed']} | skipped={s['skipped']} | "
            f"retried={s['retried']}"
        )


# ---------------------------------------------------------------------------
# Campaign entry point
# ---------------------------------------------------------------------------

async def run_campaign() -> None:
    _validate_env()

    rows = _load_targets(CAMPAIGN_INPUT_CSV)
    total = len(rows)
    if total == 0:
        logger.warning("No rows in CSV — nothing to do.")
        return

    dnc   = _load_dnc(DNC_LIST_FILE)
    state = DialerState(DIALER_STATE_FILE)

    circuit      = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET_SECS)
    rate_limiter = RateLimiter(rate=MAX_CALLS_PER_SECOND)
    semaphore    = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    shutdown     = asyncio.Event()

    logger.info(
        f"Campaign starting | targets={total} | "
        f"concurrent={MAX_CONCURRENT_CALLS} | mode={SLOT_RELEASE_MODE} | "
        f"max_duration={MAX_CALL_DURATION_SECONDS}s | "
        f"retries={MAX_RETRIES} | rate={MAX_CALLS_PER_SECOND}/s"
    )

    # ── Graceful shutdown handler ────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _handle_signal(sig_name: str):
        logger.warning(f"[shutdown] {sig_name} received — draining in-flight calls…")
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _handle_signal(s))
        except (NotImplementedError, OSError):
            pass  # Windows doesn't support add_signal_handler

    # ── Progress logger ──────────────────────────────────────────────────────
    stop_progress = asyncio.Event()
    progress_task = asyncio.create_task(
        _log_progress(state, total, stop_progress),
        name="progress-logger",
    )

    # ── Launch all call tasks ────────────────────────────────────────────────
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_CALLS + 4)
    async with aiohttp.ClientSession(connector=connector) as http:
        tasks = []
        for i, row in enumerate(rows, start=1):
            if shutdown.is_set():
                break
            task = asyncio.create_task(
                _run_call_slot(
                    semaphore, http, row, i, total,
                    state, circuit, rate_limiter, dnc, shutdown,
                ),
                name=f"call-{i}",
            )
            tasks.append(task)

            # Smooth launch rate — semaphore does the real concurrency control
            if BETWEEN_CALL_DELAY > 0 and i < total:
                await asyncio.sleep(BETWEEN_CALL_DELAY)

        # ── Wait for all tasks; cancel cleanly on shutdown ───────────────────
        if shutdown.is_set():
            logger.warning("[shutdown] Cancelling un-started tasks…")
            for t in tasks:
                if not t.done():
                    t.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        cancelled = sum(
            1 for r in results
            if isinstance(r, (asyncio.CancelledError, Exception))
            and not isinstance(r, Exception)   # CancelledError only
        )
        errors = sum(
            1 for r in results
            if isinstance(r, Exception)
            and not isinstance(r, asyncio.CancelledError)
        )
        if errors:
            logger.warning(f"[campaign] {errors} tasks raised unexpected exceptions")

    # ── Teardown ─────────────────────────────────────────────────────────────
    stop_progress.set()
    progress_task.cancel()
    await state.save()

    s = state.stats
    logger.info(
        f"Campaign complete | "
        f"placed={s['placed']} | completed={s['completed']} | "
        f"failed={s['failed']} | skipped={s['skipped']} | "
        f"retried={s['retried']}"
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.add(
        DIALER_LOG_FILE,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        encoding="utf-8",
    )
    asyncio.run(run_campaign())