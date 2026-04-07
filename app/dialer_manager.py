"""
dialer_manager.py
-----------------
In-process async dialer for batch outbound calling campaigns.
Manages concurrency, pause/resume/stop, per-call tracking, and SSE events.
"""

import asyncio
import csv
import io
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from loguru import logger

from azure.communication.callautomation import PhoneNumberIdentifier
from app import ui_events


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_IDLE      = "idle"
STATE_RUNNING   = "running"
STATE_PAUSED    = "paused"
STATE_STOPPED   = "stopped"
STATE_COMPLETED = "completed"


def _normalize_phone(raw: str) -> str:
    """Normalize a phone number string to E.164 format."""
    digits = "".join(c for c in raw if c.isdigit())
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if raw.strip().startswith("+"):
        return f"+{digits}"
    return f"+{digits}"


# ---------------------------------------------------------------------------
# Custom resizable semaphore
# ---------------------------------------------------------------------------

class _ResizableSemaphore:
    """
    Async semaphore whose limit can be changed at runtime.
    Uses an asyncio.Event for waiters: when a slot is freed or the limit
    is raised, the event fires and all waiters re-check.
    """

    def __init__(self, max_count: int):
        self._max = max_count
        self._current = 0          # number currently acquired
        self._event = asyncio.Event()
        self._event.set()          # initially free

    @property
    def current(self) -> int:
        return self._current

    def resize(self, new_max: int) -> None:
        self._max = new_max
        # Wake up any waiters — they will re-check the condition
        self._event.set()

    async def acquire(self) -> None:
        while True:
            if self._current < self._max:
                self._current += 1
                if self._current >= self._max:
                    self._event.clear()
                return
            self._event.clear()
            await self._event.wait()

    def release(self) -> None:
        if self._current > 0:
            self._current -= 1
        self._event.set()


# ---------------------------------------------------------------------------
# DialerManager
# ---------------------------------------------------------------------------

class DialerManager:
    """
    Manages an outbound call campaign: loads CSV, places calls concurrently,
    respects pause/resume/stop, and reports progress via ui_events SSE.
    """

    def __init__(
        self,
        *,
        acs_client,
        source_phone: str,
        callback_events_uri: str,
        build_ws_url_fn,
        build_media_options_fn,
        session_registry: dict,
        active_sessions: set,
        results_dir: Path,
        vad_sensitivity: int = 50,
    ):
        self._acs_client           = acs_client
        self._source_phone         = source_phone
        self._callback_events_uri  = callback_events_uri
        self._build_ws_url_fn      = build_ws_url_fn
        self._build_media_options_fn = build_media_options_fn
        self._session_registry     = session_registry
        self._active_sessions      = active_sessions
        self._results_dir          = results_dir
        self._vad_sensitivity      = vad_sensitivity

        # Campaign data
        self._rows: list[dict]     = []
        self._rows_preview: list[dict] = []

        # Runtime state
        self._state: str           = STATE_IDLE
        self._semaphore: _ResizableSemaphore | None = None
        self._max_concurrent: int  = 3
        self._inter_call_delay: float = 1.0
        self._campaign_vad: int    = 50
        self._allow_redial: bool   = False

        # Stats
        self._stats: dict          = self._fresh_stats()
        self._start_time: float | None = None

        # Pause / stop signals
        self._pause_event          = asyncio.Event()
        self._pause_event.set()    # not paused initially
        self._stop_flag: bool      = False

        # Background tasks
        self._campaign_task: asyncio.Task | None = None
        self._progress_task: asyncio.Task | None = None

        # Ensure results dir exists
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _fresh_stats() -> dict:
        return {
            "total": 0,
            "placed": 0,
            "active": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }

    @property
    def _active_count(self) -> int:
        return self._semaphore.current if self._semaphore else 0

    # -----------------------------------------------------------------------
    # CSV loading
    # -----------------------------------------------------------------------

    def load_csv(self, content: str) -> int:
        """
        Parse CSV with columns org_name, phone_number, services, unique_id.
        Handles common aliases. Returns number of valid rows loaded.
        """
        ALIASES = {
            "phone":         "phone_number",
            "phone_no":      "phone_number",
            "phonenumber":   "phone_number",
            "services_list": "services",
            "service":       "services",
            "id":            "unique_id",
            "uid":           "unique_id",
            "org":           "org_name",
            "organization":  "org_name",
        }

        reader = csv.DictReader(io.StringIO(content.strip()))
        rows: list[dict] = []

        for raw_row in reader:
            # Normalize column names: lowercase + strip
            row = {k.strip().lower(): v.strip() for k, v in raw_row.items() if k}
            # Apply aliases
            normalized: dict = {}
            for k, v in row.items():
                canonical = ALIASES.get(k, k)
                normalized[canonical] = v

            phone_raw = normalized.get("phone_number", "").strip()
            if not phone_raw:
                continue  # skip rows with no phone

            normalized["phone_number"] = _normalize_phone(phone_raw)
            normalized.setdefault("org_name", "")
            normalized.setdefault("services", "")
            normalized.setdefault("unique_id", str(uuid.uuid4())[:8])
            rows.append(normalized)

        self._rows = rows
        self._stats = self._fresh_stats()
        self._stats["total"] = len(rows)

        # Build preview (first 20 rows, safe subset of keys)
        self._rows_preview = [
            {
                "phone_number": r["phone_number"],
                "org_name":     r["org_name"],
                "unique_id":    r["unique_id"],
            }
            for r in rows[:20]
        ]

        logger.info(f"DialerManager: loaded {len(rows)} rows from CSV")
        ui_events.emit(
            "dialer_loaded",
            count=len(rows),
            rows_preview=self._rows_preview,
        )
        return len(rows)

    # -----------------------------------------------------------------------
    # Start / Pause / Resume / Stop
    # -----------------------------------------------------------------------

    async def start(
        self,
        max_concurrent: int = 3,
        inter_call_delay: float = 1.0,
        vad_sensitivity: int = 50,
        allow_redial: bool = False,
    ) -> bool:
        """Begin campaign. Returns False if already running."""
        if self._state in (STATE_RUNNING, STATE_PAUSED):
            logger.warning("DialerManager.start() called while already running")
            return False
        if not self._rows:
            logger.warning("DialerManager.start() called with no rows loaded")
            return False

        self._max_concurrent   = max_concurrent
        self._inter_call_delay = inter_call_delay
        self._campaign_vad     = vad_sensitivity
        self._allow_redial     = allow_redial
        self._semaphore        = _ResizableSemaphore(max_concurrent)
        self._stop_flag        = False
        self._pause_event.set()   # ensure not paused
        self._state            = STATE_RUNNING
        self._start_time       = time.monotonic()
        self._stats            = self._fresh_stats()
        self._stats["total"]   = len(self._rows)

        logger.info(
            f"DialerManager: campaign started | rows={len(self._rows)} | "
            f"max_concurrent={max_concurrent} | delay={inter_call_delay}s | "
            f"vad={vad_sensitivity}"
        )
        ui_events.emit(
            "dialer_started",
            total=len(self._rows),
            max_concurrent=max_concurrent,
            inter_call_delay=inter_call_delay,
            vad_sensitivity=vad_sensitivity,
        )

        self._campaign_task = asyncio.create_task(
            self._run_campaign(), name="dialer_campaign"
        )
        self._progress_task = asyncio.create_task(
            self._progress_reporter(), name="dialer_progress_reporter"
        )
        return True

    def pause(self) -> None:
        if self._state != STATE_RUNNING:
            return
        self._state = STATE_PAUSED
        self._pause_event.clear()
        logger.info("DialerManager: paused")
        ui_events.emit("dialer_paused")

    def resume(self) -> None:
        if self._state != STATE_PAUSED:
            return
        self._state = STATE_RUNNING
        self._pause_event.set()
        logger.info("DialerManager: resumed")
        ui_events.emit("dialer_resumed")

    def stop(self) -> None:
        if self._state not in (STATE_RUNNING, STATE_PAUSED):
            return
        self._stop_flag = True
        self._pause_event.set()   # unblock any coroutines waiting on pause
        self._state = STATE_STOPPED
        # Cancel the campaign loop so asyncio.sleep/acquire_slot unblocks immediately
        if self._campaign_task and not self._campaign_task.done():
            self._campaign_task.cancel()
        if self._progress_task and not self._progress_task.done():
            self._progress_task.cancel()
        logger.info("DialerManager: stop requested")
        ui_events.emit("dialer_stopped")

    # -----------------------------------------------------------------------
    # Clear queue
    # -----------------------------------------------------------------------

    def clear_results(self) -> dict:
        """
        Delete all *.json result files from self._results_dir.
        Uses the same path object as _place_and_watch — no path ambiguity.
        """
        deleted = 0
        errors  = 0
        logger.info(f"DialerManager: clearing results from {self._results_dir}")
        if self._results_dir.exists():
            for f in self._results_dir.glob("*.json"):
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete {f.name}: {e}")
                    errors += 1
        else:
            logger.warning(f"DialerManager: results dir missing — {self._results_dir}")
        logger.info(f"DialerManager: cleared {deleted} result file(s), {errors} error(s)")
        ui_events.emit("dialer_results_cleared", deleted=deleted, errors=errors)
        return {"deleted": deleted, "errors": errors}

    def clear_queue(self) -> bool:
        """Reset loaded rows. Returns False if campaign is running/paused."""
        if self._state in (STATE_RUNNING, STATE_PAUSED):
            return False
        count = len(self._rows)
        self._rows = []
        self._rows_preview = []
        self._stats = self._fresh_stats()
        self._state = STATE_IDLE
        logger.info(f"DialerManager: queue cleared ({count} rows removed)")
        ui_events.emit("dialer_queue_cleared", count=count)
        return True

    # -----------------------------------------------------------------------
    # Dynamic concurrency
    # -----------------------------------------------------------------------

    def set_concurrency(self, n: int) -> None:
        n = max(1, n)
        self._max_concurrent = n
        if self._semaphore:
            self._semaphore.resize(n)
        logger.info(f"DialerManager: concurrency changed to {n}")
        ui_events.emit(
            "dialer_concurrency_changed",
            max_concurrent=n,
            active=self._active_count,
        )

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        elapsed = (
            round(time.monotonic() - self._start_time, 1)
            if self._start_time is not None
            else 0
        )
        stats = dict(self._stats)
        stats["active"] = self._active_count
        results_dir = str(self._results_dir.resolve())
        result_files = len(list(self._results_dir.glob("*.json"))) if self._results_dir.exists() else 0
        return {
            "state":            self._state,
            "max_concurrent":   self._max_concurrent,
            "inter_call_delay": self._inter_call_delay,
            "vad_sensitivity":  self._campaign_vad,
            "allow_redial":     self._allow_redial,
            "stats":            stats,
            "elapsed_s":        elapsed,
            "rows_preview":     self._rows_preview,
            "results_dir":      results_dir,
            "result_files":     result_files,
        }

    # -----------------------------------------------------------------------
    # Slot management
    # -----------------------------------------------------------------------

    async def _acquire_slot(self) -> None:
        assert self._semaphore is not None
        await self._semaphore.acquire()

    def _release_slot(self) -> None:
        if self._semaphore:
            self._semaphore.release()

    # -----------------------------------------------------------------------
    # Campaign loop
    # -----------------------------------------------------------------------

    async def _run_campaign(self) -> None:
        try:
            for row in self._rows:
                # Respect stop flag
                if self._stop_flag:
                    break

                # Respect pause
                await self._pause_event.wait()
                if self._stop_flag:
                    break

                # Acquire a concurrency slot (blocks until one is free)
                await self._acquire_slot()

                if self._stop_flag:
                    self._release_slot()
                    break

                # Launch per-call task (slot released inside finally of task)
                asyncio.create_task(
                    self._place_and_watch(row), name=f"dialer_call_{row.get('unique_id', '')}"
                )

                # Inter-call delay
                if self._inter_call_delay > 0:
                    await asyncio.sleep(self._inter_call_delay)

            # Wait for all active calls to finish
            if not self._stop_flag:
                while self._active_count > 0:
                    await asyncio.sleep(1.0)

            if not self._stop_flag and self._state != STATE_STOPPED:
                self._state = STATE_COMPLETED
                logger.info("DialerManager: campaign completed")
                ui_events.emit("dialer_completed", stats=dict(self._stats))

        except asyncio.CancelledError:
            logger.info("DialerManager: campaign task cancelled")
        except Exception as e:
            logger.error(f"DialerManager: campaign loop error | {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cancel progress reporter when campaign ends
            if self._progress_task and not self._progress_task.done():
                self._progress_task.cancel()

    # -----------------------------------------------------------------------
    # Per-call task
    # -----------------------------------------------------------------------

    async def _place_and_watch(self, row: dict) -> None:
        uid        = row.get("unique_id", "")
        phone      = row.get("phone_number", "")
        org        = row.get("org_name", "")
        services   = row.get("services", "")

        try:
            # Skip if result already exists (unless re-dial is enabled)
            result_file = self._results_dir / f"{uid}.json"
            if not self._allow_redial and result_file.exists():
                logger.info(f"DialerManager: skipping {uid} — result exists")
                self._stats["skipped"] += 1
                ui_events.emit(
                    "dialer_call_skipped",
                    unique_id=uid,
                    phone_number=phone,
                    org_name=org,
                    reason="result_exists",
                )
                return

            session_id = str(uuid.uuid4())
            context_params = urlencode({
                "org_name":        org,
                "phone_number":    phone,
                "services":        services,
                "unique_id":       uid,
                "session_id":      session_id,
                "vad_sensitivity": self._campaign_vad,
            })

            callback_uri = f"{self._callback_events_uri}/{session_id}?{context_params}"
            ws_url       = self._build_ws_url_fn(context_params)
            media_opts   = self._build_media_options_fn(ws_url)

            logger.info(
                f"DialerManager: placing call | uid={uid} | phone={phone} | "
                f"org={org} | session={session_id[:8]}"
            )
            ui_events.emit(
                "dialer_call_placing",
                session_id=session_id,
                phone_number=phone,
                org_name=org,
                unique_id=uid,
            )

            result = self._acs_client.create_call(
                target_participant=PhoneNumberIdentifier(phone),
                source_caller_id_number=PhoneNumberIdentifier(self._source_phone),
                callback_url=callback_uri,
                media_streaming=media_opts,
                operation_context=context_params,
            )

            self._session_registry[session_id] = result.call_connection_id
            self._active_sessions.add(session_id)
            self._stats["placed"] += 1

            ui_events.emit(
                "call_initiated",
                session_id=session_id,
                phone_number=phone,
                org_name=org,
                unique_id=uid,
            )

            # Wait for call to complete (session removed from active_sessions on disconnect)
            timeout      = 300  # 5 minutes max
            elapsed      = 0
            call_done    = False
            placed_at_wc = time.time()  # wall-clock time at placement (for st_mtime comparison)

            while elapsed < timeout:
                await asyncio.sleep(1)
                elapsed += 1
                if session_id not in self._active_sessions:
                    call_done = True
                    break
                # Result file is written before WebSocket closes — check it as
                # an early completion signal (avoids the 30s receive() timeout).
                # Only count it if the file was written AFTER we placed the call
                # so that pre-existing result files (from previous campaigns) are ignored.
                result_file = self._results_dir / f"{uid}.json"
                if result_file.exists():
                    try:
                        if result_file.stat().st_mtime >= placed_at_wc:
                            call_done = True
                            break
                    except OSError:
                        pass

            if call_done:
                self._stats["completed"] += 1
                logger.info(
                    f"DialerManager: call finished | uid={uid} | "
                    f"session={session_id[:8]} | elapsed={elapsed}s"
                )
            else:
                # Timed out — clean up and count as failed
                self._stats["failed"] += 1
                self._active_sessions.discard(session_id)
                self._session_registry.pop(session_id, None)
                logger.warning(
                    f"DialerManager: call timed out after {timeout}s | "
                    f"uid={uid} | session={session_id[:8]}"
                )
                ui_events.emit(
                    "dialer_call_failed",
                    unique_id=uid,
                    phone_number=phone,
                    org_name=org,
                    error=f"timed_out after {timeout}s",
                )

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(
                f"DialerManager: call failed | uid={uid} | phone={phone} | {e}"
            )
            ui_events.emit(
                "dialer_call_failed",
                unique_id=uid,
                phone_number=phone,
                org_name=org,
                error=str(e),
            )
        finally:
            self._release_slot()

    # -----------------------------------------------------------------------
    # Progress reporter
    # -----------------------------------------------------------------------

    async def _progress_reporter(self) -> None:
        try:
            while self._state in (STATE_RUNNING, STATE_PAUSED):
                await asyncio.sleep(5.0)
                if self._state not in (STATE_RUNNING, STATE_PAUSED):
                    break
                stats = dict(self._stats)
                stats["active"] = self._active_count
                elapsed = (
                    round(time.monotonic() - self._start_time, 1)
                    if self._start_time is not None
                    else 0
                )
                ui_events.emit(
                    "dialer_progress",
                    stats=stats,
                    state=self._state,
                    elapsed_s=elapsed,
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"DialerManager: progress reporter error | {e}")
