"""
main.py
-------
FastAPI server for the Samantha OUTBOUND-ONLY voice verification agent.
Replaces Azure Voice Live with a Pipecat pipeline:
  Deepgram STT → GPT-4o LLM → ElevenLabs TTS

Endpoints:
  GET  /                          Health check
  POST /api/outboundCall          Trigger a single outbound call via HTTP
  POST /api/callbacks/{sessionId} ACS call lifecycle events
  POST /api/hangup/{connId}       Explicit ACS hangup
  WS   /ws                        ACS bidirectional PCM audio stream

Outbound campaign:
  Run app/dialer.py to place calls from campaign_input.csv
"""

import sys
import os

# Fix Windows UTF-8 encoding for log file (must be before any imports that log)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketState
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.runner import PipelineRunner

# Write all logs to file as well as terminal
logger.add(
    "server_logs.txt",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    encoding="utf-8",
    errors="replace",
)

# Intercept standard Python logging (uvicorn, fastapi, etc.)
# so their output also goes into server_logs.txt
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Suppress noisy DEBUG loggers — websockets binary frames, httpx request details
for noisy in ("websockets", "httpx", "httpcore", "hpack"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Suppress Windows ProactorEventLoop ConnectionResetError on WebSocket close
import asyncio
_orig_proactor_exc_handler = None
def _suppress_connection_reset(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    loop.default_exception_handler(context)
asyncio.get_event_loop().set_exception_handler(_suppress_connection_reset)

# Show pipecat LLM metrics so cache token usage is visible in logs
logging.getLogger("pipecat.services.openai").setLevel(logging.DEBUG)

from azure.communication.callautomation import (
    CallAutomationClient,
    PhoneNumberIdentifier,
    MediaStreamingOptions,
    AudioFormat,
    MediaStreamingTransportType,
    MediaStreamingContentType,
    MediaStreamingAudioChannelType,
)

from app import ui_events
from app.acs_transport import (
    ACSTransport,
    ACSTransportParams,
    acs_send_pcm_chunk,
    acs_send_stop_audio,
)
from app.call_session import CallSession
from app.pipecat_pipeline import create_pipeline
from app.dialer_manager import DialerManager

load_dotenv()

# ---------------------------------------------------------------------------
# Env validation — fail fast
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Set it in your .env file before starting the server."
        )
    return val


ACS_CONNECTION_STRING   = _require_env("ACS_CONNECTION_STRING")
ACS_SOURCE_PHONE_NUMBER = _require_env("ACS_SOURCE_PHONE_NUMBER")
CALLBACK_URI_HOST       = _require_env("CALLBACK_URI_HOST")
CALLBACK_EVENTS_URI     = CALLBACK_URI_HOST + "/api/callbacks"

RESULTS_DIR = Path(os.getenv("CALL_RESULTS_DIR", "./call_results"))
TRANSCRIPTS_DIR = Path(os.getenv("CALL_TRANSCRIPTS_DIR", "./call_transcripts"))
CALL_RECORDINGS_DIR = Path(os.getenv("CALL_RECORDINGS_DIR", "./call_recordings"))
CALL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "system_prompt.txt"

# Validate STT/LLM/TTS keys at startup — fail fast rather than silently
# dropping every WebSocket connection with a cryptic close.
# _require_env already checks for empty; here we additionally reject
# un-edited placeholder values from .env.template.
def _validate_api_key(name: str) -> str:
    val = _require_env(name)
    if val.startswith("your_"):
        raise RuntimeError(
            f"Environment variable {name} still contains the placeholder value "
            f"'{val}'. Edit .env and set a real API key before starting."
        )
    return val

_validate_api_key("OPENAI_API_KEY")
_validate_api_key("DEEPGRAM_API_KEY")
# TTS provider key is validated by the pipeline factory based on AGENT_SETTINGS.

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Samantha — ACS Outbound Pipecat Voice Agent")

# Serve the local monitor UI
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

acs_client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)

# session_id → call_connection_id
# Populated by create_call() response AND by CallConnected callback —
# whichever arrives first. The WS hangup closure re-reads this at call time
# so it always gets the latest value even if CallConnected is slightly late.
_session_registry: dict[str, str] = {}

# Track which sessions are considered "active" for dialer coordination.
# This is used by /api/call-status so a concurrent dialer can start the next
# call immediately after a call ends.
_active_sessions: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_websocket_url(params: str = "") -> str:
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


# Global DialerManager instance for batch campaigns (UI-driven)
_dialer_mgr = DialerManager(
    acs_client=acs_client,
    source_phone=ACS_SOURCE_PHONE_NUMBER,
    callback_events_uri=CALLBACK_EVENTS_URI,
    build_ws_url_fn=_build_websocket_url,
    build_media_options_fn=_build_media_streaming_options,
    session_registry=_session_registry,
    active_sessions=_active_sessions,
    results_dir=RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return JSONResponse({"message": "Samantha Outbound Voice Agent — ready."})


@app.get("/ui")
async def monitor_ui():
    return FileResponse(_static_dir / "index.html")


@app.get("/dashboard")
async def dashboard_ui():
    return FileResponse(_static_dir / "dashboard.html")


# ---------------------------------------------------------------------------
# Dashboard API — config, call history, recordings, batch
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    from app.agent_settings import load_config, get_system_prompt
    cfg = load_config()
    cfg["system_prompt"] = get_system_prompt()
    return JSONResponse(cfg)


@app.post("/api/config")
async def save_config(request: Request):
    try:
        body = await request.json()
        if "system_prompt" in body:
            SYSTEM_PROMPT_PATH.write_text(body.pop("system_prompt"), encoding="utf-8")
        CONFIG_PATH.write_text(json.dumps(body, indent=2), encoding="utf-8")
        return JSONResponse({"status": "saved"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/calls")
async def list_calls():
    calls = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.name.endswith(".partial.json"):
                continue
            try:
                calls.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    return JSONResponse(calls)


@app.get("/api/calls/{session_id}")
async def get_call_detail(session_id: str):
    result = None
    transcript_lines = []
    recording_file = None
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            if f.name.endswith(".partial.json"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("session_id", "").startswith(session_id) or data.get("unique_id") == session_id:
                    result = data
                    break
            except Exception:
                pass
    if result:
        uid = result.get("unique_id", "")
        sid_short = result.get("session_id", "")[:8]
        t_file = TRANSCRIPTS_DIR / f"{uid}_{sid_short}.txt"
        if t_file.exists():
            for line in t_file.read_text(encoding="utf-8").splitlines():
                parts = line.split(" | ", 1)
                if len(parts) == 2:
                    ts = parts[0]
                    speaker_text = parts[1].split(": ", 1)
                    if len(speaker_text) == 2:
                        transcript_lines.append({"ts": ts, "speaker": speaker_text[0], "text": speaker_text[1]})
        r_file = CALL_RECORDINGS_DIR / f"{uid}_{sid_short}.wav"
        if r_file.exists():
            recording_file = f"{uid}_{sid_short}.wav"
    return JSONResponse({"result": result, "transcript": transcript_lines, "recording": recording_file})


@app.get("/api/recordings/{filename}")
async def serve_recording(filename: str):
    path = CALL_RECORDINGS_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/api/batch")
async def list_batches():
    return JSONResponse([])


# ---------------------------------------------------------------------------
# Dialer Manager API — powers the Batch Calls dashboard page
# ---------------------------------------------------------------------------

@app.post("/api/dialer/load")
async def dialer_load(request: Request):
    """Load a CSV text payload into the dialer queue."""
    body = await request.json()
    csv_text = body.get("csv", "").strip()
    if not csv_text:
        return JSONResponse({"error": "csv field required"}, status_code=400)
    try:
        count = _dialer_mgr.load_csv(csv_text)
        return JSONResponse({"rows": count, "status": _dialer_mgr.get_status()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/dialer/status")
async def dialer_status():
    return JSONResponse(_dialer_mgr.get_status())


@app.post("/api/dialer/start")
async def dialer_start(request: Request):
    body = await request.json()
    ok = await _dialer_mgr.start(
        max_concurrent=int(body.get("max_concurrent", 3)),
        inter_call_delay=float(body.get("inter_call_delay", 1.0)),
        allow_redial=bool(body.get("allow_redial", False)),
    )
    return JSONResponse({"started": ok, "status": _dialer_mgr.get_status()})


@app.post("/api/dialer/pause")
async def dialer_pause():
    _dialer_mgr.pause()
    return JSONResponse({"status": _dialer_mgr.get_status()})


@app.post("/api/dialer/resume")
async def dialer_resume():
    _dialer_mgr.resume()
    return JSONResponse({"status": _dialer_mgr.get_status()})


@app.post("/api/dialer/stop")
async def dialer_stop():
    _dialer_mgr.stop()
    return JSONResponse({"status": _dialer_mgr.get_status()})


@app.post("/api/dialer/clear-queue")
async def dialer_clear_queue():
    ok = _dialer_mgr.clear_queue()
    return JSONResponse({"cleared": ok, "status": _dialer_mgr.get_status()})


@app.post("/api/dialer/clear-results")
async def dialer_clear_results():
    result = _dialer_mgr.clear_results()
    return JSONResponse(result)


@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events stream for the local call monitor UI."""
    async def stream():
        q = ui_events.subscribe()
        try:
            yield 'data: {"type":"connected"}\n\n'
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            ui_events.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Single outbound call trigger
# (The dialer.py batch runner calls ACS directly; this endpoint is for
#  one-off calls triggered via HTTP, e.g. from a webhook or test script.)
# ---------------------------------------------------------------------------

@app.post("/api/outboundCall")
async def outbound_call(request: Request):
    """
    Trigger a single outbound call.
    Body (JSON):
        {
            "phone_number": "+15551234567",
            "org_name":     "Bright Kids",
            "services":     "adoption services",
            "unique_id":    "00001"
        }
    """
    body         = await request.json()
    phone_number = body.get("phone_number", "").strip()
    org_name     = body.get("org_name", "").strip()
    services     = body.get("services", "").strip()
    unique_id    = body.get("unique_id", "").strip()

    if not phone_number:
        return JSONResponse({"error": "phone_number is required"}, status_code=400)

    # Normalize to E.164 — strip non-digits then prepend +1 if needed
    digits = "".join(c for c in phone_number if c.isdigit())
    if len(digits) == 10:
        phone_number = f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        phone_number = f"+{digits}"
    elif not phone_number.startswith("+"):
        phone_number = f"+{digits}"

    session_id = str(uuid.uuid4())
    context_params = urlencode({
        "org_name":     org_name,
        "phone_number": phone_number,
        "services":     services,
        "unique_id":    unique_id,
        "session_id":   session_id,
    })

    callback_uri = f"{CALLBACK_EVENTS_URI}/{session_id}?{context_params}"
    ws_url       = _build_websocket_url(context_params)

    logger.info(
        f"Outbound call → {phone_number} | org={org_name} | "
        f"unique_id={unique_id} | session={session_id}"
    )

    def _sdk_create_call():
        return acs_client.create_call(
            target_participant=PhoneNumberIdentifier(phone_number),
            source_caller_id_number=PhoneNumberIdentifier(ACS_SOURCE_PHONE_NUMBER),
            callback_url=callback_uri,
            media_streaming=_build_media_streaming_options(ws_url),
            operation_context=context_params,
        )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_sdk_create_call),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error(f"create_call timed out | session={session_id} | phone={phone_number}")
        return JSONResponse({"error": "ACS create_call timed out"}, status_code=504)

    _session_registry[session_id] = result.call_connection_id
    logger.info(f"Registered session {session_id} → {result.call_connection_id}")

    ui_events.emit(
        "call_initiated",
        session_id=session_id,
        phone_number=phone_number,
        org_name=org_name,
        unique_id=unique_id,
    )

    return JSONResponse({
        "call_connection_id": result.call_connection_id,
        "session_id":         session_id,
        "unique_id":          unique_id,
    })


# ---------------------------------------------------------------------------
# Call recording — client-side PCM capture, zero ACS overhead
# ---------------------------------------------------------------------------

async def _save_recording(session_id: str, unique_id: str, caller: bytes, agent: bytes) -> None:
    """Write a stereo WAV: left=caller, right=agent (time-aligned via silence padding)."""
    filename = f"{unique_id}_{session_id[:8]}.wav" if unique_id else f"{session_id}.wav"
    save_path = CALL_RECORDINGS_DIR / filename

    def _write():
        import array
        import wave
        ca = array.array("h", caller)
        aa = array.array("h", agent)
        # Pad shorter track with silence
        diff = len(ca) - len(aa)
        if diff > 0:
            aa.extend([0] * diff)
        elif diff < 0:
            ca.extend([0] * (-diff))
        # Interleave samples: L R L R ... (stereo)
        stereo = array.array("h")
        for l, r in zip(ca, aa):
            stereo.append(l)
            stereo.append(r)
        with wave.open(str(save_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(stereo.tobytes())

    try:
        await asyncio.to_thread(_write)
        duration_s = len(caller) / (16000 * 2)
        logger.info(f"Recording saved | session={session_id[:8]} | file={filename} | dur≈{duration_s:.1f}s")
    except Exception as e:
        logger.error(f"Failed to save recording | session={session_id[:8]} | {e}")


# ---------------------------------------------------------------------------
# ACS callback events
# ---------------------------------------------------------------------------

@app.post("/api/callbacks/{contextId}")
async def handle_callback(contextId: str, request: Request):
    for event in await request.json():
        event_data         = event["data"]
        call_connection_id = event_data.get("callConnectionId", "")
        event_type         = event["type"]

        logger.info(f"ACS Event: {event_type} | connectionId: {call_connection_id}")

        if event_type == "Microsoft.Communication.CallConnected":
            # contextId in the path IS the session_id
            _session_registry[contextId] = call_connection_id
            _active_sessions.add(contextId)
            logger.info(f"Registered session {contextId} → {call_connection_id}")
            ui_events.emit("call_connected", session_id=contextId, call_connection_id=call_connection_id)

        elif event_type == "Microsoft.Communication.MediaStreamingStarted":
            logger.info(
                f"Media streaming started | "
                f"status={event_data['mediaStreamingUpdate']['mediaStreamingStatus']}"
            )
            ui_events.emit("media_streaming_started", session_id=contextId)

        elif event_type == "Microsoft.Communication.MediaStreamingStopped":
            logger.info("Media streaming stopped.")

        elif event_type == "Microsoft.Communication.MediaStreamingFailed":
            logger.error(
                f"Media streaming failed | "
                f"code={event_data['resultInformation']['code']} | "
                f"msg={event_data['resultInformation']['message']}"
            )
            ui_events.emit("error", session_id=contextId, message="Media streaming failed")

        elif event_type == "Microsoft.Communication.CallDisconnected":
            logger.info(f"Call disconnected | connectionId: {call_connection_id}")
            # Mark inactive for dialers waiting to place the next call.
            # Also drop the registry mapping so /api/call-status can reflect
            # that the call has ended.
            _active_sessions.discard(contextId)
            _session_registry.pop(contextId, None)
            ui_events.emit("call_disconnected_acs", session_id=contextId)

    return JSONResponse({}, status_code=200)


# ---------------------------------------------------------------------------
# Call status (used by concurrent dialers to release slots precisely)
# ---------------------------------------------------------------------------

@app.get("/api/call-status/{session_id}")
async def call_status(session_id: str):
    return JSONResponse({
        "session_id": session_id,
        "active": (session_id in _active_sessions) or (session_id in _session_registry),
    })


# ---------------------------------------------------------------------------
# Explicit ACS hangup
# ---------------------------------------------------------------------------

@app.post("/api/hangup/{call_connection_id}")
async def hangup_call(call_connection_id: str):
    try:
        acs_client.get_call_connection(call_connection_id).hang_up(is_for_everyone=True)
        logger.info(f"Hung up | connectionId: {call_connection_id}")
        return JSONResponse({"status": "hung_up"})
    except Exception as e:
        logger.error(f"Hangup failed | connectionId: {call_connection_id} | {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket — one connection per active outbound call
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    ACS streams bidirectional PCM24K audio here after the outbound call
    is answered. Query params carry per-call context set by the dialer
    or /api/outboundCall.
    """
    await websocket.accept()

    params        = dict(websocket.query_params)
    org_name      = params.get("org_name", "")
    phone_number  = params.get("phone_number", "")
    services_list = params.get("services", "")
    unique_id     = params.get("unique_id", "")
    session_id    = params.get("session_id", str(uuid.uuid4()))

    call_connection_id = _session_registry.get(session_id, "")
    _active_sessions.add(session_id)

    # PCM capture buffers — appending bytes is negligible overhead.
    # _agent_chunks is kept time-aligned with the caller stream via silence padding.
    _caller_chunks: list[bytes] = []
    _agent_chunks:  list[bytes] = []
    _caller_bytes = [0]  # running byte count (list so closure can mutate)
    _agent_bytes  = [0]

    def _tts_capture(chunk: bytes) -> None:
        # Pad agent track with silence for any time Samantha wasn't speaking,
        # so agent audio stays time-aligned with the caller track.
        gap = _caller_bytes[0] - _agent_bytes[0]
        if gap > 0:
            silence = bytes(gap)
            _agent_chunks.append(silence)
            _agent_bytes[0] += gap
        _agent_chunks.append(chunk)
        _agent_bytes[0] += len(chunk)

    logger.info(
        f"WebSocket connected | org={org_name} | phone={phone_number} | "
        f"unique_id={unique_id} | session={session_id} | "
        f"conn_id={call_connection_id or 'pending'}"
    )
    ui_events.emit("websocket_connected", session_id=session_id, phone=phone_number, org=org_name)

    # ── Hangup callback ──────────────────────────────────────────────────────
    # Re-reads registry at call time so CallConnected latency doesn't matter.
    async def hangup_after(delay_seconds: int = 10):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        conn_id = _session_registry.get(session_id, "") or call_connection_id
        if not conn_id:
            logger.warning(f"Cannot hang up session {session_id[:8]} — no conn_id")
            return
        try:
            acs_client.get_call_connection(conn_id).hang_up(is_for_everyone=True)
            logger.info(f"Hung up | session={session_id[:8]} | conn_id={conn_id}")
        except Exception as e:
            logger.error(f"Hangup failed | session={session_id[:8]} | {e}")

    # Prerecorded voicemail: mono s16le @ 16 kHz (see app/assets/voicemail_message.pcm)
    voicemail_pcm_path = Path(__file__).resolve().parent / "assets" / "voicemail_message.pcm"
    voicemail_chunk_bytes = int(16000 * 0.02) * 2  # 20 ms frames

    async def play_prerecorded_voicemail():
        await acs_send_stop_audio(websocket)
        # Give ACS a moment to apply StopAudio before we start streaming PCM.
        # Without this, the first syllable/words can be clipped.
        await asyncio.sleep(0.2)
        try:
            pcm = voicemail_pcm_path.read_bytes()
        except OSError as e:
            logger.error(f"Voicemail PCM read failed | session={session_id[:8]} | {e}")
            return
        # Helpful INFO logs so we can verify playback attempts in server_logs.txt.
        duration_s = len(pcm) / (16000 * 2)  # s16le mono @ 16kHz
        logger.info(
            f"Playing prerecorded voicemail | session={session_id[:8]} | "
            f"bytes={len(pcm)} | dur≈{duration_s:.2f}s"
        )
        # Add a short silence lead-in to prevent clipping at the start.
        # 200ms of silence @ 16kHz mono s16le.
        silence_bytes = b"\x00\x00" * int(16000 * 0.2)
        for i in range(0, len(silence_bytes), voicemail_chunk_bytes):
            await acs_send_pcm_chunk(websocket, silence_bytes[i : i + voicemail_chunk_bytes])
            await asyncio.sleep(0.02)
        for i in range(0, len(pcm), voicemail_chunk_bytes):
            await acs_send_pcm_chunk(websocket, pcm[i : i + voicemail_chunk_bytes])
            # Pace the stream in real time (20 ms of audio per chunk).
            await asyncio.sleep(0.02)
        logger.info(f"Finished prerecorded voicemail | session={session_id[:8]}")

    # ── Create CallSession ───────────────────────────────────────────────────
    session = CallSession(
        org_name=org_name,
        phone_number=phone_number,
        services_list=services_list,
        unique_id=unique_id,
        session_id=session_id,
        results_dir=RESULTS_DIR,
        transcripts_dir=TRANSCRIPTS_DIR,
        hangup_fn=hangup_after,
        play_voicemail_fn=play_prerecorded_voicemail,
    )

    # ── Create ACS transport ─────────────────────────────────────────────────
    transport = ACSTransport(
        websocket=websocket,
        params=ACSTransportParams(sample_rate=16000),
        tts_capture_fn=_tts_capture,
    )

    # ── Create Pipecat pipeline ──────────────────────────────────────────────
    try:
        pipeline, task = create_pipeline(
            transport=transport,
            session=session,
        )
    except Exception as e:
        logger.error(f"Pipeline creation failed | session={session_id[:8]} | {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()
        return

    # ── Start timers (voicemail silence timeout + fallback hangup) ───────────
    session.start_timers()

    # ── Trigger opening immediately on pickup ───────────────────────────────
    # No caller-speaks-first gate — speak as soon as the receiver answers.
    asyncio.create_task(session.trigger_opening())

    # ── Run pipeline in background ───────────────────────────────────────────
    async def run_pipeline():
        try:
            runner = PipelineRunner(handle_sigint=False)
            await runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error | session={session_id[:8]} | {e}")
            import traceback
            traceback.print_exc()

    pipeline_task = asyncio.create_task(run_pipeline())

    # ── Read ACS audio directly and feed into pipeline ───────────────────────
    # This runs in the main ws_endpoint coroutine so it starts immediately
    # when the WebSocket opens — before the pipeline task even starts.
    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.info(f"WebSocket receive ended: {e}")
                break

            # Handle disconnect
            if message.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnected | session={session_id[:8]}")
                break

            raw = message.get("text") or message.get("bytes")
            if not raw:
                continue

            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except Exception:
                    continue

            try:
                data = json.loads(raw)
            except Exception:
                continue

            kind = data.get("kind", "")

            if kind == "AudioData":
                audio_data = data.get("audioData", {})
                b64 = audio_data.get("data", "")
                if b64:
                    try:
                        pcm_bytes = base64.b64decode(b64)
                        _caller_chunks.append(pcm_bytes)
                        _caller_bytes[0] += len(pcm_bytes)
                        pcm_for_pipeline = session.mute_inbound_pcm_if_needed(pcm_bytes)
                        frame = InputAudioRawFrame(
                            audio=pcm_for_pipeline,
                            sample_rate=16000,
                            num_channels=1,
                        )
                        await task.queue_frames([frame])
                    except Exception as e:
                        logger.warning(f"Audio frame error: {e}")

            elif kind == "StopAudio":
                logger.debug("[ACS] StopAudio received")

    finally:
        logger.info(f"WebSocket loop ended | session={session_id[:8]}")
        await session.handle_call_disconnected()
        session.cancel_timers()
        _active_sessions.discard(session_id)
        _session_registry.pop(session_id, None)
        if not pipeline_task.done():
            pipeline_task.cancel()
        if _caller_chunks or _agent_chunks:
            asyncio.create_task(_save_recording(
                session_id, unique_id,
                b"".join(_caller_chunks),
                b"".join(_agent_chunks),
            ))