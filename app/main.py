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
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
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

from app.acs_transport import ACSTransport, ACSTransportParams
from app.call_session import CallSession
from app.pipecat_pipeline import create_pipeline

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


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return JSONResponse({"message": "Samantha Outbound Voice Agent — ready."})


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

    result = acs_client.create_call(
        target_participant=PhoneNumberIdentifier(phone_number),
        source_caller_id_number=PhoneNumberIdentifier(ACS_SOURCE_PHONE_NUMBER),
        callback_url=callback_uri,
        media_streaming=_build_media_streaming_options(ws_url),
        operation_context=context_params,
    )

    _session_registry[session_id] = result.call_connection_id
    logger.info(f"Registered session {session_id} → {result.call_connection_id}")

    return JSONResponse({
        "call_connection_id": result.call_connection_id,
        "session_id":         session_id,
        "unique_id":          unique_id,
    })


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

        elif event_type == "Microsoft.Communication.MediaStreamingStarted":
            logger.info(
                f"Media streaming started | "
                f"status={event_data['mediaStreamingUpdate']['mediaStreamingStatus']}"
            )

        elif event_type == "Microsoft.Communication.MediaStreamingStopped":
            logger.info("Media streaming stopped.")

        elif event_type == "Microsoft.Communication.MediaStreamingFailed":
            logger.error(
                f"Media streaming failed | "
                f"code={event_data['resultInformation']['code']} | "
                f"msg={event_data['resultInformation']['message']}"
            )

        elif event_type == "Microsoft.Communication.CallDisconnected":
            logger.info(f"Call disconnected | connectionId: {call_connection_id}")
            # Mark inactive for dialers waiting to place the next call.
            # Also drop the registry mapping so /api/call-status can reflect
            # that the call has ended.
            _active_sessions.discard(contextId)
            _session_registry.pop(contextId, None)

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

    logger.info(
        f"WebSocket connected | org={org_name} | phone={phone_number} | "
        f"unique_id={unique_id} | session={session_id} | "
        f"conn_id={call_connection_id or 'pending'}"
    )

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
    )

    # ── Create ACS transport ─────────────────────────────────────────────────
    transport = ACSTransport(
        websocket=websocket,
        params=ACSTransportParams(sample_rate=16000),
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
                        frame = InputAudioRawFrame(
                            audio=pcm_bytes,
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