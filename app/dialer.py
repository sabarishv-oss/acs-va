"""
dialer.py
---------
Reads campaign_input.csv and places outbound ACS calls for each row.

CSV columns expected:
    org_name, phone_number, services, unique_id

A session_id (UUID) is generated per call and embedded in BOTH the
WebSocket URL and the callback URI.  When ACS fires CallConnected, the
server stores session_id → call_connection_id so the WebSocket handler
can resolve it for clean hangup.

Batching / throttling is controlled via env vars:
    CALLS_PER_BATCH      (default 10)
    BATCH_DELAY_SECONDS  (default 10)
"""

import csv
import os
import time
import uuid
from typing import Iterable
from urllib.parse import urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from loguru import logger
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

ACS_CONNECTION_STRING   = os.getenv("ACS_CONNECTION_STRING")
ACS_SOURCE_PHONE_NUMBER = os.getenv("ACS_SOURCE_PHONE_NUMBER")
CALLBACK_URI_HOST       = os.getenv("CALLBACK_URI_HOST")
CALLBACK_EVENTS_URI     = (CALLBACK_URI_HOST or "") + "/api/callbacks"

CAMPAIGN_INPUT_CSV  = os.getenv("CAMPAIGN_INPUT_CSV", "./campaign_input.csv")
CALLS_PER_BATCH     = int(os.getenv("CALLS_PER_BATCH", "10"))
BATCH_DELAY_SECONDS = int(os.getenv("BATCH_DELAY_SECONDS", "10"))


def validate_env() -> None:
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


def load_targets(path: str) -> Iterable[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _e164(number: str) -> str:
    """Normalise to E.164 — prepend +1 if no country code present."""
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


def place_calls() -> None:
    validate_env()
    client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)

    logger.info(f"Starting outbound campaign | source: {ACS_SOURCE_PHONE_NUMBER}")
    logger.info(f"Reading targets from: {CAMPAIGN_INPUT_CSV}")

    batch_count = 0
    batch_calls = 0

    for row in load_targets(CAMPAIGN_INPUT_CSV):
        raw_number = (
            row.get("phone_number") or row.get("phone") or row.get("to") or ""
        ).strip()

        if not raw_number:
            logger.warning(f"Skipping row — no phone number: {row}")
            continue

        phone_number = _e164(raw_number)
        org_name     = (row.get("org_name") or "").strip()
        services     = (
            row.get("services") or
            row.get("services_listed") or
            row.get("services_list") or ""
        ).strip()
        unique_id    = (row.get("unique_id") or row.get("id") or str(uuid.uuid4())).strip()

        session_id = str(uuid.uuid4())

        ws_params = urlencode({
            "org_name":     org_name,
            "phone_number": phone_number,
            "services":     services,
            "unique_id":    unique_id,
            "session_id":   session_id,
        })

        callback_uri  = f"{CALLBACK_EVENTS_URI}/{session_id}?{ws_params}"
        websocket_url = _build_websocket_url(ws_params)

        logger.info(
            f"Placing call → {phone_number} | org={org_name} | "
            f"unique_id={unique_id} | session_id={session_id}"
        )
        logger.debug(f"  callback_uri : {callback_uri}")
        logger.debug(f"  websocket_url: {websocket_url}")

        try:
            result = client.create_call(
                target_participant=PhoneNumberIdentifier(phone_number),
                source_caller_id_number=PhoneNumberIdentifier(ACS_SOURCE_PHONE_NUMBER),
                callback_url=callback_uri,
                media_streaming=_build_media_streaming_options(websocket_url),
                operation_context=ws_params,
            )
            logger.info(
                f"Call placed | connection_id={result.call_connection_id} | "
                f"session_id={session_id} | unique_id={unique_id}"
            )
        except Exception as e:
            logger.error(f"Failed to place call to {phone_number}: {e}")
            continue

        batch_calls += 1
        if batch_calls >= CALLS_PER_BATCH:
            batch_count += 1
            logger.info(
                f"Batch {batch_count} done ({batch_calls} calls). "
                f"Sleeping {BATCH_DELAY_SECONDS}s before next batch…"
            )
            time.sleep(BATCH_DELAY_SECONDS)
            batch_calls = 0

    logger.info("Campaign complete.")


if __name__ == "__main__":
    place_calls()