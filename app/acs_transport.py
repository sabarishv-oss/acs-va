"""
acs_transport.py
----------------
Bridges Azure Communication Services bidirectional PCM audio WebSocket
into Pipecat's frame-based pipeline.

ACS sends audio as JSON:
  { "kind": "AudioData", "audioData": { "data": "<base64-pcm>" } }

We decode → emit InputAudioRawFrame upstream into Pipecat.

Pipecat sends OutputAudioRawFrame chunks → we base64-encode and push
back to ACS as:
  { "kind": "AudioData", "audioData": { "data": "<base64-pcm>",
    "timestamp": "...", "silent": false } }

ACS uses PCM 24kHz 16-bit mono (PCM24_K_MONO).

Issue 3 fix: ACSAudioInput now reads sample_rate and channels from the
params object instead of module-level constants, so changing the rate
in one place (ACSTransportParams) propagates correctly everywhere.

Issue 5 fix: All ACS JSON keys use lowercase schema (kind/audioData/data)
matching what ACS actually sends and expects.
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport


# Module-level defaults — used only as fallback, never read directly by processors
_DEFAULT_SAMPLE_RATE = 24000
_DEFAULT_CHANNELS    = 1


@dataclass
class ACSTransportParams:
    """
    Parameters for ACSTransport.

    Issue 3 fix: sample_rate and channels are explicit fields here.
    ACSAudioInput reads from this object, not from module constants,
    so changing these values propagates correctly to InputAudioRawFrame.
    """
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    channels: int    = _DEFAULT_CHANNELS


class ACSAudioInput(FrameProcessor):
    """
    Reads JSON messages from the ACS WebSocket and emits
    InputAudioRawFrame frames into the pipeline.

    Issue 3 fix: uses params.sample_rate / params.channels throughout.
    Issue 5 fix: JSON key schema matches ACS spec (lowercase kind/audioData/data).
    """

    def __init__(self, websocket: WebSocket, params: ACSTransportParams, **kwargs):
        super().__init__(**kwargs)
        self._websocket   = websocket
        self._sample_rate = params.sample_rate
        self._channels    = params.channels
        self._running     = False
        self._read_task: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        self._running   = True
        self._read_task = asyncio.create_task(self._read_loop())

    async def stop(self, frame: EndFrame):
        self._running = False
        if self._read_task:
            self._read_task.cancel()

    async def cancel(self, frame: CancelFrame):
        self._running = False
        if self._read_task:
            self._read_task.cancel()

    async def _read_loop(self):
        try:
            while self._running:
                if self._websocket.client_state != WebSocketState.CONNECTED:
                    break
                try:
                    # Use receive() instead of receive_json() so we handle
                    # both text and binary frames without crashing
                    message = await asyncio.wait_for(
                        self._websocket.receive(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    continue

                # ACS sends audio as text frames containing JSON
                raw = message.get("text") or message.get("bytes")
                if not raw:
                    continue

                # If bytes came in, decode to string first
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except Exception:
                        continue

                # Parse JSON
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                kind = data.get("kind", "")

                if kind == "AudioData":
                    audio_data = data.get("audioData", {})
                    b64        = audio_data.get("data", "")
                    if b64:
                        try:
                            pcm_bytes = base64.b64decode(b64)
                        except Exception:
                            continue
                        frame = InputAudioRawFrame(
                            audio=pcm_bytes,
                            sample_rate=self._sample_rate,
                            num_channels=self._channels,
                        )
                        await self.push_frame(frame)

                elif kind == "StopAudio":
                    logger.debug("[ACS] StopAudio received from ACS")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[ACS Input] Read loop error: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class ACSAudioOutput(FrameProcessor):
    """
    Receives OutputAudioRawFrame from the pipeline and sends
    base64-encoded PCM back to the ACS WebSocket.

    Issue 5 fix: outbound JSON uses lowercase keys (kind/audioData/data)
    matching the ACS schema. V1's communication_handler used mixed-case
    (Kind/AudioData/StopAudio) which was a latent bug.
    """

    def __init__(self, websocket: WebSocket, **kwargs):
        super().__init__(**kwargs)
        self._websocket = websocket

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ElevenLabs emits TTSAudioRawFrame; some other transports emit OutputAudioRawFrame.
        # For ACS we just need raw PCM bytes, so handle both.
        if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
            await self._send_audio(frame.audio)
        # Pipecat emits InterruptionFrame on barge-in; StartInterruptionFrame is deprecated.
        elif isinstance(frame, (InterruptionFrame, StartInterruptionFrame)):
            await self._stop_audio()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _send_audio(self, pcm_bytes: bytes):
        if self._websocket.client_state != WebSocketState.CONNECTED:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        # Issue 5: lowercase keys — matches what ACS expects
        msg = json.dumps({
            "kind": "AudioData",
            "audioData": {
                "data": b64,
                "timestamp": str(int(time.time() * 1000)),
                "silent": False,
            },
        })
        try:
            await self._websocket.send_text(msg)
        except Exception as e:
            logger.warning(f"[ACS Output] Send error: {e}")

    async def _stop_audio(self):
        """Send StopAudio to ACS — triggered on barge-in / interruption."""
        if self._websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            # Issue 5: lowercase key
            await self._websocket.send_text(json.dumps({"kind": "StopAudio"}))
            logger.debug("[ACS Output] StopAudio sent to ACS")
        except Exception as e:
            logger.warning(f"[ACS Output] StopAudio send error: {e}")


class ACSTransport(BaseTransport):
    """
    Thin transport wrapper exposing .input() and .output()
    for use in a Pipecat Pipeline.

    Issue 3 fix: params object is passed into ACSAudioInput so
    sample_rate and channels come from one authoritative source.
    """

    def __init__(self, websocket: WebSocket, params: ACSTransportParams | None = None):
        self._params = params or ACSTransportParams()
        # Pipecat 0.0.106: BaseTransport.__init__ takes no arguments
        super().__init__()
        self._input  = ACSAudioInput(
            websocket=websocket,
            params=self._params,
            name="ACSAudioInput",
        )
        self._output = ACSAudioOutput(websocket=websocket, name="ACSAudioOutput")

    def input(self) -> ACSAudioInput:
        return self._input

    def output(self) -> ACSAudioOutput:
        return self._output