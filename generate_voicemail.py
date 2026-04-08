"""
generate_voicemail.py
---------------------
Standalone script to generate a voicemail message using Inworld TTS
(same voice/model as the Samantha agent: Sarah, inworld-tts-1.5-max).

Saves output to:
  voice_mail/voicemail.pcm  — raw PCM s16le 16kHz mono (for pipeline use)
  voice_mail/voicemail.wav  — WAV (for listening)
  voice_mail/voicemail.mp3  — MP3 (requires ffmpeg)

Usage:
  python generate_voicemail.py
"""

import asyncio
import base64
import json
import os
import uuid
import wave
from pathlib import Path

import websockets
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
VOICEMAIL_TEXT = (
    "Hi, we are calling from GroundGame dot Health, and we wanted to confirm "
    "a few details about your organization. Please give us a call back at "
    "813 851 0780. Thank you and have a great day."
)

VOICE       = "Sarah"
MODEL       = "inworld-tts-1.5-max"
SAMPLE_RATE = 16000
ENCODING    = "LINEAR16"

OUTPUT_DIR  = Path("voice_mail")
PCM_FILE    = OUTPUT_DIR / "voicemail.pcm"
WAV_FILE    = OUTPUT_DIR / "voicemail.wav"
MP3_FILE    = OUTPUT_DIR / "voicemail.mp3"

WS_URL      = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
# ---------------------------------------------------------------------------


async def generate():
    api_key = os.getenv("INWORLD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("INWORLD_API_KEY not set in .env")

    OUTPUT_DIR.mkdir(exist_ok=True)

    request_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())

    headers = [
        ("Authorization", f"Basic {api_key}"),
        ("X-Request-Id", request_id),
    ]

    audio_chunks: list[bytes] = []

    print("Connecting to Inworld TTS...")

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:

        # 1. Create context (voice/model/audio config)
        await ws.send(json.dumps({
            "contextId": context_id,
            "create": {
                "voiceId": VOICE,
                "modelId": MODEL,
                "audioConfig": {
                    "audioEncoding": ENCODING,
                    "sampleRateHertz": SAMPLE_RATE,
                },
                "autoMode": True,
            }
        }))

        # 2. Send text
        await ws.send(json.dumps({
            "contextId": context_id,
            "send_text": {"text": VOICEMAIL_TEXT},
        }))

        # 3. Flush to trigger synthesis
        await ws.send(json.dumps({
            "contextId": context_id,
            "flush": {},
        }))

        # 4. Close context to signal end
        await ws.send(json.dumps({
            "contextId": context_id,
            "close": {},
        }))

        # 5. Receive all audio until contextClosed or timeout
        print("Receiving audio...")
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=15.0)
            except asyncio.TimeoutError:
                print("Timeout — stopping receive.")
                break

            msg = json.loads(message)
            result = msg.get("result", {})

            # Error check
            status = result.get("status", {})
            if status.get("code", 0) != 0:
                raise RuntimeError(f"Inworld error {status['code']}: {status.get('message')}")

            # Collect audio chunks
            if "audioChunk" in result:
                audio_chunk = result["audioChunk"]
                pcm_b64 = audio_chunk.get("audioContent", "")
                if pcm_b64:
                    audio = base64.b64decode(pcm_b64)
                    # Strip 44-byte WAV/RIFF header if present
                    if len(audio) > 44 and audio.startswith(b"RIFF"):
                        audio = audio[44:]
                    audio_chunks.append(audio)

            # Stop when flush completed (all audio sent) or context closed
            if "flushCompleted" in result or "contextClosed" in result:
                print("Done receiving audio.")
                break

    if not audio_chunks:
        raise RuntimeError("No audio received — check API key and credits")

    pcm_data = b"".join(audio_chunks)

    # Save PCM
    PCM_FILE.write_bytes(pcm_data)
    print(f"Saved PCM  -> {PCM_FILE}  ({len(pcm_data):,} bytes)")

    # Save WAV
    with wave.open(str(WAV_FILE), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    print(f"Saved WAV  -> {WAV_FILE}")

    # Convert to MP3 via ffmpeg
    ret = os.system(f'ffmpeg -y -i "{WAV_FILE}" -codec:a libmp3lame -qscale:a 2 "{MP3_FILE}" -loglevel quiet')
    if ret == 0:
        print(f"Saved MP3  -> {MP3_FILE}")
    else:
        print("ffmpeg not found — MP3 skipped. WAV is playable directly.")

    duration_s = len(pcm_data) / (SAMPLE_RATE * 2)
    print(f"\nDuration   : {duration_s:.1f}s")
    print(f"Text       : {VOICEMAIL_TEXT}")


if __name__ == "__main__":
    asyncio.run(generate())
