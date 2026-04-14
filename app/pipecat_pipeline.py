"""
pipecat_pipeline.py
-------------------
Builds the Pipecat STT → LLM → TTS pipeline for one Samantha call.

Pipeline order:
  ACSAudioInput
      → Deepgram STT
      → InboundVADGateProcessor  (first N seconds: silence to VAD only; STT already
                                  saw real audio — avoids 1–2 word false barge-in)
      → TranscriptProcessor  (drives CallSession: voicemail/IVR detection,
                               caller-speaks-first gate, human guard; drops <3 word
                               guard interim/final so TranscriptionUserTurnStartStrategy
                               cannot broadcast interruption)
      → SileroVAD (inside LLMContextAggregatorPair)
      → user context aggregator
      → OpeningGuardDownstreamInterruptionGate  (drops downstream InterruptionFrame
                                                 during guard — VAD gate only saw upstream)
      → OpenAI GPT-4o        (function calling: extract_call_details)
      → Inworld TTS
      → ACSAudioOutput
      → assistant context aggregator

All V1 behavioural logic lives in CallSession / TranscriptProcessor.
This file only wires services and registers the function handler.
"""

import asyncio
import dataclasses
import os
from pathlib import Path
from typing import Callable, Tuple

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TextAggregationMode

from pipecat.frames.frames import AudioRawFrame, Frame, InterruptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.acs_transport import ACSTransport
from app.agent_settings import get_agent_settings, get_system_prompt, SAMANTHA_TOOLS
from app.call_session import CallSession
from app.transcript_processor import TranscriptProcessor


class InboundVADGateProcessor(FrameProcessor):
    """
    Opening guard (first N seconds, until 3+ caller words):
    - DOWNSTREAM: replace PCM with silence so Silero VAD does not see speech energy.
    - UPSTREAM: drop InterruptionFrame from user turn / VAD so TTS is not stopped
      on short utterances like \"Hello\" (Pipecat can still emit interruption before
      transcript word-count runs).
    Deepgram still receives real audio; STT is unchanged.
    """

    def __init__(self, session: CallSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, InterruptionFrame) and self._session.should_suppress_vad_during_guard():
                logger.debug(
                    "[InboundVADGate] Suppressed upstream InterruptionFrame during opening guard "
                    "(need 3+ words or window end before barge-in)"
                )
                return
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, AudioRawFrame) and self._session.should_suppress_vad_during_guard():
            muted = dataclasses.replace(frame, audio=b"\x00" * len(frame.audio))
            await self.push_frame(muted, direction)
            return
        await self.push_frame(frame, direction)


class OpeningGuardDownstreamInterruptionGate(FrameProcessor):
    """
    LLMUserAggregator.broadcast_interruption() sends InterruptionFrame downstream
    (to TTS) and upstream. InboundVADGateProcessor only suppresses the upstream
    copy. During the opening guard, drop the downstream copy so playback is not
    stopped on short utterances that still triggered a user-turn strategy.
    """

    def __init__(self, session: CallSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InterruptionFrame)
            and self._session.should_suppress_vad_during_guard()
        ):
            logger.debug(
                "[OpeningGuardInterruptionGate] Suppressed downstream InterruptionFrame "
                "during opening guard"
            )
            return
        await self.push_frame(frame, direction)


class SamanthaTextLogger(FrameProcessor):
    """Logs Samantha's text responses and forwards them to CallSession."""

    def __init__(self, session: CallSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            text = frame.text or ""
            self._buffer += text
            # Log complete sentences as they come
            if any(self._buffer.rstrip().endswith(p) for p in (".", "?", "!", ",")):
                text = self._buffer.strip()
                logger.info(f"[SAMANTHA]: {text}")
                self._session.on_samantha_text(text)
                self._buffer = ""
        await self.push_frame(frame, direction)


def create_pipeline(
    transport: ACSTransport,
    session: CallSession,
) -> Tuple[Pipeline, PipelineTask]:
    """
    Instantiate and wire the full Samantha Pipecat pipeline.

    Args:
        transport:  ACSTransport for this call
        session:    CallSession holding all per-call state and V1 logic

    Returns:
        (pipeline, task) — pass task to PipelineRunner.run()
    """
    _settings = get_agent_settings()
    stt_cfg   = _settings["stt"]
    llm_cfg   = _settings["llm"]
    tts_cfg   = _settings["tts"]
    vad_cfg   = _settings["vad"]
    audio_cfg = _settings["audio"]

    # ── Build personalised system prompt ────────────────────────────────────
    # Phone number is already formatted for speech inside CallSession
    system_prompt = get_system_prompt().format(
        org_name=session.org_name or "the organization",
        phone_number=session.phone_for_speech,
        services_list=session.services_list or "the listed services",
    )

    # ── STT: Deepgram ────────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        # Opening guard + Silero handle barge-in; avoid duplicate interrupt if Deepgram vad_events on.
        should_interrupt=False,
        settings=DeepgramSTTService.Settings(
            model=stt_cfg["model"],
            language=stt_cfg["language"],
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing=stt_cfg["endpointing"],
            utterance_end_ms=stt_cfg["utterance_end_ms"],
        ),
    )

    # ── LLM: OpenAI GPT-4o ──────────────────────────────────────────────────
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=llm_cfg["model"],
        ),
    )

    # ── TTS: Inworld ────────────────────────────────────────────────────────
    inworld_api_key = (os.getenv("INWORLD_API_KEY") or "").strip()
    if not inworld_api_key:
        raise RuntimeError(
            "Missing required environment variable: INWORLD_API_KEY\n"
            "Set it in your .env file before starting the server."
        )

    tts = InworldTTSService(
        api_key=inworld_api_key,
        settings=InworldTTSService.Settings(
            voice=tts_cfg["voice"],
            model=tts_cfg["model"],
        ),
        sample_rate=tts_cfg["sample_rate"],
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )

    # ── extract_call_details function handler ────────────────────────────────
    async def handle_extract_call_details(params: FunctionCallParams):
        # Snapshot args into CallSession immediately — before any async work.
        # This means if the caller disconnects during the handler, the
        # disconnect path will have the best available partial data.
        session.update_pending_tool_args(params.arguments)
        result = await session.handle_extract_call_details(params.arguments)
        await params.result_callback(result)

    llm.register_function("extract_call_details", handle_extract_call_details)

    # ── LLM context ─────────────────────────────────────────────────────────
    # System prompt only at init; per-call opening context is injected later
    # by CallSession.on_first_caller_speech() — exactly as V1 did
    messages = [{"role": "system", "content": system_prompt}]
    context  = LLMContext(messages=messages, tools=SAMANTHA_TOOLS)

    # ── Context aggregator pair with Silero VAD ──────────────────────────────
    aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=audio_cfg["sample_rate"],  # 16kHz — matches pipeline input
                params=VADParams(
                    confidence=vad_cfg["confidence"],
                    start_secs=vad_cfg["start_secs"],
                    stop_secs=vad_cfg["stop_secs"],
                    min_volume=vad_cfg["min_volume"],
                ),
            ),
        ),
    )

    user_aggregator      = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    # ── TranscriptProcessor + Samantha logger ───────────────────────────────
    vad_gate = InboundVADGateProcessor(session=session, name="InboundVADGateProcessor")
    transcript_proc = TranscriptProcessor(session=session, name="TranscriptProcessor")
    opening_guard_interrupt_gate = OpeningGuardDownstreamInterruptionGate(
        session=session, name="OpeningGuardDownstreamInterruptionGate"
    )
    samantha_logger = SamanthaTextLogger(session=session, name="SamanthaTextLogger")

    # ── Assemble pipeline ────────────────────────────────────────────────────
    pipeline = Pipeline([
        stt,                     # Deepgram STT → TranscriptionFrame
        vad_gate,                # Opening guard: optional silence to VAD path only
        transcript_proc,         # Caller-speaks-first gate + voicemail/IVR detection
        user_aggregator,         # Buffer user text → LLM context
        opening_guard_interrupt_gate,
        llm,                     # GPT-4o → text + function calls
        samantha_logger,         # Log [SAMANTHA]: lines
        tts,                     # Inworld TTS → PCM audio
        transport.output(),      # PCM audio out → ACS
        assistant_aggregator,    # Buffer assistant text → LLM context
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=audio_cfg["sample_rate"],   # 16kHz
            audio_out_sample_rate=audio_cfg["sample_rate"],  # 16kHz
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Give CallSession a reference to the task + context so it can
    # queue LLMRunFrames and inject context messages at the right moment
    session.attach_pipeline(task, context)

    logger.info(
        f"[PIPELINE] Created | org={session.org_name} | "
        f"session={session.session_id[:8]} | "
        f"STT={stt_cfg['model']} | LLM={llm_cfg['model']} | TTS=inworld"
    )

    return pipeline, task