"""
transcript_processor.py
------------------------
A Pipecat FrameProcessor that sits between Deepgram STT and the
LLM context aggregator.

It intercepts TranscriptionFrame (finalised STT text) and calls the
CallSession methods that implement V1's behavioural logic:

  1. on_transcript(text)       — voicemail/IVR detection + human guard

All other frames are passed through transparently.
"""

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.call_session import CallSession


class TranscriptProcessor(FrameProcessor):
    """
    Intercepts finalised STT transcripts to drive CallSession logic.
    Passes all frames downstream unchanged.
    """

    def __init__(self, session: CallSession, **kwargs):
        super().__init__(**kwargs)
        self._session          = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                # Always run transcript logic (voicemail/IVR detection, human guard)
                await self._session.on_transcript(text)
                # If voicemail/IVR was detected, CallSession ends the call.
                # Do not forward this transcription into the LLM/TTS pipeline,
                # otherwise Samantha can respond and interrupt the voicemail flow.
                if self._session.call_ended:
                    return
                # Re-intro path: session added the caller's text to context manually
                # and set this flag so we don't double-add it via the user aggregator.
                if self._session.should_drop_transcript_from_pipeline():
                    return

        # Pass every frame downstream unchanged
        await self.push_frame(frame, direction)