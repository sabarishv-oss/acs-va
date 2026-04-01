"""
call_session.py
---------------
CallSession manages all per-call state and behavioural logic
that lived in V1's CommunicationHandler, now adapted for Pipecat.

Responsibilities:
  1. Caller-speaks-first gate
       Samantha does NOT speak until the caller has said something.
       The Pipecat pipeline is started but LLMRunFrame is held until
       first STT transcript arrives.

  2. Voicemail keyword detection
       Every caller transcript (before human_confirmed=True) is scanned
       for voicemail/IVR keywords. On match, result is saved and call ends.

  3. 15-second silence timeout
       If no caller speech arrives within 15s, assume voicemail picked up.

  4. Human confirmed guard
       After the first real caller transcript, keyword detection is disabled
       permanently for the rest of the call to prevent false positives.

  5. extract_call_details handler
       Saves structured JSON result, tells GPT-4o to say goodbye,
       schedules ACS hangup.

  6. Partial result capture on mid-call disconnect
       If the caller hangs up before extract_call_details fires,
       saves whatever partial state is available with
       call_outcome = call_disconnected.

  7. Fallback hangup
       300-second hard ceiling — if the call never ends cleanly,
       hang up anyway.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import time

from loguru import logger
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, InterruptionFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext

from app.agent_settings import (
    AGENT_SETTINGS,
    VOICEMAIL_KEYWORDS,
    IVR_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_keyword(text: str, keywords: list[str]) -> str | None:
    """Return the first matched keyword found in text (case-insensitive), else None."""
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return kw
    return None


def _format_phone_for_speech(phone: str) -> str:
    """
    Strip +1 country code and format as 'XXX XXX XXXX' for natural speech.
    Matches V1 _format_phone_for_speech exactly.
    """
    digits = (
        phone.strip()
        .replace("-", "").replace(" ", "")
        .replace("(", "").replace(")", "")
    )
    if digits.startswith("+1"):
        digits = digits[2:]
    elif digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    return digits


# ---------------------------------------------------------------------------
# CallSession
# ---------------------------------------------------------------------------

class CallSession:
    """
    One instance per active ACS call.
    Holds all mutable state and orchestrates the V1 behavioural logic.
    """

    def __init__(
        self,
        org_name: str,
        phone_number: str,
        services_list: str,
        unique_id: str,
        session_id: str,
        results_dir: Path,
        transcripts_dir: Path,
        hangup_fn,            # async callable(delay_seconds: int)
        play_voicemail_fn,    # async callable() -> None — streams prerecorded PCM to ACS
    ):
        self.org_name       = org_name
        self.phone_number   = phone_number
        self.phone_for_speech = _format_phone_for_speech(phone_number)
        self.services_list  = services_list
        self.unique_id      = unique_id
        self.session_id     = session_id
        self.results_dir    = results_dir
        self.transcripts_dir = transcripts_dir
        self._hangup_fn           = hangup_fn
        self._play_voicemail_fn   = play_voicemail_fn

        # Incremental result snapshot (updated during call)
        self._incremental_uid = (unique_id or "").strip() or session_id
        self._incremental_result_file = self.results_dir / f"{self._incremental_uid}.partial.json"
        self._last_incremental_write_at = 0.0
        self._last_caller_text: str = ""
        self._last_samantha_text: str = ""

        safe_uid = (unique_id or "").strip()
        if safe_uid:
            transcript_name = f"{safe_uid}_{session_id[:8]}.txt"
        else:
            transcript_name = f"{session_id}.txt"
        self._transcript_file = self.transcripts_dir / transcript_name
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # ── V1 flags (exact equivalents) ────────────────────────────────────
        # _response_triggered  → True after first caller speech detected
        #                        (Samantha's opening is queued at that point)
        self._response_triggered: bool = False

        # _opening_spoken → True once TTSSpeakFrame has been queued.
        # Guards on_transcript so it does NOT queue an extra LLMRunFrame
        # on the first caller turn — the assistant aggregator already does
        # that automatically once it finishes aggregating the opening audio.
        self._opening_spoken: bool = False

        # _opening_deaf_until → monotonic timestamp until which incoming
        # caller transcripts are silently dropped. Prevents early "hello" or
        # org-name greetings spoken before/during the opening audio from
        # interfering with the pipeline before Samantha has finished speaking.
        self._opening_deaf_until: float = 0.0
        self._opening_deaf_secs: float  = 5.0  # seconds to ignore after opening is queued

        # _human_confirmed     → True after first real caller transcript
        #                        Once True, keyword detection is disabled
        self._human_confirmed: bool    = False

        # _call_ended          → guards all result-saving paths so only one runs
        self._call_ended: bool         = False

        # _goodbye_done        → True after extract_call_details fires;
        #                        used to know hangup should follow greeting
        self._goodbye_done: bool       = False

        # ── Pipeline references (set after pipeline is created) ──────────────
        self._pipeline_task: PipelineTask | None  = None
        self._llm_context: LLMContext | None      = None

        # ── Background tasks ─────────────────────────────────────────────────
        self._silence_timeout_task: asyncio.Task | None  = None
        self._fallback_hangup_task: asyncio.Task | None  = None

        # Prevents scheduling the goodbye-safety hangup more than once
        self._goodbye_safety_scheduled: bool = False

        # ── Partial result buffer ─────────────────────────────────────────────
        # Populated by the pipeline the moment GPT-4o invokes extract_call_details
        # (even before the handler finishes), so handle_call_disconnected() can
        # recover whatever args were captured if the caller hangs up mid-tool-call.
        self._pending_tool_args: dict = {}

    def _build_incremental_snapshot(self) -> dict:
        return {
            "unique_id": self.unique_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "org_name": self.org_name,
            "dialed_number": self.phone_number,
            "dialed_services": self.services_list,
            "human_confirmed": self._human_confirmed,
            "response_triggered": self._response_triggered,
            "goodbye_done": self._goodbye_done,
            "call_ended": self._call_ended,
            "last_caller_text": self._last_caller_text,
            "last_samantha_text": self._last_samantha_text,
            "pending_tool_args": self._pending_tool_args,
            "partial_capture": True,
        }

    def _maybe_save_incremental(self, *, force: bool = False) -> None:
        if self._call_ended and not force:
            return
        now = time.monotonic()
        # Throttle writes to avoid hammering disk during streaming TTS/STT.
        if not force and (now - self._last_incremental_write_at) < 2.0:
            return
        self._last_incremental_write_at = now
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self._incremental_result_file.write_text(
                json.dumps(self._build_incremental_snapshot(), indent=2),
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            logger.debug(f"[CALL RESULT] Incremental save failed: {e}")

    def _append_transcript_line(self, speaker: str, text: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts} | {speaker}: {text.strip()}\n"
        try:
            with open(self._transcript_file, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)
        except Exception as e:
            logger.debug(f"[TRANSCRIPT] Append failed: {e}")

    # Farewell phrases that signal the agent is ending the conversation.
    # If extract_call_details was already called, _call_ended is True and the
    # safety task below does nothing.  If the LLM skips the tool, the task
    # force-saves a partial result and hangs up after GOODBYE_SAFETY_DELAY_S.
    _GOODBYE_PATTERNS = (
        "have a great day",
        "have a good day",
        "goodbye",
        "good bye",
        "take care",
        "thanks for your time",
        "thank you for your time",
    )
    GOODBYE_SAFETY_DELAY_S = 8  # seconds to wait before force-hanging-up

    def on_samantha_text(self, text: str) -> None:
        if text and not self._call_ended:
            self._last_samantha_text = text.strip()
            self._append_transcript_line("SAMANTHA", text)
            self._maybe_save_incremental()

            if not self._goodbye_safety_scheduled:
                lowered = self._last_samantha_text.lower()
                if any(p in lowered for p in self._GOODBYE_PATTERNS):
                    self._goodbye_safety_scheduled = True
                    asyncio.create_task(self._goodbye_safety_hangup())

    async def _goodbye_safety_hangup(self) -> None:
        """
        Safety net: fires GOODBYE_SAFETY_DELAY_S seconds after Samantha says a
        farewell phrase.  If extract_call_details was called first (normal path),
        _call_ended is already True and this is a no-op.  If the LLM skipped the
        tool, we force-save whatever partial data we have and hang up so the call
        does not stay open indefinitely.
        """
        await asyncio.sleep(self.GOODBYE_SAFETY_DELAY_S)
        if self._call_ended:
            return

        logger.warning(
            f"[SESSION {self.session_id[:8]}] extract_call_details was NOT called before "
            f"goodbye — force-saving partial result and hanging up | "
            f"unique_id={self.unique_id}"
        )
        self._call_ended = True
        self.cancel_timers()

        # Build result from whatever partial state was captured
        pa = self._pending_tool_args
        result = {
            "unique_id":            self.unique_id,
            "session_id":           self.session_id,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "phone_status":         pa.get("phone_status", "unknown"),
            "is_correct_number":    pa.get("is_correct_number", "unknown"),
            "org_valid":            pa.get("org_valid", "unknown"),
            "call_outcome":         pa.get("call_outcome", "other"),
            "call_summary": (
                pa.get("call_summary") or
                f"Call with {self.org_name} ended without extract_call_details. "
                f"Last caller: '{self._last_caller_text}'. "
                f"Last agent: '{self._last_samantha_text}'."
            ),
            "other_numbers":        pa.get("other_numbers"),
            "services_confirmed":   pa.get("services_confirmed", "unknown"),
            "available_services":   pa.get("available_services", []),
            "unavailable_services": pa.get("unavailable_services", []),
            "other_services":       pa.get("other_services", []),
            "mentioned_funding":    pa.get("mentioned_funding", "no"),
            "mentioned_callback":   pa.get("mentioned_callback", "no"),
            "dialed_number":        self.phone_number,
            "dialed_services":      self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)
        await self._hangup_fn(0)

    # ------------------------------------------------------------------
    # Called by pipeline factory after task is created
    # ------------------------------------------------------------------

    def attach_pipeline(self, task: PipelineTask, context: LLMContext):
        self._pipeline_task = task
        self._llm_context   = context

    def update_pending_tool_args(self, args: dict):
        """
        Called by the pipeline the moment GPT-4o fires extract_call_details,
        BEFORE the async handler finishes. This snapshot means that if the
        caller disconnects mid-tool-call, handle_call_disconnected() will have
        the best available args rather than an empty dict.

        Equivalent to V1's _fn_args_buf which accumulated streamed JSON deltas.
        In Pipecat, the LLM service delivers complete arguments at once, so we
        snapshot the whole dict here rather than accumulating deltas.
        """
        self._pending_tool_args = dict(args)
        self._maybe_save_incremental(force=True)

    # ------------------------------------------------------------------
    # Session start — kick off timers
    # ------------------------------------------------------------------

    def start_timers(self):
        """
        Start the voicemail silence timeout and fallback hangup.
        Call this as soon as the WebSocket is open and pipeline is running.
        """
        cfg = AGENT_SETTINGS["call"]
        self._silence_timeout_task = asyncio.create_task(
            self._voicemail_silence_timeout(cfg["voicemail_silence_timeout_seconds"])
        )
        self._fallback_hangup_task = asyncio.create_task(
            self._fallback_hangup(cfg["fallback_hangup_seconds"])
        )
        logger.info(
            f"[SESSION {self.session_id[:8]}] Timers started | "
            f"silence_timeout={cfg['voicemail_silence_timeout_seconds']}s | "
            f"fallback_hangup={cfg['fallback_hangup_seconds']}s"
        )

    def cancel_timers(self):
        """Cancel background timer tasks on clean call end."""
        for task in (self._silence_timeout_task, self._fallback_hangup_task):
            if task and not task.done():
                task.cancel()

    # ------------------------------------------------------------------
    # Opening trigger
    # ------------------------------------------------------------------

    def _build_opening_text(self) -> str:
        """
        Construct Samantha's deterministic opening line.

        The opening is fully static except for org_name and phone_for_speech —
        no LLM needed. We always use the two-question variant at pickup time
        because we don't know yet whether the caller will say the org name.
        GPT-4o handles the branching logic from the caller's first reply onward.
        """
        return (
            f"Hi, this is Samantha from GroundGame dot Health. "
            f"Just to confirm, are we speaking to {self.org_name}? "
            f"And is {self.phone_for_speech} the best number to reach you?"
        )

    def _inject_call_context(self, opening_already_spoken: bool = False):
        """Inject per-call facts into LLM context. Called before either opening path."""
        if self._llm_context is None:
            return
        if opening_already_spoken:
            note = (
                f"IMPORTANT: The opening greeting was already spoken to the caller "
                f"via a pre-rendered TTS step — you did NOT generate it. "
                f"Do NOT repeat the introduction or re-ask any question the "
                f"opening already covered. When the caller responds, pick up "
                f"from their reply and follow the branching logic in your system prompt."
            )
        else:
            note = (
                f"IMPORTANT: The caller spoke before the opening greeting was played. "
                f"You must now introduce yourself as Samantha from GroundGame dot Health "
                f"and ask BOTH opening questions: (1) confirm org name, (2) confirm phone number. "
                f"Do NOT skip either question unless the caller explicitly confirmed them. "
                f"A greeting like 'hello' or 'who is this' is NOT a confirmation of anything — "
                f"you must still ask both questions after introducing yourself."
            )
        self._llm_context.messages.append({
            "role": "system",
            "content": (
                f"[CALL CONTEXT — do not read aloud]\n"
                f"Organization to verify: {self.org_name}\n"
                f"Phone number dialed (say as): {self.phone_for_speech}\n"
                f"Services to verify: {self.services_list}\n"
                f"Unique ID for this call (never say aloud): {self.unique_id}\n\n"
                f"{note}"
            ),
        })

    async def trigger_opening(self):
        """
        Opening strategy:
        - Immediately queue TTSSpeakFrame (no LLM, zero latency).
        - If caller speaks during playback → on_transcript cancels the TTS
          and falls back to LLM which adapts based on what caller said.
        """
        if self._response_triggered or self._call_ended:
            return
        self._response_triggered = True

        if self._call_ended:
            return

        # Inject call context into LLM — needed for both TTS and LLM paths.
        self._inject_call_context(opening_already_spoken=True)

        opening_text = self._build_opening_text()

        if self._pipeline_task is not None:
            self._opening_spoken = True
            # Deaf period = estimated time for opening audio to finish playing.
            self._opening_deaf_until = time.monotonic() + self._opening_deaf_secs
            self._append_transcript_line("SAMANTHA", opening_text)
            self._last_samantha_text = opening_text
            await self._pipeline_task.queue_frames([
                TTSSpeakFrame(text=opening_text, append_to_context=True),
            ])
            logger.info(
                f"[SESSION {self.session_id[:8]}] TTSSpeakFrame queued immediately | "
                f"opening='{opening_text[:80]}…'"
            )
            logger.info(f"[SAMANTHA]: {opening_text}")

    # ------------------------------------------------------------------
    # Transcript processing — voicemail / IVR detection + human guard
    # Called by TranscriptProcessor for every finalised STT transcript
    # ------------------------------------------------------------------

    async def on_transcript(self, text: str):
        """
        Equivalent to V1 conversation.item.input_audio_transcription.completed handler.

        Before human_confirmed:
          - Check for voicemail keywords → handle_voicemail
          - Check for IVR keywords      → handle_ivr
        After first real transcript: set human_confirmed = True (disables checks)
        """
        if self._call_ended:
            return

        logger.info(f"[CALLER]: {text}")
        self._last_caller_text = text.strip()
        self._append_transcript_line("CALLER", text)
        self._maybe_save_incremental()

        if not self._human_confirmed:
            vm_match = _contains_keyword(text, VOICEMAIL_KEYWORDS)
            if vm_match:
                logger.info(
                    f"[SESSION {self.session_id[:8]}] Voicemail keyword: '{vm_match}'"
                )
                await self.handle_voicemail_detected(reason=f"keyword: {vm_match}")
                return

            if not self._call_ended:
                ivr_match = _contains_keyword(text, IVR_KEYWORDS)
                if ivr_match:
                    logger.info(
                        f"[SESSION {self.session_id[:8]}] IVR keyword: '{ivr_match}'"
                    )
                    await self.handle_ivr_detected(matched_keyword=ivr_match)
                    return

        # Set AFTER keyword check — first real transcript still gets checked,
        # all subsequent ones skip keyword detection entirely
        if not self._human_confirmed and not self._call_ended:
            self._human_confirmed = True
            logger.info(
                f"[SESSION {self.session_id[:8]}] Human confirmed — "
                f"keyword detection disabled for rest of call"
            )
            # If the opening was pre-rendered via TTSSpeakFrame, the assistant
            # aggregator will push a context frame (and LLM call) automatically
            # once it aggregates the opening audio. Do NOT queue an extra
            # LLMRunFrame here — that would cause a duplicate GPT-4o response.
            if self._opening_spoken:
                # Caller spoke before/during the opening — stop the audio.
                # Remove the "opening already spoken" context so GPT-4o doesn't
                # skip the org/phone questions thinking they were already answered.
                logger.info(
                    f"[SESSION {self.session_id[:8]}] Caller spoke during opening — "
                    f"stopping audio, GPT-4o will re-introduce"
                )
                if self._llm_context is not None:
                    # Strip the call context message injected at opening time
                    self._llm_context.messages = [
                        m for m in self._llm_context.messages
                        if not (
                            m.get("role") == "system" and
                            "CALL CONTEXT" in m.get("content", "")
                        )
                    ]
                    # Re-inject without the "already spoken" instruction
                    self._inject_call_context(opening_already_spoken=False)
                if self._pipeline_task is not None:
                    await self._pipeline_task.queue_frames([InterruptionFrame()])
                    await self._pipeline_task.queue_frames([LLMRunFrame()])
                return

    # ------------------------------------------------------------------
    # Voicemail silence timeout
    # ------------------------------------------------------------------

    async def _voicemail_silence_timeout(self, seconds: int):
        """
        If no caller speech within `seconds`, assume voicemail picked up silently.
        Equivalent to V1 _voicemail_silence_timeout.
        """
        await asyncio.sleep(seconds)
        if not self._response_triggered and not self._call_ended:
            logger.info(
                f"[SESSION {self.session_id[:8]}] No speech after {seconds}s "
                f"— assuming voicemail"
            )
            await self.handle_voicemail_detected(reason="silence_timeout")

    # ------------------------------------------------------------------
    # Fallback hangup
    # ------------------------------------------------------------------

    async def _fallback_hangup(self, seconds: int):
        """
        Hard ceiling — hang up after `seconds` regardless of call state.
        Prevents zombie calls if something goes wrong.
        """
        await asyncio.sleep(seconds)
        if not self._call_ended:
            logger.warning(
                f"[SESSION {self.session_id[:8]}] Fallback hangup after {seconds}s"
            )
            self._call_ended = True
            await self._hangup_fn(0)

    # ------------------------------------------------------------------
    # Voicemail handler
    # Equivalent to V1 _handle_voicemail_detected
    # ------------------------------------------------------------------

    async def handle_voicemail_detected(self, reason: str = "keyword"):
        """Save voicemail result, play prerecorded PCM via play_voicemail_fn, then hang up."""
        if self._call_ended:
            return
        self._call_ended = True

        logger.info(
            f"[SESSION {self.session_id[:8]}] Voicemail detected ({reason}) | "
            f"unique_id={self.unique_id}"
        )

        result = {
            "unique_id":            self.unique_id,
            "session_id":           self.session_id,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "phone_status":         "sent_to_voicemail",
            "is_correct_number":    "unknown",
            "org_valid":            "unknown",
            "call_outcome":         "no_answer_voicemail",
            "call_summary": (
                f"Reached voicemail for {self.org_name}. "
                f"Detection reason: {reason}. Left prerecorded voicemail message."
            ),
            "other_numbers":        None,
            "services_confirmed":   "unknown",
            "available_services":   [],
            "unavailable_services": [],
            "other_services":       [],
            "mentioned_funding":    "no",
            "mentioned_callback":   "no",
            "dialed_number":        self.phone_number,
            "dialed_services":      self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

        asyncio.create_task(self._voicemail_prerecorded_then_hangup())

    async def _voicemail_prerecorded_then_hangup(self):
        cfg = AGENT_SETTINGS["call"]
        trailing = cfg.get("voicemail_trailing_silence_seconds", 3)

        if self._pipeline_task is not None:
            await self._pipeline_task.queue_frames([InterruptionFrame()])

        try:
            await self._play_voicemail_fn()
        except Exception as e:
            logger.error(
                f"[SESSION {self.session_id[:8]}] play_voicemail_fn failed: {e}"
            )

        await asyncio.sleep(trailing)
        await self._hangup_fn(0)

    # ------------------------------------------------------------------
    # IVR handler
    # Equivalent to V1 _handle_ivr_detected
    # ------------------------------------------------------------------

    async def handle_ivr_detected(self, matched_keyword: str):
        """Save IVR result and hang up immediately — no Voice interaction needed."""
        if self._call_ended:
            return
        self._call_ended = True

        logger.info(
            f"[SESSION {self.session_id[:8]}] IVR/call centre detected | "
            f"keyword='{matched_keyword}' | unique_id={self.unique_id}"
        )

        result = {
            "unique_id":            self.unique_id,
            "session_id":           self.session_id,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "phone_status":         "invalid",
            "is_correct_number":    "unknown",
            "org_valid":            "unknown",
            "call_outcome":         "other",
            "call_summary": (
                f"Reached IVR or call centre for {self.org_name}. "
                f"Detected keyword: '{matched_keyword}'. Hung up without verifying."
            ),
            "other_numbers":        None,
            "services_confirmed":   "unknown",
            "available_services":   [],
            "unavailable_services": [],
            "other_services":       [],
            "mentioned_funding":    "no",
            "mentioned_callback":   "no",
            "dialed_number":        self.phone_number,
            "dialed_services":      self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

        # IVR: hang up immediately (V1: 1s delay)
        asyncio.create_task(self._hangup_fn(1))

    # ------------------------------------------------------------------
    # extract_call_details handler
    # Equivalent to V1 _handle_extract_call_details
    # ------------------------------------------------------------------

    async def handle_extract_call_details(self, args: dict) -> dict:
        """
        Called when GPT-4o invokes the extract_call_details tool.
        Fills defaults, saves JSON, triggers goodbye, schedules hangup.
        Returns the result_callback payload for Pipecat.
        """
        if self._call_ended:
            logger.info(
                f"[SESSION {self.session_id[:8]}] extract_call_details already captured — skipping"
            )
            return {"status": "already_captured"}

        self._call_ended   = True
        self._goodbye_done = True

        # Fill defaults for any missing fields (exact same logic as V1)
        args.setdefault("unique_id",            self.unique_id)
        args.setdefault("phone_status",         "unknown")
        args.setdefault("is_correct_number",    "unknown")
        args.setdefault("org_valid",            "unknown")
        args.setdefault("call_outcome",         "other")
        args.setdefault("call_summary",         "")
        args.setdefault("other_numbers",        None)
        args.setdefault("services_confirmed",   "unknown")
        args.setdefault("available_services",   [])
        args.setdefault("unavailable_services", [])
        args.setdefault("other_services",       [])
        args.setdefault("mentioned_funding",    "no")
        args.setdefault("mentioned_callback",   "no")

        args["session_id"]      = self.session_id
        args["timestamp"]       = datetime.now(timezone.utc).isoformat()
        args["dialed_number"]   = self.phone_number
        args["dialed_services"] = self.services_list

        self._save_result(args)
        self._maybe_save_incremental(force=True)

        # Cancel fallback timer now that we have a clean result
        self.cancel_timers()

        # Schedule hangup after Samantha finishes her goodbye (V1: 10s)
        cfg = AGENT_SETTINGS["call"]
        asyncio.create_task(self._hangup_fn(cfg["hangup_delay_seconds"]))

        # Return "captured" so GPT-4o knows to say goodbye
        return {"status": "captured"}

    # ------------------------------------------------------------------
    # Disconnect handler — partial result capture
    # Equivalent to V1 _handle_call_disconnected
    # ------------------------------------------------------------------

    async def handle_call_disconnected(self):
        """
        Called when the ACS WebSocket closes before extract_call_details fired.
        Saves whatever partial state was captured.
        Equivalent to V1 _handle_call_disconnected.
        """
        if self._call_ended:
            return
        self._call_ended = True

        self.cancel_timers()

        logger.info(
            f"[SESSION {self.session_id[:8]}] Caller disconnected mid-call | "
            f"unique_id={self.unique_id}"
        )

        # Try to recover partial tool args (equivalent to V1 _fn_args_buf recovery)
        partial = self._pending_tool_args

        result = {
            "unique_id":            partial.get("unique_id",           self.unique_id),
            "session_id":           self.session_id,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "phone_status":         partial.get("phone_status",        "unknown"),
            "is_correct_number":    partial.get("is_correct_number",   "unknown"),
            "org_valid":            partial.get("org_valid",           "unknown"),
            "call_outcome":         "call_disconnected",   # always override
            "call_summary":         partial.get(
                                        "call_summary",
                                        f"Caller disconnected mid-call for {self.org_name}. "
                                        f"Partial data captured where available."
                                    ),
            "other_numbers":        partial.get("other_numbers",       None),
            "services_confirmed":   partial.get("services_confirmed",  "unknown"),
            "available_services":   partial.get("available_services",  []),
            "unavailable_services": partial.get("unavailable_services",[]),
            "other_services":       partial.get("other_services",      []),
            "mentioned_funding":    partial.get("mentioned_funding",   "no"),
            "mentioned_callback":   partial.get("mentioned_callback",  "no"),
            "partial_capture":      True,
            "dialed_number":        self.phone_number,
            "dialed_services":      self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

    # ------------------------------------------------------------------
    # Internal: save result JSON to disk
    # ------------------------------------------------------------------

    def _save_result(self, result: dict):
        # Issue 2 fix: unique_id may be "" for inbound calls or edge cases.
        # Always fall back to a generated UUID so the filename is never
        # ".json" and two calls never silently overwrite each other.
        raw_uid = result.get("unique_id") or self.unique_id or ""
        uid = raw_uid.strip() if raw_uid.strip() else str(uuid.uuid4())

        logger.info(
            f"[CALL RESULT] unique_id={result.get('unique_id')} | "
            f"org_valid={result.get('org_valid')} | "
            f"phone_status={result.get('phone_status')} | "
            f"is_correct_number={result.get('is_correct_number')} | "
            f"services_confirmed={result.get('services_confirmed')} | "
            f"outcome={result.get('call_outcome')} | "
            f"other_numbers={result.get('other_numbers')} | "
            f"available={result.get('available_services')} | "
            f"unavailable={result.get('unavailable_services')} | "
            f"other_services={result.get('other_services')} | "
            f"summary={result.get('call_summary')}"
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        result_file = self.results_dir / f"{uid}.json"
        try:
            result_file.write_text(json.dumps(result, indent=2))
            logger.info(f"Result saved → {result_file}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")