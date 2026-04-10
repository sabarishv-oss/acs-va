"""
call_session.py
---------------
CallSession manages all per-call state and behavioural logic
that lived in V1's CommunicationHandler, now adapted for Pipecat.

This version adds an interruption-aware intro state machine:
  - the greeting is split into short deterministic chunks
  - if the caller interrupts, audio is stopped immediately
  - the LLM receives hidden context describing which intro facts/chunks were
    already completed and which still remain
  - the LLM then continues naturally from the remaining intro content instead
    of restarting from "Hi, this is Samantha..."
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from loguru import logger
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, InterruptionFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext

from app.agent_settings import (
    get_agent_settings,
    VOICEMAIL_KEYWORDS,
    IVR_KEYWORDS,
)
from app import ui_events


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

    INTRO_SECONDS_PER_WORD = 0.34
    INTRO_SECONDS_BUFFER = 0.45

    def __init__(
        self,
        org_name: str,
        phone_number: str,
        services_list: str,
        unique_id: str,
        session_id: str,
        results_dir: Path,
        transcripts_dir: Path,
        hangup_fn,
        play_voicemail_fn,
    ):
        self.org_name = org_name
        self.phone_number = phone_number
        self.phone_for_speech = _format_phone_for_speech(phone_number)
        self.services_list = services_list
        self.unique_id = unique_id
        self.session_id = session_id
        self.results_dir = results_dir
        self.transcripts_dir = transcripts_dir
        self._hangup_fn = hangup_fn
        self._play_voicemail_fn = play_voicemail_fn

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

        self._response_triggered: bool = False
        self._opening_spoken: bool = False
        self._human_confirmed: bool = False
        self._call_ended: bool = False
        self._goodbye_done: bool = False

        self._pipeline_task: PipelineTask | None = None
        self._llm_context: LLMContext | None = None

        self._last_transcript_time: float = 0.0
        self._drop_next_transcript_from_pipeline: bool = False

        self._silence_timeout_task: asyncio.Task | None = None
        self._fallback_hangup_task: asyncio.Task | None = None
        self._goodbye_safety_scheduled: bool = False
        self._pending_tool_args: dict = {}

        # Intro state machine
        self._intro_task: asyncio.Task | None = None
        self._intro_state = self._create_intro_state()

    # ------------------------------------------------------------------
    # Intro state machine
    # ------------------------------------------------------------------

    def _create_intro_state(self) -> dict[str, Any]:
        # Short spoken phrases (comma / sentence boundaries). The chunk that
        # completes a logical fact still carries the fact marker for LLM context.
        chunks = [
            {"key": "intro_hi",        "text": "Hi,", "fact": None},
            {"key": "intro_identity",  "text": "This is Samantha,", "fact": None},
            {
                "key": "intro_ggh",
                "text": "Calling on behalf of GroundGame dot Health,",
                "fact": "identity_and_org",
            },
            {
                "key": "intro_warmth",
                "text": "I hope you're doing well today,",
                "fact": "warm_opening",
            },
            {
                "key": "intro_recording",
                "text": (
                    "Before we get started, I'd like to let you know that this call may be recorded "
                    "for quality assurance and training purposes,"
                ),
                "fact": "recording_notice",
            },
            {
                "key": "intro_org_q",
                "text": (
                    f"With that, May I please confirm that I am speaking with {self.org_name}?"
                ),
                "fact": "org_question_asked",
            },
        ]
        for chunk in chunks:
            chunk["estimated_seconds"] = self._estimate_chunk_seconds(chunk["text"])

        return {
            "active": False,
            "completed": False,
            "interrupted": False,
            "current_index": -1,
            "completed_chunks": [],
            "remaining_chunks": [c["key"] for c in chunks],
            "facts_completed": [],
            "chunks": chunks,
        }

    def _estimate_chunk_seconds(self, text: str) -> float:
        words = max(1, len(text.split()))
        return (words * self.INTRO_SECONDS_PER_WORD) + self.INTRO_SECONDS_BUFFER

    def _cancel_intro_task(self) -> None:
        if self._intro_task and not self._intro_task.done():
            self._intro_task.cancel()
        self._intro_task = None

    async def _play_intro_chunks(self) -> None:
        if self._pipeline_task is None:
            return

        self._intro_state["active"] = True
        self._intro_state["interrupted"] = False

        for idx, chunk in enumerate(self._intro_state["chunks"]):
            if self._call_ended or self._intro_state["interrupted"]:
                break

            self._intro_state["current_index"] = idx
            await self._pipeline_task.queue_frames([TTSSpeakFrame(text=chunk["text"])])
            logger.info(
                f"[SESSION {self.session_id[:8]}] Intro chunk queued | "
                f"idx={idx} | key={chunk['key']} | text={chunk['text']}"
            )
            ui_events.emit(
                "opening_chunk_queued",
                session_id=self.session_id,
                chunk_key=chunk["key"],
                text=chunk["text"],
            )

            try:
                await asyncio.sleep(chunk["estimated_seconds"])
            except asyncio.CancelledError:
                raise

            if self._call_ended or self._intro_state["interrupted"]:
                break

            self._mark_intro_chunk_completed(idx)

        self._intro_state["active"] = False
        if not self._intro_state["interrupted"] and len(self._intro_state["completed_chunks"]) == len(self._intro_state["chunks"]):
            self._intro_state["completed"] = True
            logger.info(f"[SESSION {self.session_id[:8]}] Intro completed without interruption")

    def _mark_intro_chunk_completed(self, idx: int) -> None:
        chunk = self._intro_state["chunks"][idx]
        if chunk["key"] not in self._intro_state["completed_chunks"]:
            self._intro_state["completed_chunks"].append(chunk["key"])
        if chunk["fact"] and chunk["fact"] not in self._intro_state["facts_completed"]:
            self._intro_state["facts_completed"].append(chunk["fact"])
        self._intro_state["remaining_chunks"] = [
            c["key"] for c in self._intro_state["chunks"]
            if c["key"] not in self._intro_state["completed_chunks"]
        ]
        self._last_samantha_text = chunk["text"]
        self._append_transcript_line("SAMANTHA", chunk["text"])
        self._maybe_save_incremental()

    def _remove_call_context_messages(self) -> None:
        if self._llm_context is None:
            return
        self._llm_context.messages[:] = [
            m for m in self._llm_context.messages
            if not (
                m.get("role") == "system"
                and "[CALL CONTEXT — do not read aloud]" in m.get("content", "")
            )
        ]

    def _build_intro_runtime_note(self) -> str:
        completed_chunks = self._intro_state["completed_chunks"]
        remaining_chunks = self._intro_state["remaining_chunks"]
        completed_text = [
            c["text"] for c in self._intro_state["chunks"]
            if c["key"] in completed_chunks
        ]
        remaining_text = [
            c["text"] for c in self._intro_state["chunks"]
            if c["key"] in remaining_chunks
        ]

        if self._intro_state["completed"]:
            note = (
                f"The deterministic opening finished before the caller spoke. "
                f"Opening facts already covered: {', '.join(self._intro_state['facts_completed']) or 'none'}. "
                f"Exact completed intro text: {completed_text}. "
                f"Do NOT restart the intro. Treat the caller as responding to the org-name question. "
                f"If they confirm the org, your next question is: 'And is {self.phone_for_speech} the best number to reach you?' "
                f"If their answer is ambiguous, clarify naturally without replaying the intro."
            )
        else:
            # Intro did not finish (typically: caller interrupted). Split fully delivered,
            # in-progress (partially spoken — not in completed_chunks), and not-yet-started
            # so the LLM does not ignore the phrase that was cut off mid-utterance.
            if self._intro_state["interrupted"]:
                chunks = self._intro_state["chunks"]
                done = set(completed_chunks)
                cur_i = self._intro_state["current_index"]
                in_prog = None
                if 0 <= cur_i < len(chunks) and chunks[cur_i]["key"] not in done:
                    in_prog = chunks[cur_i]
                if 0 <= cur_i < len(chunks):
                    not_started_texts = [chunks[i]["text"] for i in range(cur_i + 1, len(chunks))]
                else:
                    not_started_texts = list(remaining_text)

                if in_prog:
                    fact = in_prog.get("fact")
                    fact_part = f" Fact tied to this phrase: {fact!r}." if fact else ""
                    in_progress_line = (
                        f"IN-PROGRESS when interrupted (likely only partly heard — continue naturally from "
                        f"here; do NOT restart this phrase from the beginning): key={in_prog['key']!r} "
                        f"text={in_prog['text']!r}.{fact_part}"
                    )
                else:
                    in_progress_line = (
                        "IN-PROGRESS chunk: none (interruption between chunks or state edge case)."
                    )

                note = (
                    f"The deterministic opening was interrupted by the caller. "
                    f"Fully completed chunk keys (do not repeat): {completed_chunks or ['none']}. "
                    f"Exact text fully delivered: {completed_text or ['none']}. "
                    f"Completed intro facts: {self._intro_state['facts_completed'] or ['none']}. "
                    f"{in_progress_line} "
                    f"Phrases not yet started (still to cover if relevant): {not_started_texts or ['none']}. "
                    f"Continue from the in-progress and/or not-yet-started content; do not restart from 'Hi'. "
                    f"Do NOT repeat already completed chunks. "
                    f"If the org question chunk was not completed, you still need to ask the org question once. "
                    f"If the org question chunk was already completed, do NOT ask it again; respond to the caller's answer and then ask: 'And is {self.phone_for_speech} the best number to reach you?'"
                )
            else:
                note = (
                    f"Intro did not complete (unexpected state: interrupted flag false). "
                    f"Completed chunks: {completed_chunks or ['none']}. "
                    f"Remaining chunk keys: {remaining_chunks or ['none']}. "
                    f"Remaining text: {remaining_text or ['none']}."
                )
        return note

    def _inject_call_context(self) -> None:
        if self._llm_context is None:
            return
        self._remove_call_context_messages()
        self._llm_context.messages.append({
            "role": "system",
            "content": (
                f"[CALL CONTEXT — do not read aloud]\n"
                f"Organization to verify: {self.org_name}\n"
                f"Phone number dialed (say as): {self.phone_for_speech}\n"
                f"Services to verify: {self.services_list}\n"
                f"Unique ID for this call (never say aloud): {self.unique_id}\n\n"
                f"{self._build_intro_runtime_note()}"
            ),
        })

    async def _handoff_to_llm_after_interrupt(self, caller_text: str) -> None:
        if self._llm_context is None or self._pipeline_task is None:
            return

        self._inject_call_context()
        self._llm_context.messages.append({"role": "user", "content": caller_text})
        self._drop_next_transcript_from_pipeline = True
        await self._pipeline_task.queue_frames([InterruptionFrame()])
        await self._pipeline_task.queue_frames([LLMRunFrame()])

    # ------------------------------------------------------------------
    # Generic session helpers
    # ------------------------------------------------------------------

    @property
    def call_ended(self) -> bool:
        return self._call_ended

    def should_drop_transcript_from_pipeline(self) -> bool:
        if self._drop_next_transcript_from_pipeline:
            self._drop_next_transcript_from_pipeline = False
            return True
        return False

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
            "intro_state": {
                "active": self._intro_state["active"],
                "completed": self._intro_state["completed"],
                "interrupted": self._intro_state["interrupted"],
                "current_index": self._intro_state["current_index"],
                "completed_chunks": list(self._intro_state["completed_chunks"]),
                "remaining_chunks": list(self._intro_state["remaining_chunks"]),
                "facts_completed": list(self._intro_state["facts_completed"]),
            },
            "partial_capture": True,
        }

    def _maybe_save_incremental(self, *, force: bool = False) -> None:
        if self._call_ended and not force:
            return
        now = time.monotonic()
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

    _GOODBYE_PATTERNS = (
        "have a great day",
        "have a good day",
        "goodbye",
        "good bye",
        "take care",
        "thanks for your time",
        "thank you for your time",
    )
    GOODBYE_SAFETY_DELAY_S = 8

    def on_samantha_text(self, text: str) -> None:
        if text and not self._call_ended:
            self._last_samantha_text = text.strip()
            self._append_transcript_line("SAMANTHA", text)
            self._maybe_save_incremental()

            latency_s = None
            if self._last_transcript_time > 0:
                latency_s = round(time.time() - self._last_transcript_time, 2)
                self._last_transcript_time = 0.0
            ui_events.emit("samantha_text", session_id=self.session_id, text=text.strip(), latency_s=latency_s)

            if not self._goodbye_safety_scheduled:
                lowered = self._last_samantha_text.lower()
                if any(p in lowered for p in self._GOODBYE_PATTERNS):
                    self._goodbye_safety_scheduled = True
                    asyncio.create_task(self._goodbye_safety_hangup())

    async def _goodbye_safety_hangup(self) -> None:
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
        self._cancel_intro_task()

        pa = self._pending_tool_args
        result = {
            "unique_id": self.unique_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_status": pa.get("phone_status", "unknown"),
            "is_correct_number": pa.get("is_correct_number", "unknown"),
            "org_valid": pa.get("org_valid", "unknown"),
            "call_outcome": pa.get("call_outcome", "other"),
            "call_summary": (
                pa.get("call_summary") or
                f"Call with {self.org_name} ended without extract_call_details. "
                f"Last caller: '{self._last_caller_text}'. "
                f"Last agent: '{self._last_samantha_text}'."
            ),
            "other_numbers": pa.get("other_numbers"),
            "services_confirmed": pa.get("services_confirmed", "unknown"),
            "available_services": pa.get("available_services", []),
            "unavailable_services": pa.get("unavailable_services", []),
            "other_services": pa.get("other_services", []),
            "mentioned_funding": pa.get("mentioned_funding", "no"),
            "mentioned_callback": pa.get("mentioned_callback", "no"),
            "dialed_number": self.phone_number,
            "dialed_services": self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)
        await self._hangup_fn(0)

    def attach_pipeline(self, task: PipelineTask, context: LLMContext):
        self._pipeline_task = task
        self._llm_context = context

    def update_pending_tool_args(self, args: dict):
        self._pending_tool_args = dict(args)
        self._maybe_save_incremental(force=True)

    def start_timers(self):
        cfg = get_agent_settings()["call"]
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
        for task in (self._silence_timeout_task, self._fallback_hangup_task):
            if task and not task.done():
                task.cancel()

    async def trigger_opening(self):
        if self._response_triggered or self._call_ended:
            return
        self._response_triggered = True

        if self._call_ended:
            return

        self._opening_spoken = True
        self._intro_task = asyncio.create_task(self._play_intro_chunks())
        logger.info(f"[SESSION {self.session_id[:8]}] Intro state machine started")
        ui_events.emit(
            "opening_queued",
            session_id=self.session_id,
            text=" ".join(chunk["text"] for chunk in self._intro_state["chunks"]),
        )

    async def on_transcript(self, text: str):
        if self._call_ended:
            return

        logger.info(f"[CALLER]: {text}")
        self._last_caller_text = text.strip()
        self._last_transcript_time = time.time()
        self._append_transcript_line("CALLER", text)
        self._maybe_save_incremental()
        ui_events.emit("caller_transcript", session_id=self.session_id, text=text.strip())

        if not self._human_confirmed:
            vm_match = _contains_keyword(text, VOICEMAIL_KEYWORDS)
            if vm_match:
                logger.info(f"[SESSION {self.session_id[:8]}] Voicemail keyword: '{vm_match}'")
                await self.handle_voicemail_detected(reason=f"keyword: {vm_match}")
                return

            if not self._call_ended:
                ivr_match = _contains_keyword(text, IVR_KEYWORDS)
                if ivr_match:
                    logger.info(f"[SESSION {self.session_id[:8]}] IVR keyword: '{ivr_match}'")
                    await self.handle_ivr_detected(matched_keyword=ivr_match)
                    return

        first_human_turn = not self._human_confirmed and not self._call_ended
        if first_human_turn:
            self._human_confirmed = True
            logger.info(
                f"[SESSION {self.session_id[:8]}] Human confirmed — "
                f"keyword detection disabled for rest of call"
            )

        if self._opening_spoken and not self._call_ended:
            intro_active = self._intro_state["active"] and not self._intro_state["completed"]
            if intro_active:
                _chunks = self._intro_state["chunks"]
                _done_keys = set(self._intro_state["completed_chunks"])
                read_texts = [c["text"] for c in _chunks if c["key"] in _done_keys]
                pending_texts = [c["text"] for c in _chunks if c["key"] not in _done_keys]
                _cur = self._intro_state["current_index"]
                _cur_key = _chunks[_cur]["key"] if 0 <= _cur < len(_chunks) else "?"
                logger.info(
                    f"[SESSION {self.session_id[:8]}] Caller interrupted intro | "
                    f"caller_stt={text!r} | "
                    f"current_chunk_idx={_cur} key={_cur_key} | "
                    f"read ({len(read_texts)} parts): {' | '.join(read_texts) or '(none yet)'} | "
                    f"pending ({len(pending_texts)} parts): {' | '.join(pending_texts) or '(none)'}"
                )
                self._intro_state["interrupted"] = True
                self._cancel_intro_task()
                await self._handoff_to_llm_after_interrupt(text)
                return

            if self._intro_state["completed"] and first_human_turn:
                self._inject_call_context()

    async def _voicemail_silence_timeout(self, seconds: int):
        await asyncio.sleep(seconds)
        if not self._response_triggered and not self._call_ended:
            logger.info(
                f"[SESSION {self.session_id[:8]}] No speech after {seconds}s "
                f"— assuming voicemail"
            )
            await self.handle_voicemail_detected(reason="silence_timeout")

    async def _fallback_hangup(self, seconds: int):
        await asyncio.sleep(seconds)
        if not self._call_ended:
            logger.warning(f"[SESSION {self.session_id[:8]}] Fallback hangup after {seconds}s")
            self._call_ended = True
            self._cancel_intro_task()
            await self._hangup_fn(0)

    async def handle_voicemail_detected(self, reason: str = "keyword"):
        if self._call_ended:
            return
        self._call_ended = True
        self._cancel_intro_task()

        logger.info(
            f"[SESSION {self.session_id[:8]}] Voicemail detected ({reason}) | "
            f"unique_id={self.unique_id}"
        )
        ui_events.emit("voicemail_detected", session_id=self.session_id, reason=reason)

        result = {
            "unique_id": self.unique_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_status": "sent_to_voicemail",
            "is_correct_number": "unknown",
            "org_valid": "unknown",
            "call_outcome": "no_answer_voicemail",
            "call_summary": (
                f"Reached voicemail for {self.org_name}. "
                f"Detection reason: {reason}. Left prerecorded voicemail message."
            ),
            "other_numbers": None,
            "services_confirmed": "unknown",
            "available_services": [],
            "unavailable_services": [],
            "other_services": [],
            "mentioned_funding": "no",
            "mentioned_callback": "no",
            "dialed_number": self.phone_number,
            "dialed_services": self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

        asyncio.create_task(self._voicemail_prerecorded_then_hangup())

    async def _voicemail_prerecorded_then_hangup(self):
        cfg = get_agent_settings()["call"]
        trailing = cfg.get("voicemail_trailing_silence_seconds", 3)
        start_delay = float(cfg.get("voicemail_recording_start_delay_seconds", 4.0))

        if self._pipeline_task is not None:
            await self._pipeline_task.queue_frames([InterruptionFrame()])

        try:
            self._append_transcript_line(
                "SYSTEM",
                "Voicemail detected — playing prerecorded voicemail message.",
            )
            if start_delay > 0:
                await asyncio.sleep(start_delay)
            await self._play_voicemail_fn()
            self._append_transcript_line(
                "SYSTEM",
                "Finished playing prerecorded voicemail message.",
            )
        except Exception as e:
            logger.error(f"[SESSION {self.session_id[:8]}] play_voicemail_fn failed: {e}")
            self._append_transcript_line(
                "SYSTEM",
                f"Failed to play prerecorded voicemail message: {e}",
            )

        await asyncio.sleep(trailing)
        await self._hangup_fn(0)

    async def handle_ivr_detected(self, matched_keyword: str):
        if self._call_ended:
            return
        self._call_ended = True
        self._cancel_intro_task()

        logger.info(
            f"[SESSION {self.session_id[:8]}] IVR/call centre detected | "
            f"keyword='{matched_keyword}' | unique_id={self.unique_id}"
        )
        ui_events.emit("ivr_detected", session_id=self.session_id, keyword=matched_keyword)

        result = {
            "unique_id": self.unique_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_status": "invalid",
            "is_correct_number": "unknown",
            "org_valid": "unknown",
            "call_outcome": "other",
            "call_summary": (
                f"Reached IVR or call centre for {self.org_name}. "
                f"Detected keyword: '{matched_keyword}'. Hung up without verifying."
            ),
            "other_numbers": None,
            "services_confirmed": "unknown",
            "available_services": [],
            "unavailable_services": [],
            "other_services": [],
            "mentioned_funding": "no",
            "mentioned_callback": "no",
            "dialed_number": self.phone_number,
            "dialed_services": self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

        asyncio.create_task(self._hangup_fn(1))

    async def handle_extract_call_details(self, args: dict) -> dict:
        if self._call_ended:
            logger.info(f"[SESSION {self.session_id[:8]}] extract_call_details already captured — skipping")
            return {"status": "already_captured"}

        self._call_ended = True
        self._goodbye_done = True
        self._cancel_intro_task()

        args.setdefault("unique_id", self.unique_id)
        args.setdefault("phone_status", "unknown")
        args.setdefault("is_correct_number", "unknown")
        args.setdefault("org_valid", "unknown")
        args.setdefault("call_outcome", "other")
        args.setdefault("call_summary", "")
        args.setdefault("other_numbers", None)
        args.setdefault("services_confirmed", "unknown")
        args.setdefault("available_services", [])
        args.setdefault("unavailable_services", [])
        args.setdefault("other_services", [])
        args.setdefault("mentioned_funding", "no")
        args.setdefault("mentioned_callback", "no")

        args["session_id"] = self.session_id
        args["timestamp"] = datetime.now(timezone.utc).isoformat()
        args["dialed_number"] = self.phone_number
        args["dialed_services"] = self.services_list

        self._save_result(args)
        self._maybe_save_incremental(force=True)

        ui_events.emit(
            "call_result",
            session_id=self.session_id,
            outcome=args.get("call_outcome"),
            summary=args.get("call_summary", ""),
            org_valid=args.get("org_valid"),
            phone_status=args.get("phone_status"),
        )

        self.cancel_timers()

        cfg = get_agent_settings()["call"]
        asyncio.create_task(self._hangup_fn(cfg["hangup_delay_seconds"]))
        return {"status": "captured"}

    async def handle_call_disconnected(self):
        if self._call_ended:
            return
        self._call_ended = True
        self._cancel_intro_task()

        self.cancel_timers()

        logger.info(
            f"[SESSION {self.session_id[:8]}] Caller disconnected mid-call | "
            f"unique_id={self.unique_id}"
        )
        ui_events.emit("call_disconnected", session_id=self.session_id)

        partial = self._pending_tool_args

        result = {
            "unique_id": partial.get("unique_id", self.unique_id),
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_status": partial.get("phone_status", "unknown"),
            "is_correct_number": partial.get("is_correct_number", "unknown"),
            "org_valid": partial.get("org_valid", "unknown"),
            "call_outcome": "call_disconnected",
            "call_summary": partial.get(
                "call_summary",
                f"Caller disconnected mid-call for {self.org_name}. Partial data captured where available.",
            ),
            "other_numbers": partial.get("other_numbers", None),
            "services_confirmed": partial.get("services_confirmed", "unknown"),
            "available_services": partial.get("available_services", []),
            "unavailable_services": partial.get("unavailable_services", []),
            "other_services": partial.get("other_services", []),
            "mentioned_funding": partial.get("mentioned_funding", "no"),
            "mentioned_callback": partial.get("mentioned_callback", "no"),
            "partial_capture": True,
            "dialed_number": self.phone_number,
            "dialed_services": self.services_list,
        }
        self._save_result(result)
        self._maybe_save_incremental(force=True)

    def _save_result(self, result: dict):
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
        result_file = self.results_dir / f"{uid}_{self.session_id[:8]}.json"
        try:
            result_file.write_text(json.dumps(result, indent=2))
            logger.info(f"Result saved → {result_file}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")