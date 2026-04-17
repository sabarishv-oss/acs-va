"""
Post-hangup extraction: when the caller disconnects before extract_call_details runs,
call OpenAI once in a fresh context with the full transcript and produce the same
field bundle as the live tool.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.agent_settings import get_agent_settings

_MAX_TRANSCRIPT_CHARS = 120_000

_SYSTEM = """You are a call-quality analyst. The phone assistant (Samantha) was verifying contact information for an organization when the caller hung up before the system could save structured fields.

You will receive:
- Call metadata: organization name being verified, dialed E.164 phone number, services list, unique_id (must be echoed exactly in output).
- A full transcript with lines like: TIMESTAMP | SPEAKER: text (SPEAKER is CALLER or SAMANTHA).

Task: Output ONE JSON object only (no markdown) with exactly these keys and value types:
- phone_status: string, one of: "valid", "invalid", "sent_to_voicemail", "not_verified"
- is_correct_number: string, one of: "yes", "no", "unknown"
- org_valid: string, one of: "correct_org", "incorrect_org", "unknown"
- call_outcome: string, one of: "confirmed_correct", "provided_alternative", "not_org_wrong_number", "no_answer_voicemail", "call_disconnected", "refused", "busy_callback_requested", "other"
- call_summary: string, 1-3 sentences factual summary of what happened
- unique_id: string (must equal the provided unique_id exactly)
- other_numbers: JSON array of strings, or null if none mentioned
- services_confirmed: string, one of: "yes", "no", "partially", "unknown"
- available_services: JSON array of strings
- unavailable_services: JSON array of strings
- other_services: JSON array of strings
- mentioned_funding: string, "yes" or "no"
- mentioned_callback: string, "yes" or "no"

Rules:
- Base conclusions only on the transcript; do not invent facts.
- If the caller disconnected mid-conversation and verification was incomplete, prefer call_outcome "call_disconnected" and phone_status "not_verified" unless the transcript clearly supports another outcome (e.g. voicemail, wrong org).
- If you cannot tell, use "unknown" for enum fields that allow it.
- Arrays may be empty. other_numbers may be null.
"""


def _truncate_transcript(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_TRANSCRIPT_CHARS:
        return text
    omitted = len(text) - _MAX_TRANSCRIPT_CHARS
    return (
        f"[Transcript truncated: omitted first {omitted} characters]\n\n"
        + text[-_MAX_TRANSCRIPT_CHARS:]
    )


def _normalize_result(raw: dict[str, Any], unique_id: str) -> dict[str, Any]:
    """Ensure required keys exist with safe defaults."""
    out: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    out["unique_id"] = (out.get("unique_id") or unique_id or "").strip() or unique_id

    def _s(key: str, default: str) -> str:
        v = out.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return default

    def _arr(key: str) -> list:
        v = out.get(key)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    out["phone_status"] = _s("phone_status", "not_verified")
    out["is_correct_number"] = _s("is_correct_number", "unknown")
    out["org_valid"] = _s("org_valid", "unknown")
    out["call_outcome"] = _s("call_outcome", "call_disconnected")
    out["call_summary"] = _s("call_summary", "Caller disconnected mid-call.")
    on = out.get("other_numbers")
    if on is not None and not isinstance(on, list):
        on = None
    out["other_numbers"] = on if on is None else [str(x).strip() for x in on if str(x).strip()]
    out["services_confirmed"] = _s("services_confirmed", "unknown")
    out["available_services"] = _arr("available_services")
    out["unavailable_services"] = _arr("unavailable_services")
    out["other_services"] = _arr("other_services")
    mf = out.get("mentioned_funding")
    out["mentioned_funding"] = mf if mf in ("yes", "no") else "no"
    mc = out.get("mentioned_callback")
    out["mentioned_callback"] = mc if mc in ("yes", "no") else "no"

    return out


async def extract_disconnect_fields_from_transcript(
    *,
    transcript_text: str,
    org_name: str,
    phone_number: str,
    services_list: str,
    unique_id: str,
    timeout_s: float = 60.0,
) -> dict[str, Any] | None:
    """
    Call OpenAI with a fresh context. Returns normalized field dict, or None on failure/empty transcript.
    """
    text = (transcript_text or "").strip()
    if not text:
        logger.info("disconnect_llm_extract: empty transcript, skipping")
        return None

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("disconnect_llm_extract: OPENAI_API_KEY missing, skipping")
        return None

    model = get_agent_settings().get("llm", {}).get("model", "gpt-4o")
    body = _truncate_transcript(text)
    user_msg = (
        f"unique_id (echo in JSON): {unique_id}\n"
        f"organization_name: {org_name}\n"
        f"dialed_number: {phone_number}\n"
        f"services_list: {services_list}\n\n"
        f"--- TRANSCRIPT ---\n{body}"
    )

    client = AsyncOpenAI(api_key=api_key)
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("disconnect_llm_extract: OpenAI call timed out")
        return None
    except Exception as e:
        logger.warning(f"disconnect_llm_extract: OpenAI error: {e}")
        return None

    raw_content = (resp.choices[0].message.content or "").strip()
    if not raw_content:
        logger.warning("disconnect_llm_extract: empty model response")
        return None

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.warning(f"disconnect_llm_extract: JSON parse failed: {e}")
        return None

    if not isinstance(parsed, dict):
        return None

    return _normalize_result(parsed, unique_id)
