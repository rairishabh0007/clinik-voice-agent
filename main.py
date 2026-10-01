"""Outbound healthcare voice agent worker.

Run modes:
    python main.py console                      talk over your microphone, no telephony
    python main.py dev                          connect to LiveKit and wait for dispatch
    python main.py download-files               fetch VAD / turn-detector weights

A call is triggered by dispatching a job with JSON metadata:
    {"patient_id": "P001", "phone_number": "+91...", "transfer_to": "+91..."}
See scripts/dispatch_call.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

from agent.patient_agent import PatientOutreachAgent
from agent.prompts import greeting
from analysis.post_call import analyse_call
from observability.opik_tracer import OpikCallTracer
from services import patients
from services.call_state import CallState
from services.model_config import llm_config

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("outbound-agent")

AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-caller")
DEFAULT_PATIENT = os.getenv("DEFAULT_PATIENT_ID", "P001")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram")
MAX_CALL_DURATION_S = int(os.getenv("MAX_CALL_DURATION_S", "600"))
RECORDING_FILENAME = "audio.ogg"

# SIP status -> the outcome we record. Anything unmapped is a trunk problem.
_SIP_OUTCOMES = {486: "rejected", 603: "rejected", 408: "no_answer", 480: "no_answer"}


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    # Opik's package graph is large; importing it lazily inside the job blocks the event loop
    # for ~0.75s on the first call. Pay that cost here instead.
    try:
        import opik  # noqa: F401
    except Exception:
        logger.debug("opik not importable at prewarm", exc_info=True)


def _build_tts() -> Any:
    if TTS_PROVIDER == "openai":
        return openai.TTS(voice=os.getenv("OPENAI_TTS_VOICE", "shimmer"))
    if TTS_PROVIDER == "cartesia":
        return cartesia.TTS(
            voice=os.getenv("CARTESIA_VOICE_ID", "794f9389-aac1-45b6-b726-9d9369183238")
        )
    return deepgram.TTS(model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en"))


def _build_llm() -> Any:
    """Gemini and OpenAI both run through the OpenAI plugin — Gemini via its compatible endpoint."""
    config = llm_config()
    if config.base_url:
        return openai.LLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    return openai.LLM(model=config.model, api_key=config.api_key or None)


def _parse_metadata(ctx: JobContext) -> dict[str, Any]:
    raw = (ctx.job.metadata or "").strip()
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        logger.error("job metadata is not valid JSON: %r", raw)
        return {}
    if not isinstance(data, dict):
        logger.error("job metadata is not a JSON object: %r", raw)
        return {}
    return data


WEB_ROOM_PREFIX = "web-"


def _patient_from_room(room_name: str) -> str | None:
    """A web-console room is named `web-P001-a1b2c3`, so a job without metadata can still find
    its patient from the first segment."""
    if room_name.startswith(WEB_ROOM_PREFIX):
        return room_name[len(WEB_ROOM_PREFIX):].split("-")[0] or None
    return None


async def _wait_for_recording(path: Path, timeout: float = 8.0) -> str | None:
    """The recorder finalises the file as the session closes, which can race our shutdown."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return str(path)
        await asyncio.sleep(0.25)
    logger.warning("recording not available at %s", path)
    return None


async def entrypoint(ctx: JobContext) -> None:
    dial_info = _parse_metadata(ctx)
    phone_number = dial_info.get("phone_number")
    patient_id = (
        dial_info.get("patient_id") or _patient_from_room(ctx.room.name) or DEFAULT_PATIENT
    )
    # The browser console brings the patient to us, so we wait for them rather than dialling.
    web_call = not phone_number and ctx.room.name.startswith(WEB_ROOM_PREFIX)

    try:
        patient = patients.get_patient(patient_id)
    except patients.PatientNotFound:
        logger.error("unknown patient %s — aborting job", patient_id)
        ctx.shutdown()
        return

    state = CallState(patient=patient, room_name=ctx.room.name)
    variables = patients.build_call_variables(patient)
    variables["transfer_to"] = dial_info.get("transfer_to") or os.getenv("TRANSFER_TO")

    ctx.log_context_fields = {"room": ctx.room.name, "patient": patient.id}

    stt_model = os.getenv("DEEPGRAM_MODEL", "nova-3")
    session = AgentSession(
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(),
        stt=deepgram.STT(model=stt_model, language="en"),
        llm=_build_llm(),
        tts=_build_tts(),
        turn_detection=EnglishModel(),
        user_away_timeout=float(os.getenv("USER_AWAY_TIMEOUT", "20")),
    )

    recording_path = ctx.session_directory / RECORDING_FILENAME

    async def build_summary(transcript: list[dict[str, Any]]) -> dict[str, Any]:
        """Everything the Opik module needs, with no Opik knowledge leaking into the app."""
        state.ended_at = datetime.now(timezone.utc)
        state.recording_path = await _wait_for_recording(recording_path)
        result = await analyse_call(state, transcript)
        return {
            "outcome": result.analysis.outcome,
            "analysis": result.to_dict(),
            "call_record": state.to_dict(),
            "recording_path": state.recording_path,
            "analysis_model": result.model,
            "analysis_usage": result.usage,
        }

    model = llm_config()
    tracer = OpikCallTracer.from_env(
        call_id=ctx.room.name,
        variables=variables,
        metadata={
            "patient_id": patient.id,
            "care_program": patient.care_program,
            "ordering_clinician": patient.ordering_clinician,
            "llm_model": model.model,
            "llm_provider": model.provider,
            "stt_model": stt_model,
            "tts_provider": TTS_PROVIDER,
            "agent_name": AGENT_NAME,
            "mode": (
                "outbound_sip" if phone_number else "web_call" if web_call else "console"
            ),
        },
        tags=["voice", "outbound", "healthcare", f"patient:{patient.id}"],
    )
    tracer.attach(session, ctx, finalise=build_summary)

    @session.on("error")
    def _on_session_error(event: Any) -> None:
        state.errors.append(str(getattr(event, "error", event)))

    background: set[asyncio.Task[Any]] = set()

    def run_in_background(coro: Any) -> None:
        task = asyncio.create_task(coro)
        background.add(task)
        task.add_done_callback(background.discard)

    async def hang_up(reason: str) -> None:
        state.end_reason = reason
        logger.info("hanging up %s: %s", ctx.room.name, reason)
        try:
            await ctx.delete_room()
        except Exception:
            logger.exception("failed to delete room")

    # user_away_timeout only marks the caller "away"; it never ends the call. Check in once, and
    # hang up if the line stays silent, so a dead line cannot hold the agent open.
    silent_checks = 0

    @session.on("user_state_changed")
    def _on_user_state(event: Any) -> None:
        nonlocal silent_checks
        if event.new_state == "speaking":
            silent_checks = 0
        elif event.new_state == "away" and state.answered_at:
            silent_checks += 1
            if silent_checks == 1:
                session.say("Are you still there?")
            else:
                run_in_background(hang_up("no_response"))

    @ctx.room.on("participant_disconnected")
    def _on_disconnect(participant: rtc.RemoteParticipant) -> None:
        reason = getattr(participant, "disconnect_reason", None)
        state.disconnect_reason = str(reason) if reason is not None else "unknown"
        logger.info("participant %s left: %s", participant.identity, state.disconnect_reason)

    await ctx.connect()

    agent = PatientOutreachAgent(state=state, variables=variables)
    room_input = RoomInputOptions()
    if phone_number and os.getenv("LIVEKIT_URL", "").endswith("livekit.cloud"):
        try:
            from livekit.plugins import noise_cancellation

            room_input = RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony())
        except Exception:
            logger.info("Krisp noise cancellation unavailable — continuing without it")

    session_task = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            record={"audio": True},
            room_input_options=room_input,
        )
    )

    async def abort(dial_error: str) -> None:
        state.dial_error = dial_error
        session_task.cancel()
        await tracer.finalise()
        ctx.shutdown()

    if phone_number:
        trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
        if not trunk_id:
            logger.error("SIP_OUTBOUND_TRUNK_ID is not set — cannot place an outbound call")
            return await abort("sip_failure")

        logger.info("dialling %s for patient %s", phone_number, patient.id)
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=f"patient-{patient.id}",
                    participant_name=patient.name,
                    wait_until_answered=True,
                    krisp_enabled=True,
                    max_call_duration=_duration(MAX_CALL_DURATION_S),
                )
            )
        except api.SipCallError as exc:
            status = getattr(exc, "sip_status_code", None)
            state.sip_status_code = status
            outcome = _SIP_OUTCOMES.get(status or 0, "sip_failure")
            logger.warning("call not connected (SIP %s) -> %s", status, outcome)
            return await abort(outcome)
        except Exception:
            logger.exception("outbound dial failed")
            return await abort("sip_failure")
        logger.info("call answered by %s", phone_number)
    elif web_call:
        logger.info("waiting for the caller to join %s", ctx.room.name)
        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(), timeout=float(os.getenv("ANSWER_TIMEOUT_S", "60"))
            )
        except asyncio.TimeoutError:
            logger.warning("nobody joined %s — treating as unanswered", ctx.room.name)
            return await abort("no_answer")
        logger.info("caller %s joined", participant.identity)
    state.answered_at = datetime.now(timezone.utc)

    if not phone_number:
        # SIP enforces max_call_duration on the phone leg; other calls need their own cap.
        async def cap_duration() -> None:
            await asyncio.sleep(MAX_CALL_DURATION_S)
            await hang_up("max_duration")

        run_in_background(cap_duration())

    await session_task
    try:
        await session.say(greeting(patient.name)).wait_for_playout()
    except RuntimeError:
        logger.info("session closed before the greeting — the caller left early")


def _duration(seconds: int) -> Any:
    from google.protobuf.duration_pb2 import Duration

    return Duration(seconds=seconds)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
            # Post-call analysis and the Opik flush run in the shutdown callback. The default
            # grace period kills the process mid-write, losing the trace for exactly the calls
            # worth inspecting — the slow, retrying ones.
            shutdown_process_timeout=float(os.getenv("SHUTDOWN_TIMEOUT_S", "90")),
        )
    )
