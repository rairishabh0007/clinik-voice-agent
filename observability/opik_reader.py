"""Read completed calls back out of Opik.

The write side (opik_tracer.py) is the standalone deliverable and stays free of this. This is the
read side, used by the web console: the traces Opik already holds are the system of record for a
call, so nothing is stored twice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

WEB_ROOT = "https://www.comet.com/opik"


@dataclass
class CallSummary:
    trace_id: str
    thread_id: str | None
    name: str
    started_at: datetime | None
    patient_id: str | None
    outcome: str
    mode: str
    tags: list[str]
    scores: dict[str, float]
    duration_s: float | None


@dataclass
class CallDetail:
    summary: CallSummary
    variables: dict[str, Any]
    analysis: dict[str, Any]
    call_record: dict[str, Any]
    metadata: dict[str, Any]
    transcript: list[dict[str, str]] = field(default_factory=list)
    tool_spans: list[dict[str, Any]] = field(default_factory=list)
    score_reasons: dict[str, str] = field(default_factory=dict)


def _project_name() -> str:
    return os.getenv("OPIK_PROJECT_NAME", "livekit-voice-agent")


def _client():
    import opik

    return opik.Opik(project_name=_project_name())


def configured() -> bool:
    return bool(os.getenv("OPIK_API_KEY") or os.getenv("OPIK_URL_OVERRIDE"))


def _project(rest) -> Any:
    return rest.projects.retrieve_project(name=_project_name())


def _logs_url(project_id: str) -> str:
    return f"{WEB_ROOT}/{os.getenv('OPIK_WORKSPACE', 'default')}/projects/{project_id}/logs"


def trace_url(project_id: str, trace_id: str) -> str:
    return f"{_logs_url(project_id)}?logsType=traces&trace={trace_id}"


def project_id() -> str:
    return str(_project(_client().rest_client).id)


def _summarise(trace: Any) -> CallSummary:
    meta = trace.metadata or {}
    output = trace.output or {}
    record = meta.get("call_record") or {}
    return CallSummary(
        trace_id=str(trace.id),
        thread_id=meta.get("thread_id"),
        name=trace.name or "(unnamed)",
        started_at=trace.start_time,
        patient_id=meta.get("patient_id"),
        outcome=output.get("outcome") or "unknown",
        mode=meta.get("mode") or "unknown",
        tags=list(trace.tags or []),
        scores={f.name: f.value for f in (trace.feedback_scores or [])},
        duration_s=record.get("talk_time_seconds") or record.get("duration_seconds"),
    )


def find_call_by_room(room_name: str) -> CallSummary | None:
    """Locate the call trace for a room. Returns None until the agent has finished logging it."""
    rest = _client().rest_client
    page = rest.traces.get_traces_by_project(project_id=str(_project(rest).id), size=100)
    for trace in page.content or []:
        if trace.name == "outbound_call" and (trace.metadata or {}).get("call_id") == room_name:
            return _summarise(trace)
    return None


def get_call(trace_id: str) -> CallDetail:
    rest = _client().rest_client
    project = _project(rest)
    trace = rest.traces.get_trace_by_id(id=trace_id)
    summary = _summarise(trace)

    meta = trace.metadata or {}
    output = trace.output or {}
    detail = CallDetail(
        summary=summary,
        variables=(trace.input or {}).get("variables") or {},
        analysis=output.get("analysis") or {},
        call_record=meta.get("call_record") or {},
        metadata=meta,
        score_reasons={
            f.name: (f.reason or "") for f in (trace.feedback_scores or []) if f.reason
        },
    )

    spans = rest.spans.get_spans_by_project(
        project_id=str(project.id), trace_id=trace_id, size=100
    ).content or []
    detail.tool_spans = [
        {
            "name": s.name,
            "input": s.input or {},
            "output": s.output or {},
            "metadata": s.metadata or {},
        }
        for s in spans
        if s.type == "tool"
    ]

    if summary.thread_id:
        detail.transcript = _transcript(rest, str(project.id), summary.thread_id)
    return detail


def _transcript(rest, project_id: str, thread_id: str) -> list[dict[str, str]]:
    """Rebuild the conversation from the turn traces that share this thread_id."""
    page = rest.traces.get_traces_by_project(project_id=project_id, size=400)
    turns = [
        t for t in (page.content or [])
        if t.name == "turn" and t.thread_id == thread_id
    ]
    turns.sort(key=lambda t: (t.metadata or {}).get("turn_index", 0))

    lines: list[dict[str, str]] = []
    for turn in turns:
        for role, text in (
            ("patient", (turn.input or {}).get("user", "")),
            ("agent", (turn.output or {}).get("agent", "")),
        ):
            if text.strip():
                lines.append({"role": role, "text": text.strip()})
    return lines
