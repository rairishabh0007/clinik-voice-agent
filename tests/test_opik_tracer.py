import asyncio
from types import SimpleNamespace

from observability.opik_tracer import OpikCallTracer


class FakeSession:
    def __init__(self):
        self.handlers = {}

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)

    def emit(self, event, payload):
        for handler in self.handlers.get(event, []):
            handler(payload)


class RecordingClient:
    """Stands in for opik.Opik and records what would have been sent."""

    def __init__(self):
        self.traces = []

    def trace(self, **kwargs):
        self.traces.append(kwargs)
        return SimpleNamespace(span=lambda **_: None)

    def flush(self, timeout):
        pass


def _tracer(**overrides):
    tracer = OpikCallTracer(
        call_id="room-1", enabled=False, **{"sdk_scoring": False, **overrides}
    )
    tracer._enabled, tracer._client = True, RecordingClient()
    return tracer


def _say(session, role, text):
    item = SimpleNamespace(role=role, text_content=text, interrupted=False)
    session.emit("conversation_item_added", SimpleNamespace(item=item, created_at=None))


async def test_a_hung_analysis_still_logs_the_call():
    """The whole point of per-stage timeouts: a slow analysis must not cost the trace."""

    async def hangs(_transcript):
        await asyncio.sleep(10)

    tracer, session = _tracer(callback_timeout=0.05), FakeSession()
    tracer.attach(session, finalise=hangs)
    _say(session, "assistant", "Hello, this is Riya.")
    _say(session, "user", "Yes, speaking.")

    await tracer.finalise()

    names = [t["name"] for t in tracer._client.traces]
    assert names.count("outbound_call") == 1 and "turn" in names


async def test_finalise_twice_logs_once():
    async def summary(_transcript):
        return {"outcome": "declined", "analysis": {"outcome": "declined"}}

    tracer, session = _tracer(), FakeSession()
    tracer.attach(session, finalise=summary)
    _say(session, "user", "No thanks.")

    await tracer.finalise()
    await tracer.finalise()

    assert [t["name"] for t in tracer._client.traces].count("outbound_call") == 1


async def test_collection_runs_with_opik_disabled():
    """Disabling telemetry must not change what the host's analysis receives."""
    received = []

    async def summary(transcript):
        received.extend(transcript)
        return {}

    tracer, session = OpikCallTracer(call_id="room-2", enabled=False), FakeSession()
    tracer.attach(session, finalise=summary)
    _say(session, "user", "Hello?")

    await tracer.finalise()

    assert [item["text"] for item in received] == ["Hello?"]


async def test_protocol_judge_sees_both_sides_of_the_conversation(monkeypatch):
    """Without the patient's lines the judge cannot see identity being confirmed."""
    seen = {}

    def fake_score(agent_text, conversation, analysis):
        seen.update(agent_text=agent_text, conversation=conversation)
        return []

    tracer, session = _tracer(sdk_scoring=True), FakeSession()
    monkeypatch.setattr(tracer, "_score_sync", fake_score)
    tracer.attach(session)
    _say(session, "assistant", "Am I speaking with Vikram?")
    _say(session, "user", "Yes, this is Vikram.")

    await tracer.finalise()

    assert "patient: Yes, this is Vikram." in seen["conversation"]
    assert "Yes, this is Vikram" not in seen["agent_text"]
