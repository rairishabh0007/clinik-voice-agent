from types import SimpleNamespace

from agent.patient_agent import PatientOutreachAgent
from agent.prompts import CLOSING_LINES, DEFAULT_CLOSING
from services import patients, scheduler
from services.call_state import CallState


def _agent():
    patient = patients.get_patient("P002")
    state = CallState(patient=patient, room_name="room-test")
    return PatientOutreachAgent(state, patients.build_call_variables(patient)), state


def _context(said):
    async def done():
        pass

    def say(text, **_):
        said.append(text)
        return SimpleNamespace(wait_for_playout=done)

    return SimpleNamespace(session=SimpleNamespace(say=say), wait_for_playout=done)


async def test_end_call_never_hangs_up_on_an_unanswered_patient():
    agent, state = _agent()
    agent._chat_ctx.add_message(role="assistant", content="Would you like a callback?")
    agent._chat_ctx.add_message(role="user", content="Can you schedule a call for me?")
    said = []

    await agent.end_call(_context(said), reason="callback_requested")

    assert said == [CLOSING_LINES["callback_requested"]]
    assert state.end_reason == "callback_requested"


async def test_end_call_does_not_repeat_a_goodbye_already_said():
    agent, _ = _agent()
    agent._chat_ctx.add_message(role="user", content="Thanks, bye.")
    agent._chat_ctx.add_message(role="assistant", content="Take care, goodbye.")
    said = []

    await agent.end_call(_context(said), reason="completed")

    assert said == []


async def test_do_not_call_synonym_gets_the_do_not_call_goodbye():
    agent, state = _agent()
    agent._chat_ctx.add_message(role="user", content="Stop calling me.")
    said = []

    await agent.end_call(_context(said), reason="dnc")

    assert state.do_not_call_requested is True
    assert said == [CLOSING_LINES["do_not_call"]]
    assert DEFAULT_CLOSING not in said


def test_briefing_only_uses_words_the_rules_allow():
    briefing = patients.biomarker_briefing(patients.get_patient("P002"))
    assert "borderline" not in briefing
    assert "slightly above the normal range" in briefing


def _book(state):
    slot = scheduler.available_slots("endocrinology", limit=1)[0]
    state.bookings.append(scheduler.book(
        "P002", "Vikram Singh", "endocrinology", slot.start.date().isoformat(),
        preferred_time=f"{slot.start:%H:%M}",
    ))


async def test_end_call_waits_for_the_patient_to_hear_the_booking():
    """The model booked and said goodbye in one breath; the patient never got to object."""
    scheduler.reset()
    agent, state = _agent()
    agent._chat_ctx.add_message(role="user", content="Tomorrow morning.")
    _book(state)
    said = []

    result = await agent.end_call(_context(said), reason="completed")

    assert result and "wait for their answer" in result
    assert said == [] and state.end_reason is None


async def test_end_call_allowed_once_the_patient_answered():
    scheduler.reset()
    agent, state = _agent()
    _book(state)
    agent._chat_ctx.add_message(role="assistant", content="That is Tuesday at 10:30. Does that work?")
    agent._chat_ctx.add_message(role="user", content="Yes, perfect. Bye.")
    agent._chat_ctx.add_message(role="assistant", content="Take care, goodbye.")

    result = await agent.end_call(_context([]), reason="completed")

    assert result is None and state.end_reason == "completed"
