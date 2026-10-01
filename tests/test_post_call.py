from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from analysis import post_call
from analysis.post_call import CallAnalysis, _deterministic_only, _reconcile, analyse_call
from services import patients, scheduler
from services.call_state import CallState


@pytest.fixture(autouse=True)
def clean_bookings():
    scheduler.reset()
    yield
    scheduler.reset()


def _open_day() -> str:
    day = scheduler.now().date() + timedelta(days=2)
    while not scheduler._is_open(day):
        day += timedelta(days=1)
    return day.isoformat()


@pytest.fixture
def state() -> CallState:
    return CallState(patient=patients.get_patient("P001"), room_name="room-test")


def _analysis(**overrides) -> CallAnalysis:
    base = dict(
        outcome="declined",
        appointment_booked=False,
        appointment=None,
        patient_sentiment="neutral",
        identity_verified=False,
        call_summary="summary",
        next_action="next",
    )
    base.update(overrides)
    return CallAnalysis(**base)


class TestReconciliation:
    def test_model_claiming_a_booking_is_overruled(self, state):
        """The headline correctness property: the LLM cannot mark its own homework."""
        claimed = _analysis(outcome="appointment_booked", appointment_booked=True)
        result, corrections = _reconcile(claimed, state)

        assert result.appointment_booked is False
        assert result.appointment is None
        assert any("appointment_booked" in c for c in corrections)

    def test_booked_outcome_without_a_booking_is_not_kept(self, state):
        """Fixing the flag but keeping outcome=appointment_booked would contradict itself."""
        state.answered_at = state.started_at
        state.identity_confirmed = True
        claimed = _analysis(
            outcome="appointment_booked", appointment_booked=True, identity_verified=True
        )
        result, corrections = _reconcile(claimed, state)

        assert result.outcome == "incomplete"
        assert any(c.startswith("outcome:") for c in corrections)

    def test_model_missing_a_real_booking_is_corrected(self, state):
        booking = scheduler.book("P001", "Sunita Joshi", "endocrinology", _open_day())
        state.bookings.append(booking)

        result, corrections = _reconcile(_analysis(appointment_booked=False), state)

        assert result.appointment_booked is True
        assert result.outcome == "appointment_booked"
        assert result.appointment.confirmation_id == booking.confirmation_id
        assert any("appointment_booked" in c for c in corrections)

    def test_identity_flag_comes_from_the_tool_record(self, state):
        state.identity_confirmed = True
        result, corrections = _reconcile(_analysis(identity_verified=False), state)

        assert result.identity_verified is True
        assert any("identity_verified" in c for c in corrections)

    def test_agreement_produces_no_corrections(self, state):
        state.identity_confirmed = False
        result, corrections = _reconcile(_analysis(outcome="declined"), state)

        assert corrections == []
        assert result.outcome == "declined"

    def test_voicemail_overrides_whatever_the_model_said(self, state):
        state.voicemail_detected = True
        result, corrections = _reconcile(_analysis(outcome="declined"), state)

        assert result.outcome == "voicemail"

    def test_do_not_call_outranks_a_booking(self, state):
        state.do_not_call_requested = True
        result, _ = _reconcile(_analysis(outcome="callback_requested"), state)

        assert result.outcome == "do_not_call_requested"


class TestDeterministicOutcome:
    def test_dial_failure_short_circuits(self, state):
        state.dial_error = "no_answer"
        assert state.deterministic_outcome() == "no_answer"

    def test_answered_but_unverified(self, state):
        state.answered_at = datetime.now(timezone.utc)
        assert state.deterministic_outcome() == "identity_unverified"

    def test_normal_call_defers_to_the_model(self, state):
        state.answered_at = datetime.now(timezone.utc)
        state.identity_confirmed = True
        assert state.deterministic_outcome() is None

    def test_booking_is_authoritative(self, state):
        state.answered_at = datetime.now(timezone.utc)
        state.identity_confirmed = True
        state.bookings.append(scheduler.book("P001", "A", "endocrinology", _open_day()))
        assert state.deterministic_outcome() == "appointment_booked"


class TestFallbacks:
    async def test_empty_transcript_never_calls_the_llm(self, state):
        state.dial_error = "no_answer"
        result = await analyse_call(state, transcript=[])

        assert result.source == "deterministic"
        assert result.analysis.outcome == "no_answer"
        assert result.analysis.appointment_booked is False

    async def test_llm_failure_degrades_to_known_facts(self, state, monkeypatch):
        class ExplodingClient:
            class chat:
                class completions:
                    @staticmethod
                    async def parse(**_):
                        raise RuntimeError("openai is down")

        state.voicemail_detected = True
        result = await analyse_call(
            state,
            transcript=[{"role": "assistant", "text": "Hello, this is Riya."}],
            client=ExplodingClient(),
        )

        assert result.source == "fallback"
        assert result.analysis.outcome == "voicemail"

    async def test_retry_waits_as_long_as_the_provider_asks(self, state, monkeypatch):
        """A 429 says 'retry in 31s'; giving up after 14s of backoff would waste the call."""
        waits = []

        async def fake_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr(post_call.asyncio, "sleep", fake_sleep)
        parsed = _analysis(outcome="declined")
        calls = []

        class RateLimitedOnce:
            class chat:
                class completions:
                    @staticmethod
                    async def parse(**_):
                        calls.append(1)
                        if len(calls) == 1:
                            raise RuntimeError("Error code: 429 RESOURCE_EXHAUSTED. Please retry in 31.2s.")
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))],
                            usage=None,
                        )

        result = await analyse_call(
            state, transcript=[{"role": "user", "text": "hello"}], client=RateLimitedOnce()
        )

        assert result.source == "llm"
        assert waits == [pytest.approx(32.2)]

    async def test_retry_gives_up_beyond_budget_with_a_short_reason(self, state, monkeypatch):
        async def fake_sleep(_):
            pass

        monkeypatch.setattr(post_call.asyncio, "sleep", fake_sleep)

        class AlwaysLimited:
            class chat:
                class completions:
                    @staticmethod
                    async def parse(**_):
                        raise RuntimeError("Error code: 429 {'retryDelay': '300s'} ...long payload...")

        result = await analyse_call(
            state, transcript=[{"role": "user", "text": "hello"}], client=AlwaysLimited()
        )

        assert result.source == "fallback"
        assert result.analysis.call_summary == (
            "Analysis unavailable: the analysis model's rate limit was reached (429)."
        )

    async def test_a_hung_model_still_gives_the_known_facts(self, state, monkeypatch):
        """A request that never returns must end in the fallback, not in no analysis at all."""
        import asyncio

        monkeypatch.setattr(post_call, "_ANALYSIS_TIMEOUT_S", 0.05)

        class Hangs:
            class chat:
                class completions:
                    @staticmethod
                    async def parse(**_):
                        await asyncio.sleep(10)

        state.voicemail_detected = True
        result = await analyse_call(
            state, transcript=[{"role": "user", "text": "hello"}], client=Hangs()
        )

        assert result.source == "fallback"
        assert result.analysis.outcome == "voicemail"
        assert "did not respond in time" in result.analysis.call_summary

    def test_no_transcript_summary_is_honest(self, state):
        result = _deterministic_only(state, "The call produced no conversation.")
        assert "no conversation" in result.analysis.call_summary


class TestCallState:
    def test_tool_call_alone_is_not_success(self, state):
        state.record_tool("book_appointment", {"preferred_date": "tomorrow"}, error="slot taken")
        assert state.appointment_booked is False
        assert len(state.tool_invocations) == 1

    def test_durations(self, state):
        state.started_at = datetime.now(timezone.utc)
        state.answered_at = state.started_at + timedelta(seconds=5)
        state.ended_at = state.started_at + timedelta(seconds=65)

        assert state.duration_seconds == pytest.approx(65, abs=0.5)
        assert state.talk_time_seconds == pytest.approx(60, abs=0.5)

    def test_durations_are_none_while_live(self, state):
        assert state.duration_seconds is None
