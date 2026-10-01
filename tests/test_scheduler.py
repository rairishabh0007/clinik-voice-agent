from datetime import date, timedelta

import pytest

from services import scheduler


@pytest.fixture(autouse=True)
def clean_bookings():
    scheduler.reset()
    yield
    scheduler.reset()


def _next_open_day(offset: int = 2) -> date:
    day = scheduler.now().date() + timedelta(days=offset)
    while not scheduler._is_open(day):
        day += timedelta(days=1)
    return day


class TestResolveDate:
    def test_iso(self):
        assert scheduler.resolve_date("2026-10-15") == date(2026, 10, 15)

    def test_relative_keywords(self):
        today = date(2026, 9, 12)
        assert scheduler.resolve_date("today", today=today) == today
        assert scheduler.resolve_date("tomorrow", today=today) == date(2026, 9, 13)
        assert scheduler.resolve_date("day after tomorrow", today=today) == date(2026, 9, 14)

    def test_weekday_resolves_forward(self):
        saturday = date(2026, 9, 12)
        assert scheduler.resolve_date("tuesday", today=saturday) == date(2026, 9, 15)

    def test_same_weekday_goes_to_next_week(self):
        saturday = date(2026, 9, 12)
        assert scheduler.resolve_date("saturday", today=saturday) == date(2026, 9, 19)

    def test_unparseable_raises(self):
        with pytest.raises(scheduler.SchedulingError):
            scheduler.resolve_date("sometime next quarter")

    def test_empty_raises(self):
        with pytest.raises(scheduler.SchedulingError):
            scheduler.resolve_date("")


class TestAvailability:
    def test_returns_requested_count(self):
        slots = scheduler.available_slots("endocrinology", on=_next_open_day(), limit=3)
        assert len(slots) == 3

    def test_is_deterministic(self):
        day = _next_open_day()
        first = scheduler.available_slots("endocrinology", on=day, limit=3)
        second = scheduler.available_slots("endocrinology", on=day, limit=3)
        assert [s.key for s in first] == [s.key for s in second]

    def test_window_filter_respected(self):
        slots = scheduler.available_slots("endocrinology", on=_next_open_day(), window="morning")
        assert all(s.start.hour < 13 for s in slots)

    def test_sunday_is_skipped(self):
        day = scheduler.now().date()
        while day.weekday() != 6:
            day += timedelta(days=1)
        slots = scheduler.available_slots("endocrinology", on=day, limit=1)
        assert slots and slots[0].start.date() != day


class TestBooking:
    def test_books_and_returns_confirmation(self):
        day = _next_open_day()
        booking = scheduler.book("P001", "Sunita Joshi", "endocrinology", day.isoformat())
        assert booking.confirmation_id.startswith("APT-")
        assert booking.slot.start.date() == day

    def test_second_booking_takes_a_different_slot(self):
        day = _next_open_day()
        first = scheduler.book("P001", "A", "endocrinology", day.isoformat())
        second = scheduler.book("P002", "B", "endocrinology", day.isoformat())
        assert first.slot.key != second.slot.key

    def test_past_date_rejected(self):
        yesterday = scheduler.now().date() - timedelta(days=1)
        with pytest.raises(scheduler.SchedulingError, match="past"):
            scheduler.book("P001", "A", "endocrinology", yesterday.isoformat())

    def test_check_bookable_matches_booking_rules(self):
        today = scheduler.now().date()
        scheduler.check_bookable(today)
        scheduler.check_bookable(today + timedelta(days=scheduler.BOOKING_HORIZON_DAYS))
        with pytest.raises(scheduler.SchedulingError, match="past"):
            scheduler.check_bookable(today - timedelta(days=1))
        with pytest.raises(scheduler.SchedulingError, match="days ahead"):
            scheduler.check_bookable(today + timedelta(days=scheduler.BOOKING_HORIZON_DAYS + 1))

    def test_beyond_horizon_rejected(self):
        far = scheduler.now().date() + timedelta(days=scheduler.BOOKING_HORIZON_DAYS + 5)
        with pytest.raises(scheduler.SchedulingError, match="days ahead"):
            scheduler.book("P001", "A", "endocrinology", far.isoformat())

    def test_exhausted_day_offers_alternatives(self):
        day = _next_open_day()
        for _ in range(200):
            try:
                scheduler.book("PX", "X", "endocrinology", day.isoformat())
            except scheduler.SlotUnavailable as exc:
                assert exc.alternatives, "alternatives must be offered, not an empty failure"
                assert all(s.start.date() != day for s in exc.alternatives)
                return
        pytest.fail("expected the day to fill up")

    def test_unknown_specialty_falls_back_to_default(self):
        day = _next_open_day()
        booking = scheduler.book("P001", "A", "cardio-thoracic wizardry", day.isoformat())
        assert booking.slot.specialty == scheduler.DEFAULT_SPECIALTY

    def test_relative_date_accepted_from_model(self):
        """Models pass 'tuesday' rather than YYYY-MM-DD often enough that it must work."""
        target = _next_open_day()
        booking = scheduler.book(
            "P001", "A", "endocrinology", target.strftime("%A").lower()
        )
        assert booking.slot.start.date() == target

    def test_closed_day_says_so(self):
        day = scheduler.now().date()
        while day.weekday() != 6:
            day += timedelta(days=1)
        with pytest.raises(scheduler.SlotUnavailable, match="closed") as exc:
            scheduler.book("P001", "A", "endocrinology", day.isoformat())
        assert exc.value.alternatives, "a closed day must still offer alternatives"


class TestExactTime:
    @pytest.mark.parametrize("text,expected", [
        ("10:30", (10, 30)), ("14:00", (14, 0)), ("10:30 am", (10, 30)),
        ("2 pm", (14, 0)), ("12 pm", (12, 0)), ("12:30 a.m.", (0, 30)),
    ])
    def test_resolve_time(self, text, expected):
        at = scheduler.resolve_time(text)
        assert (at.hour, at.minute) == expected

    @pytest.mark.parametrize("text", ["", "noon-ish", "25:00", "10:75"])
    def test_resolve_time_rejects(self, text):
        with pytest.raises(scheduler.SchedulingError):
            scheduler.resolve_time(text)

    def test_books_the_exact_time_offered(self):
        slot = scheduler.available_slots("endocrinology", on=_next_open_day(), limit=1)[0]
        booking = scheduler.book(
            "P001", "A", "endocrinology", slot.start.date().isoformat(),
            preferred_time=f"{slot.start:%H:%M}",
        )
        assert booking.slot.start == slot.start

    def test_time_off_the_grid_offers_alternatives(self):
        day = _next_open_day()
        with pytest.raises(scheduler.SlotUnavailable, match="not an appointment time") as exc:
            scheduler.book("P001", "A", "endocrinology", day.isoformat(), preferred_time="10:10")
        assert exc.value.alternatives

    def test_a_taken_time_is_not_silently_moved(self):
        slot = scheduler.available_slots("endocrinology", on=_next_open_day(), limit=1)[0]
        args = ("endocrinology", slot.start.date().isoformat())
        scheduler.book("P001", "A", *args, preferred_time=f"{slot.start:%H:%M}")
        with pytest.raises(scheduler.SlotUnavailable):
            scheduler.book("P002", "B", *args, preferred_time=f"{slot.start:%H:%M}")

    def test_morning_ends_before_noon(self):
        slots = scheduler.available_slots("endocrinology", on=_next_open_day(), window="morning", limit=20)
        assert slots and all(s.start.hour < 12 for s in slots)
