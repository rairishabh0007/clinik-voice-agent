import pytest
from aiohttp.test_utils import TestClient, TestServer

from services import report

FINISHED_CALL = {
    "ready": True,
    "patient_id": "P001",
    "patient_name": "Sunita Joshi",
    "outcome": "appointment_booked",
    "appointment_booked": True,
    "appointment": {
        "clinician": "Dr. Kavita Menon",
        "starts_at": "2026-10-06T10:30:00+05:30",
        "confirmation_id": "APT-1A2B3C4D",
    },
    "summary": "Results shared <b>and</b> a consultation booked.",
    "next_action": "Send a confirmation.",
    "safety_violations": [],
}


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        self.user = user

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(report.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_USER", "clinic@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    return FakeSMTP


def test_report_text_reads_like_a_message():
    text = report.report_text(FINISHED_CALL, "Clinik Care")
    assert text.startswith("Clinik Care — call report")
    assert "Outcome: Appointment booked" in text
    assert "Appointment: Dr. Kavita Menon · Tue 06 Oct, 10:30 AM · APT-1A2B3C4D" in text
    assert "Safety: No issues found" in text


def test_report_html_escapes_model_text():
    html = report.report_html(FINISHED_CALL, "Clinik Care")
    assert "&lt;b&gt;and&lt;/b&gt;" in html and "<b>and</b>" not in html


@pytest.mark.parametrize("address,ok", [
    ("a@b.co", True), ("rishabh.rai@gmail.com", True),
    ("no-at-sign", False), ("two@@b.co", False), ("a@b", False), ("a @b.co", False),
])
def test_email_validation(address, ok):
    assert report.valid_email(address) is ok


def test_send_email_builds_text_and_html(smtp):
    report.send_email("patient@example.com", FINISHED_CALL, "Clinik Care")
    (msg,) = smtp.sent
    assert msg["To"] == "patient@example.com"
    assert msg["From"] == "clinic@example.com"
    assert "Sunita Joshi" in msg["Subject"]
    assert [p.get_content_type() for p in msg.iter_parts()] == ["text/plain", "text/html"]


class TestReportEndpoint:
    @pytest.fixture
    async def client(self, monkeypatch, smtp):
        import web.server as server

        async def payload(room_name):
            return dict(FINISHED_CALL) if room_name == "web-P001-done" else {"ready": False}

        monkeypatch.setattr(server, "_analysis_payload", payload)
        server._emails_sent.clear()
        async with TestClient(TestServer(server.build_app())) as client:
            yield client

    async def _send(self, client, **body):
        res = await client.post("/api/report", json=body)
        return res.status, await res.json()

    async def test_sends_for_a_finished_call(self, client, smtp):
        status, body = await self._send(client, room="web-P001-done", email="a@b.co")
        assert (status, body) == (200, {"ok": True}) and len(smtp.sent) == 1

    async def test_refuses_bad_input(self, client, smtp):
        assert (await self._send(client, room="web-P001-done", email="nope"))[0] == 400
        assert (await self._send(client, room="call-P001-x", email="a@b.co"))[0] == 400
        assert (await self._send(client, room="web-P001-live", email="a@b.co"))[0] == 409
        assert smtp.sent == []

    async def test_limits_emails_per_call(self, client, smtp):
        for _ in range(3):
            assert (await self._send(client, room="web-P001-done", email="a@b.co"))[0] == 200
        assert (await self._send(client, room="web-P001-done", email="a@b.co"))[0] == 429

    async def test_smtp_failure_is_reported_not_raised(self, client, monkeypatch):
        def broken(*_):
            raise OSError("auth failed")

        monkeypatch.setattr(report, "send_email", broken)
        status, body = await self._send(client, room="web-P001-done", email="a@b.co")
        assert status == 502 and "could not be sent" in body["error"]

    async def test_config_and_avatars(self, client, monkeypatch):
        assert (await (await client.get("/api/config")).json()) == {"email": True}
        monkeypatch.delenv("SMTP_PASSWORD")
        assert (await (await client.get("/api/config")).json()) == {"email": False}
        patients = await (await client.get("/api/patients")).json()
        assert patients[0]["avatar"] == "/static/avatars/P001.svg"
        assert (await client.get(patients[0]["avatar"])).status == 200
