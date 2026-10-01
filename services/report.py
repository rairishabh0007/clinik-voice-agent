"""The post-call report a patient-facing team member can send by email or WhatsApp.

One plain-text report serves both channels: WhatsApp gets it as the message body, email gets it
alongside a simple HTML version. Email goes out through any SMTP server (Gmail with an app
password works); WhatsApp is a click-to-chat link opened on the sender's own phone.
"""

from __future__ import annotations

import html
import os
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def email_configured() -> bool:
    return bool(os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD"))


def valid_email(address: str) -> bool:
    return len(address) <= 254 and bool(EMAIL_RE.match(address))


def _spoken_time(iso: str | None) -> str:
    """'2026-10-06T10:30:00+05:30' -> 'Tue 06 Oct, 10:30 AM' (built by hand: %-I is not portable)."""
    if not iso:
        return ""
    try:
        at = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{at:%a %d %b}, {at.hour % 12 or 12}:{at:%M %p}"


def _label(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").capitalize()


def report_lines(data: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, value) pairs shared by the text and HTML versions."""
    lines = [("Patient", f"{data.get('patient_name') or 'Unknown'} ({data.get('patient_id') or '-'})"),
             ("Outcome", _label(data.get("outcome")))]
    appt = data.get("appointment") or {}
    if data.get("appointment_booked") and appt:
        lines.append(("Appointment", " · ".join(filter(None, [
            appt.get("clinician"), _spoken_time(appt.get("starts_at")), appt.get("confirmation_id"),
        ]))))
    if data.get("summary"):
        lines.append(("Summary", data["summary"]))
    if data.get("next_action"):
        lines.append(("Next step", data["next_action"]))
    violations = data.get("safety_violations") or []
    lines.append(("Safety", "; ".join(violations) if violations else "No issues found"))
    return lines


def report_text(data: dict[str, Any], clinic: str) -> str:
    body = "\n".join(f"{label}: {value}" for label, value in report_lines(data))
    return f"{clinic} — call report\n\n{body}"


def report_html(data: dict[str, Any], clinic: str) -> str:
    rows = "".join(
        f'<tr><td style="padding:8px 14px 8px 0;color:#7f85a3;vertical-align:top;'
        f'white-space:nowrap">{html.escape(label)}</td>'
        f'<td style="padding:8px 0;color:#141833">{html.escape(value)}</td></tr>'
        for label, value in report_lines(data)
    )
    return (
        '<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.5;max-width:560px">'
        f'<h2 style="color:#3d4fd6;margin:0 0 12px">{html.escape(clinic)} — call report</h2>'
        f'<table style="border-collapse:collapse">{rows}</table>'
        '<p style="color:#7f85a3;font-size:12px;margin-top:18px">Sent from the voice agent '
        "console after the call was analysed.</p></div>"
    )


def send_email(to: str, data: dict[str, Any], clinic: str) -> None:
    """Blocking; call from a thread. Raises on any SMTP failure."""
    user = os.environ["SMTP_USER"]
    msg = EmailMessage()
    msg["Subject"] = f"{clinic} call report — {data.get('patient_name') or 'patient'}"
    msg["From"] = os.getenv("REPORT_FROM") or user
    msg["To"] = to
    msg.set_content(report_text(data, clinic))
    msg.add_alternative(report_html(data, clinic), subtype="html")

    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(user, os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
