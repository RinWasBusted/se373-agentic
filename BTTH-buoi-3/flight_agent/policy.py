"""Pure policy functions. They do not call tools or network services."""

from __future__ import annotations

import hashlib
import json
from datetime import timezone
from typing import Any
from zoneinfo import ZoneInfo

from .contracts import Approval, Booking, BookingRequest, Flight, RunStatus, SeatQuote, TraceEvent

LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def payload_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable approval fingerprint for an action payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def flight_matches(request: BookingRequest, flight: Flight | SeatQuote) -> bool:
    """Check every immutable flight constraint in the request."""
    local_departure = flight.depart_at.astimezone(LOCAL_TIMEZONE)
    return (
        flight.origin == request.origin
        and flight.destination == request.destination
        and local_departure.date() == request.depart_date
        and local_departure.time().replace(tzinfo=None) < request.depart_before
        and flight.price_vnd <= request.max_price_vnd
    )


def approval_matches(approval: Approval | None, action: str, payload: dict[str, Any]) -> bool:
    """Accept only an explicit approval tied to the exact action and payload."""
    return bool(
        approval
        and approval.approved
        and approval.action == action
        and approval.payload_fingerprint == payload_fingerprint(payload)
    )


def completion_failure_reason(
    request: BookingRequest,
    checked_quote: SeatQuote | None,
    booking_snapshot: Booking | None,
    flight: Flight | None,
) -> str | None:
    """Return a stable reason when a booking cannot be proven complete."""
    if booking_snapshot is None:
        return "booking_snapshot_missing"
    if booking_snapshot.snapshot_source != "get_booking":
        return "booking_not_read_back"
    if booking_snapshot.status.value != "confirmed" or not booking_snapshot.paid:
        return "booking_not_confirmed_and_paid"
    if checked_quote is None:
        return "checked_quote_missing"
    if flight is None:
        return "flight_missing"
    if booking_snapshot.flight_id != checked_quote.flight_id or booking_snapshot.flight_id != flight.flight_id:
        return "flight_id_mismatch"
    if booking_snapshot.price_vnd != checked_quote.price_vnd or booking_snapshot.price_vnd != flight.price_vnd:
        return "price_mismatch"
    if not flight_matches(request, flight) or not flight_matches(request, checked_quote):
        return "request_constraints_not_met"
    return None


def verify_completion(
    request: BookingRequest,
    checked_quote: SeatQuote | None,
    booking_snapshot: Booking | None,
    flight: Flight | None,
) -> bool:
    """Return true only for a read-back, paid, matching and eligible booking."""
    return completion_failure_reason(request, checked_quote, booking_snapshot, flight) is None


def build_handoff(
    status: RunStatus,
    traces: list[TraceEvent],
    pending_action: str | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """Produce the minimum context a human needs to resume safely."""
    attempted = [event.action for event in traces if event.action]
    side_effects = [
        event.action
        for event in traces
        if event.phase == "tool"
        and event.action in {"book_seat", "pay"}
        and event.observation is not None
        and event.observation.status.value == "ok"
    ]
    unverified_side_effects = [
        event.action
        for event in traces
        if event.phase == "tool"
        and event.action in {"book_seat", "pay"}
        and event.decision == "tool_output_malformed_after_dispatch"
    ]
    return {
        "status": status.value,
        "completed_to": traces[-1].decision if traces else "no_action_taken",
        "attempted_actions": attempted,
        "side_effects": side_effects,
        "unverified_side_effects": unverified_side_effects,
        "pending_action": pending_action,
        "question": question or "Please provide the decision needed to resume this run.",
    }
