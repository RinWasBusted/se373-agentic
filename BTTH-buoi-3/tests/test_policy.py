from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from flight_agent.contracts import (
    Approval,
    Booking,
    BookingRequest,
    BookingStatus,
    Flight,
    SeatQuote,
)
from flight_agent.policy import (
    approval_matches,
    completion_failure_reason,
    flight_matches,
    payload_fingerprint,
    verify_completion,
)

TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def request() -> BookingRequest:
    return BookingRequest(
        origin="SGN",
        destination="DAD",
        depart_date="2026-10-07",
        depart_before="12:00",
        max_price_vnd=2_000_000,
        passenger_id="student-01",
    )


def flight(*, flight_id="VN122", depart_at=None, price=1_850_000) -> Flight:
    return Flight(
        flight_id=flight_id,
        origin="SGN",
        destination="DAD",
        depart_at=depart_at or datetime(2026, 10, 7, 8, 10, tzinfo=TZ),
        price_vnd=price,
        seats_remaining=3,
        refundable=False,
    )


def quote(*, flight_id="VN122", price=1_850_000, depart_at=None) -> SeatQuote:
    return SeatQuote(
        flight_id=flight_id,
        origin="SGN",
        destination="DAD",
        depart_at=depart_at or datetime(2026, 10, 7, 8, 10, tzinfo=TZ),
        price_vnd=price,
        seats_remaining=3,
        checked_at=datetime(2026, 10, 1, 8, 0, tzinfo=TZ),
    )


def booking(*, status=BookingStatus.CONFIRMED, paid=True, source="get_booking", flight_id="VN122", price=1_850_000) -> Booking:
    return Booking(
        booking_code="AB12CD",
        flight_id=flight_id,
        seat="12A",
        price_vnd=price,
        status=status,
        paid=paid,
        idempotency_key="booking-1",
        snapshot_source=source,
    )


def test_t02_departure_boundary_is_strict():
    assert flight_matches(request(), flight(depart_at=datetime(2026, 10, 7, 11, 59, tzinfo=TZ)))
    assert not flight_matches(request(), flight(depart_at=datetime(2026, 10, 7, 12, 0, tzinfo=TZ)))


def test_t03_price_ceiling_is_inclusive():
    assert flight_matches(request(), flight(price=2_000_000))
    assert not flight_matches(request(), flight(price=2_000_001))


def test_t17_held_or_unpaid_booking_is_not_complete():
    assert not verify_completion(request(), quote(), booking(status=BookingStatus.HELD, paid=False), flight())
    assert not verify_completion(request(), quote(), booking(source=None), flight())


@pytest.mark.parametrize(
    "invalid_flight,invalid_quote,invalid_booking",
    [
        (flight(depart_at=datetime(2026, 10, 8, 8, 10, tzinfo=TZ)), quote(), booking()),
        (flight(depart_at=datetime(2026, 10, 7, 12, 0, tzinfo=TZ)), quote(), booking()),
        (flight(price=2_000_001), quote(), booking()),
        (flight(), quote(price=2_000_001), booking()),
    ],
)
def test_t18_constraints_must_remain_true(invalid_flight, invalid_quote, invalid_booking):
    assert not verify_completion(request(), invalid_quote, invalid_booking, invalid_flight)


def test_t19_read_back_booking_that_matches_everything_is_complete():
    assert verify_completion(request(), quote(), booking(), flight())


@pytest.mark.parametrize(
    "bad_quote,bad_booking,bad_flight,expected_reason",
    [
        (quote(flight_id="VJ604"), booking(), flight(), "flight_id_mismatch"),
        (quote(), booking(price=1_840_000), flight(), "price_mismatch"),
    ],
)
def test_t39_booking_must_match_checked_quote(bad_quote, bad_booking, bad_flight, expected_reason):
    assert not verify_completion(request(), bad_quote, bad_booking, bad_flight)
    assert completion_failure_reason(request(), bad_quote, bad_booking, bad_flight) == expected_reason


def test_p1_a_approval_is_bound_to_exact_action_and_payload():
    payload = {"booking_code": "AB12CD", "price_vnd": 1_850_000}
    approval = Approval(
        approved=True,
        action="pay",
        payload_fingerprint=payload_fingerprint(payload),
        reason="student approval",
    )
    assert approval_matches(approval, "pay", payload)
    assert not approval_matches(approval, "book_seat", payload)
    assert not approval_matches(approval, "pay", {**payload, "price_vnd": 1_850_001})


def test_p1_c_flight_date_uses_vietnam_timezone():
    utc_departure = datetime(2026, 10, 6, 17, 10, tzinfo=ZoneInfo("UTC"))
    assert flight_matches(request(), flight(depart_at=utc_departure))
