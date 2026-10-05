from concurrent.futures import ThreadPoolExecutor

import pytest

from flight_agent.contracts import Approval, Booking, BookingRequest, Flight, SeatQuote, ToolStatus
from flight_agent.fixtures import FIXED_NOW, default_environment
from flight_agent.mock_environment import MockFlightEnvironment
from flight_agent.policy import payload_fingerprint, verify_completion


def request(*, passenger_id: str = "student-01", max_price_vnd: int = 2_000_000) -> BookingRequest:
    return BookingRequest(
        origin="SGN",
        destination="DAD",
        depart_date="2026-10-07",
        depart_before="12:00",
        max_price_vnd=max_price_vnd,
        passenger_id=passenger_id,
    )


def book_approval(
    env: MockFlightEnvironment,
    booking_request: BookingRequest,
    flight_id: str,
    seat: str,
    price: int,
    key: str,
) -> Approval:
    payload = env.booking_approval_payload(booking_request, flight_id, seat, price, key)
    return Approval(
        approved=True,
        action="book_seat",
        payload_fingerprint=payload_fingerprint(payload),
        reason="approved in mock scenario",
    )


def pay_approval(
    env: MockFlightEnvironment, booking_request: BookingRequest, booking: Booking, key: str
) -> Approval:
    payload = env.payment_approval_payload(booking_request, booking, key)
    return Approval(
        approved=True,
        action="pay",
        payload_fingerprint=payload_fingerprint(payload),
        reason="approved in mock scenario",
    )


def booked(env: MockFlightEnvironment, *, key: str = "book-1") -> Booking:
    booking_request = request()
    response = env.book_seat(
        booking_request,
        "VN122",
        "12A",
        1_850_000,
        key,
        book_approval(env, booking_request, "VN122", "12A", 1_850_000, key),
    )
    assert response.status is ToolStatus.OK
    return Booking.model_validate(response.data["booking"])


def test_t04_search_returns_only_fixture_flights_in_stable_order():
    response = default_environment().search_flights("SGN", "DAD", "2026-10-07")
    assert response.status is ToolStatus.OK
    assert [flight["flight_id"] for flight in response.data["flights"]] == ["VN122", "VJ604", "VN134", "QH118"]


def test_t05_and_t06_distinguish_empty_from_timeout():
    env = default_environment()
    assert env.search_flights("SGN", "HUI", "2026-10-07").status is ToolStatus.EMPTY
    env.inject_fault("search_flights")
    response = env.search_flights("SGN", "DAD", "2026-10-07")
    assert response.status is ToolStatus.ERROR
    assert response.error_code == "timeout"


def test_t08_invalid_search_arguments_are_structured():
    response = default_environment().search_flights("SGN", "DAD", "07/10")
    assert response.status is ToolStatus.INVALID_PARAM
    assert response.hint


def test_t35_search_keeps_ineligible_flights_visible_for_the_harness():
    response = default_environment().search_flights("SGN", "DAD", "2026-10-07")
    assert response.status is ToolStatus.OK
    assert any(flight["price_vnd"] > 2_000_000 for flight in response.data["flights"])
    assert any(flight["depart_at"].startswith("2026-10-07T15:40") for flight in response.data["flights"])


def test_t09_check_seat_reports_zero_and_another_flight_remains_available():
    env = default_environment()
    no_seat = env.check_seat("VJ604")
    available = env.check_seat("VN122")
    assert no_seat.data["quote"]["seats_remaining"] == 0
    assert available.data["available_seats"] == ["12A", "12B"]


def test_t10_price_change_blocks_booking_without_side_effect():
    env = default_environment()
    booking_request = request()
    env.check_seat("VN122")
    env.set_price("VN122", 1_900_000)
    response = env.book_seat(
        booking_request,
        "VN122",
        "12A",
        1_850_000,
        "book-price-change",
        book_approval(env, booking_request, "VN122", "12A", 1_850_000, "book-price-change"),
    )
    assert response.status is ToolStatus.ERROR
    assert response.error_code == "price_changed"
    assert env.check_seat("VN122").data["available_seats"] == ["12A", "12B"]


def test_t11_constraints_and_approval_are_checked_before_booking():
    env = default_environment()
    booking_request = request()
    response = env.book_seat(
        booking_request,
        "QH118",
        "10A",
        1_640_000,
        "book-wrong-time",
        book_approval(env, booking_request, "QH118", "10A", 1_640_000, "book-wrong-time"),
    )
    assert response.status is ToolStatus.DENIED
    assert response.error_code == "request_constraints_not_met"
    assert env.booking_count == 0


def test_t12_missing_or_mismatched_approval_has_no_side_effect():
    env = default_environment()
    booking_request = request()
    denied = env.book_seat(booking_request, "VN122", "12A", 1_850_000, "no-approval", None)
    wrong = env.book_seat(
        booking_request,
        "VN122",
        "12A",
        1_850_000,
        "wrong-approval",
        book_approval(env, booking_request, "VN122", "12B", 1_850_000, "wrong-approval"),
    )
    assert denied.status is ToolStatus.DENIED
    assert wrong.status is ToolStatus.DENIED
    assert env.booking_count == 0


def test_t16_book_retry_and_key_conflict_are_idempotent():
    env = default_environment()
    booking_request = request()
    approval = book_approval(env, booking_request, "VN122", "12A", 1_850_000, "book-retry")
    first = env.book_seat(booking_request, "VN122", "12A", 1_850_000, "book-retry", approval)
    retry = env.book_seat(booking_request, "VN122", "12A", 1_850_000, "book-retry", None)
    conflict = env.book_seat(
        booking_request,
        "VN122",
        "12B",
        1_850_000,
        "book-retry",
        book_approval(env, booking_request, "VN122", "12B", 1_850_000, "book-retry"),
    )
    assert first == retry
    assert conflict.error_code == "idempotency_conflict"
    assert env.booking_count == 1
    assert env.check_seat("VN122").data["available_seats"] == ["12B"]


def test_t38_only_one_call_can_hold_the_last_seat():
    env = default_environment()

    def reserve(passenger_id: str, key: str):
        booking_request = request(passenger_id=passenger_id, max_price_vnd=2_200_000)
        return env.book_seat(
            booking_request,
            "VN134",
            "14A",
            2_100_000,
            key,
            book_approval(env, booking_request, "VN134", "14A", 2_100_000, key),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda item: reserve(*item), [("student-01", "last-1"), ("student-02", "last-2")]))
    assert [outcome.status for outcome in outcomes].count(ToolStatus.OK) == 1
    assert env.booking_count == 1
    assert env.check_seat("VN134").data["quote"]["seats_remaining"] == 0


def test_t16_pay_retry_creates_one_mock_payment_and_t19_get_booking_is_read_back_snapshot():
    env = default_environment()
    booking_request = request()
    held = booked(env)
    approval = pay_approval(env, booking_request, held, "pay-retry")
    first = env.pay(booking_request, held.booking_code, "pay-retry", approval)
    retry = env.pay(booking_request, held.booking_code, "pay-retry", None)
    snapshot_response = env.get_booking(held.booking_code)
    snapshot = Booking.model_validate(snapshot_response.data["booking"])
    quote = SeatQuote.model_validate(env.check_seat("VN122").data["quote"])
    flight = Flight.model_validate(env.search_flights("SGN", "DAD", "2026-10-07").data["flights"][0])
    assert first == retry
    assert len(env.payments) == 1
    assert snapshot.snapshot_source == "get_booking"
    assert verify_completion(booking_request, quote, snapshot, flight)


def test_paid_booking_cannot_create_a_second_payment_with_another_key():
    env = default_environment()
    booking_request = request()
    held = booked(env)
    first_key = "pay-first"
    first = env.pay(
        booking_request,
        held.booking_code,
        first_key,
        pay_approval(env, booking_request, held, first_key),
    )
    paid = Booking.model_validate(first.data["booking"])
    second_key = "pay-second"
    second = env.pay(
        booking_request,
        held.booking_code,
        second_key,
        pay_approval(env, booking_request, paid, second_key),
    )
    assert second.status is ToolStatus.ERROR
    assert second.error_code == "booking_not_payable"
    assert len(env.payments) == 1


def test_payment_blocks_owner_mismatch_and_price_change():
    env = default_environment()
    held = booked(env)
    wrong_request = request(passenger_id="student-02")
    wrong_owner = env.pay(
        wrong_request,
        held.booking_code,
        "pay-wrong-owner",
        pay_approval(env, wrong_request, held, "pay-wrong-owner"),
    )
    assert wrong_owner.status is ToolStatus.DENIED
    env.set_price("VN122", 1_900_000)
    booking_request = request()
    price_change = env.pay(
        booking_request,
        held.booking_code,
        "pay-price-change",
        pay_approval(env, booking_request, held, "pay-price-change"),
    )
    assert price_change.error_code == "price_changed"
    assert not env.payments


def test_get_booking_not_found_is_explicitly_empty():
    response = default_environment().get_booking("UNKNOWN")
    assert response.status is ToolStatus.EMPTY
    assert response.data == {"booking": None}


def test_invalid_flight_or_seat_never_mutates_inventory():
    env = default_environment()
    booking_request = request()
    response = env.book_seat(
        booking_request,
        "UNKNOWN",
        "12A",
        1_850_000,
        "invalid-flight",
        book_approval(env, booking_request, "UNKNOWN", "12A", 1_850_000, "invalid-flight"),
    )
    assert response.status is ToolStatus.INVALID_PARAM
    assert env.booking_count == 0
    assert env.check_seat("VN122").data["available_seats"] == ["12A", "12B"]


def test_environments_are_isolated_and_reproducible():
    first = default_environment()
    second = default_environment()
    assert first.search_flights("SGN", "DAD", "2026-10-07") == second.search_flights("SGN", "DAD", "2026-10-07")
    booked(first)
    assert first.check_seat("VN122").data["available_seats"] == ["12B"]
    assert second.check_seat("VN122").data["available_seats"] == ["12A", "12B"]


def test_fixture_rejects_inventory_that_does_not_match_flights():
    flight = Flight(
        flight_id="VN122",
        origin="SGN",
        destination="DAD",
        depart_at=FIXED_NOW,
        price_vnd=1_850_000,
        seats_remaining=1,
        refundable=False,
    )
    with pytest.raises(ValueError, match="seat inventory"):
        MockFlightEnvironment([flight], {"VN122": []}, now=lambda: FIXED_NOW)
