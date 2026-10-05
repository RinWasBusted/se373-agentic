"""Deterministic in-memory tools for the flight-booking agent exercises."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from threading import RLock
from typing import Callable, Iterable

from .contracts import (
    Approval,
    Booking,
    BookingRequest,
    BookingStatus,
    Flight,
    SeatQuote,
    ToolObservation,
    ToolStatus,
)
from .policy import LOCAL_TIMEZONE, approval_matches, flight_matches


@dataclass(frozen=True)
class PaymentRecord:
    booking_code: str
    passenger_id: str
    amount_vnd: int
    idempotency_key: str


@dataclass(frozen=True)
class _IdempotencyRecord:
    payload: tuple[tuple[str, object], ...]
    response: ToolObservation


class MockFlightEnvironment:
    """A scenario-local mock backend with no network or persistent storage."""

    def __init__(
        self,
        flights: Iterable[Flight],
        seats_by_flight: dict[str, Iterable[str]],
        *,
        now: Callable[[], datetime],
    ) -> None:
        self._lock = RLock()
        self._now = now
        fixture_flights = list(flights)
        self._flights = {flight.flight_id: flight.model_copy(deep=True) for flight in fixture_flights}
        if len(self._flights) != len(fixture_flights):
            raise ValueError("fixture contains duplicate flight_id values")
        self._seats = {flight_id: set(seats) for flight_id, seats in seats_by_flight.items()}
        if set(self._seats) != set(self._flights):
            raise ValueError("seat fixture must contain exactly one entry per flight")
        for flight_id, flight in self._flights.items():
            if len(self._seats[flight_id]) != flight.seats_remaining:
                raise ValueError("seat inventory must match Flight.seats_remaining")
        self._bookings: dict[str, Booking] = {}
        self._booking_owner: dict[str, str] = {}
        self._payments: list[PaymentRecord] = []
        self._idempotency: dict[str, _IdempotencyRecord] = {}
        self._faults: dict[str, list[ToolObservation]] = defaultdict(list)
        self._booking_sequence = 0

    @property
    def payments(self) -> tuple[PaymentRecord, ...]:
        return tuple(self._payments)

    @property
    def booking_count(self) -> int:
        return len(self._bookings)

    def inject_fault(
        self,
        tool_name: str,
        *,
        error_code: str = "timeout",
        hint: str = "Mock service is temporarily unavailable.",
        count: int = 1,
    ) -> None:
        if tool_name not in {"search_flights", "check_seat", "book_seat", "pay", "get_booking"}:
            raise ValueError("unknown tool name")
        if count <= 0:
            raise ValueError("count must be positive")
        fault = self._error(error_code, hint)
        self._faults[tool_name].extend([fault] * count)

    def set_price(self, flight_id: str, price_vnd: int) -> None:
        """Scenario control only; the agent never receives this operation as a tool."""
        if type(price_vnd) is not int or price_vnd < 0:
            raise ValueError("price_vnd must be a non-negative integer")
        with self._lock:
            flight = self._flights.get(flight_id)
            if flight is None:
                raise ValueError("unknown flight_id")
            self._flights[flight_id] = flight.model_copy(update={"price_vnd": price_vnd})

    def search_flights(self, origin: str, destination: str, depart_date: str) -> ToolObservation:
        with self._lock:
            if fault := self._take_fault("search_flights"):
                return fault
            try:
                parsed_date = date.fromisoformat(depart_date)
            except (TypeError, ValueError):
                return self._invalid("date", "Use YYYY-MM-DD, for example 2026-10-07.")
            if not self._valid_airport(origin) or not self._valid_airport(destination) or origin == destination:
                return self._invalid("route", "Use two distinct three-letter uppercase airport codes.")
            results = [
                flight
                for flight in self._flights.values()
                if flight.origin == origin
                and flight.destination == destination
                and flight.depart_at.astimezone(LOCAL_TIMEZONE).date() == parsed_date
            ]
            results.sort(key=lambda item: (item.depart_at, item.flight_id))
            data = {"flights": [self._flight_data(flight) for flight in results]}
            return self._observation(ToolStatus.OK if results else ToolStatus.EMPTY, data=data)

    def check_seat(self, flight_id: str) -> ToolObservation:
        with self._lock:
            if fault := self._take_fault("check_seat"):
                return fault
            flight = self._flights.get(flight_id)
            if flight is None:
                return self._invalid("flight_id", "Use a flight_id returned by search_flights.")
            quote = self._quote(flight)
            return self._observation(
                ToolStatus.OK,
                data={"quote": quote.model_dump(mode="json"), "available_seats": sorted(self._seats[flight_id])},
            )

    def book_seat(
        self,
        request: BookingRequest,
        flight_id: str,
        seat: str,
        quoted_price_vnd: int,
        idempotency_key: str,
        approval: Approval | None,
    ) -> ToolObservation:
        if not isinstance(request, BookingRequest):
            return self._invalid("request", "Use a validated BookingRequest.")
        payload = self.booking_approval_payload(request, flight_id, seat, quoted_price_vnd, idempotency_key)
        with self._lock:
            if replay := self._replay_if_known(idempotency_key, payload):
                return replay
            if fault := self._take_fault("book_seat"):
                return fault
            if not approval_matches(approval, "book_seat", payload):
                return self._denied("approval_required", "Approve this exact booking payload before holding a seat.")
            flight = self._flights.get(flight_id)
            if flight is None or seat not in self._seats.get(flight_id, set()):
                return self._invalid("flight_id_or_seat", "Choose an available seat from check_seat.")
            if not flight_matches(request, flight):
                return self._denied("request_constraints_not_met", "The current flight no longer meets the request constraints.")
            if type(quoted_price_vnd) is not int or quoted_price_vnd != flight.price_vnd:
                return self._error("price_changed", "Check the current seat quote and request new approval.")
            self._seats[flight_id].remove(seat)
            self._flights[flight_id] = flight.model_copy(update={"seats_remaining": len(self._seats[flight_id])})
            self._booking_sequence += 1
            booking = Booking(
                booking_code=f"BK{self._booking_sequence:04d}",
                flight_id=flight_id,
                seat=seat,
                price_vnd=flight.price_vnd,
                status=BookingStatus.HELD,
                paid=False,
                idempotency_key=idempotency_key,
            )
            self._bookings[booking.booking_code] = booking
            self._booking_owner[booking.booking_code] = request.passenger_id
            response = self._observation(ToolStatus.OK, data={"booking": booking.model_dump(mode="json")})
            self._remember(idempotency_key, payload, response)
            return response

    def pay(
        self,
        request: BookingRequest,
        booking_code: str,
        idempotency_key: str,
        approval: Approval | None,
    ) -> ToolObservation:
        if not isinstance(request, BookingRequest):
            return self._invalid("request", "Use a validated BookingRequest.")
        with self._lock:
            booking = self._bookings.get(booking_code)
            if booking is None:
                return self._invalid("booking_code", "Use a booking_code returned by book_seat.")
            payload = self.payment_approval_payload(request, booking, idempotency_key)
            if replay := self._replay_if_known(idempotency_key, payload):
                return replay
            if fault := self._take_fault("pay"):
                return fault
            if self._booking_owner[booking_code] != request.passenger_id:
                return self._denied("booking_owner_mismatch", "The booking belongs to another mock passenger.")
            if not approval_matches(approval, "pay", payload):
                return self._denied("approval_required", "Approve this exact payment payload before paying.")
            if booking.status is not BookingStatus.HELD or booking.paid:
                return self._error("booking_not_payable", "Only an unpaid held booking can be paid.")
            flight = self._flights[booking.flight_id]
            if flight.price_vnd != booking.price_vnd:
                return self._error("price_changed", "The current price changed; do not pay this held booking.")
            confirmed = booking.model_copy(update={"status": BookingStatus.CONFIRMED, "paid": True})
            self._bookings[booking_code] = confirmed
            self._payments.append(
                PaymentRecord(
                    booking_code=booking_code,
                    passenger_id=request.passenger_id,
                    amount_vnd=booking.price_vnd,
                    idempotency_key=idempotency_key,
                )
            )
            response = self._observation(ToolStatus.OK, data={"booking": confirmed.model_dump(mode="json")})
            self._remember(idempotency_key, payload, response)
            return response

    def get_booking(self, booking_code: str) -> ToolObservation:
        with self._lock:
            if fault := self._take_fault("get_booking"):
                return fault
            booking = self._bookings.get(booking_code)
            if booking is None:
                return self._observation(ToolStatus.EMPTY, data={"booking": None})
            snapshot = booking.model_copy(update={"snapshot_source": "get_booking"})
            return self._observation(ToolStatus.OK, data={"booking": snapshot.model_dump(mode="json")})

    @staticmethod
    def booking_approval_payload(
        request: BookingRequest,
        flight_id: str,
        seat: str,
        quoted_price_vnd: int,
        idempotency_key: str,
    ) -> dict[str, object]:
        return {
            "passenger_id": request.passenger_id,
            "flight_id": flight_id,
            "seat": seat,
            "quoted_price_vnd": quoted_price_vnd,
            "idempotency_key": idempotency_key,
        }

    @staticmethod
    def payment_approval_payload(
        request: BookingRequest, booking: Booking, idempotency_key: str
    ) -> dict[str, object]:
        return {
            "passenger_id": request.passenger_id,
            "booking_code": booking.booking_code,
            "price_vnd": booking.price_vnd,
            "idempotency_key": idempotency_key,
        }

    def _quote(self, flight: Flight) -> SeatQuote:
        return SeatQuote(
            flight_id=flight.flight_id,
            origin=flight.origin,
            destination=flight.destination,
            depart_at=flight.depart_at,
            price_vnd=flight.price_vnd,
            seats_remaining=flight.seats_remaining,
            checked_at=self._now(),
        )

    def _take_fault(self, tool_name: str) -> ToolObservation | None:
        return self._faults[tool_name].pop(0) if self._faults[tool_name] else None

    def _replay_if_known(self, key: str, payload: dict[str, object]) -> ToolObservation | None:
        known = self._idempotency.get(key)
        if known is None:
            return None
        if known.payload == self._canonical_payload(payload):
            return known.response.model_copy(deep=True)
        return self._error("idempotency_conflict", "Use a new idempotency key for a different payload.")

    def _remember(self, key: str, payload: dict[str, object], response: ToolObservation) -> None:
        self._idempotency[key] = _IdempotencyRecord(self._canonical_payload(payload), response.model_copy(deep=True))

    @staticmethod
    def _canonical_payload(payload: dict[str, object]) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(payload.items()))

    @staticmethod
    def _valid_airport(value: object) -> bool:
        return isinstance(value, str) and len(value) == 3 and value.isascii() and value.isupper() and value.isalpha()

    @staticmethod
    def _flight_data(flight: Flight) -> dict[str, object]:
        return flight.model_dump(mode="json")

    @staticmethod
    def _observation(status: ToolStatus, *, data: object) -> ToolObservation:
        return ToolObservation(status=status, data=data)

    @staticmethod
    def _error(error_code: str, hint: str) -> ToolObservation:
        return ToolObservation(status=ToolStatus.ERROR, error_code=error_code, hint=hint)

    @staticmethod
    def _invalid(error_code: str, hint: str) -> ToolObservation:
        return ToolObservation(status=ToolStatus.INVALID_PARAM, error_code=error_code, hint=hint)

    @staticmethod
    def _denied(error_code: str, hint: str) -> ToolObservation:
        return ToolObservation(status=ToolStatus.DENIED, error_code=error_code, hint=hint)
