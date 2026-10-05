"""Deterministic scenario fixtures for offline tests and future agent demos."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .contracts import Flight
from .mock_environment import MockFlightEnvironment

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
FIXED_NOW = datetime(2026, 10, 1, 8, 0, tzinfo=TZ)


def default_environment() -> MockFlightEnvironment:
    flights = [
        Flight(flight_id="VN122", origin="SGN", destination="DAD", depart_at=datetime(2026, 10, 7, 8, 10, tzinfo=TZ), price_vnd=1_850_000, seats_remaining=2, refundable=False),
        Flight(flight_id="VJ604", origin="SGN", destination="DAD", depart_at=datetime(2026, 10, 7, 9, 0, tzinfo=TZ), price_vnd=1_700_000, seats_remaining=0, refundable=False),
        Flight(flight_id="VN134", origin="SGN", destination="DAD", depart_at=datetime(2026, 10, 7, 10, 30, tzinfo=TZ), price_vnd=2_100_000, seats_remaining=1, refundable=True),
        Flight(flight_id="QH118", origin="SGN", destination="DAD", depart_at=datetime(2026, 10, 7, 15, 40, tzinfo=TZ), price_vnd=1_640_000, seats_remaining=1, refundable=True),
    ]
    return MockFlightEnvironment(
        flights,
        {
            "VN122": ["12A", "12B"],
            "VJ604": [],
            "VN134": ["14A"],
            "QH118": ["10A"],
        },
        now=lambda: FIXED_NOW,
    )
