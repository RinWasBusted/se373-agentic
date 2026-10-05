from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from flight_agent.contracts import (
    Approval,
    BookingRequest,
    Flight,
    ToolObservation,
    ToolStatus,
)

TZ = ZoneInfo("Asia/Ho_Chi_Minh")


@pytest.mark.parametrize(
    "payload",
    [
        {"origin": "SGN", "destination": "DAD", "depart_date": "2026-10-07", "depart_before": "12:00", "passenger_id": "p"},
        {"origin": "SG", "destination": "DAD", "depart_date": "2026-10-07", "depart_before": "12:00", "max_price_vnd": 1, "passenger_id": "p"},
        {"origin": "SGN", "destination": "SGN", "depart_date": "2026-10-07", "depart_before": "12:00", "max_price_vnd": 1, "passenger_id": "p"},
        {"origin": "SGN", "destination": "DAD", "depart_date": "07/10", "depart_before": "12:00", "max_price_vnd": 1, "passenger_id": "p"},
        {"origin": "SGN", "destination": "DAD", "depart_date": "2026-10-07", "depart_before": "12:00", "max_price_vnd": 0, "passenger_id": "p"},
        {"origin": "SGN", "destination": "DAD", "depart_date": "2026-10-07", "depart_before": "12h00", "max_price_vnd": 1, "passenger_id": "p"},
    ],
)
def test_t01_rejects_invalid_request(payload):
    with pytest.raises(ValidationError):
        BookingRequest.model_validate(payload)


def test_requires_timezone_on_flight():
    with pytest.raises(ValidationError):
        Flight(
            flight_id="VN122",
            origin="SGN",
            destination="DAD",
            depart_at=datetime(2026, 10, 7, 8, 10),
            price_vnd=1_850_000,
            seats_remaining=1,
            refundable=False,
        )


def test_observation_requires_meaningful_shape():
    with pytest.raises(ValidationError):
        ToolObservation.model_validate({})
    with pytest.raises(ValidationError):
        ToolObservation(status=ToolStatus.OK)
    with pytest.raises(ValidationError):
        ToolObservation(status=ToolStatus.ERROR, error_code="timeout")
    assert ToolObservation(status=ToolStatus.EMPTY, data=[]).status is ToolStatus.EMPTY


def test_approval_requires_sha256_fingerprint():
    with pytest.raises(ValidationError):
        Approval(approved=True, action="pay", payload_fingerprint="wrong", reason="ok")
