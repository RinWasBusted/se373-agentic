"""Validated data contracts shared by tools, the harness, and agent strategies."""

from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    INVALID_PARAM = "invalid_param"
    ERROR = "error"
    DENIED = "denied"


class BookingStatus(StrEnum):
    HELD = "held"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    SUCCESS = "success"
    AWAITING_APPROVAL = "awaiting_approval"
    NO_FLIGHTS = "no_flights"
    NO_ELIGIBLE_FLIGHT = "no_eligible_flight"
    LOOP = "loop"
    STALL = "stall"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_ERROR = "tool_error"
    INVALID_OUTPUT = "invalid_output"
    DENIED = "denied"
    MODEL_ERROR = "model_error"
    USAGE_UNAVAILABLE = "usage_unavailable"


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BookingRequest(AgentModel):
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_date: date
    depart_before: time
    max_price_vnd: int = Field(gt=0)
    passenger_id: str = Field(min_length=1, max_length=64)

    @field_validator("depart_before", mode="before")
    @classmethod
    def require_hh_mm(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) == 5 and value[2] == ":":
            return value
        raise ValueError("depart_before must use the HH:MM format")

    @model_validator(mode="after")
    def require_different_airports(self) -> "BookingRequest":
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        return self


class Flight(AgentModel):
    flight_id: str = Field(min_length=1, max_length=24)
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_at: datetime
    price_vnd: int = Field(ge=0)
    seats_remaining: int = Field(ge=0)
    refundable: bool

    @field_validator("depart_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("depart_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_different_airports(self) -> "Flight":
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        return self


class SeatQuote(AgentModel):
    flight_id: str = Field(min_length=1, max_length=24)
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_at: datetime
    price_vnd: int = Field(ge=0)
    seats_remaining: int = Field(ge=0)
    checked_at: datetime

    @field_validator("depart_at", "checked_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value


class Booking(AgentModel):
    booking_code: str = Field(min_length=1, max_length=32)
    flight_id: str = Field(min_length=1, max_length=24)
    seat: str = Field(min_length=1, max_length=8)
    price_vnd: int = Field(ge=0)
    status: BookingStatus
    paid: bool
    idempotency_key: str = Field(min_length=1, max_length=128)
    snapshot_source: Literal["get_booking"] | None = None


class ToolObservation(AgentModel):
    status: ToolStatus
    data: Any | None = None
    error_code: str | None = None
    hint: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ToolObservation":
        if self.status in {ToolStatus.OK, ToolStatus.EMPTY}:
            if self.data is None:
                raise ValueError("ok and empty observations require data")
            if self.error_code is not None or self.hint is not None:
                raise ValueError("ok and empty observations cannot carry errors")
        elif not self.error_code or not self.hint:
            raise ValueError("error observations require error_code and hint")
        return self


class Approval(AgentModel):
    approved: bool
    action: Literal["book_seat", "pay"]
    payload_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1, max_length=500)


class BudgetConfig(AgentModel):
    max_model_calls: int = Field(default=12, gt=0)
    max_tool_calls: int = Field(default=20, gt=0)
    max_seconds: float = Field(default=30.0, gt=0)
    repeat_threshold: int = Field(default=3, gt=1)
    stall_threshold: int = Field(default=5, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class TraceEvent(AgentModel):
    sequence: int = Field(gt=0)
    phase: Literal["model", "policy", "tool", "termination", "handoff"]
    action: str | None = None
    decision: str
    observation: ToolObservation | None = None
    progress: int | float | str | None = None
    stop_reason: RunStatus | None = None


class ProposedToolCall(AgentModel):
    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(AgentModel):
    tool_call_id: str
    name: str
    observation: ToolObservation


class ModelUsage(AgentModel):
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class HarnessStep(AgentModel):
    state: Literal["continue", "awaiting_approval", "stopped"]
    results: list[ToolCallResult] = Field(default_factory=list)
    pending_tool_call_id: str | None = None
    message: str | None = None
    run_result: "RunResult | None" = None


class RunResult(AgentModel):
    status: RunStatus
    booking: Booking | None = None
    traces: list[TraceEvent] = Field(default_factory=list)
    handoff: dict[str, Any] | None = None
    metrics: dict[str, int | float | str | None] = Field(default_factory=dict)


HarnessStep.model_rebuild()
