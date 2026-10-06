"""A deterministic harness that validates and dispatches model-proposed tool calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from .contracts import (
    Approval,
    Booking,
    BookingRequest,
    BookingStatus,
    BudgetConfig,
    Flight,
    HarnessStep,
    ModelUsage,
    ProposedToolCall,
    RunResult,
    RunStatus,
    SeatQuote,
    ToolCallResult,
    ToolObservation,
    ToolStatus,
    TraceEvent,
)
from .mock_environment import MockFlightEnvironment
from .policy import approval_matches, build_handoff, flight_matches, payload_fingerprint, verify_completion


@dataclass
class _PendingTurn:
    calls: list[ProposedToolCall]
    index: int
    results: list[ToolCallResult]
    action: str
    payload: dict[str, object]


class HarnessSession:
    """Stateful policy boundary between a model and ``MockFlightEnvironment``."""

    _TOOL_NAMES = {"search_flights", "check_seat", "book_seat", "pay", "get_booking"}

    def __init__(
        self,
        request: BookingRequest,
        environment: MockFlightEnvironment,
        *,
        budget: BudgetConfig | None = None,
        run_id: str = "run-1",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request = request
        self.environment = environment
        self.budget = budget or BudgetConfig()
        self.run_id = run_id
        self._clock = clock
        self._started_at = clock()
        self._traces: list[TraceEvent] = []
        self._flights: dict[str, Flight] = {}
        self._quotes: dict[str, SeatQuote] = {}
        self._available_seats: dict[str, set[str]] = {}
        self._bookings: dict[str, Booking] = {}
        self._snapshot: Booking | None = None
        self._pending: _PendingTurn | None = None
        self._terminal: RunResult | None = None
        self._seen_call_ids: set[str] = set()
        self._recent_actions: list[tuple[tuple[str, tuple[tuple[str, Any], ...]], int]] = []
        self._progress = 0
        self._stall_count = 0
        self._model_calls = 0
        self._tool_calls = 0
        self._tokens_used: int | None = 0
        self._cost_used: float | None = 0.0

    @property
    def traces(self) -> tuple[TraceEvent, ...]:
        return tuple(self._traces)

    @property
    def terminal_result(self) -> RunResult | None:
        return self._terminal

    @property
    def pending_approval(self) -> dict[str, object] | None:
        """Return a copy of the exact action awaiting human approval."""
        if self._pending is None:
            return None
        return {
            "action": self._pending.action,
            "payload": dict(self._pending.payload),
            "payload_fingerprint": payload_fingerprint(self._pending.payload),
            "tool_call_id": self._pending.calls[self._pending.index].tool_call_id,
        }

    def record_strategy_event(self, decision: str, *, action: str | None = None) -> None:
        """Record a visible planning decision without granting execution authority."""
        self._trace("model", action, decision)

    def submit_turn(
        self,
        tool_calls: Iterable[ProposedToolCall | dict[str, Any]],
        usage: ModelUsage | dict[str, Any] | None = None,
    ) -> HarnessStep:
        """Accept one model turn after its output has been generated."""
        if self._terminal:
            return self._stopped_step([])
        if self._pending:
            return self._awaiting_step([], "Approval is required before this session can continue.")
        if self._pre_model_limit():
            return self._stopped_step([])
        self._model_calls += 1
        self._trace("model", None, "model_turn_received")
        if self._record_usage(usage):
            return self._stopped_step([])
        try:
            calls = [ProposedToolCall.model_validate(call) for call in tool_calls]
        except ValidationError as error:
            self._terminate(RunStatus.MODEL_ERROR, question=f"Model returned an invalid tool call: {error.errors()[0]['msg']}")
            return self._stopped_step([])
        call_ids = [call.tool_call_id for call in calls]
        if len(call_ids) != len(set(call_ids)) or any(call_id in self._seen_call_ids for call_id in call_ids):
            self._terminate(RunStatus.MODEL_ERROR, question="Model reused a tool_call_id; provide a unique ID for each call.")
            return self._stopped_step([])
        self._seen_call_ids.update(call_ids)
        return self._execute_calls(calls, 0, [])

    def resume(self, approval: Approval) -> HarnessStep:
        """Resume the pending call only after matching approval is supplied."""
        if self._terminal:
            return self._stopped_step([])
        pending = self._pending
        if pending is None:
            return HarnessStep(state="continue", message="No action is awaiting approval.")
        if not approval.approved:
            results = pending.results + self._skipped_results(pending.calls[pending.index + 1 :])
            self._pending = None
            self._terminate(
                RunStatus.DENIED,
                pending_action=pending.action,
                question="The requested action was explicitly denied by the approver.",
            )
            return self._stopped_step(results)
        if not approval_matches(approval, pending.action, pending.payload):
            self._trace("policy", pending.action, "approval_does_not_match_pending_payload")
            return self._awaiting_step(pending.results, "Approval does not match the pending action and payload.")
        self._pending = None
        return self._execute_calls(pending.calls, pending.index, pending.results, approval)

    def model_failed(self, error: str) -> HarnessStep:
        """Record a provider/model failure without treating it as a successful answer."""
        if self._terminal:
            return self._stopped_step([])
        if self._pre_model_limit():
            return self._stopped_step([])
        self._model_calls += 1
        self._trace("model", None, "model_invocation_failed")
        self._terminate(RunStatus.MODEL_ERROR, question=f"Model invocation failed: {error}")
        return self._stopped_step([])

    def _execute_calls(
        self,
        calls: list[ProposedToolCall],
        start: int,
        results: list[ToolCallResult],
        approval: Approval | None = None,
    ) -> HarnessStep:
        for index in range(start, len(calls)):
            call = calls[index]
            outcome = self._prepare_call(call)
            if isinstance(outcome, ToolCallResult):
                results.append(outcome)
                if self._terminal:
                    results.extend(self._skipped_results(calls[index + 1 :]))
                    return self._stopped_step(results)
                continue
            if outcome is not None:
                action, payload = outcome
                if approval is None:
                    self._pending = _PendingTurn(calls, index, results, action, payload)
                    self._trace("policy", action, "approval_required_before_dispatch")
                    return self._awaiting_step(results, f"Approval is required for {action}.", call.tool_call_id)
                if not approval_matches(approval, action, payload):
                    self._pending = _PendingTurn(calls, index, results, action, payload)
                    self._trace("policy", action, "approval_does_not_match_pending_payload")
                    return self._awaiting_step(results, "Approval does not match the pending action and payload.", call.tool_call_id)
            result = self._dispatch(call, approval)
            results.append(result)
            if self._terminal:
                results.extend(self._skipped_results(calls[index + 1 :]))
                return self._stopped_step(results)
        return HarnessStep(state="continue", results=results)

    def _prepare_call(self, call: ProposedToolCall) -> ToolCallResult | tuple[str, dict[str, object]] | None:
        if call.name not in self._TOOL_NAMES:
            return self._invalid_result(call, "invalid_tool", "Use one of the five registered flight tools.")
        validation = self._validate_call_args(call)
        if validation is not None:
            return validation
        if call.name == "book_seat":
            quote = self._quotes[call.args["flight_id"]]
            payload = self.environment.booking_approval_payload(
                self.request,
                call.args["flight_id"],
                call.args["seat"],
                quote.price_vnd,
                self._idempotency_key(call),
            )
            return "book_seat", payload
        if call.name == "pay":
            booking = self._bookings[call.args["booking_code"]]
            payload = self.environment.payment_approval_payload(self.request, booking, self._idempotency_key(call))
            return "pay", payload
        return None

    def _validate_call_args(self, call: ProposedToolCall) -> ToolCallResult | None:
        expected: dict[str, type] = {
            "search_flights": {"origin": str, "destination": str, "depart_date": str},
            "check_seat": {"flight_id": str},
            "book_seat": {"flight_id": str, "seat": str},
            "pay": {"booking_code": str},
            "get_booking": {"booking_code": str},
        }[call.name]
        if set(call.args) != set(expected) or any(type(call.args[key]) is not value for key, value in expected.items()):
            return self._invalid_result(call, "invalid_param", "Tool arguments do not match the required schema.")
        if call.name == "search_flights":
            try:
                date.fromisoformat(call.args["depart_date"])
            except ValueError:
                return self._invalid_result(call, "invalid_param", "Use YYYY-MM-DD for depart_date.")
            required = {
                "origin": self.request.origin,
                "destination": self.request.destination,
                "depart_date": self.request.depart_date.isoformat(),
            }
            if call.args != required:
                return self._invalid_result(call, "request_constraints_not_met", "Search must use the original route and departure date.")
        elif call.name == "check_seat" and call.args["flight_id"] not in self._flights:
            return self._invalid_result(call, "unknown_flight", "Check only a flight returned by search_flights.")
        elif call.name == "book_seat":
            flight_id, seat = call.args["flight_id"], call.args["seat"]
            if flight_id not in self._quotes or seat not in self._available_seats.get(flight_id, set()):
                return self._invalid_result(call, "quote_required", "Check the selected flight and seat before booking.")
        elif call.name in {"pay", "get_booking"} and call.args["booking_code"] not in self._bookings:
            return self._invalid_result(call, "unknown_booking", "Use a booking_code observed in this session.")
        elif call.name == "pay":
            booking = self._bookings[call.args["booking_code"]]
            if booking.status is not BookingStatus.HELD or booking.paid:
                return self._invalid_result(call, "booking_not_payable", "Pay only an unpaid held booking.")
        return None

    def _dispatch(self, call: ProposedToolCall, approval: Approval | None) -> ToolCallResult:
        if self._pre_tool_limit():
            return self._skipped_result(call)
        self._tool_calls += 1
        response = self._call_environment(call, approval)
        observation = self._validate_observation(call.name, response)
        if observation is None:
            error_observation = self._error_observation("invalid_output", "Tool output failed schema validation.")
            self._trace(
                "tool",
                call.name,
                "tool_output_malformed_after_dispatch",
                observation=error_observation,
                progress=self._progress,
            )
            self._terminate(RunStatus.INVALID_OUTPUT, question=f"{call.name} returned malformed output.")
            return ToolCallResult(
                tool_call_id=call.tool_call_id,
                name=call.name,
                observation=error_observation,
            )
        self._trace("tool", call.name, "tool_dispatched", observation=observation, progress=self._progress)
        result = ToolCallResult(tool_call_id=call.tool_call_id, name=call.name, observation=observation)
        self._apply_observation(call, observation)
        if self._terminal:
            return result
        self._evaluate_after_observation(call, observation)
        return result

    def _call_environment(self, call: ProposedToolCall, approval: Approval | None) -> object:
        args = call.args
        if call.name == "search_flights":
            return self.environment.search_flights(**args)
        if call.name == "check_seat":
            return self.environment.check_seat(**args)
        if call.name == "book_seat":
            quote = self._quotes[args["flight_id"]]
            return self.environment.book_seat(
                self.request,
                args["flight_id"],
                args["seat"],
                quote.price_vnd,
                self._idempotency_key(call),
                approval,
            )
        if call.name == "pay":
            return self.environment.pay(self.request, args["booking_code"], self._idempotency_key(call), approval)
        return self.environment.get_booking(**args)

    def _validate_observation(self, tool_name: str, response: object) -> ToolObservation | None:
        if not isinstance(response, ToolObservation):
            return None
        try:
            observation = ToolObservation.model_validate(response)
            data = observation.data
            if observation.status in {ToolStatus.ERROR, ToolStatus.INVALID_PARAM, ToolStatus.DENIED}:
                return observation
            if tool_name == "search_flights":
                if not isinstance(data, dict) or not isinstance(data.get("flights"), list):
                    return None
                [Flight.model_validate(item) for item in data["flights"]]
            elif tool_name == "check_seat":
                if not isinstance(data, dict) or not isinstance(data.get("available_seats"), list):
                    return None
                SeatQuote.model_validate(data.get("quote"))
                if not all(isinstance(seat, str) for seat in data["available_seats"]):
                    return None
            elif tool_name in {"book_seat", "pay"}:
                if not isinstance(data, dict):
                    return None
                Booking.model_validate(data.get("booking"))
            elif tool_name == "get_booking":
                if not isinstance(data, dict) or "booking" not in data:
                    return None
                if observation.status is ToolStatus.OK:
                    Booking.model_validate(data["booking"])
                elif observation.status is ToolStatus.EMPTY and data["booking"] is not None:
                    return None
            return observation
        except (ValidationError, TypeError, ValueError):
            return None

    def _apply_observation(self, call: ProposedToolCall, observation: ToolObservation) -> None:
        if observation.status is not ToolStatus.OK:
            return
        data = observation.data
        if call.name == "search_flights":
            flights = [Flight.model_validate(item) for item in data["flights"]]
            self._flights = {flight.flight_id: flight for flight in flights}
        elif call.name == "check_seat":
            quote = SeatQuote.model_validate(data["quote"])
            self._quotes[quote.flight_id] = quote
            self._available_seats[quote.flight_id] = set(data["available_seats"])
        elif call.name in {"book_seat", "pay"}:
            booking = Booking.model_validate(data["booking"])
            self._bookings[booking.booking_code] = booking
        elif call.name == "get_booking" and data["booking"] is not None:
            self._snapshot = Booking.model_validate(data["booking"])
            self._bookings[self._snapshot.booking_code] = self._snapshot

    def _evaluate_after_observation(self, call: ProposedToolCall, observation: ToolObservation) -> None:
        if observation.status is ToolStatus.EMPTY and call.name == "search_flights":
            self._terminate(RunStatus.NO_FLIGHTS, question="No flights match the requested route and date.")
            return
        if observation.status is ToolStatus.ERROR:
            self._terminate(RunStatus.TOOL_ERROR, question=f"{call.name} failed: {observation.error_code}.")
            return
        if observation.status is ToolStatus.DENIED:
            self._terminate(RunStatus.DENIED, question=f"{call.name} was denied: {observation.error_code}.")
            return
        previous_progress = self._progress
        if observation.status is ToolStatus.OK:
            if call.name == "search_flights":
                if not any(flight_matches(self.request, flight) for flight in self._flights.values()):
                    self._terminate(RunStatus.NO_ELIGIBLE_FLIGHT, question="No returned flight satisfies all request constraints.")
                    return
                self._progress = max(self._progress, 1)
            elif call.name == "check_seat":
                quote = self._quotes[call.args["flight_id"]]
                if flight_matches(self.request, quote) and quote.seats_remaining > 0:
                    self._progress = max(self._progress, 2)
            elif call.name == "book_seat":
                self._progress = max(self._progress, 3)
            elif call.name == "pay":
                self._progress = max(self._progress, 4)
            elif call.name == "get_booking":
                snapshot = self._snapshot
                if snapshot and snapshot.status is BookingStatus.CONFIRMED and snapshot.paid:
                    self._progress = max(self._progress, 5)
                flight = self._flights.get(snapshot.flight_id) if snapshot else None
                quote = self._quotes.get(snapshot.flight_id) if snapshot else None
                if snapshot and verify_completion(self.request, quote, snapshot, flight):
                    self._terminate(RunStatus.SUCCESS)
                    return
        fingerprint = (call.name, tuple(sorted(call.args.items())))
        self._recent_actions.append((fingerprint, self._progress))
        self._recent_actions = self._recent_actions[-6:]
        occurrences = [progress for item, progress in self._recent_actions if item == fingerprint]
        if len(occurrences) >= self.budget.repeat_threshold and all(progress == self._progress for progress in occurrences):
            self._terminate(RunStatus.LOOP, question="The same tool call repeated without progress.")
            return
        self._stall_count = 0 if self._progress > previous_progress else self._stall_count + 1
        if self._stall_count >= self.budget.stall_threshold:
            self._terminate(RunStatus.STALL, question="Multiple tool calls completed without measurable progress.")

    def _pre_model_limit(self) -> bool:
        if self._time_exceeded():
            self._terminate(RunStatus.BUDGET_EXCEEDED, question="The time budget was exhausted before another model call.")
            return True
        if self._model_calls >= self.budget.max_model_calls:
            self._terminate(RunStatus.BUDGET_EXCEEDED, question="The model-call budget was exhausted.")
            return True
        return False

    def _pre_tool_limit(self) -> bool:
        if self._time_exceeded():
            self._terminate(RunStatus.BUDGET_EXCEEDED, question="The time budget was exhausted before another tool call.")
            return True
        if self._tool_calls >= self.budget.max_tool_calls:
            self._terminate(RunStatus.BUDGET_EXCEEDED, question="The tool-call budget was exhausted.")
            return True
        return False

    def _record_usage(self, usage: ModelUsage | dict[str, Any] | None) -> bool:
        try:
            parsed = ModelUsage.model_validate(usage) if usage is not None else None
        except ValidationError:
            self._terminate(RunStatus.USAGE_UNAVAILABLE, question="Model usage has an invalid schema.")
            return True
        if self.budget.max_tokens is not None:
            if parsed is None or parsed.total_tokens is None:
                self._terminate(RunStatus.USAGE_UNAVAILABLE, question="Token budget is configured but provider usage is unavailable.")
                return True
            assert self._tokens_used is not None
            self._tokens_used += parsed.total_tokens
            if self._tokens_used > self.budget.max_tokens:
                self._terminate(RunStatus.BUDGET_EXCEEDED, question="The token budget was exceeded.")
                return True
        if self.budget.max_cost_usd is not None:
            if parsed is None or parsed.cost_usd is None:
                self._terminate(RunStatus.USAGE_UNAVAILABLE, question="Cost budget is configured but provider cost is unavailable.")
                return True
            assert self._cost_used is not None
            self._cost_used += parsed.cost_usd
            if self._cost_used > self.budget.max_cost_usd:
                self._terminate(RunStatus.BUDGET_EXCEEDED, question="The cost budget was exceeded.")
                return True
        return False

    def _time_exceeded(self) -> bool:
        return self._clock() - self._started_at >= self.budget.max_seconds

    def _idempotency_key(self, call: ProposedToolCall) -> str:
        return f"{self.run_id}:{call.tool_call_id}"

    def _invalid_result(self, call: ProposedToolCall, error_code: str, hint: str) -> ToolCallResult:
        observation = ToolObservation(status=ToolStatus.INVALID_PARAM, error_code=error_code, hint=hint)
        self._trace("policy", call.name, error_code, observation=observation, progress=self._progress)
        self._evaluate_after_observation(call, observation)
        return ToolCallResult(tool_call_id=call.tool_call_id, name=call.name, observation=observation)

    def _skipped_result(self, call: ProposedToolCall) -> ToolCallResult:
        observation = ToolObservation(
            status=ToolStatus.DENIED,
            error_code="skipped_after_stop",
            hint="The harness stopped before this tool call could run.",
        )
        self._trace("policy", call.name, "skipped_after_stop", observation=observation, progress=self._progress)
        return ToolCallResult(tool_call_id=call.tool_call_id, name=call.name, observation=observation)

    def _skipped_results(self, calls: Iterable[ProposedToolCall]) -> list[ToolCallResult]:
        return [self._skipped_result(call) for call in calls]

    @staticmethod
    def _error_observation(error_code: str, hint: str) -> ToolObservation:
        return ToolObservation(status=ToolStatus.ERROR, error_code=error_code, hint=hint)

    def _trace(
        self,
        phase: str,
        action: str | None,
        decision: str,
        *,
        observation: ToolObservation | None = None,
        progress: int | None = None,
        stop_reason: RunStatus | None = None,
    ) -> None:
        self._traces.append(
            TraceEvent(
                sequence=len(self._traces) + 1,
                phase=phase,
                action=action,
                decision=decision,
                observation=observation,
                progress=self._progress if progress is None else progress,
                stop_reason=stop_reason,
            )
        )

    def _terminate(
        self,
        status: RunStatus,
        *,
        pending_action: str | None = None,
        question: str | None = None,
    ) -> None:
        if self._terminal is not None:
            return
        self._trace("termination", pending_action, "run_stopped", stop_reason=status)
        handoff = None
        if status is not RunStatus.SUCCESS:
            handoff = build_handoff(status, self._traces, pending_action=pending_action, question=question)
            self._trace("handoff", pending_action, "handoff_created", stop_reason=status)
        latest_code = next(reversed(self._bookings), None)
        self._terminal = RunResult(
            status=status,
            booking=self._snapshot or (self._bookings[latest_code] if latest_code else None),
            traces=list(self._traces),
            handoff=handoff,
            metrics={
                "model_calls": self._model_calls,
                "tool_calls": self._tool_calls,
                "tokens_used": self._tokens_used if self.budget.max_tokens is not None else "unknown",
                "cost_usd": self._cost_used if self.budget.max_cost_usd is not None else "unknown",
                "elapsed_seconds": self._clock() - self._started_at,
            },
        )

    def _awaiting_step(
        self, results: list[ToolCallResult], message: str, call_id: str | None = None
    ) -> HarnessStep:
        return HarnessStep(
            state="awaiting_approval",
            results=results,
            pending_tool_call_id=call_id or (self._pending.calls[self._pending.index].tool_call_id if self._pending else None),
            message=message,
        )

    def _stopped_step(self, results: list[ToolCallResult]) -> HarnessStep:
        return HarnessStep(state="stopped", results=results, run_result=self._terminal)
