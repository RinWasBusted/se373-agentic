from __future__ import annotations

from flight_agent.contracts import Approval, BookingRequest, RunStatus
from flight_agent.fixtures import default_environment
from flight_agent.harness import HarnessSession
from flight_agent.strategies import HybridStrategy, PlanThenExecuteStrategy, ReActStrategy


def request() -> BookingRequest:
    return BookingRequest(
        origin="SGN", destination="DAD", depart_date="2026-10-07", depart_before="12:00", max_price_vnd=2_000_000, passenger_id="phase4-test"
    )


def approved(session: HarnessSession) -> Approval:
    pending = session.pending_approval
    assert pending is not None
    return Approval(
        approved=True,
        action=pending["action"],
        payload_fingerprint=pending["payload_fingerprint"],
        reason="test approval",
    )


def drive_to_terminal(strategy) -> RunStatus:
    while True:
        run = strategy.run()
        if run.state == "awaiting_approval":
            strategy.approve(approved(strategy.session))
            continue
        assert run.result is not None
        return run.result.status


def test_react_adapts_after_observation_and_verifies_success():
    session = HarnessSession(request(), default_environment(), run_id="react-test")
    strategy = ReActStrategy(session)
    assert drive_to_terminal(strategy) is RunStatus.SUCCESS
    actions = [item.action for item in session.traces if item.phase == "tool"]
    assert actions == ["search_flights", "check_seat", "check_seat", "book_seat", "pay", "get_booking"]


def test_plan_is_visible_before_tools_and_stops_when_first_plan_is_stale():
    session = HarnessSession(request(), default_environment(), run_id="plan-test")
    result = PlanThenExecuteStrategy(session).run()
    assert result.result is not None
    assert result.result.status is RunStatus.MODEL_ERROR
    decisions = [item.decision for item in session.traces]
    plan_index = next(index for index, item in enumerate(decisions) if item.startswith("plan_created_before_execution:"))
    assert plan_index < decisions.index("tool_dispatched")


def test_hybrid_replans_after_no_seat_and_then_completes():
    session = HarnessSession(request(), default_environment(), run_id="hybrid-test")
    strategy = HybridStrategy(session)
    assert drive_to_terminal(strategy) is RunStatus.SUCCESS
    assert any(item.decision == "plan_invalidated_replanning" for item in session.traces)


def test_strategy_never_bypasses_harness_approval():
    session = HarnessSession(request(), default_environment(), run_id="approval-test")
    strategy = ReActStrategy(session)
    run = strategy.run()
    assert run.state == "awaiting_approval"
    assert session.environment.booking_count == 0
    assert session.pending_approval["action"] == "book_seat"


def test_plan_model_turn_counts_toward_budget():
    from flight_agent.contracts import BudgetConfig

    budget = BudgetConfig(max_model_calls=1, max_tool_calls=20, max_seconds=30, repeat_threshold=3, stall_threshold=5)
    session = HarnessSession(request(), default_environment(), budget=budget, run_id="budget-test")
    result = PlanThenExecuteStrategy(session).run()
    assert result.result is not None
    assert result.result.status is RunStatus.BUDGET_EXCEEDED


def test_cli_fake_mode_noninteractive_stops_at_approval(tmp_path, monkeypatch, capsys):
    from flight_agent import __main__

    path = tmp_path / "request.json"
    path.write_text(
        '{"origin":"SGN","destination":"DAD","depart_date":"2026-10-07","depart_before":"12:00","max_price_vnd":2000000,"passenger_id":"phase4-test"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["flight_agent", "--strategy", "react", "--mode", "fake", "--request-file", str(path)])
    assert __main__.main() == 0
    assert '"status": "awaiting_approval"' in capsys.readouterr().out


def test_openai_mode_requires_environment(monkeypatch):
    from flight_agent.strategies import OpenAIFlightModel

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SE373_MODEL", raising=False)
    try:
        OpenAIFlightModel()
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)
    else:
        raise AssertionError("OpenAIFlightModel should reject missing configuration")
