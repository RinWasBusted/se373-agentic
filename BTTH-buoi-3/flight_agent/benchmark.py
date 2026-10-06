"""Repeatable live-model benchmark for the three Phase 4 strategies."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Any

from .config import load_local_env
from .contracts import Approval, BookingRequest, BudgetConfig
from .fixtures import default_environment
from .harness import HarnessSession
from .strategies import GeminiFlightModel, HybridStrategy, OpenAIFlightModel, PlanThenExecuteStrategy, ReActStrategy


STRATEGIES = {"react": ReActStrategy, "plan": PlanThenExecuteStrategy, "hybrid": HybridStrategy}


def default_request() -> BookingRequest:
    return BookingRequest(
        origin="SGN", destination="DAD", depart_date="2026-10-07", depart_before="12:00", max_price_vnd=2_000_000, passenger_id="benchmark-passenger"
    )


def approve_pending(session: HarnessSession) -> Approval:
    pending = session.pending_approval
    assert pending is not None
    return Approval(
        approved=True,
        action=pending["action"],
        payload_fingerprint=pending["payload_fingerprint"],
        reason="benchmark mock approval",
    )


def run_once(
    strategy_name: str, *, provider: str = "openai", model_name: str | None = None, run_id: str = "benchmark"
) -> dict[str, Any]:
    model = OpenAIFlightModel(model_name) if provider == "openai" else GeminiFlightModel(model_name)
    budget = BudgetConfig(
        max_model_calls=20,
        max_tool_calls=20,
        max_seconds=180,
        repeat_threshold=3,
        stall_threshold=5,
        max_tokens=100_000,
    )
    session = HarnessSession(default_request(), default_environment(), budget=budget, run_id=run_id)
    strategy = STRATEGIES[strategy_name](session, model=model)
    started = time.perf_counter()
    while True:
        run = strategy.run()
        if run.state == "awaiting_approval":
            strategy.approve(approve_pending(session))
            continue
        result = run.result or session.terminal_result
        assert result is not None
        return {
            "strategy": strategy_name,
            "status": result.status.value,
            "model_calls": result.metrics["model_calls"],
            "tool_calls": result.metrics["tool_calls"],
            "tokens_used": result.metrics["tokens_used"],
            "cost_usd": result.metrics["cost_usd"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "handoff": result.handoff is not None,
            "handoff_question": result.handoff["question"] if result.handoff else None,
            "trace_decisions": [trace.decision for trace in result.traces],
        }


def main() -> int:
    load_local_env()
    parser = argparse.ArgumentParser(description="Benchmark the three flight-agent strategies with a live tool-calling model")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    records = [
        run_once(name, provider=args.provider, model_name=args.model, run_id=f"benchmark-{name}-{index}")
        for name in STRATEGIES
        for index in range(1, args.runs + 1)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["strategy"]].append(record)
    summary = {
        name: {
            "runs": len(items),
            "success_rate": sum(item["status"] == "success" for item in items) / len(items),
            "mean_model_calls": sum(item["model_calls"] for item in items) / len(items),
            "mean_tool_calls": sum(item["tool_calls"] for item in items) / len(items),
            "mean_latency_seconds": round(sum(item["latency_seconds"] for item in items) / len(items), 3),
            "statuses": [item["status"] for item in items],
        }
        for name, items in grouped.items()
    }
    default_model = "OPENAI_MODEL" if args.provider == "openai" else "gemini-3.8-flash"
    print(json.dumps({"provider": args.provider, "model": args.model or default_model, "records": records, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
