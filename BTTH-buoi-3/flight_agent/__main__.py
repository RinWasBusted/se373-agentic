"""Command line demo for the mock flight agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import Approval, BookingRequest
from .fixtures import default_environment
from .harness import HarnessSession
from .strategies import HybridStrategy, OpenAIFlightModel, PlanThenExecuteStrategy, ReActStrategy


STRATEGIES = {"react": ReActStrategy, "plan": PlanThenExecuteStrategy, "hybrid": HybridStrategy}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock flight-booking agent demo")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="react")
    parser.add_argument("--mode", choices=["fake", "openai"], default="fake")
    parser.add_argument("--request-file", required=True, type=Path)
    return parser.parse_args()


def _approval(session: HarnessSession) -> Approval | None:
    pending = session.pending_approval
    if pending is None:
        return None
    print(json.dumps({"approval_required": pending}, ensure_ascii=False, default=str))
    if not sys.stdin.isatty():
        return None
    answer = input("Duyệt hành động mock này? [yes/no]: ").strip().lower()
    return Approval(
        approved=answer in {"y", "yes"},
        action=pending["action"],
        payload_fingerprint=pending["payload_fingerprint"],
        reason="CLI approval" if answer in {"y", "yes"} else "CLI denial",
    )


def main() -> int:
    args = _parse_args()
    try:
        payload: dict[str, Any] = json.loads(args.request_file.read_text(encoding="utf-8"))
        request = BookingRequest.model_validate(payload)
        model = OpenAIFlightModel() if args.mode == "openai" else None
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    session = HarnessSession(request, default_environment(), run_id="cli-run")
    strategy = STRATEGIES[args.strategy](session, model=model)
    while True:
        run = strategy.run()
        if run.state == "awaiting_approval":
            approval = _approval(session)
            if approval is None:
                print(json.dumps({"status": "awaiting_approval", "handoff": session.pending_approval}, ensure_ascii=False, default=str))
                return 0
            strategy.approve(approval)
            continue
        result = run.result or session.terminal_result
        print(result.model_dump_json() if result else json.dumps({"status": "continue"}))
        return 0 if result and result.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
