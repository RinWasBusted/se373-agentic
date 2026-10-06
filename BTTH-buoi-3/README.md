# BTVN#3 — Agent đặt vé máy bay

## Môi trường

- Python: 3.14.7
- LangChain: 1.4.3
- LangGraph: 1.2.12
- langchain-openai: 1.6.7
- Pydantic: 2.13.5
- pytest: 9.1.1

To run the offline test suite:

```bash
.venv/bin/python -m pytest -q
```

All flight, booking, payment, and approval data in this project is mock data.
Phase 2 provides `MockFlightEnvironment` with `search_flights`, `check_seat`,
`book_seat`, `pay`, and `get_booking`. The OpenAI environment variables in
`.env.example` are only used by the live demo planned for Phase 4; tests do not
need an API key.

Phase 3 provides `HarnessSession`, which validates model-proposed tool calls,
enforces approval and budgets, records traces, and returns a verifiable terminal
result or handoff.

## Phase 4 demo

Phase 4 adds three LangGraph runners: `react`, `plan`, and `hybrid`. They all
send proposed tool calls through the same harness. The default `fake` mode is
deterministic and needs neither network access nor an API key.

Create a request file such as:

```json
{
  "origin": "SGN",
  "destination": "DAD",
  "depart_date": "2026-10-07",
  "depart_before": "12:00",
  "max_price_vnd": 2000000,
  "passenger_id": "student-01"
}
```

Run the local demo from this directory:

```bash
.venv/bin/python -m flight_agent --strategy hybrid --mode fake --request-file request.json
```

The CLI prints the exact mock payload and asks for separate approval before
`book_seat` and `pay`. With non-interactive stdin it returns
`awaiting_approval` and the pending payload without dispatching either action.

For an optional live model decision loop, set `OPENAI_API_KEY` and `SE373_MODEL`
from `.env.example`, then replace `--mode fake` with `--mode openai`. The model
only proposes tool calls; no live flight or payment API is used.
