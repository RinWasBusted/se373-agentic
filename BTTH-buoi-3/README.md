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
send proposed tool calls through the same harness. The CLI runs with a live
model provider; deterministic model doubles are reserved for unit tests.

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

Run a live Groq demo from this directory after setting `OPENAI_API_KEY`,
`OPENAI_BASE_URL` and `OPENAI_MODEL`:

```bash
.venv/bin/python -m flight_agent --strategy react --mode openai --request-file request.json
```

The command loads these values automatically from the ignored `.env` file in
the project directory. Environment variables already set in the shell take
precedence.

The CLI prints the exact mock payload and asks for separate approval before
`book_seat` and `pay`. With non-interactive stdin it returns
`awaiting_approval` and the pending payload without dispatching either action.

For Gemini, set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and optionally
`SE373_MODEL=gemini-3.8-flash`, then use `--mode gemini`. The model only
proposes tool calls; no live flight or payment API is used.

Run the live Phase 5 benchmark with the same mock fixture and approval behavior
for all three strategies:

```bash
GEMINI_API_KEY="..." .venv/bin/python -m flight_agent.benchmark --runs 3 --model gemini-3.5-flash
```

The command emits JSON records plus aggregate success rate, model/tool calls,
tokens and latency. Provider quota or availability errors are returned as a
handoff and must be reported as incomplete benchmark evidence.

For an OpenAI-compatible provider such as Groq, set `OPENAI_API_KEY`,
`OPENAI_BASE_URL` and `OPENAI_MODEL`, then run:

```bash
.venv/bin/python -m flight_agent.benchmark --provider openai --runs 3
```
