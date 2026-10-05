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
