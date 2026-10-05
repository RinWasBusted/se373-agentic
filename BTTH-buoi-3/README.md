# BTVN#3 — Agent đặt vé máy bay

## Môi trường Phase 1

- Python: 3.14.7
- LangChain: 1.4.3
- LangGraph: 1.2.12
- langchain-openai: 1.6.7
- Pydantic: 2.13.5
- pytest: 9.1.1

To run Phase 1 tests:

```bash
.venv/bin/python -m pytest -q
```

All flight, booking, payment, and approval data in this project is mock data. The
OpenAI environment variables in `.env.example` are only used by the live demo
planned for Phase 4; tests do not need an API key.
