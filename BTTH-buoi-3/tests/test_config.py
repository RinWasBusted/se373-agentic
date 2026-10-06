from __future__ import annotations

import os

from flight_agent.config import load_local_env


def test_load_local_env_reads_values_without_overriding_shell(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FIRST=value\nSECOND='quoted value'\n", encoding="utf-8")
    monkeypatch.delenv("FIRST", raising=False)
    monkeypatch.setenv("SECOND", "shell value")
    load_local_env(env_file)
    assert os.environ["FIRST"] == "value"
    assert os.environ["SECOND"] == "shell value"
