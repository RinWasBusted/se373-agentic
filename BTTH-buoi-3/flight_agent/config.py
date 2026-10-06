"""Local runtime configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from the ignored project ``.env`` file."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
